"""Per-audio manual image uploads for image slides."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageOps
from werkzeug.datastructures import FileStorage

from app.config import IMAGE_SLIDES_DIR
from app.utils.expressions.expression_assets import suggest_vivre_assets, vivre_asset_path_from_relative

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
MAX_UPLOAD_BYTES = int(os.getenv("IMAGE_SLIDE_MAX_UPLOAD_MB", "12")) * 1024 * 1024


def audio_stem(audio_id: str) -> str:
    return Path(audio_id).stem


def slides_json_path_for_audio(audio_id: str, uploads_dir: Path) -> Optional[Path]:
    """Resolve `{stem}.image_slides.json` under uploads."""
    stem = audio_stem(audio_id)
    candidate = uploads_dir / f"{stem}.image_slides.json"
    if candidate.exists():
        return candidate
    for path in uploads_dir.glob("*.image_slides.json"):
        if path.stem.startswith(stem):
            return path
    return None


def slides_images_dir(audio_id: str) -> Path:
    """Directory where uploaded slide images for this audio are stored."""
    directory = IMAGE_SLIDES_DIR / audio_stem(audio_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def slide_image_basename(slide_index: int) -> str:
    return f"slide_{slide_index + 1:03d}.jpg"


def slide_image_path(audio_id: str, slide_index: int) -> Path:
    return slides_images_dir(audio_id) / slide_image_basename(slide_index)


def load_slides(slides_json_path: str) -> List[Dict]:
    with open(slides_json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_slides(slides_json_path: str, slides: List[Dict]) -> None:
    with open(slides_json_path, "w", encoding="utf-8") as handle:
        json.dump(slides, handle, indent=2, ensure_ascii=False)


def slides_upload_status(slides: List[Dict]) -> Dict:
    total = len(slides)
    uploaded = sum(1 for slide in slides if slide.get("image_path") and os.path.exists(slide["image_path"]))
    return {
        "total": total,
        "uploaded": uploaded,
        "complete": total > 0 and uploaded == total,
    }


def _validate_extension(filename: str) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported image type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    return ext


def _save_normalized_image(source_path: Path, dest_path: Path) -> None:
    """Convert uploads to JPEG for slideshow compatibility."""
    with Image.open(source_path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest_path, "JPEG", quality=92, optimize=True)


def save_slide_upload(
    audio_id: str,
    slide_index: int,
    file_storage: FileStorage,
    slides_json_path: str,
) -> Dict:
    """
    Save an uploaded image for one slide and update the slides JSON.

    Returns the updated slide dict.
    """
    if slide_index < 0:
        raise ValueError("slide_index must be >= 0")

    slides = load_slides(slides_json_path)
    if slide_index >= len(slides):
        raise ValueError(f"slide_index {slide_index} out of range (0-{len(slides) - 1})")

    if not file_storage or not file_storage.filename:
        raise ValueError("No image file provided")

    _validate_extension(file_storage.filename)

    raw_bytes = file_storage.read()
    if not raw_bytes:
        raise ValueError("Empty image file")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Image too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")

    dest_path = slide_image_path(audio_id, slide_index)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".upload")
    try:
        temp_path.write_bytes(raw_bytes)
        _save_normalized_image(temp_path, dest_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    slide = slides[slide_index]
    slide["image_path"] = str(dest_path.resolve())
    slide["image_source"] = "upload"
    save_slides(slides_json_path, slides)

    logger.info("Saved slide %s image for %s at %s", slide_index + 1, audio_id, dest_path)
    return slide


def apply_vivre_asset_to_slide(
    audio_id: str,
    slide_index: int,
    vivre_relative: str,
    slides_json_path: str,
) -> Dict:
    """Copy a Vivre Card PNG from the pack onto a slide slot (normalized to JPEG)."""
    source = vivre_asset_path_from_relative(vivre_relative)
    if not source:
        raise ValueError(f"Vivre asset not found: {vivre_relative}")

    slides = load_slides(slides_json_path)
    if slide_index < 0 or slide_index >= len(slides):
        raise ValueError(f"slide_index {slide_index} out of range")

    dest_path = slide_image_path(audio_id, slide_index)
    temp_path = dest_path.with_suffix(".vivre.png")
    try:
        shutil.copy2(source, temp_path)
        _save_normalized_image(temp_path, dest_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    slide = slides[slide_index]
    slide["image_path"] = str(dest_path.resolve())
    slide["image_source"] = "vivre_card"
    slide["vivre_relative"] = vivre_relative
    save_slides(slides_json_path, slides)
    logger.info(
        "Applied Vivre asset %s to slide %s for %s",
        vivre_relative,
        slide_index + 1,
        audio_id,
    )
    return slide


def build_slides_response(
    audio_id: str,
    slides_json_path: str,
    slides: List[Dict],
) -> Dict:
    """API-friendly payload with per-slide upload state and preview URLs."""
    status = slides_upload_status(slides)
    items = []
    for index, slide in enumerate(slides):
        image_path = slide.get("image_path")
        has_image = bool(image_path and os.path.exists(image_path))
        visual_source = slide.get("visual_source") or "asset_search"
        query_blob = f"{slide.get('image_search_query', '')} {slide.get('summary', '')}".strip()
        use_asset = visual_source != "ai_generate"
        vivre_suggestions = (
            suggest_vivre_assets(query_blob, limit=4) if use_asset and query_blob else []
        )

        items.append({
            "index": index,
            "start_time": slide.get("start_time"),
            "end_time": slide.get("end_time"),
            "summary": slide.get("summary"),
            "subtitle_text": slide.get("subtitle_text"),
            "image_search_query": slide.get("image_search_query"),
            "visual_source": visual_source,
            "ai_image_prompt": slide.get("ai_image_prompt") or "",
            "has_image": has_image,
            "image_path": image_path if has_image else None,
            "image_source": slide.get("image_source"),
            "preview_url": (
                f"/api/v1/image-slides/preview?audio_id={audio_id}&slide_index={index}"
                if has_image
                else None
            ),
            "vivre_suggestions": vivre_suggestions,
        })
    return {
        "audio_id": audio_id,
        "slides_json": slides_json_path,
        "images_dir": str(slides_images_dir(audio_id)),
        "slides": items,
        **status,
    }
