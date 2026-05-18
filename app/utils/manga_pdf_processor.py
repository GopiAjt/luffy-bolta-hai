import json
import logging
import re
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import fitz
import numpy as np

from app.config import MANGA_PDF_DIR, UPLOADS_DIR, VIDEO_RESOLUTION

logger = logging.getLogger(__name__)

PDF_MANIFEST = "manifest.json"


def ass_format(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def _enhance_panel(image: np.ndarray) -> np.ndarray:
    """Apply mild manga-friendly contrast and sharpening."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.12, blurred, -0.12, 0)


def _box_area(box: Tuple[int, int, int, int]) -> int:
    _x, _y, w, h = box
    return max(0, w) * max(0, h)


def _is_duplicate_box(box: Tuple[int, int, int, int], boxes: List[Tuple[int, int, int, int]]) -> bool:
    x, y, w, h = box
    area = _box_area(box)
    for bx, by, bw, bh in boxes:
        ix1 = max(x, bx)
        iy1 = max(y, by)
        ix2 = min(x + w, bx + bw)
        iy2 = min(y + h, by + bh)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        smaller = max(1, min(area, _box_area((bx, by, bw, bh))))
        if inter / smaller > 0.72:
            return True
    return False


def detect_panels(page_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect manga panels as large rectangular regions.

    This intentionally favors conservative crops. If detection is weak, callers
    should fallback to the full page image.
    """
    height, width = page_image.shape[:2]
    gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 45, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = width * height * 0.035
    max_area = width * height * 0.92
    boxes: List[Tuple[int, int, int, int]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_area or area > max_area:
            continue
        if w < width * 0.18 or h < height * 0.12:
            continue

        rect_area = max(1, w * h)
        contour_area = cv2.contourArea(contour)
        fill_ratio = contour_area / rect_area
        if fill_ratio < 0.22:
            continue

        pad = max(6, int(min(width, height) * 0.006))
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(width - x, w + 2 * pad)
        h = min(height - y, h + 2 * pad)
        box = (x, y, w, h)
        if not _is_duplicate_box(box, boxes):
            boxes.append(box)

    boxes.sort(key=lambda b: (b[1] // max(1, int(height * 0.08)), b[0]))
    return boxes[:12]


def _ocr_page_text(page_image: np.ndarray) -> Tuple[str, Optional[str]]:
    try:
        import pytesseract
    except ImportError:
        return "", "pytesseract is not installed; OCR fallback skipped"

    try:
        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return _clean_text(pytesseract.image_to_string(gray, lang="eng")), None
    except Exception as exc:
        return "", f"OCR fallback failed: {exc}"


def process_manga_pdf(file_storage) -> Dict:
    pdf_id = str(uuid.uuid4())
    pdf_dir = MANGA_PDF_DIR / pdf_id
    pages_dir = pdf_dir / "pages"
    panels_dir = pdf_dir / "panels"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    source_path = pdf_dir / "source.pdf"
    file_storage.save(str(source_path))

    doc = fitz.open(source_path)
    metadata = {
        "title": doc.metadata.get("title", "") or file_storage.filename,
        "author": doc.metadata.get("author", ""),
        "page_count": len(doc),
        "file_size": source_path.stat().st_size,
    }

    page_records = []
    panel_records = []
    text_parts = []
    warnings = []

    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        embedded_text = _clean_text(page.get_text("text"))
        pix = page.get_pixmap(matrix=fitz.Matrix(2.2, 2.2), alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        page_image = _enhance_panel(image)
        page_path = pages_dir / f"page_{page_number:03d}.jpg"
        cv2.imwrite(str(page_path), page_image, [cv2.IMWRITE_JPEG_QUALITY, 94])

        text_source = "embedded"
        page_text = embedded_text
        if len(page_text) < 40:
            ocr_text, warning = _ocr_page_text(page_image)
            if warning:
                warnings.append(f"Page {page_number}: {warning}")
            if ocr_text:
                page_text = ocr_text
                text_source = "ocr"

        if page_text:
            text_parts.append(f"Page {page_number}: {page_text}")

        panel_boxes = detect_panels(page_image)
        used_full_page = False
        if len(panel_boxes) < 2:
            h, w = page_image.shape[:2]
            panel_boxes = [(0, 0, w, h)]
            used_full_page = True

        page_panel_records = []
        for panel_index, (x, y, w, h) in enumerate(panel_boxes, start=1):
            panel = page_image[y:y + h, x:x + w]
            panel_path = panels_dir / f"page_{page_number:03d}_panel_{panel_index:02d}.jpg"
            cv2.imwrite(str(panel_path), panel, [cv2.IMWRITE_JPEG_QUALITY, 94])
            record = {
                "page": page_number,
                "panel": panel_index,
                "image_path": str(panel_path),
                "bbox": [x, y, w, h],
                "source": "full_page" if used_full_page else "panel",
            }
            panel_records.append(record)
            page_panel_records.append(record)

        page_records.append({
            "page": page_number,
            "image_path": str(page_path),
            "text": page_text,
            "text_source": text_source,
            "panels": page_panel_records,
        })

    doc.close()

    extracted_text = "\n".join(text_parts).strip()
    manifest = {
        "pdf_id": pdf_id,
        "source_pdf": str(source_path),
        "metadata": metadata,
        "extracted_text": extracted_text,
        "pages": page_records,
        "panels": panel_records,
        "warnings": warnings,
    }
    manifest_path = pdf_dir / PDF_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "pdf_id": pdf_id,
        "metadata": metadata,
        "text_preview": extracted_text[:1600],
        "text_length": len(extracted_text),
        "page_count": len(page_records),
        "panel_count": len(panel_records),
        "warnings": warnings,
    }


def load_manga_pdf_manifest(pdf_id: str) -> Dict:
    if not re.fullmatch(r"[a-f0-9-]{36}", pdf_id or ""):
        raise FileNotFoundError(f"Invalid PDF id: {pdf_id}")
    manifest_path = MANGA_PDF_DIR / pdf_id / PDF_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manga PDF not found: {pdf_id}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def create_pdf_slides(pdf_id: str, audio_duration: float, output_path: Optional[Path] = None) -> Dict:
    manifest = load_manga_pdf_manifest(pdf_id)
    panels = manifest.get("panels", [])
    if not panels:
        raise ValueError("No panels or fallback page images were extracted from this PDF")

    max_slides = max(1, int(audio_duration / 1.35))
    if len(panels) > max_slides:
        indices = np.linspace(0, len(panels) - 1, max_slides, dtype=int).tolist()
        panels = [panels[i] for i in indices]

    slide_count = len(panels)
    if output_path is None:
        output_path = UPLOADS_DIR / f"{pdf_id}.image_slides.json"

    slides = []
    current = 0.0
    for idx, panel in enumerate(panels):
        remaining_slides = slide_count - idx
        remaining_duration = max(0.1, audio_duration - current)
        duration = remaining_duration / remaining_slides
        start = current
        end = audio_duration if idx == slide_count - 1 else min(audio_duration, current + duration)
        current = end
        slides.append({
            "start_time": ass_format(start),
            "end_time": ass_format(end),
            "summary": f"Manga page {panel['page']} panel {panel['panel']}",
            "image_search_query": "",
            "image_path": panel["image_path"],
            "source": "manga_pdf",
            "page": panel["page"],
            "panel": panel["panel"],
            "panel_source": panel.get("source", "panel"),
        })

    output_path.write_text(json.dumps(slides, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "slides_json": str(output_path),
        "slide_count": len(slides),
        "slides": slides,
    }


def delete_manga_pdf(pdf_id: str) -> None:
    pdf_dir = MANGA_PDF_DIR / pdf_id
    if pdf_dir.exists():
        shutil.rmtree(pdf_dir)
