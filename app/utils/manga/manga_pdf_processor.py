import json
import logging
import re
import shutil
import uuid
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import cv2
import fitz
import numpy as np
import requests

from app.config import MANGA_PDF_DIR, UPLOADS_DIR, VIDEO_RESOLUTION

logger = logging.getLogger(__name__)

PDF_MANIFEST = "manifest.json"
SESSION_MANIFEST = "session.json"
OHARA_BASE_URL = "https://thelibraryofohara.com"
MIN_USABLE_TEXT_SCORE = 0.45
_EASYOCR_READER = None

SCAN_CREDIT_PATTERNS = [
    r"\btcb\s*scans?\b",
    r"\btcbonepiecechapters\.com\b",
    r"\breall?cbscans\b",
    r"\bread on\b",
    r"\btwitter\b",
    r"\bscanlation\b",
    r"\bmanga_light\b",
]


def ass_format(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def parse_chapter_number(value: str) -> Optional[int]:
    """Extract a One Piece chapter number from a filename, title, or heading."""
    if not value:
        return None
    patterns = [
        r"chapters?\s*[-_:]?\s*(\d{3,5})",
        r"\bch(?:apter)?\.?\s*(\d{3,5})\b",
        r"\b(\d{3,5})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, value, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _strip_scan_credit_lines(text: str) -> str:
    lines = []
    for line in re.split(r"[\r\n]+|(?<=\.)\s{2,}", text or ""):
        cleaned = _clean_text(line)
        if not cleaned:
            continue
        if any(re.search(pattern, cleaned, re.IGNORECASE) for pattern in SCAN_CREDIT_PATTERNS):
            continue
        lines.append(cleaned)
    return _clean_text(" ".join(lines))


def score_text_quality(text: str, confidence: Optional[float] = None) -> Dict:
    """Score OCR usefulness so symbol soup never becomes trusted context."""
    cleaned = _strip_scan_credit_lines(text)
    chars = len(cleaned)
    alpha_chars = sum(1 for ch in cleaned if ch.isalpha())
    allowed_punct = set(".,!?;:'\"-()[]{}&/% ")
    garbage_chars = sum(1 for ch in cleaned if not ch.isalnum() and ch not in allowed_punct and not ch.isspace())
    words = re.findall(r"[A-Za-z][A-Za-z'-]{1,}", cleaned)
    tokens = [w.lower().strip("'") for w in words if w.strip("'")]

    alpha_ratio = alpha_chars / max(chars, 1)
    garbage_ratio = garbage_chars / max(chars, 1)
    word_count = len(words)
    avg_word_len = sum(len(w) for w in words) / max(word_count, 1)
    repeated_ratio = 0.0
    if tokens:
        repeated_ratio = max(tokens.count(token) for token in set(tokens)) / len(tokens)

    confidence_score = 0.5
    if confidence is not None:
        confidence_score = max(0.0, min(float(confidence) / 100.0, 1.0))

    score = 0.0
    score += min(alpha_ratio / 0.72, 1.0) * 0.25
    score += max(0.0, 1.0 - garbage_ratio / 0.18) * 0.20
    score += min(word_count / 45.0, 1.0) * 0.20
    score += max(0.0, 1.0 - repeated_ratio / 0.35) * 0.15
    score += max(0.0, min(avg_word_len / 4.0, 1.0)) * 0.08
    score += confidence_score * 0.12

    usable = score >= MIN_USABLE_TEXT_SCORE and word_count >= 8 and garbage_ratio < 0.22
    if score >= 0.72 and usable:
        level = "good"
    elif usable:
        level = "fair"
    else:
        level = "poor"

    return {
        "text": cleaned,
        "score": round(score, 3),
        "level": level,
        "usable": usable,
        "word_count": word_count,
        "char_count": chars,
        "alpha_ratio": round(alpha_ratio, 3),
        "garbage_ratio": round(garbage_ratio, 3),
        "repeated_ratio": round(repeated_ratio, 3),
        "confidence": round(float(confidence), 2) if confidence is not None else None,
    }


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


def _tesseract_ocr_page_text(page_image: np.ndarray) -> Tuple[str, Optional[float], Optional[str]]:
    try:
        import pytesseract
    except ImportError:
        return "", None, "pytesseract is not installed; OCR fallback skipped"

    try:
        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 50, 50)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        data = pytesseract.image_to_data(gray, lang="eng", output_type=pytesseract.Output.DICT)
        words = []
        confidences = []
        for raw_text, raw_conf in zip(data.get("text", []), data.get("conf", [])):
            token = _clean_text(raw_text)
            if not token:
                continue
            try:
                conf = float(raw_conf)
            except (TypeError, ValueError):
                conf = -1
            if conf >= 0:
                confidences.append(conf)
            words.append(token)
        confidence = sum(confidences) / len(confidences) if confidences else None
        return _clean_text(" ".join(words)), confidence, None
    except Exception as exc:
        return "", None, f"Tesseract OCR fallback failed: {exc}"


def _get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr
        logger.info("Loading EasyOCR reader for English manga OCR fallback")
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _EASYOCR_READER


def _easyocr_page_text(page_image: np.ndarray) -> Tuple[str, Optional[float], Optional[str]]:
    try:
        reader = _get_easyocr_reader()
        rgb = cv2.cvtColor(page_image, cv2.COLOR_BGR2RGB)
        results = reader.readtext(rgb, detail=1, paragraph=False)
        words = []
        confidences = []
        for _bbox, text, confidence in results:
            cleaned = _clean_text(text)
            if cleaned:
                words.append(cleaned)
                confidences.append(float(confidence) * 100.0)
        avg_confidence = sum(confidences) / len(confidences) if confidences else None
        return _clean_text(" ".join(words)), avg_confidence, None
    except ImportError:
        return "", None, "easyocr is not installed; EasyOCR fallback skipped"
    except Exception as exc:
        return "", None, f"EasyOCR fallback failed: {exc}"


def _select_best_text(candidates: List[Dict]) -> Dict:
    scored = []
    for candidate in candidates:
        quality = score_text_quality(candidate.get("text", ""), candidate.get("confidence"))
        scored.append({**candidate, "quality": quality, "text": quality["text"]})
    scored.sort(key=lambda item: item["quality"]["score"], reverse=True)
    if not scored:
        return {
            "text": "",
            "source": "none",
            "engine": "none",
            "confidence": None,
            "quality": score_text_quality(""),
        }
    best = scored[0]
    if not best["quality"]["usable"]:
        best["text"] = ""
    return best


def _chapter_matches_title(title: str, chapter_number: int) -> bool:
    if not title or not chapter_number:
        return False
    normalized = re.sub(r"\s+", " ", title)
    if re.search(rf"\bchapter\s+{chapter_number}\b", normalized, re.IGNORECASE):
        return True
    for start, end in re.findall(r"\b(\d{3,5})\s*[-–]\s*(\d{3,5})\b", normalized):
        if int(start) <= chapter_number <= int(end):
            return True
    return False


def _html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</p>|</h[1-6]>", "\n", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    return unescape(re.sub(r"[ \t]+", " ", text))


def _extract_ohara_chapter_section(article_text: str, chapter_number: int) -> str:
    lines = [_clean_text(line) for line in article_text.splitlines()]
    lines = [line for line in lines if line]
    start_index = None
    heading_pattern = re.compile(rf"^chapter\s+{chapter_number}\b", re.IGNORECASE)
    any_heading_pattern = re.compile(r"^chapter\s+\d{3,5}\b", re.IGNORECASE)

    for idx, line in enumerate(lines):
        if heading_pattern.search(line):
            start_index = idx
            break

    if start_index is None:
        joined = _clean_text(" ".join(lines))
        match = re.search(rf"(Chapter\s+{chapter_number}\b.*)", joined, re.IGNORECASE)
        return match.group(1)[:9000] if match else joined[:9000]

    section = []
    for line in lines[start_index:]:
        if section and any_heading_pattern.search(line):
            break
        if line.lower().startswith(("published by", "view all posts", "leave a reply")):
            break
        section.append(line)
    return _clean_text(" ".join(section))[:9000]


def fetch_ohara_context(chapter_number: Optional[int]) -> Tuple[Optional[Dict], Optional[str]]:
    if not chapter_number:
        return None, "No chapter number found for Ohara context lookup"

    try:
        search_url = f"{OHARA_BASE_URL}/?s={quote_plus(f'Chapter Secrets One Piece Chapter {chapter_number}')}"
        response = requests.get(search_url, timeout=12, headers={"User-Agent": "luffy-bolta-hai/1.0"})
        response.raise_for_status()
        search_html = response.text

        candidates = []
        link_pattern = re.compile(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
        for url, raw_title in link_pattern.findall(search_html):
            title = _clean_text(_html_to_text(raw_title))
            if "thelibraryofohara.com" not in url:
                continue
            if "chapter secrets" not in title.lower():
                continue
            if _chapter_matches_title(title, chapter_number):
                candidates.append((title, url))

        if not candidates:
            return None, f"No matching Library of Ohara article found for Chapter {chapter_number}"

        title, url = candidates[0]
        article_response = requests.get(url, timeout=12, headers={"User-Agent": "luffy-bolta-hai/1.0"})
        article_response.raise_for_status()
        article_text = _html_to_text(article_response.text)
        section = _extract_ohara_chapter_section(article_text, chapter_number)
        quality = score_text_quality(section, 95)
        if not section or quality["word_count"] < 80:
            return None, f"Library of Ohara context was too sparse for Chapter {chapter_number}"

        return {
            "source": "the_library_of_ohara",
            "title": title,
            "url": url,
            "chapter_number": chapter_number,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "text": section,
            "quality": quality,
        }, None
    except Exception as exc:
        return None, f"Library of Ohara context lookup failed: {exc}"


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
    raw_title = doc.metadata.get("title", "") or file_storage.filename
    chapter_number = parse_chapter_number(f"{raw_title} {file_storage.filename}")
    metadata = {
        "title": raw_title,
        "author": doc.metadata.get("author", ""),
        "page_count": len(doc),
        "file_size": source_path.stat().st_size,
    }

    page_records = []
    panel_records = []
    text_parts = []
    warnings = []
    page_quality_scores = []

    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        embedded_text = _clean_text(page.get_text("text"))
        pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        page_image = _enhance_panel(image)
        page_path = pages_dir / f"page_{page_number:03d}.jpg"
        cv2.imwrite(str(page_path), page_image, [cv2.IMWRITE_JPEG_QUALITY, 94])

        candidates = []
        if embedded_text:
            candidates.append({
                "text": embedded_text,
                "source": "embedded",
                "engine": "pymupdf",
                "confidence": 100.0,
            })

        needs_ocr = len(embedded_text) < 80 or not score_text_quality(embedded_text, 100)["usable"]
        if needs_ocr:
            tesseract_text, tesseract_confidence, warning = _tesseract_ocr_page_text(page_image)
            if warning:
                warnings.append(f"Page {page_number}: {warning}")
            if tesseract_text:
                candidates.append({
                    "text": tesseract_text,
                    "source": "ocr",
                    "engine": "tesseract",
                    "confidence": tesseract_confidence,
                })

        best_text = _select_best_text(candidates)
        if needs_ocr and best_text["quality"]["score"] < 0.62:
            easyocr_text, easyocr_confidence, warning = _easyocr_page_text(page_image)
            if warning:
                warnings.append(f"Page {page_number}: {warning}")
            if easyocr_text:
                candidates.append({
                    "text": easyocr_text,
                    "source": "ocr",
                    "engine": "easyocr",
                    "confidence": easyocr_confidence,
                })
                best_text = _select_best_text(candidates)

        page_text = best_text["text"]
        text_source = best_text["source"]
        ocr_engine = best_text["engine"]
        text_quality = best_text["quality"]
        page_quality_scores.append(text_quality["score"])

        if page_text and text_quality["usable"]:
            text_parts.append(f"Page {page_number}: {page_text}")
        elif best_text.get("quality", {}).get("char_count", 0):
            warnings.append(
                f"Page {page_number}: OCR text rejected as {text_quality['level']} quality "
                f"(score={text_quality['score']})"
            )

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
            "ocr_engine": ocr_engine,
            "ocr_confidence": text_quality["confidence"],
            "text_quality": text_quality,
            "panels": page_panel_records,
        })

    doc.close()

    extracted_text = "\n".join(text_parts).strip()
    usable_text_length = len(extracted_text)
    avg_quality_score = sum(page_quality_scores) / len(page_quality_scores) if page_quality_scores else 0.0
    usable_pages = sum(1 for page in page_records if page.get("text_quality", {}).get("usable"))
    text_quality = {
        "score": round(avg_quality_score, 3),
        "level": "good" if avg_quality_score >= 0.72 else "fair" if avg_quality_score >= MIN_USABLE_TEXT_SCORE else "poor",
        "usable": usable_text_length >= 350 and usable_pages >= max(2, len(page_records) // 5),
        "usable_pages": usable_pages,
        "total_pages": len(page_records),
    }

    ohara_context, ohara_warning = fetch_ohara_context(chapter_number)
    context_sources = []
    if ohara_context:
        context_sources.append({
            "source": ohara_context["source"],
            "title": ohara_context["title"],
            "url": ohara_context["url"],
            "chapter_number": ohara_context["chapter_number"],
            "fetched_at": ohara_context["fetched_at"],
            "quality": ohara_context["quality"],
        })
    elif ohara_warning:
        warnings.append(ohara_warning)

    manifest = {
        "pdf_id": pdf_id,
        "source_pdf": str(source_path),
        "metadata": metadata,
        "chapter_number": chapter_number,
        "text_quality": text_quality,
        "usable_text_length": usable_text_length,
        "extracted_text": extracted_text,
        "ohara_context": ohara_context,
        "context_sources": context_sources,
        "pages": page_records,
        "panels": panel_records,
        "warnings": warnings,
    }
    manifest_path = pdf_dir / PDF_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    update_manga_session(pdf_id, "uploaded", "completed", {
        "source_pdf": str(source_path),
        "chapter_number": chapter_number,
        "page_count": len(page_records),
        "panel_count": len(panel_records),
        "text_quality": text_quality,
        "context_sources": context_sources,
    })

    return {
        "pdf_id": pdf_id,
        "metadata": metadata,
        "chapter_number": chapter_number,
        "text_quality": text_quality,
        "usable_text_length": usable_text_length,
        "context_sources": context_sources,
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


def _manga_pdf_dir(pdf_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9-]{36}", pdf_id or ""):
        raise FileNotFoundError(f"Invalid PDF id: {pdf_id}")
    return MANGA_PDF_DIR / pdf_id


def _session_manifest_path(pdf_id: str) -> Path:
    return _manga_pdf_dir(pdf_id) / SESSION_MANIFEST


def load_manga_session(pdf_id: str) -> Dict:
    session_path = _session_manifest_path(pdf_id)
    now = datetime.now(timezone.utc).isoformat()
    if not session_path.exists():
        return {
            "pdf_id": pdf_id,
            "status": "created",
            "stage": "uploaded",
            "created_at": now,
            "updated_at": now,
            "steps": {},
        }
    return json.loads(session_path.read_text(encoding="utf-8"))


def save_manga_session(pdf_id: str, session: Dict) -> Dict:
    session_path = _session_manifest_path(pdf_id)
    session_path.parent.mkdir(parents=True, exist_ok=True)
    session["pdf_id"] = pdf_id
    session["updated_at"] = datetime.now(timezone.utc).isoformat()
    session_path.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
    return session


def update_manga_session(
    pdf_id: str,
    stage: str,
    status: str = "running",
    step_data: Optional[Dict] = None,
    error: Optional[str] = None,
) -> Dict:
    session = load_manga_session(pdf_id)
    session["stage"] = stage
    session["status"] = status
    if error:
        session["error"] = error
    else:
        session.pop("error", None)
    if step_data is not None:
        session.setdefault("steps", {})[stage] = {
            **step_data,
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    return save_manga_session(pdf_id, session)


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
