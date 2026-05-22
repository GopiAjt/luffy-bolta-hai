"""Gemini API client for image slide JSON generation."""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GEMINI_API_BASE = os.getenv(
    "GEMINI_API_BASE",
    "https://generativelanguage.googleapis.com/v1beta",
).rstrip("/")
GEMINI_IMAGE_SLIDES_MODEL = os.getenv("GEMINI_IMAGE_SLIDES_MODEL", "gemini-3.1-flash-lite")
GEMINI_CONNECT_TIMEOUT = int(os.getenv("GEMINI_CONNECT_TIMEOUT", "15"))
GEMINI_READ_TIMEOUT = int(os.getenv("GEMINI_READ_TIMEOUT", "120"))


def _extract_gemini_text(payload: dict) -> str:
    candidates = payload.get("candidates") or []
    for candidate in candidates:
        parts = candidate.get("content", {}).get("parts") or []
        chunks = [part.get("text", "") for part in parts if part.get("text")]
        if chunks:
            return "".join(chunks).strip()
    prompt_feedback = payload.get("promptFeedback") or {}
    block_reason = prompt_feedback.get("blockReason")
    if block_reason:
        raise ValueError(f"Gemini blocked the prompt: {block_reason}")
    raise ValueError("Gemini returned no text candidates")


def _check_gemini_api_reachable() -> None:
    if os.getenv("GEMINI_SKIP_CONNECTIVITY_CHECK", "").lower() in {"1", "true", "yes"}:
        return
    if not GEMINI_API_KEY:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env for image slides")
    url = f"{GEMINI_API_BASE}/models"
    try:
        response = requests.get(
            url,
            params={"key": GEMINI_API_KEY, "pageSize": 1},
            timeout=(GEMINI_CONNECT_TIMEOUT, GEMINI_CONNECT_TIMEOUT),
        )
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise TimeoutError(
            "Timed out connecting to Gemini API (generativelanguage.googleapis.com). "
            "Check VPN/firewall/DNS or set HTTPS_PROXY if you are behind a proxy."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ConnectionError(
            f"Cannot reach Gemini API: {exc}. Verify GEMINI_API_KEY and outbound HTTPS access."
        ) from exc


def call_image_slides_llm(prompt: str, model_name: Optional[str] = None) -> str:
    """Generate slide JSON text via Gemini REST API."""
    if not GEMINI_API_KEY:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env for image slides")

    model_name = model_name or GEMINI_IMAGE_SLIDES_MODEL
    _check_gemini_api_reachable()

    url = f"{GEMINI_API_BASE}/models/{model_name}:generateContent"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
        },
    }
    logger.info(
        "Calling Gemini API (model=%s, connect_timeout=%ss, read_timeout=%ss)...",
        model_name,
        GEMINI_CONNECT_TIMEOUT,
        GEMINI_READ_TIMEOUT,
    )
    started = time.time()
    try:
        response = requests.post(
            url,
            params={"key": GEMINI_API_KEY},
            json=body,
            timeout=(GEMINI_CONNECT_TIMEOUT, GEMINI_READ_TIMEOUT),
        )
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        elapsed = time.time() - started
        raise TimeoutError(
            f"Gemini request timed out after {elapsed:.1f}s (model={model_name}). "
            "Try GEMINI_READ_TIMEOUT=180 or a faster model via GEMINI_IMAGE_SLIDES_MODEL."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        detail = ""
        if exc.response is not None:
            try:
                detail = exc.response.json().get("error", {}).get("message", "")
            except Exception:
                detail = (exc.response.text or "")[:300]
        raise RuntimeError(
            f"Gemini API HTTP {getattr(exc.response, 'status_code', '?')}: {detail or exc}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ConnectionError(f"Gemini API request failed: {exc}") from exc

    text = _extract_gemini_text(response.json())
    logger.info("Gemini API responded in %.1fs (%s chars)", time.time() - started, len(text))
    return text
