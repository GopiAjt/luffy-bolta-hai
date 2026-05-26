"""Compose still images for vertical (9:16) slideshow frames."""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

logger = logging.getLogger(__name__)


def compose_vertical_subject(
    image: Image.Image,
    target_size: Tuple[int, int] = (1080, 1920),
    subject_scale: float = 0.72,
    blur_radius: int = 28,
) -> Image.Image:
    """
    Blurred full-bleed background + centered subject (faceless narration style).
    Works well for OPArchive character cards and mixed-aspect Fandom panels.
    """
    target_w, target_h = target_size
    base = image.convert("RGB")
    base = ImageOps.exif_transpose(base)

    cover = ImageOps.fit(base, (target_w, target_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    background = cover.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    background = ImageOps.autocontrast(background, cutoff=1)

    subject_max_w = int(target_w * subject_scale)
    subject_max_h = int(target_h * subject_scale)
    subject = ImageOps.contain(base, (subject_max_w, subject_max_h), method=Image.Resampling.LANCZOS)

    canvas = background.copy()
    x = (target_w - subject.width) // 2
    y = int(target_h * 0.42) - subject.height // 2
    y = max(int(target_h * 0.12), min(y, target_h - subject.height - int(target_h * 0.14)))
    canvas.paste(subject, (x, y))
    return canvas


def compose_vertical_subject_bgr(
    image_bgr: np.ndarray,
    target_size: Tuple[int, int] = (1080, 1920),
    subject_scale: float = 0.72,
    blur_radius: int = 28,
) -> np.ndarray:
    """BGR wrapper around compose_vertical_subject (EXIF handled in PIL path)."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("empty image for compose_vertical_subject_bgr")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    composed = compose_vertical_subject(
        Image.fromarray(rgb),
        target_size=target_size,
        subject_scale=subject_scale,
        blur_radius=blur_radius,
    )
    return cv2.cvtColor(np.asarray(composed), cv2.COLOR_RGB2BGR)
