"""Compose still images for vertical (9:16) slideshow frames."""

from __future__ import annotations

import logging
from typing import Tuple

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
