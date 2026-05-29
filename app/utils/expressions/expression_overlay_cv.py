"""Single-pass OpenCV compositor for expression overlays (replaces chunked FFmpeg)."""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.utils.expressions.expression_effects import EXPR_MAX_WIDTH, resolve_expression_effect

logger = logging.getLogger(__name__)

# Bottom margin (pixels) — matches expression_effects overlay_y (H-h-N)
_EFFECT_Y_FROM_BOTTOM = {
    "pop_in": 240,
    "bounce_in": 230,
    "shake_in": 220,
    "snap_in": 250,
    "fade_rise": 170,
    "slide_in": 210,
    "fade_soft": 190,
    "fade_scale": 220,
}


def _scale_factor(effect: str, t_ratio: float) -> float:
    t_ratio = max(0.0, min(1.0, t_ratio))
    if effect == "pop_in":
        return 0.72 + 0.28 * t_ratio
    if effect == "bounce_in":
        return 0.78 + 0.22 * t_ratio
    if effect == "shake_in":
        return 0.88 + 0.12 * t_ratio
    if effect == "snap_in":
        return 0.65 + 0.35 * t_ratio
    if effect == "slide_in":
        return 0.9 + 0.1 * t_ratio
    return 0.94 + 0.06 * t_ratio  # fade_scale / fade_rise / fade_soft base


def _hold_motion(local_t: float, fade_duration: float, enabled: bool) -> Tuple[float, int, int, float]:
    """Subtle motion for continuous expression holds after the entry animation."""
    if not enabled or local_t <= fade_duration:
        return 1.0, 0, 0, 1.0

    hold_t = local_t - fade_duration
    breath = math.sin(hold_t * math.tau * 0.55)
    drift = math.sin(hold_t * math.tau * 0.23)
    scale = 1.0 + 0.018 * breath
    x_offset = int(round(4 * drift))
    y_offset = int(round(5 * breath))
    alpha = 0.96 + 0.04 * (0.5 + 0.5 * math.sin(hold_t * math.tau * 0.37))
    return scale, x_offset, y_offset, alpha


def _alpha_for_time(
    local_t: float,
    interval_duration: float,
    fade_duration: float,
    effect: str,
) -> float:
    fade_in = max(0.08, min(0.32, float(fade_duration)))
    if local_t < 0 or local_t > interval_duration:
        return 0.0
    if effect in {"fade_rise", "fade_soft"}:
        if local_t < fade_in:
            return local_t / fade_in
        fade_out_start = max(0.0, interval_duration - fade_in)
        if local_t > fade_out_start:
            return max(0.0, (interval_duration - local_t) / fade_in)
        return 1.0
    # Animated-scale effects: full opacity during hold (matches FFmpeg path without alpha fade)
    if local_t < fade_in:
        return min(1.0, local_t / fade_in)
    return 1.0


def _load_rgba(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Expression image not found: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image


def _resize_rgba(image: np.ndarray, target_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return image
    if width == target_width:
        return image
    scale = target_width / width
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _composite_rgba(
    frame: np.ndarray,
    overlay: np.ndarray,
    x: int,
    y: int,
    alpha: float,
) -> None:
    if alpha <= 0.001:
        return
    fh, fw = frame.shape[:2]
    oh, ow = overlay.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + ow)
    y2 = min(fh, y + oh)
    if x1 >= x2 or y1 >= y2:
        return

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    roi = frame[y1:y2, x1:x2]
    patch = overlay[oy1:oy2, ox1:ox2]
    patch_alpha = (patch[:, :, 3:4].astype(np.float32) / 255.0) * alpha
    patch_rgb = patch[:, :, :3].astype(np.float32)
    roi[:] = (
        patch_rgb * patch_alpha + roi.astype(np.float32) * (1.0 - patch_alpha)
    ).astype(np.uint8)


def _active_specs(specs: List[Dict[str, Any]], t: float) -> List[Dict[str, Any]]:
    return [spec for spec in specs if spec["start"] <= t <= spec["end"]]


def render_expression_overlays_opencv(
    base_video_path: str,
    overlay_specs: List[Dict[str, Any]],
    output_path: str,
    duration: float,
    fps: int = 30,
    visual_style: Optional[str] = None,
) -> str:
    """
    Apply all expression overlays in one decode/encode pass.
    Much faster than sequential FFmpeg chunk re-encodes.
    """
    if not overlay_specs:
        return base_video_path

    cap = cv2.VideoCapture(base_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {base_video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions: {width}x{height}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    target_frames = max(1, int(math.ceil(duration * fps)))
    if total_frames > 0:
        target_frames = min(target_frames, total_frames)

    # Preload source images once per path
    source_cache: Dict[str, np.ndarray] = {}
    for spec in overlay_specs:
        path = spec["img_path"]
        if path not in source_cache:
            source_cache[path] = _load_rgba(path)

    fourcc = cv2.VideoWriter_fourcc(*os.getenv("OPENCV_EXPR_FOURCC", "mp4v"))
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV VideoWriter failed: {output_path}")

    logger.info(
        "OpenCV expression pass: %s overlays, %s frames @ %sfps (%sx%s)",
        len(overlay_specs),
        target_frames,
        fps,
        width,
        height,
    )

    resize_cache: Dict[Tuple[str, int], np.ndarray] = {}
    frame_idx = 0
    while frame_idx < target_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / float(fps)
        for spec in _active_specs(overlay_specs, t):
            effect = spec.get("effect") or resolve_expression_effect(
                spec["label"], visual_style
            )
            local_t = t - spec["start"]
            interval_duration = spec["interval_duration"]
            fade_duration = spec["fade_duration"]
            alpha = _alpha_for_time(local_t, interval_duration, fade_duration, effect)
            if alpha <= 0.001:
                continue

            t_ratio = min(1.0, local_t / fade_duration) if fade_duration > 0 else 1.0
            scale = _scale_factor(effect, t_ratio)
            hold_scale, hold_x, hold_y, hold_alpha = _hold_motion(
                local_t,
                fade_duration,
                bool(spec.get("continuous_hold")),
            )
            scale *= hold_scale
            alpha *= hold_alpha
            target_w = max(1, int(EXPR_MAX_WIDTH * scale))
            cache_key = (spec["img_path"], target_w)
            if cache_key not in resize_cache:
                resize_cache[cache_key] = _resize_rgba(source_cache[spec["img_path"]], target_w)

            overlay = resize_cache[cache_key]
            oh, ow = overlay.shape[:2]
            y_bottom = _EFFECT_Y_FROM_BOTTOM.get(effect, 220)
            x = (width - ow) // 2
            if effect == "shake_in":
                x += int(6 * math.sin(local_t * 24.0))
            x += hold_x
            y = height - oh - y_bottom
            y += hold_y
            _composite_rgba(frame, overlay, x, y, alpha)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 150 == 0:
            logger.info("OpenCV expression progress: %s/%s frames", frame_idx, target_frames)

    cap.release()
    writer.release()

    if frame_idx == 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("OpenCV expression overlay pass produced no output")

    logger.info("OpenCV expression pass complete: %s frames -> %s", frame_idx, output_path)
    return output_path
