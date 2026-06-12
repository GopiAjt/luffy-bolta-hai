import os
import json
import traceback
from typing import Dict, List, Optional
import numpy as np
import cv2
from PIL import Image, ImageOps
import io
import logging
import subprocess # Added this import
import threading
import random
import math
from app.config import VIDEO_RESOLUTION
from app.utils.video.visual_effects import (
    classify_beat,
    choose_motion_preset,
    choose_visual_transition,
    get_visual_preset,
    normalize_visual_style,
)
from app.utils.images.image_composition import compose_vertical_subject_bgr
from app.utils.audio.transition_sfx import save_transition_events

logger = logging.getLogger(__name__)


def x264_speed_settings(quality_mode):
    """Return fast x264 settings appropriate for slideshow/anime imagery."""
    quality_mode = (quality_mode or 'standard').lower()
    if quality_mode == 'pro':
        return os.getenv("SLIDESHOW_X264_PRESET_PRO", "fast"), os.getenv("SLIDESHOW_X264_CRF_PRO", "22")
    return os.getenv("SLIDESHOW_X264_PRESET", "veryfast"), os.getenv("SLIDESHOW_X264_CRF", "23")


def slideshow_threads():
    """Return FFmpeg encoder thread count; 0 lets x264 auto-select."""
    return os.getenv("SLIDESHOW_FFMPEG_THREADS", os.getenv("FFMPEG_THREADS", "0"))


def slideshow_video_encoder():
    """Return the configured slideshow video encoder."""
    return os.getenv("SLIDESHOW_VIDEO_ENCODER", os.getenv("VIDEO_ENCODER", "libx264")).strip()


def slideshow_encoder_args(quality_mode):
    """Build encoder args for CPU x264 or opt-in hardware encoders."""
    encoder = slideshow_video_encoder()
    preset, crf = x264_speed_settings(quality_mode)
    threads = slideshow_threads()
    if encoder == "libx264":
        return [
            "-c:v", encoder,
            "-preset", preset,
            "-crf", crf,
            "-threads", threads,
        ]
    if encoder == "h264_qsv":
        return [
            "-c:v", encoder,
            "-preset", os.getenv("SLIDESHOW_QSV_PRESET", os.getenv("QSV_PRESET", "veryfast")),
            "-global_quality", os.getenv("SLIDESHOW_QSV_GLOBAL_QUALITY", os.getenv("QSV_GLOBAL_QUALITY", "23")),
        ]
    return ["-c:v", encoder]


class RawVideoFFmpegWriter:
    """Stream OpenCV BGR frames into FFmpeg without temporary JPG files."""

    def __init__(self, output_path, resolution, fps, quality_mode, total_frames=None):
        self.output_path = output_path
        self.resolution = sanitize_size(resolution)
        self.fps = int(fps)
        self.total_frames = total_frames
        self.frames_written = 0
        self._stderr_tail = []
        self._progress_thread = None
        res_w, res_h = self.resolution
        self.command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-nostats",
            "-progress", "pipe:2",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{res_w}x{res_h}",
            "-r", str(self.fps),
            "-i", "-",
            "-an",
            *slideshow_encoder_args(quality_mode),
            "-pix_fmt", "yuv420p",
            "-r", str(self.fps),
            "-movflags", "+faststart",
            output_path,
        ]
        logger.info(f"Starting FFmpeg raw frame writer: {' '.join(self.command)}")
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._progress_thread = threading.Thread(target=self._read_progress, daemon=True)
        self._progress_thread.start()

    def _read_progress(self):
        if self.process.stderr is None:
            return
        progress = {}
        for raw_line in self.process.stderr:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            self._stderr_tail.append(line)
            self._stderr_tail = self._stderr_tail[-80:]
            if "=" not in line:
                logger.debug(f"FFmpeg slideshow: {line}")
                continue
            key, value = line.split("=", 1)
            progress[key] = value
            if key == "progress":
                frame = int(progress.get("frame", "0") or 0)
                fps_value = progress.get("fps", "?")
                speed = progress.get("speed", "?")
                if self.total_frames:
                    percent = min(100.0, (frame / max(1, self.total_frames)) * 100.0)
                    remaining = max(0, self.total_frames - frame)
                    eta = "?"
                    try:
                        current_fps = float(fps_value)
                        if current_fps > 0:
                            eta = f"{remaining / current_fps:.0f}s"
                    except ValueError:
                        pass
                    logger.info(
                        "FFmpeg slideshow progress: frame=%s/%s %.1f%% fps=%s speed=%s eta=%s",
                        frame,
                        self.total_frames,
                        percent,
                        fps_value,
                        speed,
                        eta,
                    )
                else:
                    logger.info("FFmpeg slideshow progress: frame=%s fps=%s speed=%s", frame, fps_value, speed)
                progress = {}

    def write(self, frame):
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg stdin is not available")
        frame = ensure_frame_for_writer(frame, self.resolution)
        self.process.stdin.write(frame.tobytes())
        self.frames_written += 1
        if self.frames_written % max(1, self.fps * 10) == 0:
            logger.info(f"Streamed {self.frames_written} frames to FFmpeg")

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        return_code = self.process.wait()
        if self._progress_thread:
            self._progress_thread.join(timeout=2)
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, self.command, stderr="\n".join(self._stderr_tail))
        logger.info(f"FFmpeg raw frame writer finished after {self.frames_written} frames")


def ensure_frame_for_writer(frame, resolution):
    """Return a contiguous BGR uint8 frame at the output resolution."""
    frame = fit_frame_to_resolution(frame, resolution)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def load_slide_bgr(img_path: str) -> Optional[np.ndarray]:
    """Load a slide with EXIF orientation applied (PIL), returned as BGR uint8."""
    try:
        with Image.open(img_path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode == "RGBA":
                background = Image.new("RGB", pil_img.size, (255, 255, 255))
                background.paste(pil_img, mask=pil_img.split()[-1])
                pil_img = background
            elif pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            rgb = np.asarray(pil_img)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        logger.warning("PIL load failed for %s (%s), falling back to cv2.imread", img_path, exc)
        return cv2.imread(img_path)


def slide_aspect_ratio(img_bgr: np.ndarray) -> float:
    h, w = img_bgr.shape[:2]
    return w / max(h, 1)


def should_use_subject_layout(img_bgr: np.ndarray, resolution) -> bool:
    """
    Use blurred-background + centered subject for uploads that are not close
    to 9:16: tall cards, squares, and wide panels (Vivre, manga, screenshots).

    Safe contain is the default because generated/search images are mixed aspect
    ratios, and hard cover crops can push the useful subject outside the frame.
    """
    res_w, res_h = sanitize_size(resolution)
    aspect = slide_aspect_ratio(img_bgr)
    target_aspect = res_w / max(res_h, 1)
    safe_contain = os.getenv("SLIDE_SAFE_CONTAIN_MODE", "true").lower() not in {
        "0",
        "false",
        "no",
    }
    narrow_threshold = float(os.getenv("SLIDE_PORTRAIT_ASPECT_THRESHOLD", "0.72"))
    max_wide = float(os.getenv("SLIDE_SUBJECT_MAX_ASPECT", "2.1"))
    if safe_contain and abs(aspect - target_aspect) > 0.08:
        return True
    if aspect >= max_wide:
        return False
    if aspect <= narrow_threshold:
        return True
    if aspect > target_aspect * 1.05:
        return True
    return False


def transition_progress(frame_index: int, total_frames: int) -> float:
    """Map transition frame index to progress in [0, 1] (0 = current, 1 = next)."""
    if total_frames <= 1:
        return 1.0 if frame_index > 0 else 0.0
    return max(0.0, min(1.0, frame_index / (total_frames - 1)))

def parse_time(t):
    """Converts '0:00:06.28' to seconds."""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


# Default easing to apply across transition effects
DEFAULT_EASING = 'cosine'
PAN_EASING = 'smoothstep'

# Gamma used for approximate sRGB linearization
GAMMA = 2.2

def to_linear_bgr(img: np.ndarray) -> np.ndarray:
    """Convert uint8 BGR sRGB image to linear float32 [0,1]."""
    x = img.astype(np.float32) / 255.0
    # Simple gamma approximation
    return np.power(x, GAMMA)

def to_srgb_bgr(img_linear: np.ndarray) -> np.ndarray:
    """Convert linear float32 [0,1] to uint8 BGR sRGB image."""
    x = np.clip(img_linear, 0.0, 1.0)
    x = np.power(x, 1.0 / GAMMA)
    return (x * 255.0 + 0.5).astype(np.uint8)

def blend_linear_bgr(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Gamma-correct (linear space) blend between a and b with weight t (0..1)."""
    t = float(max(0.0, min(1.0, t)))
    la = to_linear_bgr(a)
    lb = to_linear_bgr(b)
    lc = la * (1.0 - t) + lb * t
    return to_srgb_bgr(lc)


def blend_mask_linear_bgr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Gamma-correct blend using a per-pixel mask where 0=a and 1=b."""
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if mask.ndim == 2:
        mask = mask[..., np.newaxis]
    la = to_linear_bgr(a)
    lb = to_linear_bgr(b)
    return to_srgb_bgr(la * (1.0 - mask) + lb * mask)


def sanitize_size(size):
    """Ensure size is a tuple of two ints (width, height)."""
    try:
        w, h = size[:2]
        return int(float(w)), int(float(h))
    except Exception as e:
        print(f"ERROR: Could not sanitize size: {size}, error: {e}")
        raise


def apply_blur(image, blur_amount):
    """Apply Gaussian blur to an image.
    
    Args:
        image: Input image as a numpy array
        blur_amount: Radius of the Gaussian blur (must be odd and positive)
        
    Returns:
        Blurred image as a numpy array
    """
    if blur_amount > 0:
        # Ensure blur_amount is odd and at least 1
        blur_amount = max(1, blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1  # Make it odd
        return cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)
    return image


def normalize_blur_amount(blur_amount, max_kernel=15):
    """Clamp blur to a practical odd kernel size."""
    blur_amount = max(0, int(blur_amount))
    if blur_amount <= 0:
        return 0
    blur_amount = min(blur_amount, max_kernel)
    return blur_amount if blur_amount % 2 == 1 else blur_amount + 1


def prepare_slide_canvas(img_path, resolution, blur_amount=0):
    """Load and scale a slide image once for panned frame rendering."""
    res_w, res_h = resolution
    img = load_slide_bgr(img_path)
    if img is None:
        logger.error(f"Failed to load image: {img_path}")
        return None, res_w

    if should_use_subject_layout(img, resolution):
        composed = compose_vertical_subject_bgr(img, (res_w, res_h))
        blur_amount = normalize_blur_amount(blur_amount, max_kernel=9)
        if blur_amount > 0:
            composed = apply_blur(composed, blur_amount)
        return composed, res_w

    img_h, img_w = img.shape[:2]
    scale = res_h / img_h
    scaled_w = max(1, int(round(img_w * scale)))
    scaled = cv2.resize(img, (scaled_w, res_h), interpolation=cv2.INTER_CUBIC)

    if scaled_w < res_w:
        pad = res_w - scaled_w
        left = pad // 2
        right = pad - left
        scaled = cv2.copyMakeBorder(
            scaled, 0, 0, left, right, cv2.BORDER_REPLICATE)
        scaled_w = res_w

    blur_amount = normalize_blur_amount(blur_amount, max_kernel=9)
    if blur_amount > 0:
        scaled = apply_blur(scaled, blur_amount)

    return scaled, scaled_w


def crop_panned_frame(scaled, scaled_w, resolution, idx, t, pan_strength=1.0):
    """Return a sub-pixel accurate pan crop for smoother motion."""
    res_w, res_h = resolution
    max_x = max(0.0, float(scaled_w - res_w))
    t = ease_progress(t, PAN_EASING)
    pan_strength = max(0.0, min(1.0, float(pan_strength)))

    if max_x > 0:
        travel = max_x * pan_strength
        start_x = (max_x - travel) / 2.0
        x = start_x + (t * travel if idx % 2 == 0 else (1.0 - t) * travel)
    else:
        x = 0.0

    center = (float(x + res_w / 2.0), float(res_h / 2.0))
    frame = cv2.getRectSubPix(scaled, (res_w, res_h), center)
    if frame is None or frame.size == 0:
        logger.warning(f"Empty panned crop generated at x={x:.2f}, using black frame")
        return np.zeros((res_h, res_w, 3), dtype=np.uint8)
    return frame


def crop_static_frame(scaled, resolution):
    """Return a centered frame with no Ken Burns or pan movement."""
    res_w, res_h = resolution
    height, width = scaled.shape[:2]
    center = (float(width / 2.0), float(height / 2.0))
    frame = cv2.getRectSubPix(scaled, (res_w, res_h), center)
    if frame is None or frame.size == 0:
        logger.warning("Empty static crop generated, using black frame")
        return np.zeros((res_h, res_w, 3), dtype=np.uint8)
    return frame


def _wrap_overlay_text(text: str, max_chars: int) -> List[str]:
    words = (text or "").strip().split()
    lines: List[str] = []
    current: List[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and len(candidate) > max_chars:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines[:3]


def _draw_label(
    out: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_scale: float,
    color=(238, 198, 88),
    thickness: int = 2,
) -> None:
    if not text:
        return
    cv2.putText(
        out,
        text.upper(),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_text_panel(frame: np.ndarray, text: str, resolution, mode: str) -> np.ndarray:
    """Draw production-grade title/evidence/quote overlays within safe bounds."""
    if not text:
        return frame
    res_w, res_h = resolution
    out = frame.copy()
    is_horizontal = res_w > res_h
    mode = (mode or "title_card").strip().lower()

    if is_horizontal and mode in {"horizontal_feature", "split_context"}:
        panel_w = int(res_w * 0.36)
        x0 = int(res_w * 0.055)
        y0 = int(res_h * 0.14)
        max_chars = 26
    elif is_horizontal and mode in {"evidence_card", "quote_card", "lower_third"}:
        panel_w = int(res_w * 0.44)
        x0 = int(res_w * 0.055)
        y0 = int(res_h * 0.60)
        max_chars = 30
    else:
        panel_w = int(res_w * (0.62 if is_horizontal else 0.86))
        x0 = (res_w - panel_w) // 2
        y0 = int(res_h * (0.06 if is_horizontal else 0.08))
        max_chars = 32 if is_horizontal else 24

    is_title = mode in {"title_card", "section_card"}
    is_quote = mode == "quote_card"
    display_text = text.upper() if is_title else f'"{text}"' if is_quote else text
    lines = _wrap_overlay_text(display_text, max_chars)
    if not lines:
        return out

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.76 if is_horizontal else 0.92
    if is_title:
        font_scale = 0.90 if is_horizontal else 1.04
    thickness = 2
    line_h = int(42 * font_scale) + 16
    panel_h = max(92, line_h * len(lines) + (58 if is_horizontal else 40))
    y0 = max(24, min(y0, res_h - panel_h - 32))

    overlay = out.copy()
    if is_title:
        cv2.rectangle(overlay, (0, 0), (res_w, int(res_h * (0.25 if is_horizontal else 0.19))), (8, 10, 16), -1)
        out = cv2.addWeighted(overlay, 0.48, out, 0.52, 0)
    else:
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (10, 12, 18), -1)
        out = cv2.addWeighted(overlay, 0.66, out, 0.34, 0)

    accent = (238, 198, 88)
    if mode == "evidence_card":
        accent = (96, 185, 255)
    elif mode == "quote_card":
        accent = (230, 230, 230)
    elif mode in {"horizontal_feature", "split_context"}:
        accent = (142, 220, 168)

    cv2.rectangle(out, (x0, y0), (x0 + panel_w, y0 + panel_h), accent, 2)
    if mode == "evidence_card":
        _draw_label(out, "Evidence", x0 + 22, y0 + 30, 0.54 if is_horizontal else 0.62, accent, 2)
    elif mode in {"section_card", "title_card"}:
        cv2.line(out, (x0 + 24, y0 + panel_h - 16), (x0 + panel_w - 24, y0 + panel_h - 16), accent, 3)
    elif mode == "quote_card":
        _draw_label(out, "Quote", x0 + 22, y0 + 30, 0.54 if is_horizontal else 0.62, accent, 2)

    y = y0 + (48 if mode in {"evidence_card", "quote_card"} else 36)
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        if mode in {"horizontal_feature", "split_context", "evidence_card", "quote_card"}:
            tx = x0 + 24
        else:
            tx = x0 + max(22, (panel_w - tw) // 2)
        cv2.putText(out, line, (tx, y + th), font, font_scale, (245, 245, 245), thickness, cv2.LINE_AA)
        y += line_h
    return out


def apply_production_overlay(frame: np.ndarray, slide_meta: Optional[Dict], resolution) -> np.ndarray:
    if not slide_meta:
        return frame
    text = (slide_meta.get("text_overlay") or "").strip()
    layout = (slide_meta.get("layout_mode") or "").strip().lower()
    role = (slide_meta.get("visual_role") or "").strip().lower()
    if not text:
        return frame
    mode = layout if layout else role
    return _draw_text_panel(frame, text, resolution, mode)


KEN_BURNS_ZOOM = {
    "slow_push": (1.0, 1.14),
    "impact_zoom": (1.06, 1.22),
    "hold_still": (1.0, 1.04),
    "pull_out": (1.16, 1.0),
    "stable_pan": (1.0, 1.10),
    "diagonal_pan": (1.0, 1.12),
    "subject_push": (1.0, 1.10),
    "evidence_hold": (1.0, 1.035),
    "reveal_zoom": (1.04, 1.18),
    "wide_pan": (1.0, 1.08),
    "title_card_hold": (1.0, 1.0),
}

KEN_BURNS_CENTER_LOCKED = {
    "slow_push",
    "impact_zoom",
    "hold_still",
    "pull_out",
    "subject_push",
    "evidence_hold",
    "reveal_zoom",
    "title_card_hold",
}
STATIC_HOLD_MOTIONS = {"static_hold", "still_frame", "no_ken_burns", "title_card_hold"}
FRAME_SAFE_TRANSITIONS = {
    "fade",
    "fade_eased",
    "crossfade",
    "zoom_dissolve",
    "iris_wipe",
    "radial_wipe",
}


def safe_transition_name(name: str) -> str:
    """Map harsh directional transitions to frame-safe equivalents."""
    key = (name or "crossfade").strip().lower()
    if key in FRAME_SAFE_TRANSITIONS or key == "none":
        return key
    if key.startswith("whip_pan") or key.startswith("motion_slide") or key.startswith("slide_"):
        return "crossfade"
    if key.startswith("cube_rotation"):
        return "iris_wipe"
    if key.startswith("page_curl") or key == "water_ripple":
        return "fade_eased"
    return key


def ken_burns_zoom_range(motion: str = "slow_push"):
    """Return sanitized start/end zoom for a Ken Burns motion preset."""
    z0, z1 = KEN_BURNS_ZOOM.get(motion or "slow_push", (1.0, 1.10))
    return max(0.01, float(z0)), max(0.01, float(z1))


def ken_burns_headroom_for_motion(motion: str = "slow_push", padding: float = 0.025) -> float:
    """Scale source only as much as the selected motion needs, plus small safety padding."""
    z0, z1 = ken_burns_zoom_range(motion)
    return max(1.0, z0, z1) + max(0.0, float(padding))


def ken_burns_enabled() -> bool:
    return os.getenv("ENABLE_KEN_BURNS", "true").lower() not in {"0", "false", "no"}


def motion_uses_ken_burns(motion: str) -> bool:
    return ken_burns_enabled() and (motion or "slow_push") not in STATIC_HOLD_MOTIONS


def prepare_ken_burns_canvas(img_path, resolution, blur_amount=0, headroom: float = None, motion: str = "slow_push"):
    """Cover-scale image with zoom headroom for Ken Burns (vertical Shorts)."""
    res_w, res_h = resolution
    img = load_slide_bgr(img_path)
    if img is None:
        logger.error(f"Failed to load image for Ken Burns: {img_path}")
        return None, res_w, res_h

    if should_use_subject_layout(img, resolution):
        img = compose_vertical_subject_bgr(img, (res_w, res_h))

    img_h, img_w = img.shape[:2]
    if headroom is None:
        headroom = ken_burns_headroom_for_motion(motion)
    cover_scale = max(res_w / max(img_w, 1), res_h / max(img_h, 1)) * headroom
    scaled_w = max(res_w, int(round(img_w * cover_scale)))
    scaled_h = max(res_h, int(round(img_h * cover_scale)))
    scaled = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)

    blur_amount = normalize_blur_amount(blur_amount, max_kernel=9)
    if blur_amount > 0:
        scaled = apply_blur(scaled, blur_amount)

    return scaled, scaled_w, scaled_h


def ken_burns_viewport(
    scaled_w,
    scaled_h,
    resolution,
    idx: int,
    t: float,
    motion: str = "slow_push",
    pan_strength: float = 1.0,
):
    """Return x, y, view_w, view_h for the current Ken Burns frame."""
    res_w, res_h = resolution
    t = ease_progress(t, PAN_EASING)
    motion = motion or "slow_push"
    z0, z1 = ken_burns_zoom_range(motion)
    zoom = z0 + (z1 - z0) * t

    view_w = min(float(scaled_w), res_w / max(zoom, 0.01))
    view_h = min(float(scaled_h), res_h / max(zoom, 0.01))
    max_x = max(0.0, float(scaled_w) - view_w)
    max_y = max(0.0, float(scaled_h) - view_h)
    pan_strength = max(0.0, min(1.0, float(pan_strength)))

    center_x = max_x / 2.0
    center_y = max_y / 2.0

    if motion in KEN_BURNS_CENTER_LOCKED:
        return center_x, center_y, view_w, view_h

    travel_x = max_x * pan_strength
    travel_y_factor = 0.45 if motion == "diagonal_pan" else 0.15
    travel_y = max_y * pan_strength * travel_y_factor

    base_x = (max_x - travel_x) / 2.0
    base_y = (max_y - travel_y) / 2.0

    if motion == "diagonal_pan":
        x = base_x + t * travel_x
        y = base_y + (1.0 - t) * travel_y
    elif motion == "stable_pan" and max_x > 0:
        x = base_x + (t * travel_x if idx % 2 == 0 else (1.0 - t) * travel_x)
        y = base_y
    else:
        x = base_x
        y = base_y

    x = max(0.0, min(x, max_x))
    y = max(0.0, min(y, max_y))
    return x, y, view_w, view_h


def crop_ken_burns_frame(
    scaled,
    scaled_w,
    scaled_h,
    resolution,
    idx: int,
    t: float,
    motion: str = "slow_push",
    pan_strength: float = 1.0,
):
    """Ken Burns: slow zoom with optional pan (faceless anime slideshow)."""
    res_w, res_h = resolution
    x, y, view_w, view_h = ken_burns_viewport(
        scaled_w,
        scaled_h,
        resolution,
        idx,
        t,
        motion=motion,
        pan_strength=pan_strength,
    )

    center = (float(x + view_w / 2.0), float(y + view_h / 2.0))
    frame = cv2.getRectSubPix(scaled, (int(round(view_w)), int(round(view_h))), center)
    if frame is None or frame.size == 0:
        return np.zeros((res_h, res_w, 3), dtype=np.uint8)
    if frame.shape[0] != res_h or frame.shape[1] != res_w:
        frame = cv2.resize(frame, (res_w, res_h), interpolation=cv2.INTER_CUBIC)
    return frame


def fade_effect(current_frame, next_frame, progress):
    """
    Simple fade effect that blends between two frames based on progress
        current_frame: Current frame as numpy array
        next_frame: Next frame as numpy array
        progress: Float between 0 and 1 representing transition progress
        
    Returns:
        Transitioned frame as numpy array
    """
    p = ease_progress(progress, DEFAULT_EASING)
    # Gamma-correct blend for better midtones
    return blend_linear_bgr(current_frame, next_frame, p)


def page_curl_effect(current_frame, next_frame, progress, corner='top-right'):
    """
    Creates a page curl effect from the specified corner.
    
    Args:
        current_frame: Current frame as numpy array (BGR)
        next_frame: Next frame as numpy array (BGR)
        progress: Float between 0 and 1
        corner: Corner from which to curl ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        
    Returns:
        Transitioned frame as numpy array
    """
    h, w = current_frame.shape[:2]
    result = np.zeros_like(current_frame, dtype=np.float32)
    
    # Convert to float for processing
    current = current_frame.astype(np.float32) / 255.0
    next_img = next_frame.astype(np.float32) / 255.0
    
    # Calculate curl parameters based on corner
    if corner == 'top-left':
        center = (0, 0)
        x_curve = lambda p: (1 - p) * w / 2
        y_curve = lambda p: (1 - p) * h / 2
    elif corner == 'top-right':
        center = (w, 0)
        x_curve = lambda p: w - (1 - p) * w / 2
        y_curve = lambda p: (1 - p) * h / 2
    elif corner == 'bottom-left':
        center = (0, h)
        x_curve = lambda p: (1 - p) * w / 2
        y_curve = lambda p: h - (1 - p) * h / 2
    else:  # bottom-right
        center = (w, h)
        x_curve = lambda p: w - (1 - p) * w / 2
        y_curve = lambda p: h - (1 - p) * h / 2
    
    # Create coordinate grids
    y, x = np.ogrid[:h, :w]
    
    # Calculate distance from corner
    dx = x - center[0]
    dy = y - center[1]
    distance = np.sqrt(dx*dx + dy*dy)
    max_dist = np.sqrt(w*w + h*h)
    
    # Calculate angle
    angle = np.arctan2(dy, dx)
    
    # Create curl mask
    curve_x = x_curve(progress)
    curve_y = y_curve(progress)
    curl_mask = ((x - center[0])**2 / (curve_x**2 + 1e-6) + 
                 (y - center[1])**2 / (curve_y**2 + 1e-6)) <= 1.0
    
    # Create shadow
    shadow = np.ones_like(current) * 0.3  # Darker shadow
    shadow_mask = (distance / max_dist) < (1 - progress * 0.8)
    
    # Apply the effect
    result = np.where(curl_mask[..., None], next_img, current)
    result = np.where(shadow_mask[..., None], result * shadow, result)
    
    # Add highlight on the curl
    highlight = np.ones_like(current) * 1.2
    highlight_mask = curl_mask & ((distance / max_dist) > (1 - progress * 0.5))
    result = np.where(highlight_mask[..., None], result * highlight, result)
    
    # Clamp values
    result = np.clip(result, 0, 1)
    
    return (result * 255).astype(np.uint8)


def water_ripple_effect(current_frame, next_frame, progress, center=None, amplitude=10, wavelength=30):
    """
    Creates a water ripple effect between two frames.
    
    Args:
        current_frame: Current frame as numpy array (BGR)
        next_frame: Next frame as numpy array (BGR)
        progress: Float between 0 and 1
        center: (x, y) coordinates for ripple center (None for random)
        amplitude: Maximum displacement of the ripple
        wavelength: Wavelength of the ripple
        
    Returns:
        Transitioned frame as numpy array
    """
    h, w = current_frame.shape[:2]
    
    # Use provided center or random if not specified
    if center is None:
        cx = random.randint(w//4, 3*w//4)
        cy = random.randint(h//4, 3*h//4)
        center = (cx, cy)
    
    # Create coordinate grids
    y, x = np.ogrid[:h, :w]
    
    # Calculate distance from center
    dx = x - center[0]
    dy = y - center[1]
    distance = np.sqrt(dx*dx + dy*dy)
    
    # Calculate ripple displacement
    ripple = amplitude * np.sin(2 * np.pi * distance / wavelength - progress * 2 * np.pi)
    
    # Create displacement maps
    scale = progress * ripple / (distance + 1e-6)  # Avoid division by zero
    map_x = x + dx * scale
    map_y = y + dy * scale
    
    # Ensure coordinates are within bounds
    map_x = np.clip(map_x, 0, w-1).astype(np.float32)
    map_y = np.clip(map_y, 0, h-1).astype(np.float32)
    
    # Remap the images
    current_remap = cv2.remap(current_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    next_remap = cv2.remap(next_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    # Blend based on progress
    p = ease_progress(progress, 'ease_in_out_quad')
    result = (1 - p) * current_remap.astype(np.float32) + p * next_remap.astype(np.float32)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def crossfade_effect(current_frame, next_frame, progress, blur_amount=5):
    """Crossfade with blur effect between frames.
    
    Args:
        current_frame: Current frame as numpy array
        next_frame: Next frame as numpy array
        progress: Float between 0 and 1 representing transition progress
        blur_amount: Amount of blur to apply during transition
        
    Returns:
        Transitioned frame as numpy array
    """
    # Log the crossfade progress
    logger.debug(f"Crossfade effect - Progress: {progress:.2f}, Blur amount: {blur_amount}")
    
    # Apply easing to progress for weight computation
    p = ease_progress(progress, DEFAULT_EASING)
    # Apply blur to both frames during transition
    if 0 < p < 1:
        # Ensure Gaussian kernel is a positive odd integer
        k = normalize_blur_amount(blur_amount, max_kernel=15)
        if k <= 0:
            k = 1
        current_blurred = cv2.GaussianBlur(current_frame, (k, k), 0)
        next_blurred = cv2.GaussianBlur(next_frame, (k, k), 0)
        # Gamma-correct blend
        result = blend_linear_bgr(current_blurred, next_blurred, p)
        logger.debug(f"Crossfade frame - Current weight: {1 - p:.2f}, Next weight: {p:.2f}")
        return result
        
    if p >= 1:
        logger.debug("Crossfade complete - Showing next frame")
        return next_frame
    else:
        logger.debug("Crossfade starting - Showing current frame")
        return current_frame


def apply_panoramic_pan(temp_frame_dir, frame_counter, img_path, duration, resolution, idx, fps=30, blur_amount=0, start_frame=0, pan_strength=1.0):
    """
    Fits the image height to the video height, then pans horizontally
    so that over the course of `duration` seconds, the window moves
    from one edge of the image to the other.

    - Even idx: pan left→right
    - Odd  idx: pan right→left
    
    Args:
        temp_frame_dir: Directory to save temporary frames
        frame_counter: Starting frame number
        img_path: Path to the source image
        duration: Duration to display the slide (in seconds)
        resolution: Tuple of (width, height) for output frames
        idx: Slide index (used to determine pan direction)
        fps: Frames per second
        blur_amount: Radius of the Gaussian blur (0 = no blur, higher = more blur)
        start_frame: Frame number to start from (used to skip frames after transition)
    """
    res_w, res_h = resolution
    num_frames = int(duration * fps)
    
    # Log the panning direction and blur status with more details
    direction = "left→right" if idx % 2 == 0 else "right→left"
    effective_blur = normalize_blur_amount(blur_amount, max_kernel=9)
    blur_status = f"with blur (kernel size: {effective_blur})" if effective_blur > 0 else "without blur"
    logger.info(f"Processing slide {os.path.basename(img_path)} with index {idx}, panning {direction} {blur_status}, pan_strength={pan_strength:.2f}")
    logger.debug(f"  - Start frame: {start_frame}, Total frames: {num_frames}, Duration: {duration:.2f}s")

    scaled, scaled_w = prepare_slide_canvas(img_path, resolution, effective_blur)
    if scaled is None:
        return frame_counter
    
    # 2. For each frame, compute horizontal crop window
    frames_actually_written = 0
    
    # Calculate the maximum possible x position (0 if image is narrower than video width)
    max_x = max(0, scaled_w - res_w)
    
    if start_frame >= num_frames:
        logger.warning(f"start_frame ({start_frame}) is greater than total frames ({num_frames}). Using last frame.")
        start_frame = max(0, num_frames - 1)
    
    # Generate frames starting from start_frame
    effective_frames = max(1, num_frames - start_frame)
    for i in range(start_frame, num_frames):
        # Calculate progress using absolute index so skipping frames advances t
        t = (i / (num_frames - 1)) if num_frames > 1 else 0.0
        # Clamp t to [0.0, 1.0]
        t = max(0.0, min(1.0, t))
        
        logger.debug(f"Frame {i}: t={t:.2f}, max_x={max_x}, direction={'left→right' if idx % 2 == 0 else 'right→left'}")
        frame = crop_panned_frame(scaled, scaled_w, resolution, idx, t, pan_strength=pan_strength)
        
        frame_path = os.path.join(temp_frame_dir, f"frame_{frame_counter:05d}.jpg")
        try:
            success = cv2.imwrite(frame_path, frame)
            if not success:
                logger.error(f"Failed to write frame {frame_path}")
                # Skip this frame but continue with the rest
                continue
            frame_counter += 1
            frames_actually_written += 1
        except Exception as e:
            logger.error(f"Error writing frame {frame_path}: {str(e)}")
            continue
    
    return frame_counter # Return updated frame_counter


def emit_panoramic_pan(
    frame_writer,
    frame_counter,
    img_path,
    duration,
    resolution,
    idx,
    fps=30,
    blur_amount=0,
    start_frame=0,
    pan_strength=1.0,
    motion: str = "slow_push",
    slide_meta: Optional[Dict] = None,
):
    """Render slide frames with Ken Burns (default) or legacy horizontal pan."""
    num_frames = int(duration * fps)
    logger.debug(
        "emit_panoramic_pan: %s, frames=%s, motion=%s, ken_burns=%s",
        img_path,
        num_frames,
        motion,
        ken_burns_enabled(),
    )
    if num_frames <= 0:
        logger.warning(f"Skipping slide with non-positive frame count: {img_path}")
        return frame_counter, None

    if slide_meta and slide_meta.get("composition"):
        from app.utils.video.frame_compositor import render_composed_frame
        img_bgr = load_slide_bgr(img_path)
        if img_bgr is None:
            logger.error(f"Could not prepare slide canvas for {img_path}")
            return frame_counter, None
            
        if start_frame >= num_frames:
            start_frame = max(0, num_frames - 1)
            
        last_frame = None
        for i in range(start_frame, num_frames):
            t = (i / (num_frames - 1)) if num_frames > 1 else 0.0
            frame = render_composed_frame(img_bgr, slide_meta, resolution, idx, t)
            frame = apply_production_overlay(frame, slide_meta, resolution)
            frame_writer.write(frame)
            frame_counter += 1
            last_frame = frame
            
        return frame_counter, last_frame

    use_kb = motion_uses_ken_burns(motion)
    if use_kb:
        scaled, scaled_w, scaled_h = prepare_ken_burns_canvas(
            img_path, resolution, blur_amount=blur_amount, motion=motion
        )
    else:
        scaled, scaled_w = prepare_slide_canvas(img_path, resolution, blur_amount=blur_amount)
        scaled_h = resolution[1]

    if scaled is None:
        logger.error(f"Could not prepare slide canvas for {img_path}")
        return frame_counter, None

    if start_frame >= num_frames:
        logger.warning(f"start_frame ({start_frame}) is greater than total frames ({num_frames}). Using last frame.")
        start_frame = max(0, num_frames - 1)

    last_frame = None
    for i in range(start_frame, num_frames):
        t = (i / (num_frames - 1)) if num_frames > 1 else 0.0
        if use_kb:
            frame = crop_ken_burns_frame(
                scaled,
                scaled_w,
                scaled_h,
                resolution,
                idx,
                t,
                motion=motion,
                pan_strength=pan_strength,
            )
        elif (motion or "") in STATIC_HOLD_MOTIONS:
            frame = crop_static_frame(scaled, resolution)
        else:
            frame = crop_panned_frame(scaled, scaled_w, resolution, idx, t, pan_strength=pan_strength)
        frame = apply_production_overlay(frame, slide_meta, resolution)
        frame_writer.write(frame)
        frame_counter += 1
        last_frame = frame

    return frame_counter, last_frame


def render_panned_frame(
    img_path,
    resolution,
    idx,
    t,
    blur_amount=0,
    pan_strength=1.0,
    motion: str = "slow_push",
    slide_meta: Optional[Dict] = None,
):
    """Render a single frame at pan progress t (matches emit_panoramic_pan)."""
    res_w, res_h = resolution
    if slide_meta and slide_meta.get("composition"):
        from app.utils.video.frame_compositor import render_composed_frame
        img_bgr = load_slide_bgr(img_path)
        if img_bgr is None:
            logger.warning(f"render_panned_frame: Failed to load {img_path}, using black frame")
            return np.zeros((res_h, res_w, 3), dtype=np.uint8)
        frame = render_composed_frame(img_bgr, slide_meta, resolution, idx, t)
        return apply_production_overlay(frame, slide_meta, resolution)

    if motion_uses_ken_burns(motion):
        scaled, scaled_w, scaled_h = prepare_ken_burns_canvas(
            img_path, resolution, blur_amount, motion=motion
        )
        if scaled is None:
            logger.warning(f"render_panned_frame: Failed to load {img_path}, using black frame")
            return np.zeros((res_h, res_w, 3), dtype=np.uint8)
        frame = crop_ken_burns_frame(
            scaled, scaled_w, scaled_h, resolution, idx, t, motion=motion, pan_strength=pan_strength
        )
        return apply_production_overlay(frame, slide_meta, resolution)

    scaled, scaled_w = prepare_slide_canvas(img_path, resolution, blur_amount)
    if scaled is None:
        logger.warning(f"render_panned_frame: Failed to load {img_path}, using black frame")
        return np.zeros((res_h, res_w, 3), dtype=np.uint8)
    if (motion or "") in STATIC_HOLD_MOTIONS:
        frame = crop_static_frame(scaled, resolution)
    else:
        frame = crop_panned_frame(scaled, scaled_w, resolution, idx, t, pan_strength=pan_strength)
    return apply_production_overlay(frame, slide_meta, resolution)


def weighted_choice(weighted_items):
    """Pick an item from a list of (item, weight) tuples."""
    total = sum(w for _, w in weighted_items)
    if total <= 0:
        # fallback to uniform choice over items
        items = [it for it, _ in weighted_items]
        return random.choice(items)
    r = random.uniform(0, total)
    upto = 0
    for item, weight in weighted_items:
        upto += weight
        if upto >= r:
            return item
    # Fallback (floating point edge)
    return weighted_items[-1][0]


LEFT_DIR_NAMES = {"slide_left", "motion_slide_left", "whip_pan_left"}
RIGHT_DIR_NAMES = {"slide_right", "motion_slide_right", "whip_pan_right"}


def choose_transition(slides_processed, next_slide_duration, last_selected_transition, avoid_same_direction_horizontal=True):
    """Heuristic to choose a transition effect.
    
    Args:
        slides_processed: Number of slides processed so far
        next_slide_duration: Duration of the next slide in seconds
        last_selected_transition: Last used transition effect
        avoid_same_direction_horizontal: If True, avoid same horizontal direction transitions
        
    Returns:
        Name of the selected transition effect
    """
    # Adjust weights based on next slide duration
    if next_slide_duration < 2.0:
        # For very short slides, prefer softer transitions
        soft_mix = [
            ("fade_eased", 0.38),
            ("crossfade", 0.32),
            ("fade", 0.18),
            ("iris_wipe", 0.12),
        ]
    elif next_slide_duration < 4.0:
        # For medium duration slides, mix in more dynamic transitions
        soft_mix = [
            ("crossfade", 0.34),
            ("fade_eased", 0.28),
            ("zoom_dissolve", 0.18),
            ("iris_wipe", 0.12),
            ("radial_wipe", 0.08),
        ]
    else:
        # For longer slides, use more dynamic transitions
        soft_mix = [
            ("crossfade", 0.30),
            ("fade_eased", 0.24),
            ("zoom_dissolve", 0.20),
            ("iris_wipe", 0.14),
            ("radial_wipe", 0.12),
        ]

    candidates = soft_mix
    
    # Avoid repeating the same transition
    if last_selected_transition:
        candidates = [(t, w) for t, w in candidates if t != last_selected_transition]
    
    # Avoid same direction horizontal transitions if requested
    if avoid_same_direction_horizontal and last_selected_transition:
        if 'right' in last_selected_transition:
            candidates = [(t, w) for t, w in candidates if 'right' not in t]
        elif 'left' in last_selected_transition:
            candidates = [(t, w) for t, w in candidates if 'left' not in t]
    
    # Normalize weights
    total_weight = sum(w for _, w in candidates)
    if total_weight > 0:
        candidates = [(t, w/total_weight) for t, w in candidates]
    
    # Select a transition
    if not candidates:
        return "crossfade"  # Fallback
        
    return weighted_choice(candidates)


def choose_slide_motion(
    visual_style: Optional[str],
    beat: str,
    index: int,
    duration: float,
    previous_motion: Optional[str] = None,
) -> str:
    """Randomized, beat-aware slide motion. Some slides intentionally use no Ken Burns."""
    duration = max(0.0, float(duration))
    if duration < 1.4:
        candidates = [
            ("static_hold", 0.55),
            ("hold_still", 0.25),
            ("slow_push", 0.20),
        ]
    elif beat in {"hook", "reveal"}:
        candidates = [
            ("impact_zoom", 0.28),
            ("slow_push", 0.24),
            ("static_hold", 0.18),
            ("diagonal_pan", 0.16),
            ("pull_out", 0.14),
        ]
    elif beat == "payoff":
        candidates = [
            ("static_hold", 0.34),
            ("pull_out", 0.28),
            ("slow_push", 0.22),
            ("hold_still", 0.16),
        ]
    elif beat == "cta":
        candidates = [
            ("static_hold", 0.62),
            ("hold_still", 0.24),
            ("slow_push", 0.14),
        ]
    else:
        candidates = [
            ("slow_push", 0.26),
            ("static_hold", 0.22),
            ("stable_pan", 0.18),
            ("diagonal_pan", 0.14),
            ("hold_still", 0.12),
            ("pull_out", 0.08),
        ]

    if visual_style:
        style_motion = choose_motion_preset(visual_style, beat, index)
        if style_motion and style_motion not in STATIC_HOLD_MOTIONS:
            candidates.append((style_motion, 0.16))

    if previous_motion and len(candidates) > 1:
        candidates = [(motion, weight) for motion, weight in candidates if motion != previous_motion] or candidates

    return weighted_choice(candidates)


# Normalize horizontal transition names to match upcoming pan direction
# next_pan: 'lr' (left→right) or 'rl' (right→left)
def normalize_horizontal_transition(name, next_pan):
    if not isinstance(name, str):
        return name
    # Map desired direction based on pan
    desired = 'right' if next_pan == 'lr' else 'left'
    # Simple replacements for known directional transitions
    mapping = {
        'slide_left': 'slide_right' if desired == 'right' else 'slide_left',
        'slide_right': 'slide_right' if desired == 'right' else 'slide_left',
        'motion_slide_left': 'motion_slide_right' if desired == 'right' else 'motion_slide_left',
        'motion_slide_right': 'motion_slide_right' if desired == 'right' else 'motion_slide_left',
        'whip_pan_left': 'whip_pan_right' if desired == 'right' else 'whip_pan_left',
        'whip_pan_right': 'whip_pan_right' if desired == 'right' else 'whip_pan_left',
    }
    return mapping.get(name, name)


def flip_horizontal_transition(name):
    """Flip left<->right variants for horizontal directional transitions."""
    mapping = {
        'slide_left': 'slide_right',
        'slide_right': 'slide_left',
        'motion_slide_left': 'motion_slide_right',
        'motion_slide_right': 'motion_slide_left',
        'whip_pan_left': 'whip_pan_right',
        'whip_pan_right': 'whip_pan_left',
    }
    return mapping.get(name, name)


# ------------------ New Transition Effects ------------------

def cube_rotation_effect(current_frame, next_frame, progress, direction='right'):
    """Cube-inspired directional reveal with stable endpoints and soft shading.
    
    Args:
        current_frame: Current frame as numpy array (BGR)
        next_frame: Next frame as numpy array (BGR)
        progress: Float between 0 and 1
        direction: Rotation direction ('left', 'right', 'up', 'down')
        
    Returns:
        Transitioned frame as numpy array
    """
    p = ease_progress(progress, 'smoothstep')
    if p <= 0.0:
        return current_frame.copy()
    if p >= 1.0:
        return next_frame.copy()

    h, w = current_frame.shape[:2]
    feather = max(8.0, min(w, h) * 0.035)
    yy, xx = np.mgrid[0:h, 0:w]

    if direction in ("left", "right"):
        seam = w * (1.0 - p) if direction == "right" else w * p
        signed_distance = xx - seam if direction == "right" else seam - xx
    else:
        seam = h * (1.0 - p) if direction == "down" else h * p
        signed_distance = yy - seam if direction == "down" else seam - yy

    mask = np.clip((signed_distance / feather) + 0.5, 0.0, 1.0)
    frame = blend_mask_linear_bgr(current_frame, next_frame, mask)

    seam_shadow = np.exp(-((signed_distance / max(feather, 1.0)) ** 2))
    fold_strength = 0.18 * math.sin(math.pi * p)
    shaded = frame.astype(np.float32) * (1.0 - fold_strength * seam_shadow[..., np.newaxis])
    return np.clip(shaded, 0, 255).astype(np.uint8)

def iris_wipe_effect(current_frame, next_frame, progress, center=None, feather=0.1):
    """Iris wipe transition effect with soft edges.
    
    Args:
        current_frame: Current frame as numpy array (BGR)
        next_frame: Next frame as numpy array (BGR)
        progress: Float between 0 and 1
        center: (x,y) center point of the iris (None for center of image)
        feather: Feather amount for soft edges (0-1)
        
    Returns:
        Transitioned frame as numpy array
    """
    h, w = current_frame.shape[:2]
    raw_progress = max(0.0, min(1.0, float(progress)))
    if raw_progress <= 0.0:
        return current_frame.copy()
    if raw_progress >= 1.0:
        return next_frame.copy()
    
    # Default to center of image
    if center is None:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = center
    
    # Ease the progress for smoother motion
    p = ease_progress(progress, 'smoothstep')
    
    # Create distance map from center
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_dist = np.sqrt((w/2)**2 + (h/2)**2)  # Maximum possible distance
    
    # Calculate radius based on progress
    radius = p * max_dist * 1.2  # Slightly overshoot to ensure full coverage
    
    # Create soft mask with feathering
    mask = np.clip((radius - dist) / (feather * max_dist) + 0.5, 0, 1)
    
    # Convert to 3-channel mask
    mask_3d = mask[..., np.newaxis]
    
    # Convert to linear color space for better blending
    current_lin = to_linear_bgr(current_frame)
    next_lin = to_linear_bgr(next_frame)
    
    # Blend images using the mask
    result = current_lin * (1 - mask_3d) + next_lin * mask_3d
    
    # Convert back to sRGB
    return to_srgb_bgr(result)

def ease_progress(p, kind='cosine'):
    p = max(0.0, min(1.0, float(p)))
    if kind == 'cosine':
        return 0.5 - 0.5 * math.cos(math.pi * p)
    if kind == 'smoothstep':
        return p * p * (3 - 2 * p)
    if kind == 'sine':
        return math.sin(p * math.pi * 0.5)
    if kind == 'ease_in_out_quad':
        return 2 * p * p if p < 0.5 else 1 - ((-2 * p + 2) ** 2) / 2
    return p


def center_crop_to(frame, size_wh):
    """Cover-scale then center-crop to exact size (preserves aspect ratio)."""
    w, h = size_wh
    fh, fw = frame.shape[:2]
    if fw == w and fh == h:
        return frame
    scale = max(w / max(fw, 1), h / max(fh, 1))
    new_w = max(w, int(round(fw * scale)))
    new_h = max(h, int(round(fh * scale)))
    if new_w != fw or new_h != fh:
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    x0 = max(0, (frame.shape[1] - w) // 2)
    y0 = max(0, (frame.shape[0] - h) // 2)
    return frame[y0 : y0 + h, x0 : x0 + w]


def fit_frame_to_resolution(frame, resolution):
    """Fit any BGR frame to output resolution without stretching."""
    res_w, res_h = sanitize_size(resolution)
    if frame is None:
        return np.zeros((res_h, res_w, 3), dtype=np.uint8)
    return center_crop_to(frame, (res_w, res_h))


def zoom_dissolve_effect(current_frame, next_frame, progress, resolution, zoom=1.05, easing='cosine'):
    p = ease_progress(progress, easing)
    res_w, res_h = resolution
    # scale next frame by factor 1 -> zoom
    scale = 1.0 + (zoom - 1.0) * p
    nh, nw = next_frame.shape[:2]
    scaled = cv2.resize(next_frame, (int(nw * scale), int(nh * scale)), interpolation=cv2.INTER_CUBIC)
    scaled = center_crop_to(scaled, (res_w, res_h))
    # Mild unsharp mask for crispness after upscaling
    us_k = 3
    if us_k % 2 == 0:
        us_k += 1
    blurred = cv2.GaussianBlur(scaled, (us_k, us_k), 0)
    sharpened = cv2.addWeighted(scaled, 1.15, blurred, -0.15, 0)
    # Gamma-correct blend
    out = blend_linear_bgr(current_frame, sharpened, p)
    return out


def motion_blur_kernel(length, orientation='horizontal'):
    k = max(1, int(length))
    if k % 2 == 0:
        k += 1
    if orientation == 'horizontal':
        kernel = np.zeros((1, k), dtype=np.float32)
        kernel[0, :] = 1.0 / k
    else:
        kernel = np.zeros((k, 1), dtype=np.float32)
        kernel[:, 0] = 1.0 / k
    return kernel


def motion_blur_slide_effect(current_frame, next_frame, progress, direction='right', strength=15):
    h, w = current_frame.shape[:2]
    p = ease_progress(progress, 'smoothstep')
    if p <= 0.0:
        return current_frame.copy()
    if p >= 1.0:
        return next_frame.copy()

    yy, xx = np.mgrid[0:h, 0:w]
    feather = max(6.0, min(w, h) * 0.02)
    if direction == 'right':
        seam = w * (1.0 - p)
        mask = np.clip((xx - seam) / feather + 0.5, 0.0, 1.0)
        orientation = 'horizontal'
    elif direction == 'left':
        seam = w * p
        mask = np.clip((seam - xx) / feather + 0.5, 0.0, 1.0)
        orientation = 'horizontal'
    elif direction == 'down':
        seam = h * (1.0 - p)
        mask = np.clip((yy - seam) / feather + 0.5, 0.0, 1.0)
        orientation = 'vertical'
    else:  # up
        seam = h * p
        mask = np.clip((seam - yy) / feather + 0.5, 0.0, 1.0)
        orientation = 'vertical'
    result = blend_mask_linear_bgr(current_frame, next_frame, mask)
    # Apply motion blur proportional to progress
    k = int(strength * max(0.1, min(1.0, 2 * abs(p - 0.5))))  # strongest mid-way
    kernel = motion_blur_kernel(max(3, k), 'horizontal' if orientation == 'horizontal' else 'vertical')
    blurred = cv2.filter2D(result, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return blurred


def slide_push_effect(current_frame, next_frame, progress, direction='right'):
    """Push one frame out while the next frame enters from the chosen direction."""
    h, w = current_frame.shape[:2]
    p = ease_progress(progress, 'smoothstep')
    if p <= 0.0:
        return current_frame.copy()
    if p >= 1.0:
        return next_frame.copy()

    result = np.zeros_like(current_frame)
    if direction == 'right':
        offset = int(round(w * p))
        if offset > 0:
            result[:, :offset] = next_frame[:, w - offset:]
        if offset < w:
            result[:, offset:] = current_frame[:, :w - offset]
    elif direction == 'left':
        offset = int(round(w * p))
        if offset < w:
            result[:, :w - offset] = current_frame[:, offset:]
        if offset > 0:
            result[:, w - offset:] = next_frame[:, :offset]
    elif direction == 'down':
        offset = int(round(h * p))
        if offset > 0:
            result[:offset, :] = next_frame[h - offset:, :]
        if offset < h:
            result[offset:, :] = current_frame[:h - offset, :]
    else:  # up
        offset = int(round(h * p))
        if offset < h:
            result[:h - offset, :] = current_frame[offset:, :]
        if offset > 0:
            result[h - offset:, :] = next_frame[:offset, :]

    return result


def fade_effect_eased(current_frame, next_frame, progress, easing='cosine'):
    p = ease_progress(progress, easing)
    return blend_linear_bgr(current_frame, next_frame, p)


def radial_wipe_effect(current_frame, next_frame, progress, feather=0.05):
    # Build a circular mask from center that grows with progress
    h, w = current_frame.shape[:2]
    raw_progress = max(0.0, min(1.0, float(progress)))
    if raw_progress <= 0.0:
        return current_frame.copy()
    if raw_progress >= 1.0:
        return next_frame.copy()
    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2)
    # Apply easing to radius growth
    p = ease_progress(progress, DEFAULT_EASING)
    r = max(0.0, min(1.0, float(p))) * max_r
    feather_px = feather * max(h, w)
    # Mask grows from 0 to 1 with a feathered edge
    mask = (dist - (r - feather_px)) / (feather_px + 1e-6)
    mask = np.clip(mask, 0.0, 1.0)
    mask = 1.0 - mask  # inside is 1, outside is 0
    mask = np.clip(mask, 0.0, 1.0)
    return blend_mask_linear_bgr(current_frame, next_frame, mask.astype(np.float32))


def whip_pan_effect(current_frame, next_frame, progress, direction='right', blur_strength=25):
    h, w = current_frame.shape[:2]
    p = ease_progress(progress, 'sine')
    if p <= 0.0:
        return current_frame.copy()
    if p >= 1.0:
        return next_frame.copy()

    yy, xx = np.mgrid[0:h, 0:w]
    feather = max(10.0, min(w, h) * 0.04)
    if direction == 'right':
        seam = w * (1.0 - p)
        mask = np.clip((xx - seam) / feather + 0.5, 0.0, 1.0)
        orientation = 'horizontal'
    elif direction == 'left':
        seam = w * p
        mask = np.clip((seam - xx) / feather + 0.5, 0.0, 1.0)
        orientation = 'horizontal'
    elif direction == 'down':
        seam = h * (1.0 - p)
        mask = np.clip((yy - seam) / feather + 0.5, 0.0, 1.0)
        orientation = 'vertical'
    else:
        seam = h * p
        mask = np.clip((seam - yy) / feather + 0.5, 0.0, 1.0)
        orientation = 'vertical'
    result = blend_mask_linear_bgr(current_frame, next_frame, mask)
    # Strong motion blur regardless of p
    kernel = motion_blur_kernel(max(5, int(blur_strength)), 'horizontal' if orientation == 'horizontal' else 'vertical')
    blurred = cv2.filter2D(result, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return blurred

def get_video_duration(filepath):
    """Get the duration of a video file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        filepath
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"Error getting video duration for {filepath}: {e}")
        return 0

def convert_to_jpg(image_path, quality=95):
    """Convert any image to JPG format.
    
    Args:
        image_path: Path to the input image
        quality: JPEG quality (1-100)
        
    Returns:
        Path to the converted JPG file or None if conversion fails
    """
    try:
        # If already a JPG, return as is
        if image_path.lower().endswith(('.jpg', '.jpeg')):
            return image_path
            
        # Generate output path with .jpg extension
        base_path = os.path.splitext(image_path)[0]
        output_path = f"{base_path}.jpg"
        
        # Skip if already converted
        if os.path.exists(output_path):
            return output_path
            
        # Open and convert the image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (e.g., for PNG with transparency)
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Save as JPG
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
            
        logger.info(f"Converted {image_path} to JPG format")
        return output_path
        
    except Exception as e:
        logger.error(f"Error converting {image_path} to JPG: {e}")
        return None


def create_placeholder_slide(image_dir, resolution, label="One Piece"):
    """Create a simple fallback image so missing downloads do not shorten videos."""
    os.makedirs(image_dir, exist_ok=True)
    output_path = os.path.join(image_dir, "slide_placeholder.jpg")
    if os.path.exists(output_path):
        return output_path

    res_w, res_h = resolution
    frame = np.zeros((res_h, res_w, 3), dtype=np.uint8)
    frame[:] = (18, 18, 24)

    font = cv2.FONT_HERSHEY_SIMPLEX
    title = label[:32] if label else "One Piece"
    subtitle = "image unavailable"
    cv2.putText(frame, title, (max(24, res_w // 12), res_h // 2 - 20), font, 1.5, (245, 245, 245), 3, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (max(24, res_w // 12), res_h // 2 + 40), font, 0.9, (180, 180, 190), 2, cv2.LINE_AA)
    cv2.imwrite(output_path, frame)
    return output_path


def build_visual_quality_report(
    slides: List[Dict],
    resolution,
    frames_written: int,
    expected_frames: int,
    transition_events: List[Dict],
    output_path: str,
) -> Dict:
    """Summarize visual quality risks for generated slideshow artifacts."""
    res_w, res_h = sanitize_size(resolution)
    warnings: List[str] = []
    duplicate_runs = 0
    previous_query = None
    current_run = 1
    overlong_holds = 0
    missing_assets = 0
    low_confidence = 0

    for slide in slides:
        query = (slide.get("image_search_query") or "").strip().lower()
        if query and query == previous_query:
            current_run += 1
            if current_run > 2:
                duplicate_runs += 1
        else:
            current_run = 1
        previous_query = query

        try:
            duration = parse_time(slide["end_time"]) - parse_time(slide["start_time"])
        except Exception:
            duration = 0
        if duration > (8.0 if res_w > res_h else 5.5):
            overlong_holds += 1
        if not slide.get("image_path") and not slide.get("image_search_query"):
            missing_assets += 1
        if float(slide.get("asset_confidence") or 1.0) < 0.55:
            low_confidence += 1

    if frames_written != expected_frames:
        warnings.append(f"Frame count mismatch: wrote {frames_written}, expected {expected_frames}")
    if duplicate_runs:
        warnings.append(f"Repeated visual query runs over limit: {duplicate_runs}")
    if overlong_holds:
        warnings.append(f"Overlong slide holds detected: {overlong_holds}")
    if missing_assets:
        warnings.append(f"Slides missing image query/path: {missing_assets}")
    if low_confidence:
        warnings.append(f"Low-confidence visual matches: {low_confidence}")
    if len(transition_events) > max(2, len(slides) * 0.85):
        warnings.append("Transition density is high; consider calmer pacing")

    return {
        "output_path": output_path,
        "resolution": {"width": res_w, "height": res_h},
        "frames_written": frames_written,
        "expected_frames": expected_frames,
        "slides": len(slides),
        "transitions": len(transition_events),
        "duplicate_query_runs_over_limit": duplicate_runs,
        "overlong_holds": overlong_holds,
        "missing_asset_references": missing_assets,
        "low_confidence_assets": low_confidence,
        "warnings": warnings,
        "status": "ok" if not warnings else "review",
    }


def write_visual_quality_report(report: Dict, output_path: str) -> str:
    report_path = os.path.splitext(output_path)[0] + ".visual_quality.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report_path


def main(json_path, image_dir, output_path, total_duration=None, resolution=VIDEO_RESOLUTION, fps=30, blur_amount=5, transition_type='auto', transition_duration=0.5, flip_horizontal_once=False, avoid_same_direction_horizontal=True, slide_blur_amount=0, quality_mode='standard', pan_strength=1.0, visual_style=None):
    """
    Generates the final slideshow video using OpenCV.
    
    Args:
        json_path: Path to the JSON file containing slide timings
        image_dir: Directory containing slide images
        output_path: Path where the output video will be saved
        total_duration: Total duration of the video in seconds
        resolution: Video resolution as (width, height)
        fps: Frames per second for the output video
        blur_amount: Amount of blur to apply during blur-based transitions
        slide_blur_amount: Optional blur applied to the slide images themselves
        transition_type: Type of transition between slides ('fade', 'crossfade', 'slide_left', 'slide_right', 'slide_up', 'slide_down')
        transition_duration: Duration of transition in seconds
    """
    resolution = sanitize_size(resolution)
    res_w, res_h = resolution
    if visual_style:
        visual_style = normalize_visual_style(visual_style)
        logger.info("Using visual style for slideshow: %s", visual_style)
    
    quality_mode = (quality_mode or 'standard').lower()
    if quality_mode == 'pro':
        transition_type = 'pro' if transition_type == 'auto' else transition_type
        transition_duration = min(float(transition_duration), 0.42)
        pan_strength = min(float(pan_strength), 0.35)
        slide_blur_amount = 0
        logger.info("Using pro slideshow mode: restrained transitions, subtle pan, no slide blur")

    blur_amount = normalize_blur_amount(blur_amount, max_kernel=15)
    slide_blur_amount = normalize_blur_amount(slide_blur_amount, max_kernel=9)
    pan_strength = max(0.0, min(1.0, float(pan_strength)))
    if blur_amount > 0:
        logger.info(f"Applying transition blur with kernel: {blur_amount}")
    if slide_blur_amount > 0:
        logger.info(f"Applying slide blur with kernel: {slide_blur_amount}")

    # Calculate total frames needed
    if total_duration is None:
        # If no total duration provided, calculate from slides
        with open(json_path, "r", encoding="utf-8") as f:
            slides = json.load(f)
        total_duration = sum(slide['end'] - slide['start'] for slide in slides)
    total_output_frames = int(total_duration * fps)
    logger.info(f"Total video duration: {total_duration:.2f}s, {total_output_frames} frames")
    logger.info(f"Target total frames: {total_output_frames} for a duration of {total_duration:.2f}s at {fps} fps.")

    with open(json_path, "r", encoding="utf-8") as f:
        slides = json.load(f)

    last_slide_end_time = 0
    # Track the actual number of slides processed for panning direction
    slides_processed = 0
    
    logger.info(f"Starting to process {len(slides)} slides...")

    transition_events: List[Dict] = []

    # -------- Prepass: select transitions and allocate frames to avoid tail padding --------
    last_selected_transition = None
    flip_remaining = 1 if flip_horizontal_once else 0
    boundary_transition_types = []  # per slide index (same as processing index)
    content_frames_float = []       # desired written content frames per slide (before rounding)
    start_skips = []                # 0 for first slide or no transition, else 1
    transition_frames_list = []     # frames written for transition before this slide

    # First, select transitions per boundary deterministically for this run
    for idx, slide in enumerate(slides):
        # Parse timing
        start = parse_time(slide["start_time"]) 
        end = parse_time(slide["end_time"]) 
        duration = end - start
        # Decide transition type into THIS slide (except first)
        selected_transition_type = 'none'
        if idx > 0 and transition_duration > 0:
            explicit_transition = (slide.get("transition_in") or slide.get("transition_type") or "").strip().lower()
            if explicit_transition:
                selected_transition_type = explicit_transition
            elif isinstance(transition_type, str):
                tt = transition_type.lower()
                if tt == 'pro':
                    if visual_style:
                        slide_text = slide.get("summary") or slide.get("text") or ""
                        beat = classify_beat(slide_text, idx, len(slides))
                        preset = get_visual_preset(visual_style)
                        transitions_pool = list(preset["transitions"])
                        weights = preset.get("transition_weights")
                        if weights and len(weights) == len(transitions_pool):
                            pool = [t for t in transitions_pool if t != last_selected_transition] or transitions_pool
                            pool_weights = [
                                weights[transitions_pool.index(t)]
                                for t in pool
                            ]
                            selected_transition_type = random.choices(pool, weights=pool_weights, k=1)[0]
                        else:
                            selected_transition_type = choose_visual_transition(
                                visual_style, beat, last_selected_transition, idx
                            )
                    else:
                        transitions_pool = ['crossfade', 'zoom_dissolve', 'fade_eased']
                        selected_transition_type = random.choices(
                            transitions_pool,
                            weights=[0.55, 0.30, 0.15],
                            k=1
                        )[0]
                elif tt == 'random':
                    transitions_pool = [
                        'fade', 'crossfade', 'fade_eased', 'zoom_dissolve', 'radial_wipe',
                        'slide_up', 'slide_down',
                        'iris_wipe',
                        'cube_rotation_left', 'cube_rotation_right',
                        'cube_rotation_up', 'cube_rotation_down',
                        'page_curl_tl', 'page_curl_tr', 'page_curl_bl', 'page_curl_br',
                        'water_ripple'
                    ]
                    # Filter out the last used transition if we have alternatives
                    pool = [t for t in transitions_pool if t != last_selected_transition] or transitions_pool
                    selected_transition_type = random.choice(pool)
                elif tt == 'auto':
                    expected_content_duration = max(0.1, duration - transition_duration)
                    selected_transition_type = choose_transition(idx, expected_content_duration, last_selected_transition, avoid_same_direction_horizontal)
                else:
                    selected_transition_type = transition_type
            selected_transition_type = safe_transition_name(selected_transition_type)
            # Normalize horizontal transitions to match upcoming pan (unless that would violate avoidance rule)
            if selected_transition_type and selected_transition_type != 'none':
                next_pan = 'lr' if (idx % 2 == 0) else 'rl'
                normalized = normalize_horizontal_transition(selected_transition_type, next_pan)
                # If avoidance is enabled, do not normalize into same-direction
                if avoid_same_direction_horizontal:
                    if (next_pan == 'lr' and normalized in RIGHT_DIR_NAMES) or (next_pan == 'rl' and normalized in LEFT_DIR_NAMES):
                        logger.info(
                            f"Skipped normalization to '{normalized}' to avoid same-direction with pan {next_pan}; keeping '{selected_transition_type}'"
                        )
                    else:
                        if normalized != selected_transition_type:
                            logger.info(f"Normalized transition '{selected_transition_type}' to '{normalized}' to match upcoming pan {next_pan}")
                        selected_transition_type = normalized
                else:
                    if normalized != selected_transition_type:
                        logger.info(f"Normalized transition '{selected_transition_type}' to '{normalized}' to match upcoming pan {next_pan}")
                    selected_transition_type = normalized
                # Optionally flip once per run for creative variety
                if flip_remaining > 0:
                    flipped = flip_horizontal_transition(selected_transition_type)
                    if flipped != selected_transition_type:
                        logger.info(f"Flipped horizontal transition once: '{selected_transition_type}' -> '{flipped}'")
                        selected_transition_type = flipped
                        flip_remaining -= 1
            if selected_transition_type != 'none':
                last_selected_transition = selected_transition_type
        boundary_transition_types.append(selected_transition_type)

        # Compute desired content frames (the frames we intend to WRITE for the slide's content)
        content_duration = duration - (transition_duration if idx > 0 and selected_transition_type != 'none' else 0.0)
        content_frames_float.append(max(0.0, content_duration * fps))
        # Determine if we'll skip first content frame due to transition overlap
        start_skips.append(1 if (idx > 0 and selected_transition_type != 'none') else 0)
        transition_frames_list.append(int(transition_duration * fps) if (idx > 0 and selected_transition_type != 'none') else 0)

    # Allocate integer content frames using largest remainder so that total == target - transitions
    total_transition_frames = sum(transition_frames_list)
    target_content_frames = total_output_frames - total_transition_frames
    content_floor = [int(np.floor(x)) for x in content_frames_float]
    remainders = [x - f for x, f in zip(content_frames_float, content_floor)]
    allocated = content_floor[:]
    deficit = target_content_frames - sum(allocated)
    if deficit > 0:
        # Distribute +1 to the largest remainders
        order = sorted(range(len(remainders)), key=lambda i: remainders[i], reverse=True)
        for i in order[:deficit]:
            allocated[i] += 1
    elif deficit < 0:
        # Remove frames from smallest remainders (or anywhere) to match target
        order = sorted(range(len(remainders)), key=lambda i: remainders[i])
        for i in order[:abs(deficit)]:
            if allocated[i] > 0:
                allocated[i] -= 1

    # Compute planned slide_frames passed to pan function (accounting for double skip in implementation)
    planned_slide_frames = [alloc + 2 * start_skips[i] for i, alloc in enumerate(allocated)]

    # Initialize frames_written counter
    frames_written = 0
    last_valid_img_path = None
    last_output_frame = None
    prev_img_path = None
    prev_motion = "slow_push"
    prev_slide_meta = None
    prev_slide_idx = 0
    frame_writer = RawVideoFFmpegWriter(output_path, resolution, fps, quality_mode, total_frames=total_output_frames)

    for idx, slide in enumerate(slides):
        try:
            start = parse_time(slide["start_time"])
            end = parse_time(slide["end_time"])
            duration = end - start
            last_slide_end_time = max(last_slide_end_time, end)
            
            logger.info(f"Processing slide {idx+1} (slides_processed={slides_processed}): start={start:.2f}s, end={end:.2f}s, duration={duration:.2f}s, transition={transition_type}")

            # First try to get the image path from the slide data if it exists
            img_path = slide.get('image_path')
            
            # If not in slide data or path doesn't exist, try to find the image
            if not img_path or not os.path.exists(img_path):
                # Get the image hash if available
                image_hash = slide.get('image_hash', '')[:8] if 'image_hash' in slide else ''
                
                # List of possible extensions to try
                extensions = ["jpg", "jpeg", "png"]
                
                # Try different naming patterns in order of preference
                patterns = [
                    f"slide_{idx+1:03d}_{image_hash}.{{ext}}" if image_hash else None,  # With hash
                    f"slide_{idx+1:03d}.{{ext}}",                                        # Just number
                    f"slide_{idx+1}_{image_hash}.{{ext}}" if image_hash else None,      # Old format with hash
                    f"slide_{idx+1}.{{ext}}"                                            # Old format just number
                ]
                
                # Try each pattern with each extension
                img_path = None
                for pattern in patterns:
                    if not pattern:
                        continue
                        
                    for ext in extensions:
                        candidate = os.path.join(image_dir, pattern.format(ext=ext))
                        if os.path.exists(candidate):
                            img_path = candidate
                            logger.info(f"Found image for slide {idx+1} at: {img_path}")
                            break
                    if img_path:
                        break
            
            # If still no image found, try to find any file starting with the slide number
            if not img_path or not os.path.exists(img_path):
                try:
                    # List all files in the directory
                    all_files = os.listdir(image_dir)
                    # Look for files that start with the slide number
                    for f in all_files:
                        if f.startswith(f"slide_{idx+1}"):
                            candidate = os.path.join(image_dir, f)
                            if os.path.isfile(candidate):
                                img_path = candidate
                                logger.info(f"Found image for slide {idx+1} using fallback pattern: {img_path}")
                                break
                    
                    # If still not found, try with zero-padded number
                    if not img_path and idx + 1 < 100:  # Only try if it makes sense to pad
                        padded_num = f"{idx+1:03d}"  # 1 -> 001, 12 -> 012
                        for f in all_files:
                            if f.startswith(f"slide_{padded_num}"):
                                candidate = os.path.join(image_dir, f)
                                if os.path.isfile(candidate):
                                    img_path = candidate
                                    logger.info(f"Found image for slide {idx+1} using padded fallback: {img_path}")
                                    break
                except Exception as e:
                    logger.warning(f"Error while searching for slide {idx+1}: {e}")
            
            # If still no image found, reuse the previous visual so timing stays intact.
            if not img_path or not os.path.exists(img_path):
                if last_valid_img_path and os.path.exists(last_valid_img_path):
                    logger.warning(
                        f"Image for slide {idx+1} not found in {image_dir}. "
                        f"Reusing previous image to preserve slide timing: {last_valid_img_path}"
                    )
                    img_path = last_valid_img_path
                else:
                    label = slide.get("summary") or slide.get("image_search_query") or f"Slide {idx+1}"
                    img_path = create_placeholder_slide(image_dir, resolution, label=label)
                    logger.warning(
                        f"Image for slide {idx+1} not found in {image_dir}. "
                        f"Using generated placeholder: {img_path}"
                    )
                logger.debug(f"Searched for patterns: slide_{idx+1:03d}_*.{{jpg,png,jpeg}}, slide_{idx+1}*.{{jpg,png,jpeg}}")

            # Convert image to JPG if it's not already
            _, ext = os.path.splitext(img_path)
            ext = ext.lower()
            if ext != '.jpg' and ext != '.jpeg':
                jpg_path = convert_to_jpg(img_path)
                if jpg_path and jpg_path != img_path:
                    logger.info(f"Using converted JPG: {jpg_path}")
                    img_path = jpg_path

            beat = (slide.get("beat_type") or classify_beat(slide.get("summary", ""), idx, len(slides))).strip().lower()
            motion_plan = slide.get("motion_plan") if isinstance(slide.get("motion_plan"), dict) else {}
            requested_motion = (
                slide.get("motion_preset")
                or motion_plan.get("preset")
                or ""
            ).strip().lower()
            slide_motion = requested_motion or choose_slide_motion(
                visual_style,
                beat,
                idx,
                duration,
                previous_motion=prev_motion,
            )
            slide["_motion"] = slide_motion

            logger.info(f"Processing image: {img_path} (motion={slide_motion})")
            previous_img_path = last_valid_img_path
            same_as_previous = (
                previous_img_path is not None
                and os.path.abspath(previous_img_path) == os.path.abspath(img_path)
            )
            last_valid_img_path = img_path
            # Use preselected transition for this boundary to keep frame plan consistent
            selected_transition_type = boundary_transition_types[idx]
            if idx > 0 and selected_transition_type and selected_transition_type != "none":
                slide["transition_in"] = selected_transition_type
            skipped_duplicate_transition_frames = 0
            if same_as_previous and selected_transition_type and selected_transition_type != 'none' and idx > 0:
                skipped_duplicate_transition_frames = transition_frames_list[idx]
                selected_transition_type = 'none'
                slide.pop("transition_in", None)
                logger.info(
                    "Skipping transition for duplicate slide image: %s -> %s; preserving %d frames as slide hold",
                    os.path.basename(previous_img_path),
                    os.path.basename(img_path),
                    skipped_duplicate_transition_frames,
                )
            if selected_transition_type and selected_transition_type != 'none' and idx > 0:
                logger.info(f"Using preselected transition: {selected_transition_type}")

            # Calculate transition frames if not the first slide
            transition_frames = 0
            if slides_processed > 0 and transition_duration > 0 and selected_transition_type != 'none':
                transition_frames = int(transition_duration * fps)
                logger.info(
                    f"Preparing {selected_transition_type} transition between slides {slides_processed} and {slides_processed+1} | "
                    f"from: {os.path.basename(slides[idx-1].get('image_path', 'unknown')) if idx>0 else 'N/A'} -> "
                    f"to: {os.path.basename(img_path)} | duration: {transition_duration:.2f}s, frames: {transition_frames}, fps: {fps}"
                )
                
                # Adjust duration for transition
                original_duration = duration
                duration -= transition_duration
                if duration < 0:
                    duration = 0.1  # Minimum duration for the slide
                    logger.warning(f"Adjusted slide duration to minimum (0.1s) to accommodate transition")
                
                logger.debug(f"Original slide duration: {original_duration:.2f}s, After transition: {duration:.2f}s")
                
                slide_motion = slide.get("_motion", "slow_push")
                next_frame = render_panned_frame(
                    img_path,
                    resolution,
                    idx=slides_processed,
                    t=0.0,
                    blur_amount=slide_blur_amount,
                    pan_strength=pan_strength,
                    motion=slide_motion,
                    slide_meta=slide,
                )
                logger.debug(
                    "Next transition endpoint: %s t=0.0 motion=%s idx=%s",
                    os.path.basename(img_path),
                    slide_motion,
                    slides_processed,
                )

                if slides_processed == 0:
                    current_frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
                elif prev_img_path and os.path.exists(prev_img_path):
                    current_frame = render_panned_frame(
                        prev_img_path,
                        resolution,
                        idx=prev_slide_idx,
                        t=1.0,
                        blur_amount=slide_blur_amount,
                        pan_strength=pan_strength,
                        motion=prev_motion,
                        slide_meta=prev_slide_meta,
                    )
                    logger.debug(
                        "Current transition endpoint: %s t=1.0 motion=%s idx=%s",
                        os.path.basename(prev_img_path),
                        prev_motion,
                        prev_slide_idx,
                    )
                else:
                    current_frame = last_output_frame
                    if current_frame is None:
                        logger.warning("No previous slide frame available; using next frame as fallback.")
                        current_frame = next_frame.copy()
                
                # Log frame details
                logger.debug(
                    f"Current frame shape: {current_frame.shape if current_frame is not None else 'None'} | "
                    f"Next frame shape: {next_frame.shape if next_frame is not None else 'None'}"
                )
                
                # Apply transition effect with error handling
                logger.info(
                    f"Applying {selected_transition_type} transition over {transition_frames} frames "
                    f"between slides {slides_processed} and {slides_processed+1} | "
                    f"frame range: {frames_written:05d}-{frames_written + transition_frames - 1:05d}"
                )

                visual_transition_time = frames_written / max(1, fps)
                sfx_cue = (slide.get("sfx_cue") or selected_transition_type or "default").strip().lower()
                transition_events.append(
                    {
                        "time": visual_transition_time,
                        "transition": sfx_cue,
                        "visual_transition": selected_transition_type,
                        "slide_index": idx + 1,
                        "duration": transition_frames / max(1, fps),
                        "frame": frames_written,
                        "nominal_slide_start": parse_time(slide["start_time"]),
                    }
                )
                logger.info(
                    "Transition SFX cue aligned to video frame %s (%.3fs): %s",
                    frames_written,
                    visual_transition_time,
                    sfx_cue,
                )
                
                current_frame = fit_frame_to_resolution(current_frame, resolution)
                next_frame = fit_frame_to_resolution(next_frame, resolution)
                
                transition_params = {}
                if selected_transition_type == 'iris_wipe' and random.random() < 0.3:
                    h, w = current_frame.shape[:2]
                    transition_params['center'] = (
                        int(w * (0.2 + 0.6 * random.random())),
                        int(h * (0.2 + 0.6 * random.random()))
                    )
                elif selected_transition_type == 'water_ripple':
                    h, w = current_frame.shape[:2]
                    transition_params['center'] = (
                        int(w * (0.25 + 0.5 * random.random())),
                        int(h * (0.25 + 0.5 * random.random()))
                    )

                try:
                    # Write transition frames
                    transition_written = 0
                    for i in range(transition_frames):
                        try:
                            progress = transition_progress(i, transition_frames)
                            frame_num = frames_written + i
                            
                            # Generate transition frame
                            try:
                                if selected_transition_type == 'fade':
                                    frame = fade_effect(current_frame, next_frame, progress)
                                elif selected_transition_type == 'fade_eased':
                                    frame = fade_effect_eased(current_frame, next_frame, progress, easing='cosine')
                                elif selected_transition_type == 'crossfade':
                                    frame = crossfade_effect(current_frame, next_frame, progress, blur_amount)
                                elif selected_transition_type == 'zoom_dissolve':
                                    frame = zoom_dissolve_effect(current_frame, next_frame, progress, resolution)
                                elif selected_transition_type == 'radial_wipe':
                                    frame = radial_wipe_effect(current_frame, next_frame, progress, feather=0.06)
                                elif selected_transition_type == 'iris_wipe':
                                    frame = iris_wipe_effect(
                                        current_frame,
                                        next_frame,
                                        progress,
                                        center=transition_params.get('center'),
                                        feather=0.08
                                    )
                                elif selected_transition_type.startswith('cube_rotation_'):
                                    direction = selected_transition_type.split('_')[2]
                                    frame = cube_rotation_effect(current_frame, next_frame, progress, direction=direction)
                                elif selected_transition_type.startswith('page_curl_'):
                                    corner_key = selected_transition_type.split('_')[2]  # tl, tr, bl, or br
                                    corner_map = {
                                        'tl': 'top-left',
                                        'tr': 'top-right',
                                        'bl': 'bottom-left',
                                        'br': 'bottom-right',
                                    }
                                    frame = page_curl_effect(
                                        current_frame,
                                        next_frame,
                                        progress,
                                        corner=corner_map.get(corner_key, 'top-right')
                                    )
                                elif selected_transition_type == 'water_ripple':
                                    frame = water_ripple_effect(
                                        current_frame,
                                        next_frame,
                                        progress,
                                        center=transition_params.get('center')
                                    )
                                elif selected_transition_type.startswith('motion_slide_'):
                                    direction = selected_transition_type.split('_')[2]
                                    frame = motion_blur_slide_effect(
                                        current_frame,
                                        next_frame,
                                        progress,
                                        direction=direction,
                                        strength=10
                                    )
                                elif selected_transition_type.startswith('whip_pan_'):
                                    direction = selected_transition_type.split('_')[2]
                                    frame = whip_pan_effect(
                                        current_frame,
                                        next_frame,
                                        progress,
                                        direction=direction,
                                        blur_strength=18
                                    )
                                elif selected_transition_type.startswith('slide_'):
                                    direction = selected_transition_type.split('_')[1]
                                    frame = slide_push_effect(
                                        current_frame,
                                        next_frame,
                                        progress,
                                        direction=direction
                                    )
                                else:
                                    # Fallback to crossfade
                                    frame = crossfade_effect(current_frame, next_frame, progress, blur_amount)
                            except Exception as e:
                                logger.error(f"Error applying transition '{selected_transition_type}': {str(e)}")
                                logger.error(traceback.format_exc())
                                # Fallback to crossfade on error
                                frame = crossfade_effect(current_frame, next_frame, progress, blur_amount)
                            
                            # Log progress at key checkpoints to avoid log spam
                            if i in {0, max(0, transition_frames//4), max(0, transition_frames//2), max(0, 3*transition_frames//4), transition_frames-1}:
                                logger.debug(
                                    f"Transition frame {i+1}/{transition_frames} | global #{frame_num:05d} | progress={progress:.2f}"
                                )
                            
                            # Ensure frame is in the correct format
                            if frame is None:
                                logger.warning(f"Generated frame is None, using black frame instead")
                                frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
                            
                            frame = fit_frame_to_resolution(frame, resolution)
                            
                            frame_writer.write(frame)
                            last_output_frame = frame
                            transition_written += 1
                            
                        except Exception as e:
                            logger.error(f"Error in transition frame {i+1}/{transition_frames} between slides {slides_processed} and {slides_processed+1}: {str(e)}")
                            logger.error(traceback.format_exc())
                            # Skip this frame but continue with the rest
                            # Continue with next frame even if one fails
                            continue
                    
                    start_idx = frames_written
                    frames_written += transition_frames
                    end_idx = frames_written - 1
                    logger.info(
                        f"Transition complete. Wrote {transition_written}/{transition_frames} frames | "
                        f"range: {start_idx:05d}-{end_idx:05d}"
                    )
                    
                except Exception as e:
                    logger.error(f"Critical error during transition: {str(e)}")
                    logger.error(traceback.format_exc())
                    # Skip transition frames to prevent getting stuck
                    frames_written += transition_frames
            
            # Calculate planned frames for this slide's pan (based on pre-allocation)
            slide_frames = planned_slide_frames[idx]
            if skipped_duplicate_transition_frames:
                allocated_content_frames = max(0, slide_frames - 2 * start_skips[idx])
                slide_frames = allocated_content_frames + skipped_duplicate_transition_frames
            
            # For the first slide, start from frame 0
            # For subsequent slides with transitions, we need to account for the transition frames
            if slides_processed == 0:
                start_frame = 0
            else:
                # If we had a transition, we need to start from frame 1 to avoid duplicating the last transition frame
                start_frame = 1 if transition_frames > 0 else 0
            
            # Last-slide correction: ensure we exactly hit total_output_frames
            if idx == len(slides) - 1:
                remaining_needed = total_output_frames - frames_written
                corrected_slide_frames = remaining_needed + 2 * (1 if (slides_processed > 0 and transition_frames > 0) else 0)
                if corrected_slide_frames != slide_frames and corrected_slide_frames > 0:
                    logger.info(
                        f"Adjusting last slide frames from {slide_frames} to {corrected_slide_frames} to meet target {total_output_frames}."
                    )
                    slide_frames = corrected_slide_frames

            # Only generate frames if we have frames left after accounting for the start frame
            try:
                if slide_frames > start_frame:
                    # Calculate the adjusted duration to ensure the function writes (slide_frames - 2*start_frame)
                    adjusted_duration = (slide_frames - start_frame) / fps
                    
                    logger.info(f"Processing slide {os.path.basename(img_path)} - frames: {slide_frames}, start: {start_frame}, duration: {adjusted_duration:.2f}s")
                    
                    frames_before = frames_written
                    frames_written, last_pan_frame = emit_panoramic_pan(
                        frame_writer=frame_writer,
                        frame_counter=frames_written,
                        img_path=img_path,
                        duration=adjusted_duration,
                        resolution=resolution,
                        idx=slides_processed,
                        fps=fps,
                        blur_amount=slide_blur_amount,
                        start_frame=start_frame,
                        pan_strength=pan_strength,
                        motion=slide_motion,
                        slide_meta=slide,
                    )
                    if last_pan_frame is not None:
                        last_output_frame = last_pan_frame
                    
                    frames_written_this_slide = frames_written - frames_before
                    planned_content_frames = slide_frames - 2 * start_frame
                    logger.info(
                        f"Successfully wrote {frames_written_this_slide} frames for slide {slides_processed + 1} (planned={planned_content_frames}, start_skip={start_frame})"
                    )
                    
                else:
                    logger.warning(f"Skipping slide {slides_processed + 1} - not enough frames (need {start_frame + 1}, have {slide_frames})")
                    fallback = last_output_frame if last_output_frame is not None else np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
                    for _ in range(max(0, slide_frames)):
                        frame_writer.write(fallback)
                        frames_written += 1

                prev_img_path = img_path
                prev_motion = slide_motion
                prev_slide_meta = slide
                prev_slide_idx = slides_processed
                slides_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing slide {slides_processed + 1}: {str(e)}")
                logger.error(traceback.format_exc())
                # Skip to next slide to prevent getting stuck
                fallback = last_output_frame if last_output_frame is not None else np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
                for _ in range(max(1, slide_frames - start_frame)):
                    frame_writer.write(fallback)
                    frames_written += 1
                slides_processed += 1

        except Exception as e:
            logger.error(f"Error processing slide {idx+1}: {e}")
            traceback.print_exc()

    logger.info(f"Frames written after generation: {frames_written}")
    if frames_written != total_output_frames:
        logger.warning(f"Frame count mismatch: wrote {frames_written}, expected {total_output_frames}. Consider re-checking allocation.")

    try:
        frame_writer.close()
        report = build_visual_quality_report(
            slides,
            resolution,
            frames_written,
            total_output_frames,
            transition_events,
            output_path,
        )
        report_path = write_visual_quality_report(report, output_path)
        logger.info("Visual quality report written to %s (status=%s)", report_path, report["status"])
        logger.info(f"Video successfully generated at {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg raw frame writer failed: {e}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        raise

    try:
        transitions_path = save_transition_events(json_path, transition_events)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(slides, f, indent=2, ensure_ascii=False)
        logger.info(
            "Saved %s transition events to %s",
            len(transition_events),
            transitions_path,
        )
    except Exception as exc:
        logger.warning("Could not save transition metadata: %s", exc)


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Generate a slideshow video from images and a JSON file using OpenCV.")
    parser.add_argument("--json", required=True,
                        help="Path to slides JSON file")
    parser.add_argument("--images", required=True,
                        help="Directory with images")
    parser.add_argument(
        "--duration", type=float, default=None, help="Total duration for the video. If provided, it will be padded."
    )
    parser.add_argument("--resolution", type=str, default=f"{VIDEO_RESOLUTION[0]}x{VIDEO_RESOLUTION[1]}",
                        help=f"Video resolution as WxH (e.g., {VIDEO_RESOLUTION[0]}x{VIDEO_RESOLUTION[1]})")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second for the output video.")

    args = parser.parse_args()

    try:
        w, h = map(int, args.resolution.split('x'))
        resolution_arg = (w, h)
    except ValueError:
        print("Invalid resolution format. Use WxH (e.g., 576x1024).")
        exit(1)

    output_dir = "app/output"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"slideshow_{timestamp}.mp4")

    main(args.json, args.images, output_path, total_duration=args.duration,
         resolution=resolution_arg, fps=args.fps)
    
