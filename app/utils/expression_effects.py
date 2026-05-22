"""Emotion-based FFmpeg overlay effects for character expression PNGs."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

from app.utils.visual_effects import expression_entry_for_visual_style, normalize_visual_style

EXPRESSION_EMOTION_EFFECTS: Dict[str, str] = {
    "neutral": "fade_scale",
    "serious": "fade_scale",
    "happy": "pop_in",
    "excited": "bounce_in",
    "angry": "shake_in",
    "surprised": "snap_in",
    "sad": "fade_rise",
    "worried": "fade_rise",
    "smirking": "slide_in",
    "confident": "pop_in",
    "intense": "shake_in",
    "embarrassed": "fade_soft",
}

STYLE_FALLBACK_EFFECT = "fade_scale"

# Max width for expression overlay on 1080-wide vertical video
EXPR_MAX_WIDTH = int(os.getenv("EXPRESSION_MAX_WIDTH", "420"))


def resolve_expression_effect(expression: str, visual_style: Optional[str] = None) -> str:
    label = (expression or "neutral").strip().lower()
    if label in EXPRESSION_EMOTION_EFFECTS:
        return EXPRESSION_EMOTION_EFFECTS[label]
    return expression_entry_for_visual_style(visual_style) or STYLE_FALLBACK_EFFECT


def _clamp_fade(fade_duration: float, interval_duration: float) -> Tuple[float, float, float]:
    fade_in = max(0.08, min(0.32, float(fade_duration)))
    fade_out_start = max(0.0, float(interval_duration) - fade_in)
    return fade_in, fade_out_start, float(interval_duration)


def build_expression_overlay_chain(
    expression: str,
    fade_duration: float,
    interval_duration: float,
    visual_style: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build reliable FFmpeg filters (scale to fixed width, rgba, fade).
    Avoids rotate / dynamic overlay-y expressions that often fail in filter_complex.
    """
    effect = resolve_expression_effect(expression, visual_style)
    fade_in, fade_out_start, _duration = _clamp_fade(fade_duration, interval_duration)
    fd = fade_in
    fos = fade_out_start
    max_w = EXPR_MAX_WIDTH

    # Base: scale to max width, keep alpha
    base_scale = f"scale='min({max_w}\\,iw)':-1"

    if effect == "pop_in":
        scale = (
            f"{base_scale},scale='min({max_w}\\,iw*(0.72+0.28*min(1\\,t/{fd:.3f})))':-1"
        )
        overlay_y = "H-h-240"
    elif effect == "bounce_in":
        scale = (
            f"{base_scale},scale='min({max_w}\\,iw*(0.78+0.22*min(1\\,t/{fd:.3f})))':-1"
        )
        overlay_y = "H-h-230"
    elif effect == "shake_in":
        scale = (
            f"{base_scale},scale='min({max_w}\\,iw*(0.88+0.12*min(1\\,t/{fd:.3f})))':-1"
        )
        overlay_y = "H-h-220"
    elif effect == "snap_in":
        scale = (
            f"{base_scale},scale='min({max_w}\\,iw*(0.65+0.35*min(1\\,t/{fd:.3f})))':-1"
        )
        overlay_y = "H-h-250"
    elif effect == "fade_rise":
        scale = f"{base_scale},scale='min({max_w}\\,iw)':-1"
        overlay_y = "H-h-170"
    elif effect == "slide_in":
        scale = (
            f"{base_scale},scale='min({max_w}\\,iw*(0.9+0.1*min(1\\,t/{fd:.3f})))':-1"
        )
        overlay_y = "H-h-210"
    elif effect == "fade_soft":
        scale = f"{base_scale},scale='min({max_w}\\,iw)':-1"
        overlay_y = "H-h-190"
    else:  # fade_scale
        scale = (
            f"{base_scale},scale='min({max_w}\\,iw*(0.94+0.06*min(1\\,t/{fd:.3f})))':-1"
        )
        overlay_y = "H-h-220"

    fade = f"fade=t=in:st=0:d={fd:.3f}:alpha=1,fade=t=out:st={fos:.3f}:d={fd:.3f}:alpha=1"

    return {
        "effect": effect,
        "scale_expr": scale,
        "fade_expr": fade,
        "overlay_y": overlay_y,
    }


def format_expression_filter_step(
    img_label: str,
    expr_label: str,
    expression: str,
    fade_duration: float,
    interval_duration: float,
    start: float,
    visual_style: Optional[str] = None,
) -> str:
    chain = build_expression_overlay_chain(
        expression, fade_duration, interval_duration, visual_style=visual_style
    )
    return (
        f"{img_label}{chain['scale_expr']},format=rgba,"
        f"{chain['fade_expr']},setpts=PTS+{start:.3f}/TB{expr_label}"
    )


def format_expression_overlay(
    last_label: str,
    expr_label: str,
    expression: str,
    fade_duration: float,
    interval_duration: float,
    start: float,
    end: float,
    overlay_step: int,
    visual_style: Optional[str] = None,
) -> Tuple[str, str]:
    chain = build_expression_overlay_chain(
        expression, fade_duration, interval_duration, visual_style=visual_style
    )
    enable_expr = f"between(t,{start},{end})"
    out_label = f"[bg{overlay_step}]"
    overlay = (
        f"{last_label}{expr_label}overlay=x=(W-w)/2:y={chain['overlay_y']}:"
        f"enable='{enable_expr}':eval=frame{out_label}"
    )
    return overlay, out_label
