"""Shared visual-effect preset helpers for final video generation."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional


DEFAULT_VISUAL_STYLE = "clean_pro"

VISUAL_STYLE_PRESETS: Dict[str, Dict] = {
    "clean_pro": {
        "transition_mood": "clean",
        "transitions": ["crossfade", "fade_eased", "zoom_dissolve"],
        "transition_weights": [0.56, 0.30, 0.14],
        "motion": ["slow_push", "stable_pan", "hold_still"],
        "global_fx": [],
        "subtitle_style": "clean_pro",
        "expression_entry": "fade_scale",
        "color_grade": None,
    },
    "manga_hype": {
        "transition_mood": "manga",
        "transitions": ["zoom_dissolve", "whip_pan_right", "whip_pan_left", "radial_wipe", "iris_wipe"],
        "transition_weights": [0.34, 0.16, 0.16, 0.18, 0.16],
        "motion": ["impact_zoom", "diagonal_pan", "slow_push"],
        "global_fx": ["grain", "vignette"],
        "subtitle_style": "manga_panel",
        "expression_entry": "pop_in",
        "color_grade": "manga",
    },
    "emotional": {
        "transition_mood": "dramatic",
        "transitions": ["fade_eased", "crossfade", "iris_wipe"],
        "transition_weights": [0.46, 0.40, 0.14],
        "motion": ["hold_still", "slow_push", "pull_out"],
        "global_fx": ["vignette"],
        "subtitle_style": "emotional",
        "expression_entry": "fade_rise",
        "color_grade": "warm",
    },
    "dark_lore": {
        "transition_mood": "dramatic",
        "transitions": ["crossfade", "iris_wipe", "zoom_dissolve", "radial_wipe"],
        "transition_weights": [0.34, 0.24, 0.24, 0.18],
        "motion": ["slow_push", "hold_still", "diagonal_pan"],
        "global_fx": ["vignette", "cool_grade"],
        "subtitle_style": "void_century",
        "expression_entry": "fade_scale",
        "color_grade": "cool",
    },
    "action": {
        "transition_mood": "hype",
        "transitions": ["whip_pan_right", "whip_pan_left", "motion_slide_right", "motion_slide_left", "zoom_dissolve"],
        "transition_weights": [0.22, 0.22, 0.20, 0.20, 0.16],
        "motion": ["impact_zoom", "diagonal_pan", "slow_push"],
        "global_fx": ["grain"],
        "subtitle_style": "yonko_hype",
        "expression_entry": "pop_in",
        "color_grade": "contrast",
    },
}

REVEAL_TERMS = {
    "reveal", "revealed", "truth", "secret", "hidden", "clue", "twist",
    "shocking", "dangerous", "impossible", "real reason",
}
PAYOFF_TERMS = {"payoff", "means", "proves", "changes", "because", "therefore"}
CTA_TERMS = {"follow", "comment", "tell me", "subscribe", "like"}
EMOTIONAL_TERMS = {"farewell", "death", "promise", "cry", "heart", "dream", "nakama"}


def normalize_visual_style(visual_style: Optional[str]) -> str:
    style = (visual_style or DEFAULT_VISUAL_STYLE).strip().lower()
    return style if style in VISUAL_STYLE_PRESETS else DEFAULT_VISUAL_STYLE


def get_visual_preset(visual_style: Optional[str]) -> Dict:
    return VISUAL_STYLE_PRESETS[normalize_visual_style(visual_style)]


def classify_beat(text: str, index: int = 0, total: int = 1) -> str:
    lowered = (text or "").lower()
    if any(term in lowered for term in CTA_TERMS) or index >= max(0, total - 1):
        return "cta"
    if any(term in lowered for term in REVEAL_TERMS):
        return "reveal"
    if any(term in lowered for term in PAYOFF_TERMS):
        return "payoff"
    if any(term in lowered for term in EMOTIONAL_TERMS):
        return "payoff"
    if index == 0:
        return "hook"
    return "evidence"


def deterministic_pick(items: List[str], seed_text: str, offset: int = 0) -> str:
    if not items:
        return ""
    digest = hashlib.sha1(f"{seed_text}:{offset}".encode("utf-8")).hexdigest()
    return items[int(digest[:8], 16) % len(items)]


def choose_motion_preset(visual_style: Optional[str], beat: str, index: int = 0) -> str:
    style = normalize_visual_style(visual_style)
    if beat == "hook":
        return "slow_push"
    if beat == "reveal":
        return "impact_zoom" if style in {"manga_hype", "action"} else "slow_push"
    if beat == "payoff":
        return "hold_still" if style == "emotional" else "pull_out"
    if beat == "cta":
        return "hold_still"
    return deterministic_pick(get_visual_preset(style)["motion"], style, index) or "stable_pan"


def choose_visual_transition(
    visual_style: Optional[str],
    beat: str = "evidence",
    last_transition: Optional[str] = None,
    index: int = 0,
) -> str:
    preset = get_visual_preset(visual_style)
    transitions = list(preset["transitions"])
    if beat in {"hook", "payoff"}:
        transitions = [t for t in transitions if t in {"fade", "fade_eased", "crossfade", "iris_wipe"}] or transitions
    if beat == "reveal":
        transitions = [t for t in transitions if t in {"zoom_dissolve", "whip_pan_right", "whip_pan_left", "radial_wipe"}] or transitions
    if last_transition and len(transitions) > 1:
        transitions = [t for t in transitions if t != last_transition] or transitions
    return deterministic_pick(transitions, f"{normalize_visual_style(visual_style)}:{beat}", index)


KEYWORD_SUBTITLE_STYLES = frozenset(
    {"clean_pro", "manga_panel", "manga_hype", "emotional", "dark_lore", "void_century", "action", "yonko_hype"}
)


def subtitle_style_for_visual_style(visual_style: Optional[str], requested_style: Optional[str] = None) -> str:
    requested = (requested_style or "").strip().lower()
    if requested and requested not in {"pro", ""}:
        return requested
    return get_visual_preset(visual_style)["subtitle_style"]


def build_global_ffmpeg_filter(visual_style: Optional[str]) -> str:
    preset = get_visual_preset(visual_style)
    filters = []
    if "vignette" in preset["global_fx"]:
        filters.append("vignette=PI/5")
    if "grain" in preset["global_fx"]:
        filters.append("noise=alls=6:allf=t+u")
    if "cool_grade" in preset["global_fx"]:
        filters.append("eq=saturation=0.92:contrast=1.06:brightness=-0.015")
    elif preset.get("color_grade") == "warm":
        filters.append("eq=saturation=1.05:contrast=1.03:brightness=0.005")
    elif preset.get("color_grade") == "contrast":
        filters.append("eq=saturation=1.08:contrast=1.08")
    elif preset.get("color_grade") == "manga":
        filters.append("eq=saturation=1.10:contrast=1.07")
    return ",".join(filters)


def expression_entry_for_visual_style(visual_style: Optional[str]) -> str:
    return get_visual_preset(visual_style)["expression_entry"]

