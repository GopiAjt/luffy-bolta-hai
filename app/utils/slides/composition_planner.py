"""Composition planner — multi-layer visual compositions for storyboard beats.

Production editors don't show a single image per beat.  They layer:
**background → foreground/character → effects → motion → text placement**.

This module upgrades the simple ``composition_layers`` field into a full
multi-layer production composition.

Example::

    >>> planner = CompositionPlanner()
    >>> comp = planner.plan(beat)
    >>> for layer in comp.layers:
    ...     print(f"  z={layer.z_index} [{layer.category}] {layer.name}: {layer.source}")
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class CompositionLayer:
    """A single layer in the visual composition stack."""

    name: str                  # layer identifier
    category: str              # background | foreground | effect | text | motion
    z_index: int               # render order (0 = bottom, higher = on top)
    source: str                # asset query, text content, or effect name
    layer_type: str            # image, shape, text, particle, gradient, blur, vignette
    fit: str                   # cover, contain, safe_contain, center, stretch
    opacity: float             # 0–1
    blend_mode: str            # normal, multiply, screen, overlay, soft_light
    position: Dict[str, Any]   # {x, y, width, height, anchor}
    animation: Dict[str, Any]  # {type, duration, delay, easing}
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["opacity"] = round(d["opacity"], 2)
        return d


@dataclass
class Composition:
    """Full multi-layer composition for a single beat."""

    layers: List[CompositionLayer]
    beat_type: str
    visual_role: str
    layout_mode: str
    reasoning: str
    layer_count: int = 0

    def __post_init__(self):
        self.layer_count = len(self.layers)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layers": [l.to_dict() for l in self.layers],
            "beat_type": self.beat_type,
            "visual_role": self.visual_role,
            "layout_mode": self.layout_mode,
            "layer_count": self.layer_count,
            "reasoning": self.reasoning,
        }

    @property
    def by_category(self) -> Dict[str, List[CompositionLayer]]:
        groups: Dict[str, List[CompositionLayer]] = {}
        for layer in self.layers:
            groups.setdefault(layer.category, []).append(layer)
        return groups


# ── Position presets ────────────────────────────────────────────────

_POS_FULL = {"x": 0, "y": 0, "width": "100%", "height": "100%", "anchor": "top_left"}
_POS_CENTER = {"x": "50%", "y": "50%", "width": "80%", "height": "80%", "anchor": "center"}
_POS_LOWER_THIRD = {"x": 0, "y": "70%", "width": "100%", "height": "30%", "anchor": "top_left"}
_POS_UPPER_THIRD = {"x": 0, "y": 0, "width": "100%", "height": "35%", "anchor": "top_left"}
_POS_LEFT_HALF = {"x": 0, "y": 0, "width": "50%", "height": "100%", "anchor": "top_left"}
_POS_RIGHT_HALF = {"x": "50%", "y": 0, "width": "50%", "height": "100%", "anchor": "top_left"}
_POS_HEADLINE = {"x": "10%", "y": "15%", "width": "80%", "height": "auto", "anchor": "top_left"}
_POS_SUBTITLE = {"x": "5%", "y": "80%", "width": "90%", "height": "auto", "anchor": "top_left"}
_POS_BADGE = {"x": "75%", "y": "5%", "width": "20%", "height": "auto", "anchor": "top_left"}

# ── Animation presets ───────────────────────────────────────────────

_ANIM_NONE = {"type": "none", "duration": 0, "delay": 0, "easing": "linear"}
_ANIM_FADE_IN = {"type": "fade_in", "duration": 0.4, "delay": 0, "easing": "ease_out"}
_ANIM_SLIDE_UP = {"type": "slide_up", "duration": 0.5, "delay": 0.2, "easing": "ease_out_cubic"}
_ANIM_ZOOM_IN = {"type": "zoom_in", "duration": 0.6, "delay": 0, "easing": "ease_in_out"}
_ANIM_POP = {"type": "pop_scale", "duration": 0.3, "delay": 0.1, "easing": "ease_out_back"}
_ANIM_DRIFT = {"type": "slow_drift", "duration": 3.0, "delay": 0, "easing": "linear"}
_ANIM_PULSE = {"type": "pulse_glow", "duration": 2.0, "delay": 0.3, "easing": "ease_in_out_sine"}
_ANIM_TYPEWRITER = {"type": "typewriter", "duration": 1.2, "delay": 0.4, "easing": "linear"}


# ── Beat × role composition rules ──────────────────────────────────

# Maps (beat_type, visual_role) → composition strategy name.
_COMPOSITION_STRATEGIES: Dict[Tuple[str, str], str] = {
    # Hook beats — maximum impact.
    ("hook", "character"):    "hero_reveal",
    ("hook", "title_card"):   "title_splash",
    ("hook", "section_card"): "title_splash",
    ("hook", "symbol"):       "symbol_dramatic",
    ("hook", "location"):     "establishing_wide",
    # Setup — context building.
    ("setup", "character"):   "character_context",
    ("setup", "location"):    "establishing_wide",
    ("setup", "evidence"):    "evidence_inspect",
    # Evidence — proof and detail.
    ("evidence", "evidence"):     "evidence_inspect",
    ("evidence", "character"):    "character_context",
    ("evidence", "object"):       "object_focus",
    ("evidence", "quote_card"):   "quote_emphasis",
    # Reveal — dramatic unveiling.
    ("reveal", "character"):  "hero_reveal",
    ("reveal", "symbol"):     "symbol_dramatic",
    ("reveal", "evidence"):   "evidence_inspect",
    # Payoff — climactic impact.
    ("payoff", "character"):  "hero_reveal",
    ("payoff", "symbol"):     "symbol_dramatic",
    ("payoff", "location"):   "establishing_wide",
    # Warning — dread.
    ("warning", "character"): "character_dread",
    ("warning", "symbol"):    "symbol_dramatic",
    # CTA — clean and readable.
    ("cta", "cta_card"):      "cta_clean",
    ("cta", "section_card"):  "cta_clean",
    # Comparison / timeline.
    ("evidence", "comparison"): "split_compare",
    ("evidence", "timeline"):   "timeline_progression",
}

# ── Effect palettes by emotion ──────────────────────────────────────

_EMOTION_EFFECTS: Dict[str, List[Dict[str, Any]]] = {
    "grief": [
        {"name": "desaturation", "type": "color_filter", "params": {"saturation": 0.3, "brightness": 0.85}},
        {"name": "rain_particles", "type": "particle", "params": {"density": 0.4, "speed": "slow"}},
    ],
    "tension": [
        {"name": "vignette_dark", "type": "vignette", "params": {"intensity": 0.6, "color": "#000000"}},
        {"name": "film_grain", "type": "texture", "params": {"intensity": 0.25}},
    ],
    "fear": [
        {"name": "vignette_red", "type": "vignette", "params": {"intensity": 0.5, "color": "#330000"}},
        {"name": "chromatic_shift", "type": "distortion", "params": {"offset": 2}},
    ],
    "rage": [
        {"name": "red_tint", "type": "color_filter", "params": {"hue_shift": -15, "saturation": 1.3}},
        {"name": "impact_lines", "type": "overlay", "params": {"style": "speed_lines"}},
    ],
    "shock": [
        {"name": "flash_white", "type": "flash", "params": {"duration": 0.15}},
        {"name": "shake_effect", "type": "camera_shake", "params": {"amplitude": 4}},
    ],
    "hope": [
        {"name": "warm_glow", "type": "color_filter", "params": {"temperature": 20, "brightness": 1.1}},
        {"name": "light_rays", "type": "particle", "params": {"style": "god_rays", "opacity": 0.3}},
    ],
    "triumph": [
        {"name": "golden_glow", "type": "color_filter", "params": {"temperature": 30, "saturation": 1.2}},
        {"name": "sparkle_particles", "type": "particle", "params": {"style": "sparkle", "density": 0.3}},
    ],
    "intrigue": [
        {"name": "vignette_subtle", "type": "vignette", "params": {"intensity": 0.35}},
        {"name": "mist_overlay", "type": "particle", "params": {"style": "mist", "opacity": 0.2}},
    ],
    "revelation": [
        {"name": "light_bloom", "type": "bloom", "params": {"intensity": 0.4, "threshold": 0.7}},
        {"name": "radial_blur", "type": "blur", "params": {"center": True, "amount": 3}},
    ],
}

_DEFAULT_EFFECTS = [
    {"name": "subtle_vignette", "type": "vignette", "params": {"intensity": 0.2}},
]


# ── CompositionPlanner ──────────────────────────────────────────────


class CompositionPlanner:
    """Plans multi-layer visual compositions for storyboard beats.

    Upgrades from ``1 beat = 1 image`` to production-quality:
    ``background + foreground/character + effects + motion + text placement``.

    Parameters
    ----------
    video_profile : str
        "short_vertical" or "long_youtube" (affects layout).
    max_layers : int
        Maximum layers to generate (default 8).
    enable_effects : bool
        Whether to add emotion-driven effects (default True).
    """

    def __init__(
        self,
        video_profile: str = "short_vertical",
        max_layers: int = 8,
        enable_effects: bool = True,
    ):
        self.video_profile = video_profile.strip().lower()
        if self.video_profile not in ("short_vertical", "long_youtube"):
            self.video_profile = "short_vertical"
        self.max_layers = max(3, max_layers)
        self.enable_effects = enable_effects

    def plan(self, beat: Dict[str, Any]) -> Composition:
        """Plan a multi-layer composition for a single beat/beat.

        Parameters
        ----------
        beat : dict
            A storyboard beat with beat_type, visual_role, emotion_state,
            image_search_query, text_overlay, emphasis_words, etc.

        Returns
        -------
        Composition
        """
        beat_type = (beat.get("beat_type") or "evidence").strip().lower()
        visual_role = (beat.get("visual_role") or "character").strip().lower()
        layout_mode = (beat.get("layout_mode") or "safe_subject").strip().lower()
        emotion_state = beat.get("emotion_state") or {}
        emotion = ""
        intensity = 0.4
        if isinstance(emotion_state, dict):
            emotion = (emotion_state.get("emotion") or "").strip().lower()
            intensity = float(emotion_state.get("intensity") or 0.4)

        # Resolve composition strategy.
        strategy = _COMPOSITION_STRATEGIES.get(
            (beat_type, visual_role),
            _COMPOSITION_STRATEGIES.get(("evidence", visual_role), "character_context"),
        )

        layers: List[CompositionLayer] = []
        z = 0

        # ── Layer 1: Background ──────────────────────────────────────
        bg_layer = self._build_background(beat, beat_type, visual_role, layout_mode, strategy)
        bg_layer.z_index = z
        layers.append(bg_layer)
        z += 1

        # ── Layer 2: Background treatment (gradient/blur overlay) ────
        treatment = self._build_bg_treatment(beat_type, emotion, intensity, strategy)
        if treatment:
            treatment.z_index = z
            layers.append(treatment)
            z += 1

        # ── Layer 3: Foreground / character ──────────────────────────
        fg_layer = self._build_foreground(beat, beat_type, visual_role, strategy)
        if fg_layer:
            fg_layer.z_index = z
            layers.append(fg_layer)
            z += 1

        # ── Layer 4: Effects (emotion-driven) ────────────────────────
        if self.enable_effects:
            effect_layers = self._build_effects(emotion, intensity, beat_type)
            for eff in effect_layers:
                if z < self.max_layers:
                    eff.z_index = z
                    layers.append(eff)
                    z += 1

        # ── Layer 5: Editorial panel / shape ─────────────────────────
        panel = self._build_editorial_panel(beat, layout_mode, strategy)
        if panel and z < self.max_layers:
            panel.z_index = z
            layers.append(panel)
            z += 1

        # ── Layer 6: Text placement ──────────────────────────────────
        text_layers = self._build_text_layers(beat, beat_type, visual_role, strategy)
        for tl in text_layers:
            if z < self.max_layers:
                tl.z_index = z
                layers.append(tl)
                z += 1

        # ── Layer 7: Motion metadata ─────────────────────────────────
        motion_layer = self._build_motion_layer(beat, beat_type, emotion, intensity)
        if motion_layer and z < self.max_layers:
            motion_layer.z_index = z
            layers.append(motion_layer)
            z += 1

        reasoning = self._build_reasoning(strategy, beat_type, visual_role, emotion, len(layers))

        return Composition(
            layers=layers,
            beat_type=beat_type,
            visual_role=visual_role,
            layout_mode=layout_mode,
            reasoning=reasoning,
        )

    def plan_sequence(
        self,
        beats: Sequence[Dict[str, Any]],
    ) -> List[Composition]:
        """Plan compositions for a full storyboard beat sequence."""
        return [self.plan(beat) for beat in beats]

    # ── Background layer ─────────────────────────────────────────────

    def _build_background(
        self, beat: Dict, beat_type: str, visual_role: str,
        layout_mode: str, strategy: str,
    ) -> CompositionLayer:
        query = beat.get("image_search_query", "")
        entities = (beat.get("context_entities") or [])

        # For character-centric strategies, background is the environment.
        if strategy in ("hero_reveal", "character_context", "character_dread"):
            bg_source = f"environment: {query}" if query else "abstract background"
        elif strategy == "establishing_wide":
            bg_source = f"wide shot: {query}" if query else "landscape establishing"
        else:
            bg_source = query or "abstract background"

        fit = "cover"
        if layout_mode in ("safe_subject", "horizontal_feature"):
            fit = "safe_contain"

        return CompositionLayer(
            name="background",
            category="background",
            z_index=0,
            source=bg_source,
            layer_type="image",
            fit=fit,
            opacity=1.0,
            blend_mode="normal",
            position=dict(_POS_FULL),
            animation=dict(_ANIM_DRIFT),
            metadata={"role": "environment", "search_entities": entities[:3]},
        )

    # ── Background treatment ─────────────────────────────────────────

    def _build_bg_treatment(
        self, beat_type: str, emotion: str, intensity: float, strategy: str,
    ) -> Optional[CompositionLayer]:
        if strategy in ("cta_clean", "title_splash"):
            return CompositionLayer(
                name="bg_darken",
                category="background",
                z_index=1,
                source="gradient_overlay",
                layer_type="gradient",
                fit="cover",
                opacity=0.55,
                blend_mode="multiply",
                position=dict(_POS_FULL),
                animation=dict(_ANIM_NONE),
                metadata={"gradient": "top_transparent_to_bottom_black"},
            )

        if emotion in ("grief", "tension", "fear", "dread"):
            return CompositionLayer(
                name="bg_mood",
                category="background",
                z_index=1,
                source="mood_overlay",
                layer_type="gradient",
                fit="cover",
                opacity=min(0.5, intensity * 0.5),
                blend_mode="multiply",
                position=dict(_POS_FULL),
                animation=dict(_ANIM_NONE),
                metadata={"mood": emotion},
            )

        if beat_type in ("reveal", "payoff"):
            return CompositionLayer(
                name="bg_blur",
                category="background",
                z_index=1,
                source="depth_blur",
                layer_type="blur",
                fit="cover",
                opacity=0.3,
                blend_mode="normal",
                position=dict(_POS_FULL),
                animation=dict(_ANIM_NONE),
                metadata={"blur_radius": 8, "purpose": "focus_subject"},
            )

        return None

    # ── Foreground / character layer ─────────────────────────────────

    def _build_foreground(
        self, beat: Dict, beat_type: str, visual_role: str, strategy: str,
    ) -> Optional[CompositionLayer]:
        if strategy == "cta_clean":
            return None  # CTA has no foreground character.

        query = beat.get("image_search_query", "")
        entities = beat.get("context_entities") or []

        if strategy in ("hero_reveal", "character_context", "character_dread"):
            char_name = entities[0] if entities else "character"
            position = dict(_POS_CENTER)
            animation = dict(_ANIM_ZOOM_IN) if strategy == "hero_reveal" else dict(_ANIM_FADE_IN)
            if strategy == "character_dread":
                animation = {"type": "slow_drift", "duration": 4.0, "delay": 0, "easing": "ease_in_sine"}

            return CompositionLayer(
                name="foreground_character",
                category="foreground",
                z_index=2,
                source=f"character: {char_name}",
                layer_type="image",
                fit="contain",
                opacity=1.0,
                blend_mode="normal",
                position=position,
                animation=animation,
                metadata={"character": char_name, "cutout": True},
            )

        if strategy == "symbol_dramatic":
            return CompositionLayer(
                name="foreground_symbol",
                category="foreground",
                z_index=2,
                source=f"symbol: {query}",
                layer_type="image",
                fit="contain",
                opacity=1.0,
                blend_mode="normal",
                position=dict(_POS_CENTER),
                animation=dict(_ANIM_POP),
                metadata={"type": "symbol"},
            )

        if strategy == "object_focus":
            return CompositionLayer(
                name="foreground_object",
                category="foreground",
                z_index=2,
                source=f"object: {query}",
                layer_type="image",
                fit="contain",
                opacity=1.0,
                blend_mode="normal",
                position=dict(_POS_CENTER),
                animation=dict(_ANIM_ZOOM_IN),
                metadata={"type": "object", "inspect": True},
            )

        if strategy in ("evidence_inspect", "quote_emphasis"):
            return CompositionLayer(
                name="foreground_evidence",
                category="foreground",
                z_index=2,
                source=f"evidence: {query}",
                layer_type="image",
                fit="safe_contain",
                opacity=1.0,
                blend_mode="normal",
                position=dict(_POS_CENTER),
                animation=dict(_ANIM_FADE_IN),
                metadata={"type": "evidence"},
            )

        if strategy == "split_compare":
            return CompositionLayer(
                name="foreground_compare_left",
                category="foreground",
                z_index=2,
                source=f"compare_a: {query}",
                layer_type="image",
                fit="contain",
                opacity=1.0,
                blend_mode="normal",
                position=dict(_POS_LEFT_HALF),
                animation=dict(_ANIM_FADE_IN),
                metadata={"type": "comparison", "side": "left"},
            )

        if strategy == "establishing_wide":
            return None  # Background IS the foreground for wide shots.

        return None

    # ── Effects layers ───────────────────────────────────────────────

    def _build_effects(
        self, emotion: str, intensity: float, beat_type: str,
    ) -> List[CompositionLayer]:
        layers: List[CompositionLayer] = []

        effect_defs = _EMOTION_EFFECTS.get(emotion, _DEFAULT_EFFECTS)

        # Scale number of effects by intensity.
        max_effects = 1 if intensity < 0.5 else 2
        for i, effect_def in enumerate(effect_defs[:max_effects]):
            layers.append(CompositionLayer(
                name=effect_def["name"],
                category="effect",
                z_index=0,  # Will be reassigned.
                source=effect_def["type"],
                layer_type=effect_def["type"],
                fit="cover",
                opacity=min(0.8, intensity * 0.6 + 0.1),
                blend_mode="screen" if effect_def["type"] in ("particle", "bloom", "flash") else "normal",
                position=dict(_POS_FULL),
                animation=dict(_ANIM_PULSE) if effect_def["type"] == "particle" else dict(_ANIM_FADE_IN),
                metadata=effect_def.get("params", {}),
            ))

        return layers

    # ── Editorial panel ──────────────────────────────────────────────

    def _build_editorial_panel(
        self, beat: Dict, layout_mode: str, strategy: str,
    ) -> Optional[CompositionLayer]:
        panel_layouts = {
            "title_card", "section_card", "evidence_card",
            "quote_card", "split_context",
        }
        if layout_mode not in panel_layouts and strategy not in ("title_splash", "cta_clean", "quote_emphasis"):
            return None

        style = layout_mode if layout_mode in panel_layouts else "title_card"

        return CompositionLayer(
            name="editorial_panel",
            category="text",
            z_index=0,
            source="panel",
            layer_type="shape",
            fit="cover",
            opacity=0.75,
            blend_mode="multiply",
            position=dict(_POS_LOWER_THIRD) if style in ("evidence_card", "quote_card") else dict(_POS_FULL),
            animation=dict(_ANIM_FADE_IN),
            metadata={"style": style, "corner_radius": 0},
        )

    # ── Text layers ──────────────────────────────────────────────────

    def _build_text_layers(
        self, beat: Dict, beat_type: str, visual_role: str, strategy: str,
    ) -> List[CompositionLayer]:
        layers: List[CompositionLayer] = []

        # Headline / text overlay.
        text_overlay = beat.get("text_overlay", "")
        if text_overlay:
            position = dict(_POS_HEADLINE) if strategy in ("title_splash", "cta_clean") else dict(_POS_LOWER_THIRD)
            animation = dict(_ANIM_TYPEWRITER) if beat_type == "hook" else dict(_ANIM_SLIDE_UP)

            layers.append(CompositionLayer(
                name="headline",
                category="text",
                z_index=0,
                source=text_overlay,
                layer_type="text",
                fit="contain",
                opacity=1.0,
                blend_mode="normal",
                position=position,
                animation=animation,
                metadata={
                    "font_weight": "bold",
                    "font_size": "xl" if beat_type in ("hook", "payoff") else "lg",
                    "text_align": "center",
                    "text_shadow": True,
                    "max_lines": 2,
                },
            ))

        # Emphasis words / tags.
        emphasis = beat.get("emphasis_words") or []
        if emphasis:
            layers.append(CompositionLayer(
                name="emphasis_tags",
                category="text",
                z_index=0,
                source=", ".join(emphasis[:4]),
                layer_type="text_tags",
                fit="contain",
                opacity=0.9,
                blend_mode="normal",
                position=dict(_POS_BADGE) if text_overlay else dict(_POS_LOWER_THIRD),
                animation=dict(_ANIM_POP),
                metadata={
                    "font_weight": "semibold",
                    "font_size": "sm",
                    "style": "pill_tags",
                    "max_tags": 4,
                },
            ))

        # Subtitle placement hint (always present for voiceover videos).
        layers.append(CompositionLayer(
            name="subtitle_zone",
            category="text",
            z_index=0,
            source="subtitle",
            layer_type="reserved_zone",
            fit="contain",
            opacity=0.0,
            blend_mode="normal",
            position=dict(_POS_SUBTITLE),
            animation=dict(_ANIM_NONE),
            metadata={"purpose": "subtitle_safe_area", "avoid_overlap": True},
        ))

        return layers

    # ── Motion layer ─────────────────────────────────────────────────

    def _build_motion_layer(
        self, beat: Dict, beat_type: str, emotion: str, intensity: float,
    ) -> Optional[CompositionLayer]:
        motion_preset = (beat.get("motion_preset") or "").strip().lower()

        # Derive camera motion parameters.
        camera_style = {
            "hook": "push_in_fast",
            "reveal": "slow_zoom_reveal",
            "payoff": "dramatic_push",
            "setup": "slow_pan_right",
            "evidence": "subtle_drift",
            "escalation": "accelerating_push",
            "warning": "uneasy_drift",
            "cta": "static_hold",
        }.get(beat_type, "subtle_drift")

        speed = min(1.0, 0.3 + intensity * 0.5)

        return CompositionLayer(
            name="camera_motion",
            category="motion",
            z_index=0,
            source=camera_style,
            layer_type="camera",
            fit="cover",
            opacity=1.0,
            blend_mode="normal",
            position=dict(_POS_FULL),
            animation={
                "type": camera_style,
                "duration": 3.0 + (1.0 - speed) * 2.0,
                "delay": 0,
                "easing": "ease_in_out_cubic" if speed < 0.6 else "ease_out_expo",
            },
            metadata={
                "motion_preset": motion_preset or camera_style,
                "speed": round(speed, 2),
                "intensity": round(intensity, 2),
                "parallax": beat_type in ("hook", "reveal", "payoff"),
            },
        )

    # ── Reasoning ────────────────────────────────────────────────────

    def _build_reasoning(
        self, strategy: str, beat_type: str, visual_role: str,
        emotion: str, layer_count: int,
    ) -> str:
        return (
            f"Strategy '{strategy}' for {beat_type}/{visual_role} "
            f"(emotion: {emotion or 'neutral'}). "
            f"{layer_count} layers: bg + treatment + foreground + effects + text + motion."
        )

    # ── Introspection ────────────────────────────────────────────────

    @staticmethod
    def available_strategies() -> List[str]:
        return sorted(set(_COMPOSITION_STRATEGIES.values()))

    @staticmethod
    def supported_emotions() -> List[str]:
        return sorted(_EMOTION_EFFECTS.keys())


# ── Convenience function ─────────────────────────────────────────────


def plan_composition(
    beat: Dict[str, Any],
    video_profile: str = "short_vertical",
    **kwargs,
) -> Composition:
    """Shortcut: plan composition for a single beat."""
    return CompositionPlanner(video_profile=video_profile, **kwargs).plan(beat)
