"""Motion planner — maps emotion + visual intent to optimal motion style.

Takes emotion state and visual intent as input and selects the best
motion style from a curated rule matrix, with intensity scaling and
anti-repetition logic.

Examples::

    >>> planner = MotionPlanner()
    >>> planner.plan("fear", "curiosity_gap")
    MotionPlan(style='dramatic_push', intensity=0.85, ...)
    >>> planner.plan("intrigue", "reversal")
    MotionPlan(style='reveal_zoom', intensity=0.88, ...)
    >>> planner.plan("curious", "proof")
    MotionPlan(style='slow_pan', intensity=0.45, ...)
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class MotionPlan:
    """Output of the motion planner for a single beat."""

    style: str                 # primary motion style name
    intensity: float           # motion intensity [0, 1]
    camera_goal: str           # what the camera should achieve
    transition: str            # transition into this beat
    easing: str                # easing curve name
    duration_modifier: float   # 1.0 = normal, 1.5 = slow, 0.7 = snappy
    reasoning: str             # human-readable why this was chosen

    def to_dict(self) -> Dict[str, Any]:
        return {
            "style": self.style,
            "intensity": round(self.intensity, 3),
            "camera_goal": self.camera_goal,
            "transition": self.transition,
            "easing": self.easing,
            "duration_modifier": round(self.duration_modifier, 2),
            "reasoning": self.reasoning,
        }


# ── Motion style catalog ────────────────────────────────────────────

@dataclass(frozen=True)
class MotionStyle:
    """Definition of a single motion style."""
    name: str
    camera_goal: str
    base_intensity: float
    easing: str
    duration_modifier: float
    transition: str
    description: str


# All available motion styles with their properties.
MOTION_STYLES: Dict[str, MotionStyle] = {
    "dramatic_push": MotionStyle(
        name="dramatic_push",
        camera_goal="urgent_face_lock",
        base_intensity=0.85,
        easing="ease_out_expo",
        duration_modifier=0.8,
        transition="zoom_dissolve",
        description="Fast push into subject — urgency, dread, confrontation",
    ),
    "reveal_zoom": MotionStyle(
        name="reveal_zoom",
        camera_goal="slow_reveal_to_subject",
        base_intensity=0.88,
        easing="ease_in_out_cubic",
        duration_modifier=1.1,
        transition="zoom_dissolve",
        description="Controlled zoom revealing key information",
    ),
    "slow_pan": MotionStyle(
        name="slow_pan",
        camera_goal="inspect_evidence",
        base_intensity=0.45,
        easing="ease_in_out_quad",
        duration_modifier=1.3,
        transition="crossfade",
        description="Gentle lateral pan — evidence, details, calm exploration",
    ),
    "subject_push": MotionStyle(
        name="subject_push",
        camera_goal="face_lock_push",
        base_intensity=0.65,
        easing="ease_out_quad",
        duration_modifier=1.0,
        transition="crossfade",
        description="Standard push towards subject — moderate emphasis",
    ),
    "wide_pan": MotionStyle(
        name="wide_pan",
        camera_goal="wide_context_sweep",
        base_intensity=0.50,
        easing="ease_in_out_sine",
        duration_modifier=1.4,
        transition="crossfade",
        description="Wide sweeping pan — establishing shots, landscapes",
    ),
    "tension_drift": MotionStyle(
        name="tension_drift",
        camera_goal="uneasy_slow_drift",
        base_intensity=0.70,
        easing="ease_in_sine",
        duration_modifier=1.2,
        transition="crossfade",
        description="Slow uneasy drift — suspense, mystery, unease",
    ),
    "impact_shake": MotionStyle(
        name="impact_shake",
        camera_goal="impact_reaction",
        base_intensity=0.92,
        easing="ease_out_bounce",
        duration_modifier=0.6,
        transition="hard_cut",
        description="Sharp impact shake — shock, rage, explosion",
    ),
    "hopeful_rise": MotionStyle(
        name="hopeful_rise",
        camera_goal="ascending_reveal",
        base_intensity=0.60,
        easing="ease_out_cubic",
        duration_modifier=1.1,
        transition="fade_eased",
        description="Gentle upward movement — hope, triumph, dawn",
    ),
    "grief_hold": MotionStyle(
        name="grief_hold",
        camera_goal="still_contemplation",
        base_intensity=0.35,
        easing="ease_in_out_quad",
        duration_modifier=1.5,
        transition="fade_eased",
        description="Near-still hold — grief, loss, silence",
    ),
    "title_card_hold": MotionStyle(
        name="title_card_hold",
        camera_goal="readable_hold",
        base_intensity=0.15,
        easing="linear",
        duration_modifier=1.0,
        transition="fade_eased",
        description="Static or minimal hold for text readability",
    ),
    "evidence_hold": MotionStyle(
        name="evidence_hold",
        camera_goal="inspect_detail",
        base_intensity=0.40,
        easing="ease_in_out_quad",
        duration_modifier=1.2,
        transition="crossfade",
        description="Steady hold with subtle push — evidence examination",
    ),
    "climax_zoom": MotionStyle(
        name="climax_zoom",
        camera_goal="dramatic_climax_zoom",
        base_intensity=0.95,
        easing="ease_in_expo",
        duration_modifier=0.7,
        transition="zoom_dissolve",
        description="Maximum intensity zoom — payoff, climax, transformation",
    ),
    "contrast_cut": MotionStyle(
        name="contrast_cut",
        camera_goal="abrupt_juxtaposition",
        base_intensity=0.75,
        easing="ease_out_quad",
        duration_modifier=0.8,
        transition="hard_cut",
        description="Hard cut between contrasting visuals",
    ),
    "orbit_scan": MotionStyle(
        name="orbit_scan",
        camera_goal="circular_scan",
        base_intensity=0.55,
        easing="ease_in_out_sine",
        duration_modifier=1.3,
        transition="crossfade",
        description="Orbital scan around subject — comparison, timeline",
    ),
    "static_hold": MotionStyle(
        name="static_hold",
        camera_goal="no_motion",
        base_intensity=0.10,
        easing="linear",
        duration_modifier=1.0,
        transition="fade_eased",
        description="No motion — CTA, end card, simple message",
    ),
}


# ── Emotion → motion mapping matrix ─────────────────────────────────

# Primary lookup: emotion → best motion style.
# Each emotion maps to (primary_style, fallback_style).
_EMOTION_MOTION_MAP: Dict[str, Tuple[str, str]] = {
    # Negative / high-arousal
    "fear":       ("dramatic_push",  "tension_drift"),
    "tension":    ("tension_drift",  "dramatic_push"),
    "rage":       ("impact_shake",   "dramatic_push"),
    "shock":      ("impact_shake",   "contrast_cut"),
    "dread":      ("tension_drift",  "dramatic_push"),
    "danger":     ("dramatic_push",  "tension_drift"),
    # Negative / low-arousal
    "grief":      ("grief_hold",     "slow_pan"),
    "sadness":    ("grief_hold",     "slow_pan"),
    "loss":       ("grief_hold",     "tension_drift"),
    "loneliness": ("grief_hold",     "slow_pan"),
    # Positive / high-arousal
    "hope":       ("hopeful_rise",   "subject_push"),
    "joy":        ("hopeful_rise",   "reveal_zoom"),
    "triumph":    ("climax_zoom",    "hopeful_rise"),
    "freedom":    ("hopeful_rise",   "wide_pan"),
    "excitement": ("dramatic_push",  "climax_zoom"),
    # Positive / low-arousal
    "bond":       ("slow_pan",       "subject_push"),
    "calm":       ("slow_pan",       "evidence_hold"),
    "resolution": ("static_hold",    "title_card_hold"),
    # Neutral / cognitive
    "curious":    ("slow_pan",       "evidence_hold"),
    "intrigue":   ("tension_drift",  "reveal_zoom"),
    "revelation": ("reveal_zoom",    "climax_zoom"),
    "mystery":    ("tension_drift",  "slow_pan"),
    "neutral":    ("subject_push",   "slow_pan"),
}

# ── Visual intent → motion mapping ──────────────────────────────────

# Visual intent (from VisualIntentClassifier) → preferred motion style.
_INTENT_MOTION_MAP: Dict[str, str] = {
    "curiosity_gap":    "dramatic_push",
    "context_setup":    "wide_pan",
    "proof":            "slow_pan",
    "evidence":         "evidence_hold",
    "reversal":         "reveal_zoom",
    "takeaway":         "climax_zoom",
    "conversion":       "static_hold",
    "support":          "subject_push",
    "comparison":       "orbit_scan",
    "timeline":         "orbit_scan",
    "emotional_anchor": "grief_hold",
    "contrast":         "contrast_cut",
}

# ── Beat type overrides (strongest signal for certain beats) ────────

_BEAT_OVERRIDES: Dict[str, str] = {
    "hook":    "dramatic_push",
    "reveal":  "reveal_zoom",
    "payoff":  "climax_zoom",
    "cta":     "static_hold",
}

# ── Visual role → camera goal refinements ───────────────────────────

_ROLE_CAMERA_GOALS: Dict[str, str] = {
    "character":    "face_lock_push",
    "evidence":     "inspect_detail",
    "object":       "object_reveal",
    "location":     "wide_context_pan",
    "symbol":       "centered_icon_hold",
    "comparison":   "two_subject_scan",
    "timeline":     "left_to_right_progression",
    "section_card": "readable_hold",
    "quote_card":   "readable_hold",
    "cta_card":     "readable_hold",
    "title_card":   "readable_hold",
}


# ── MotionPlanner ───────────────────────────────────────────────────


class MotionHistoryTracker:
    def __init__(self):
        self.history = []
        self.counts = {}

    def add_motion(self, motion: str):
        self.history.append(motion)
        self.counts[motion] = self.counts.get(motion, 0) + 1
        if len(self.history) > 10:
            self.history.pop(0)

    def is_violation(self, motion: str, total_beats: int = 10) -> bool:
        # Prevent appearing more than twice consecutively
        if len(self.history) >= 2:
            if self.history[-1] == motion and self.history[-2] == motion:
                return True
                
        # Frequency limit: reveal_zoom max 30%
        if motion == "reveal_zoom":
            if self.counts.get(motion, 0) >= max(2, int(total_beats * 0.30)):
                return True
                
        return False

class MotionPlanner:
    """Plans optimal motion style from emotion + visual intent.

    Resolution priority:
    1. Beat-type overrides (hook/reveal/payoff/cta always get their signature motion)
    2. Emotion × intent cross-match (when both point to the same style)
    3. Emotion primary (emotion is the strongest creative signal)
    4. Intent fallback (when emotion is neutral/unknown)
    5. Default (subject_push)

    Anti-repetition: if the resolved style matches the previous beat,
    the planner picks the fallback to maintain visual variety.
    """

    def __init__(self, styles: Optional[Dict[str, MotionStyle]] = None):
        self.styles = styles or MOTION_STYLES
        self.tracker = MotionHistoryTracker()

    def plan(
        self,
        emotion: str,
        visual_intent: str = "",
        beat_type: str = "",
        visual_role: str = "",
        intensity: Optional[float] = None,
        previous_style: str = "",
        total_beats: int = 10,
    ) -> MotionPlan:
        """Select the best motion style for a beat.

        Parameters
        ----------
        emotion : str
            Emotion label (e.g., "fear", "hope", "grief").
        visual_intent : str
            Visual intent from classifier (e.g., "curiosity_gap", "proof").
        beat_type : str
            Story beat type (e.g., "hook", "reveal", "evidence").
        visual_role : str
            Visual role (e.g., "character", "evidence", "symbol").
        intensity : float, optional
            Override intensity. If None, derived from emotion + style.
        previous_style : str
            Previous beat's motion style (for anti-repetition).

        Returns
        -------
        MotionPlan
        """
        emotion_norm = (emotion or "neutral").strip().lower()
        intent_norm = (visual_intent or "").strip().lower()
        beat_norm = (beat_type or "").strip().lower()
        role_norm = (visual_role or "").strip().lower()

        style_name, reasoning = self._resolve_style(
            emotion_norm, intent_norm, beat_norm, previous_style, total_beats
        )

        self.tracker.add_motion(style_name)

        # Look up the style definition.
        style_def = self.styles.get(style_name, self.styles["subject_push"])

        # Compute intensity.
        final_intensity = self._compute_intensity(
            style_def, emotion_norm, intensity
        )

        # Refine camera goal based on visual role.
        camera_goal = _ROLE_CAMERA_GOALS.get(role_norm, style_def.camera_goal)
        # If the style has a strong camera goal, prefer it.
        if style_name in ("dramatic_push", "reveal_zoom", "climax_zoom", "impact_shake"):
            camera_goal = style_def.camera_goal

        return MotionPlan(
            style=style_def.name,
            intensity=final_intensity,
            camera_goal=camera_goal,
            transition=style_def.transition,
            easing=style_def.easing,
            duration_modifier=style_def.duration_modifier,
            reasoning=reasoning,
        )

    def plan_sequence(
        self,
        beats: Sequence[Dict[str, Any]],
    ) -> List[MotionPlan]:
        """Plan motion for a full sequence of beats.

        Each beat dict should have: ``emotion`` (or ``emotion_state``),
        ``visual_intent`` (or ``primary_intent``), ``beat_type``,
        ``visual_role``.

        Automatically handles anti-repetition across the sequence.
        """
        plans: List[MotionPlan] = []
        prev_style = ""

        for beat in beats:
            # Extract emotion.
            emotion_raw = beat.get("emotion") or beat.get("emotion_state") or {}
            if isinstance(emotion_raw, dict):
                emotion = emotion_raw.get("emotion", "neutral")
                beat_intensity = emotion_raw.get("intensity")
            else:
                emotion = str(emotion_raw)
                beat_intensity = None

            # Extract visual intent.
            intent = (
                beat.get("visual_intent")
                or beat.get("primary_intent")
                or (beat.get("visual_intent_data") or {}).get("primary_intent", "")
            )

            plan = self.plan(
                emotion=emotion,
                visual_intent=intent,
                beat_type=beat.get("beat_type", ""),
                visual_role=beat.get("visual_role", ""),
                intensity=beat_intensity,
                previous_style=prev_style,
                total_beats=len(beats),
            )
            plans.append(plan)
            prev_style = plan.style

        return plans

    # ── Internal resolution ──────────────────────────────────────────

    def _resolve_style(
        self,
        emotion: str,
        intent: str,
        beat_type: str,
        previous_style: str,
        total_beats: int = 10,
    ) -> Tuple[str, str]:
        """Resolve the best motion style and return (style_name, reasoning)."""

        # Priority 1: Beat-type override.
        if beat_type in _BEAT_OVERRIDES:
            style = _BEAT_OVERRIDES[beat_type]
            if not self.tracker.is_violation(style, total_beats) and style != previous_style:
                return style, f"Beat override: {beat_type} → {style}"

        # Priority 2: Emotion × intent agreement.
        emotion_primary, emotion_fallback = _EMOTION_MOTION_MAP.get(
            emotion, ("subject_push", "slow_pan")
        )
        intent_style = _INTENT_MOTION_MAP.get(intent, "")

        if intent_style and intent_style == emotion_primary:
            style = intent_style
            if not self.tracker.is_violation(style, total_beats) and style != previous_style:
                return style, f"Emotion+intent agree: {emotion}+{intent} → {style}"
            # Both agree but repeats — use emotion fallback.
            if not self.tracker.is_violation(emotion_fallback, total_beats):
                return emotion_fallback, f"Anti-repeat: {emotion}+{intent} agreed on {style}, using fallback {emotion_fallback}"

        # Priority 3: Emotion primary.
        if not self.tracker.is_violation(emotion_primary, total_beats) and emotion_primary != previous_style:
            return emotion_primary, f"Emotion-driven: {emotion} → {emotion_primary}"
        # Emotion primary repeats — try intent.
        if intent_style and not self.tracker.is_violation(intent_style, total_beats) and intent_style != previous_style:
            return intent_style, f"Anti-repeat via intent: {intent} → {intent_style} (emotion {emotion} would repeat)"
        # Both repeat — use emotion fallback.
        if not self.tracker.is_violation(emotion_fallback, total_beats) and emotion_fallback != previous_style:
            return emotion_fallback, f"Emotion fallback: {emotion} → {emotion_fallback} (primary would repeat)"

        # Priority 4: Intent fallback.
        if intent_style and not self.tracker.is_violation(intent_style, total_beats):
            return intent_style, f"Intent fallback: {intent} → {intent_style}"

        # Priority 5: Default fallback loop to find a safe motion
        for fallback in ["subject_push", "slow_pan", "stable_pan", "hold_still"]:
            if not self.tracker.is_violation(fallback, total_beats) and fallback != previous_style:
                return fallback, f"System default fallback: {fallback}"
                
        return "slow_push", "System default fallback: exhausted all constraints"

    def _compute_intensity(
        self,
        style_def: MotionStyle,
        emotion: str,
        override: Optional[float],
    ) -> float:
        """Compute final motion intensity."""
        if override is not None:
            # Blend override with style base.
            return max(0.0, min(1.0, 0.6 * override + 0.4 * style_def.base_intensity))

        # High-arousal emotions boost intensity.
        arousal_boost = {
            "fear": 0.12, "rage": 0.15, "shock": 0.18, "tension": 0.08,
            "excitement": 0.10, "triumph": 0.12, "danger": 0.10, "dread": 0.08,
        }.get(emotion, 0.0)

        # Low-arousal emotions reduce intensity.
        arousal_damp = {
            "grief": -0.10, "calm": -0.15, "resolution": -0.20,
            "bond": -0.08, "sadness": -0.12, "loneliness": -0.10,
        }.get(emotion, 0.0)

        return max(0.05, min(1.0, style_def.base_intensity + arousal_boost + arousal_damp))

    # ── Introspection ────────────────────────────────────────────────

    def available_styles(self) -> List[Dict[str, Any]]:
        """List all available motion styles."""
        return [
            {
                "name": s.name,
                "camera_goal": s.camera_goal,
                "base_intensity": s.base_intensity,
                "description": s.description,
            }
            for s in sorted(self.styles.values(), key=lambda s: s.name)
        ]

    def supported_emotions(self) -> List[str]:
        """List all supported emotion inputs."""
        return sorted(_EMOTION_MOTION_MAP.keys())

    def supported_intents(self) -> List[str]:
        """List all supported visual intent inputs."""
        return sorted(_INTENT_MOTION_MAP.keys())


# ── Convenience function ─────────────────────────────────────────────


def plan_motion(
    emotion: str,
    visual_intent: str = "",
    beat_type: str = "",
    **kwargs,
) -> MotionPlan:
    """Shortcut: plan motion for a single beat."""
    return MotionPlanner().plan(emotion, visual_intent, beat_type, **kwargs)
