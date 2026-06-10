"""Tests for the MotionPlanner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.slides.motion_planner import (
    MOTION_STYLES,
    MotionPlan,
    MotionPlanner,
    plan_motion,
)


@pytest.fixture
def planner():
    return MotionPlanner()


# ── User-specified examples ──────────────────────────────────────────


class TestUserExamples:
    """The three examples from the user request."""

    def test_fear_gives_dramatic_push(self, planner):
        plan = planner.plan("fear", "curiosity_gap")
        assert plan.style == "dramatic_push"

    def test_reveal_gives_reveal_zoom(self, planner):
        plan = planner.plan("intrigue", "reversal", beat_type="reveal")
        assert plan.style == "reveal_zoom"

    def test_evidence_gives_slow_pan(self, planner):
        plan = planner.plan("curious", "proof", beat_type="evidence")
        assert plan.style == "slow_pan"


# ── Emotion → style mapping ─────────────────────────────────────────


class TestEmotionMapping:
    @pytest.mark.parametrize("emotion,expected_style", [
        ("fear", "dramatic_push"),
        ("grief", "grief_hold"),
        ("hope", "hopeful_rise"),
        ("rage", "impact_shake"),
        ("shock", "impact_shake"),
        ("tension", "tension_drift"),
        ("intrigue", "tension_drift"),
        ("triumph", "climax_zoom"),
        ("calm", "slow_pan"),
        ("resolution", "static_hold"),
    ])
    def test_emotion_to_style(self, planner, emotion, expected_style):
        plan = planner.plan(emotion)
        assert plan.style == expected_style, (
            f"Expected {emotion} → {expected_style}, got {plan.style}"
        )

    def test_unknown_emotion_defaults(self, planner):
        plan = planner.plan("zzz_unknown_zzz")
        assert plan.style in MOTION_STYLES


# ── Beat type overrides ──────────────────────────────────────────────


class TestBeatOverrides:
    def test_hook_overrides_to_dramatic_push(self, planner):
        plan = planner.plan("calm", "context_setup", beat_type="hook")
        assert plan.style == "dramatic_push"

    def test_reveal_overrides_to_reveal_zoom(self, planner):
        plan = planner.plan("calm", "context_setup", beat_type="reveal")
        assert plan.style == "reveal_zoom"

    def test_payoff_overrides_to_climax_zoom(self, planner):
        plan = planner.plan("hope", "takeaway", beat_type="payoff")
        assert plan.style == "climax_zoom"

    def test_cta_overrides_to_static_hold(self, planner):
        plan = planner.plan("hope", "conversion", beat_type="cta")
        assert plan.style == "static_hold"


# ── Visual intent mapping ───────────────────────────────────────────


class TestIntentMapping:
    def test_curiosity_gap_intent(self, planner):
        plan = planner.plan("neutral", "curiosity_gap")
        # Emotion is neutral → should pick intent's style.
        assert plan.style in ("dramatic_push", "subject_push")

    def test_proof_intent_gives_slow_pan(self, planner):
        plan = planner.plan("curious", "proof")
        assert plan.style == "slow_pan"

    def test_reversal_intent_gives_reveal_zoom(self, planner):
        plan = planner.plan("revelation", "reversal")
        assert plan.style == "reveal_zoom"

    def test_conversion_intent_gives_static(self, planner):
        plan = planner.plan("resolution", "conversion")
        assert plan.style == "static_hold"


# ── Anti-repetition ─────────────────────────────────────────────────


class TestAntiRepetition:
    def test_avoids_repeating_previous(self, planner):
        plan = planner.plan("fear", "curiosity_gap", previous_style="dramatic_push")
        assert plan.style != "dramatic_push", "Should avoid repeating previous style"

    def test_uses_fallback_on_repeat(self, planner):
        plan = planner.plan("fear", "", previous_style="dramatic_push")
        # Fear's fallback is tension_drift.
        assert plan.style == "tension_drift"

    def test_sequence_avoids_consecutive_repeats(self, planner):
        beats = [
            {"emotion": "fear", "beat_type": "hook"},
            {"emotion": "fear", "visual_intent": "curiosity_gap"},
            {"emotion": "fear", "visual_intent": "curiosity_gap"},
        ]
        plans = planner.plan_sequence(beats)
        # No two consecutive plans should have the same style.
        for i in range(1, len(plans)):
            if plans[i].style == plans[i - 1].style:
                # This is acceptable if the beat type forces it.
                beat_type = beats[i].get("beat_type", "")
                assert beat_type in ("hook", "reveal", "payoff", "cta"), (
                    f"Non-override beats should not repeat: [{i-1}]={plans[i-1].style} [{i}]={plans[i].style}"
                )


# ── Intensity ────────────────────────────────────────────────────────


class TestIntensity:
    def test_intensity_in_valid_range(self, planner):
        for emotion in ("fear", "grief", "hope", "calm", "rage", "neutral"):
            plan = planner.plan(emotion)
            assert 0.0 <= plan.intensity <= 1.0, f"{emotion}: intensity={plan.intensity}"

    def test_high_arousal_boosts_intensity(self, planner):
        calm = planner.plan("calm")
        fear = planner.plan("fear")
        assert fear.intensity > calm.intensity

    def test_grief_lower_than_rage(self, planner):
        grief = planner.plan("grief")
        rage = planner.plan("rage")
        assert rage.intensity > grief.intensity

    def test_override_intensity_blended(self, planner):
        plan = planner.plan("neutral", intensity=0.9)
        # Should blend override with style base.
        assert plan.intensity > 0.5


# ── MotionPlan properties ───────────────────────────────────────────


class TestMotionPlanProperties:
    def test_has_all_fields(self, planner):
        plan = planner.plan("fear", "curiosity_gap", beat_type="hook")
        assert plan.style
        assert plan.camera_goal
        assert plan.transition
        assert plan.easing
        assert plan.duration_modifier > 0
        assert plan.reasoning

    def test_reasoning_explains_choice(self, planner):
        plan = planner.plan("fear", "curiosity_gap")
        assert "fear" in plan.reasoning.lower() or "dramatic" in plan.reasoning.lower()

    def test_to_dict_serializable(self, planner):
        plan = planner.plan("hope", "takeaway", beat_type="payoff")
        d = plan.to_dict()
        assert "style" in d
        assert "intensity" in d
        serialized = json.dumps(d)
        assert isinstance(serialized, str)


# ── Sequence planning ───────────────────────────────────────────────


class TestSequencePlanning:
    def test_basic_sequence(self, planner):
        beats = [
            {"emotion": "fear", "beat_type": "hook", "visual_intent": "curiosity_gap"},
            {"emotion": "curious", "beat_type": "setup", "visual_intent": "context_setup"},
            {"emotion": "tension", "beat_type": "escalation", "visual_intent": "proof"},
            {"emotion": "revelation", "beat_type": "reveal", "visual_intent": "reversal"},
            {"emotion": "hope", "beat_type": "payoff", "visual_intent": "takeaway"},
            {"emotion": "resolution", "beat_type": "cta", "visual_intent": "conversion"},
        ]
        plans = planner.plan_sequence(beats)
        assert len(plans) == 6
        assert plans[0].style == "dramatic_push"     # hook override
        assert plans[3].style == "reveal_zoom"        # reveal override
        assert plans[4].style == "climax_zoom"        # payoff override
        assert plans[5].style == "static_hold"        # cta override

    def test_emotion_state_dict(self, planner):
        """Emotion can be a dict (from the slide pipeline)."""
        beats = [
            {"emotion_state": {"emotion": "fear", "intensity": 0.8}, "beat_type": "hook"},
        ]
        plans = planner.plan_sequence(beats)
        assert plans[0].style == "dramatic_push"
        assert plans[0].intensity > 0.7

    def test_empty_sequence(self, planner):
        plans = planner.plan_sequence([])
        assert plans == []


# ── Visual role camera goals ────────────────────────────────────────


class TestVisualRole:
    def test_character_role(self, planner):
        plan = planner.plan("curious", "proof", visual_role="character")
        assert plan.camera_goal == "face_lock_push"

    def test_location_role(self, planner):
        plan = planner.plan("curious", "proof", visual_role="location")
        assert plan.camera_goal == "wide_context_pan"

    def test_high_priority_style_keeps_own_camera(self, planner):
        plan = planner.plan("fear", "curiosity_gap", visual_role="character")
        # dramatic_push has its own camera goal that overrides role.
        assert plan.camera_goal == "urgent_face_lock"


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_emotion(self, planner):
        plan = planner.plan("")
        assert plan.style in MOTION_STYLES

    def test_none_inputs(self, planner):
        plan = planner.plan("neutral", None, None, None)
        assert plan.style in MOTION_STYLES

    def test_all_styles_valid(self, planner):
        for style_name in MOTION_STYLES:
            style = MOTION_STYLES[style_name]
            assert style.name == style_name
            assert 0.0 <= style.base_intensity <= 1.0
            assert style.duration_modifier > 0


# ── Introspection ───────────────────────────────────────────────────


class TestIntrospection:
    def test_available_styles(self, planner):
        styles = planner.available_styles()
        assert len(styles) >= 12
        names = {s["name"] for s in styles}
        assert "dramatic_push" in names
        assert "reveal_zoom" in names
        assert "slow_pan" in names

    def test_supported_emotions(self, planner):
        emotions = planner.supported_emotions()
        assert "fear" in emotions
        assert "hope" in emotions
        assert "grief" in emotions

    def test_supported_intents(self, planner):
        intents = planner.supported_intents()
        assert "curiosity_gap" in intents
        assert "reversal" in intents
        assert "proof" in intents


# ── Convenience function ─────────────────────────────────────────────


class TestConvenienceFunction:
    def test_plan_motion_shortcut(self):
        plan = plan_motion("fear", "curiosity_gap")
        assert plan.style == "dramatic_push"

    def test_plan_motion_with_beat(self):
        plan = plan_motion("hope", "takeaway", beat_type="payoff")
        assert plan.style == "climax_zoom"
