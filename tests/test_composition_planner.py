"""Tests for the CompositionPlanner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.slides.composition_planner import (
    Composition,
    CompositionLayer,
    CompositionPlanner,
    plan_composition,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_beat(
    beat_type: str = "evidence",
    visual_role: str = "character",
    layout_mode: str = "safe_subject",
    emotion: str = "curious",
    intensity: float = 0.4,
    query: str = "Luffy Wano",
    text_overlay: str = "",
    emphasis_words: list = None,
    character: str = "Luffy",
    motion_preset: str = "",
):
    return {
        "beat_type": beat_type,
        "visual_role": visual_role,
        "layout_mode": layout_mode,
        "emotion_state": {"emotion": emotion, "intensity": intensity},
        "image_search_query": query,
        "text_overlay": text_overlay,
        "emphasis_words": emphasis_words or [],
        "context_entities": [character] if character else [],
        "motion_preset": motion_preset,
    }


@pytest.fixture
def planner():
    return CompositionPlanner()


# ── Core: multi-layer output ────────────────────────────────────────


class TestMultiLayerOutput:
    """Verify that output is multi-layer, not a single image."""

    def test_always_has_background(self, planner):
        comp = planner.plan(_make_beat())
        bg_layers = [l for l in comp.layers if l.category == "background"]
        assert len(bg_layers) >= 1

    def test_has_foreground_for_character(self, planner):
        comp = planner.plan(_make_beat(visual_role="character"))
        fg_layers = [l for l in comp.layers if l.category == "foreground"]
        assert len(fg_layers) >= 1

    def test_has_effects(self, planner):
        comp = planner.plan(_make_beat(emotion="grief", intensity=0.8))
        eff_layers = [l for l in comp.layers if l.category == "effect"]
        assert len(eff_layers) >= 1

    def test_has_text_layers(self, planner):
        comp = planner.plan(_make_beat(text_overlay="WHY DID HE DO IT?"))
        text_layers = [l for l in comp.layers if l.category == "text"]
        assert len(text_layers) >= 1
        headlines = [l for l in text_layers if l.name == "headline"]
        assert len(headlines) == 1
        assert headlines[0].source == "WHY DID HE DO IT?"

    def test_has_motion_layer(self, planner):
        comp = planner.plan(_make_beat())
        motion_layers = [l for l in comp.layers if l.category == "motion"]
        assert len(motion_layers) >= 1

    def test_minimum_layer_count(self, planner):
        comp = planner.plan(_make_beat(emotion="tension", intensity=0.7,
                                         text_overlay="HOOK", visual_role="character"))
        # Should have: bg + treatment + foreground + effect + text + subtitle + motion ≥ 5
        assert comp.layer_count >= 5

    def test_always_has_subtitle_zone(self, planner):
        comp = planner.plan(_make_beat())
        subtitle = [l for l in comp.layers if l.name == "subtitle_zone"]
        assert len(subtitle) == 1
        assert subtitle[0].metadata.get("purpose") == "subtitle_safe_area"


# ── Layer categories ────────────────────────────────────────────────


class TestLayerCategories:
    def test_by_category_property(self, planner):
        comp = planner.plan(_make_beat(emotion="grief", intensity=0.8,
                                         text_overlay="LOSS", visual_role="character"))
        cats = comp.by_category
        assert "background" in cats
        assert "text" in cats

    def test_z_order_is_ascending(self, planner):
        comp = planner.plan(_make_beat(emotion="tension", intensity=0.7,
                                         text_overlay="DANGER"))
        z_indices = [l.z_index for l in comp.layers]
        assert z_indices == sorted(z_indices)


# ── Beat type × visual role strategies ──────────────────────────────


class TestStrategies:
    def test_hook_character_is_hero_reveal(self, planner):
        comp = planner.plan(_make_beat("hook", "character"))
        assert "hero_reveal" in comp.reasoning

    def test_reveal_symbol_is_symbol_dramatic(self, planner):
        comp = planner.plan(_make_beat("reveal", "symbol"))
        assert "symbol_dramatic" in comp.reasoning

    def test_cta_has_no_foreground(self, planner):
        comp = planner.plan(_make_beat("cta", "cta_card"))
        fg = [l for l in comp.layers if l.category == "foreground"]
        assert len(fg) == 0

    def test_evidence_inspect_strategy(self, planner):
        comp = planner.plan(_make_beat("evidence", "evidence"))
        assert "evidence_inspect" in comp.reasoning

    def test_establishing_wide_no_foreground(self, planner):
        comp = planner.plan(_make_beat("setup", "location"))
        assert "establishing_wide" in comp.reasoning
        # Wide shot: background IS the subject, no separate foreground.
        fg = [l for l in comp.layers if l.category == "foreground"]
        assert len(fg) == 0


# ── Emotion-driven effects ──────────────────────────────────────────


class TestEmotionEffects:
    @pytest.mark.parametrize("emotion,expected_effect", [
        ("grief", "desaturation"),
        ("tension", "vignette_dark"),
        ("fear", "vignette_red"),
        ("rage", "red_tint"),
        ("shock", "flash_white"),
        ("hope", "warm_glow"),
        ("triumph", "golden_glow"),
        ("revelation", "light_bloom"),
    ])
    def test_emotion_creates_correct_effect(self, planner, emotion, expected_effect):
        comp = planner.plan(_make_beat(emotion=emotion, intensity=0.8))
        effect_names = [l.name for l in comp.layers if l.category == "effect"]
        assert expected_effect in effect_names

    def test_high_intensity_adds_more_effects(self, planner):
        low = planner.plan(_make_beat(emotion="tension", intensity=0.3))
        high = planner.plan(_make_beat(emotion="tension", intensity=0.9))
        low_effects = len([l for l in low.layers if l.category == "effect"])
        high_effects = len([l for l in high.layers if l.category == "effect"])
        assert high_effects >= low_effects

    def test_effects_can_be_disabled(self):
        planner = CompositionPlanner(enable_effects=False)
        comp = planner.plan(_make_beat(emotion="grief", intensity=0.9))
        effect_layers = [l for l in comp.layers if l.category == "effect"]
        assert len(effect_layers) == 0


# ── Background treatment ────────────────────────────────────────────


class TestBackgroundTreatment:
    def test_cta_gets_darken_overlay(self, planner):
        comp = planner.plan(_make_beat("cta", "cta_card", "title_card"))
        bg_layers = [l for l in comp.layers if l.category == "background"]
        darken = [l for l in bg_layers if l.name == "bg_darken"]
        assert len(darken) == 1

    def test_grief_gets_mood_overlay(self, planner):
        comp = planner.plan(_make_beat(emotion="grief", intensity=0.7))
        bg_layers = [l for l in comp.layers if l.category == "background"]
        mood = [l for l in bg_layers if l.name == "bg_mood"]
        assert len(mood) == 1
        assert mood[0].metadata.get("mood") == "grief"

    def test_reveal_gets_blur(self, planner):
        comp = planner.plan(_make_beat("reveal", "character"))
        bg_layers = [l for l in comp.layers if l.category == "background"]
        blur = [l for l in bg_layers if l.name == "bg_blur"]
        assert len(blur) == 1


# ── Text placement ──────────────────────────────────────────────────


class TestTextPlacement:
    def test_headline_from_text_overlay(self, planner):
        comp = planner.plan(_make_beat(text_overlay="WHY?"))
        headlines = [l for l in comp.layers if l.name == "headline"]
        assert len(headlines) == 1
        assert headlines[0].source == "WHY?"
        assert headlines[0].metadata["font_weight"] == "bold"

    def test_emphasis_tags(self, planner):
        comp = planner.plan(_make_beat(emphasis_words=["Gear 5", "Joy Boy", "Nika"]))
        tags = [l for l in comp.layers if l.name == "emphasis_tags"]
        assert len(tags) == 1
        assert "Gear 5" in tags[0].source

    def test_hook_headline_gets_typewriter(self, planner):
        comp = planner.plan(_make_beat("hook", text_overlay="SHOCKING TRUTH"))
        headlines = [l for l in comp.layers if l.name == "headline"]
        assert len(headlines) == 1
        assert headlines[0].animation["type"] == "typewriter"

    def test_no_text_overlay_no_headline(self, planner):
        comp = planner.plan(_make_beat())
        headlines = [l for l in comp.layers if l.name == "headline"]
        assert len(headlines) == 0


# ── Motion layer ────────────────────────────────────────────────────


class TestMotionLayer:
    def test_hook_gets_fast_push(self, planner):
        comp = planner.plan(_make_beat("hook"))
        motion = [l for l in comp.layers if l.category == "motion"]
        assert len(motion) == 1
        assert "push" in motion[0].source

    def test_cta_gets_static(self, planner):
        comp = planner.plan(_make_beat("cta", "cta_card"))
        motion = [l for l in comp.layers if l.category == "motion"]
        assert len(motion) == 1
        assert "static" in motion[0].source

    def test_parallax_on_reveals(self, planner):
        comp = planner.plan(_make_beat("reveal"))
        motion = [l for l in comp.layers if l.category == "motion"]
        assert len(motion) == 1
        assert motion[0].metadata.get("parallax") is True


# ── Foreground character details ────────────────────────────────────


class TestForegroundCharacter:
    def test_character_name_in_source(self, planner):
        comp = planner.plan(_make_beat(character="Zoro"))
        fg = [l for l in comp.layers if l.category == "foreground"]
        assert len(fg) >= 1
        assert "Zoro" in fg[0].source.lower() or "zoro" in fg[0].source.lower()

    def test_cutout_metadata(self, planner):
        comp = planner.plan(_make_beat(visual_role="character", character="Luffy"))
        fg = [l for l in comp.layers if l.name == "foreground_character"]
        if fg:
            assert fg[0].metadata.get("cutout") is True

    def test_symbol_gets_pop_animation(self, planner):
        comp = planner.plan(_make_beat(visual_role="symbol", beat_type="reveal"))
        fg = [l for l in comp.layers if l.name == "foreground_symbol"]
        assert len(fg) == 1
        assert fg[0].animation["type"] == "pop_scale"


# ── Max layers ───────────────────────────────────────────────────────


class TestMaxLayers:
    def test_respects_max_layers(self):
        planner = CompositionPlanner(max_layers=4)
        comp = planner.plan(_make_beat(emotion="rage", intensity=0.95,
                                         text_overlay="RAGE", emphasis_words=["fury"]))
        assert comp.layer_count <= 4

    def test_default_max_is_8(self, planner):
        comp = planner.plan(_make_beat(emotion="rage", intensity=0.95,
                                         text_overlay="FURY", emphasis_words=["fire"]))
        assert comp.layer_count <= 8


# ── Sequence planning ───────────────────────────────────────────────


class TestSequencePlanning:
    def test_plan_sequence(self, planner):
        beats = [
            _make_beat("hook", "character", text_overlay="WHY?"),
            _make_beat("evidence", "evidence"),
            _make_beat("reveal", "symbol", text_overlay="THE TRUTH"),
            _make_beat("cta", "cta_card"),
        ]
        comps = planner.plan_sequence(beats)
        assert len(comps) == 4
        assert all(isinstance(c, Composition) for c in comps)
        assert comps[0].beat_type == "hook"
        assert comps[-1].beat_type == "cta"

    def test_empty_sequence(self, planner):
        assert planner.plan_sequence([]) == []


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_beat(self, planner):
        comp = planner.plan({})
        assert comp.layer_count >= 2  # At least background + subtitle zone

    def test_unknown_beat_type(self, planner):
        comp = planner.plan(_make_beat(beat_type="unknown_xyz"))
        assert comp.layer_count >= 2

    def test_unknown_visual_role(self, planner):
        comp = planner.plan(_make_beat(visual_role="unknown_abc"))
        assert comp.layer_count >= 2

    def test_video_profile_long_youtube(self):
        planner = CompositionPlanner(video_profile="long_youtube")
        comp = planner.plan(_make_beat())
        assert comp.layer_count >= 2


# ── Serialization ────────────────────────────────────────────────────


class TestSerialization:
    def test_layer_to_dict(self, planner):
        comp = planner.plan(_make_beat())
        d = comp.layers[0].to_dict()
        assert "name" in d
        assert "category" in d
        assert "z_index" in d
        json.dumps(d)

    def test_composition_to_dict(self, planner):
        comp = planner.plan(_make_beat(text_overlay="TEST", emotion="grief", intensity=0.8))
        d = comp.to_dict()
        assert "layers" in d
        assert "layer_count" in d
        assert "reasoning" in d
        serialized = json.dumps(d)
        assert isinstance(serialized, str)


# ── Introspection ───────────────────────────────────────────────────


class TestIntrospection:
    def test_available_strategies(self):
        strategies = CompositionPlanner.available_strategies()
        assert "hero_reveal" in strategies
        assert "cta_clean" in strategies
        assert len(strategies) >= 8

    def test_supported_emotions(self):
        emotions = CompositionPlanner.supported_emotions()
        assert "grief" in emotions
        assert "triumph" in emotions


# ── Convenience function ─────────────────────────────────────────────


class TestConvenienceFunction:
    def test_plan_composition(self):
        comp = plan_composition(_make_beat("hook", "character", text_overlay="WHY?"))
        assert comp.layer_count >= 4
        assert comp.beat_type == "hook"

    def test_custom_video_profile(self):
        comp = plan_composition(_make_beat(), video_profile="long_youtube")
        assert comp.layer_count >= 2
