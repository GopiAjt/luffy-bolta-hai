"""Tests for the EmotionCurveGenerator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.slides.emotion_curve import (
    EmotionCurve,
    EmotionCurveGenerator,
    EmotionPoint,
    generate_emotion_curve,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_beat(beat_type: str, text: str = "", index: int = 0, confidence: float = 0.6):
    return {
        "beat_type": beat_type,
        "text": text,
        "index": index,
        "confidence": confidence,
    }


@pytest.fixture
def typical_beats():
    """A typical 8-beat narrative sequence."""
    return [
        _make_beat("hook", "Why did Shanks sacrifice his arm for a child?", 0, 0.85),
        _make_beat("setup", "Before the Great Pirate Era, Roger started it all.", 1, 0.7),
        _make_beat("evidence", "Luffy trained for years to become stronger.", 2, 0.6),
        _make_beat("escalation", "The danger grew as enemies appeared everywhere.", 3, 0.65),
        _make_beat("reveal", "The truth is that Luffy carries the will of Joy Boy.", 4, 0.9),
        _make_beat("payoff", "Luffy achieved his dream of freedom and victory.", 5, 0.85),
        _make_beat("warning", "But a hidden threat still lurks in the shadows.", 6, 0.7),
        _make_beat("cta", "Subscribe for more One Piece analysis.", 7, 0.5),
    ]


@pytest.fixture
def generator():
    return EmotionCurveGenerator()


# ── Core rules ───────────────────────────────────────────────────────


class TestSpikeRules:
    """Verify the user's four rules: hooks spike, reveals spike, payoffs peak, CTA drops."""

    def test_hook_spikes(self, generator, typical_beats):
        curve = generator.generate(typical_beats)
        hook = curve.points[0]
        assert hook.beat_type == "hook"
        assert hook.score >= 0.78, f"Hook should spike (≥0.78), got {hook.score:.2f}"
        assert hook.spike is True

    def test_reveal_spikes(self, generator, typical_beats):
        curve = generator.generate(typical_beats)
        reveal = [p for p in curve.points if p.beat_type == "reveal"][0]
        assert reveal.score >= 0.82, f"Reveal should spike (≥0.82), got {reveal.score:.2f}"
        assert reveal.spike is True

    def test_payoff_peaks(self, generator, typical_beats):
        curve = generator.generate(typical_beats)
        payoff = [p for p in curve.points if p.beat_type == "payoff"][0]
        assert payoff.score >= 0.90, f"Payoff should peak (≥0.90), got {payoff.score:.2f}"
        assert payoff.spike is True
        # Payoff should be the global peak (or tied with reveal).
        assert payoff.score >= max(p.score for p in curve.points) - 0.05

    def test_cta_drops(self, generator, typical_beats):
        curve = generator.generate(typical_beats)
        cta = [p for p in curve.points if p.beat_type == "cta"][0]
        assert cta.score <= 0.30, f"CTA should drop (≤0.30), got {cta.score:.2f}"
        assert cta.spike is False

    def test_cta_is_lowest_or_near_lowest(self, generator, typical_beats):
        curve = generator.generate(typical_beats)
        scores = curve.scores
        cta_score = scores[-1]  # CTA is last
        assert cta_score == min(scores) or cta_score < scores[-2]

    def test_spike_ordering_hook_vs_setup(self, generator, typical_beats):
        """Hook score should be significantly above setup."""
        curve = generator.generate(typical_beats)
        hook_score = curve.points[0].score
        setup_score = curve.points[1].score
        assert hook_score > setup_score + 0.2


class TestBeatTypeScoring:
    """Verify that each beat type behaves as expected."""

    @pytest.mark.parametrize("beat_type,expected_min,expected_max", [
        ("hook", 0.78, 1.0),
        ("setup", 0.20, 0.60),
        ("evidence", 0.30, 0.70),
        ("escalation", 0.55, 0.90),
        ("reveal", 0.82, 1.0),
        ("payoff", 0.90, 1.0),
        ("warning", 0.60, 0.95),
        ("cta", 0.0, 0.30),
    ])
    def test_beat_type_range(self, generator, beat_type, expected_min, expected_max):
        beats = [_make_beat(beat_type, f"Test text for {beat_type}.", 0)]
        curve = generator.generate(beats)
        score = curve.points[0].score
        assert expected_min <= score <= expected_max, (
            f"{beat_type}: expected [{expected_min}, {expected_max}], got {score:.3f}"
        )


# ── Text emotion detection ───────────────────────────────────────────


class TestTextEmotionDetection:
    def test_grief_words(self, generator):
        beats = [_make_beat("evidence", "The death of Ace caused immense pain and suffering.")]
        curve = generator.generate(beats)
        assert curve.points[0].label == "grief"

    def test_tension_words(self, generator):
        beats = [_make_beat("escalation", "The villain threatens to destroy everything.")]
        curve = generator.generate(beats)
        assert curve.points[0].label == "tension"

    def test_hope_words(self, generator):
        beats = [_make_beat("payoff", "Luffy achieved his dream of freedom and hope.")]
        curve = generator.generate(beats)
        assert curve.points[0].label == "hope"

    def test_intrigue_words(self, generator):
        beats = [_make_beat("mystery", "A hidden secret behind the mystery of the Void Century.")]
        curve = generator.generate(beats)
        assert curve.points[0].label == "intrigue"

    def test_neutral_fallback(self, generator):
        beats = [_make_beat("setup", "The scene takes place at a regular location.")]
        curve = generator.generate(beats)
        assert curve.points[0].label == "curious"

    def test_grief_boosts_intensity(self, generator):
        """Grief text should boost intensity beyond the base."""
        neutral_beats = [_make_beat("evidence", "Regular text here.")]
        grief_beats = [_make_beat("evidence", "The death and sacrifice broke everyone.")]
        neutral_score = generator.generate(neutral_beats).points[0].score
        grief_score = generator.generate(grief_beats).points[0].score
        assert grief_score > neutral_score


# ── Curve shape and summary ──────────────────────────────────────────


class TestCurveSummary:
    def test_summary_present(self, generator, typical_beats):
        curve = generator.generate(typical_beats)
        s = curve.summary
        assert "beat_count" in s
        assert s["beat_count"] == 8
        assert "peak_intensity" in s
        assert "mean_intensity" in s
        assert "arc_shape" in s
        assert "dominant_emotion" in s

    def test_peak_is_payoff_or_reveal_or_hook(self, generator, typical_beats):
        curve = generator.generate(typical_beats)
        peak_type = curve.summary["peak_beat_type"]
        assert peak_type in ("payoff", "reveal", "hook")

    def test_spike_count(self, generator, typical_beats):
        curve = generator.generate(typical_beats)
        assert curve.summary["spike_count"] == 3  # hook, reveal, payoff

    def test_intensity_range_significant(self, generator, typical_beats):
        curve = generator.generate(typical_beats)
        assert curve.summary["intensity_range"] >= 0.5

    def test_arc_shape_classified(self, generator, typical_beats):
        curve = generator.generate(typical_beats)
        assert curve.summary["arc_shape"] in (
            "rising_climax", "mountain", "wave", "front_loaded", "valley", "flat"
        )

    def test_valence_trend(self, generator, typical_beats):
        curve = generator.generate(typical_beats)
        assert curve.summary["valence_trend"] in ("positive", "negative", "mixed")


class TestPeakIndex:
    def test_peak_index_property(self, generator, typical_beats):
        curve = generator.generate(typical_beats)
        peak = curve.peak_index
        assert 0 <= peak < len(curve.points)
        assert curve.points[peak].score == max(curve.scores)


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_beats(self, generator):
        curve = generator.generate([])
        assert len(curve.points) == 0
        assert curve.summary["beat_count"] == 0

    def test_single_beat(self, generator):
        beats = [_make_beat("hook", "Why did this happen?")]
        curve = generator.generate(beats)
        assert len(curve.points) == 1
        assert curve.points[0].score >= 0.78

    def test_two_beats(self, generator):
        beats = [_make_beat("hook", "Start here."), _make_beat("cta", "Subscribe now.")]
        curve = generator.generate(beats)
        assert len(curve.points) == 2
        assert curve.points[0].score > curve.points[1].score

    def test_unknown_beat_type(self, generator):
        beats = [_make_beat("unknown_type", "Some text.")]
        curve = generator.generate(beats)
        assert len(curve.points) == 1
        assert 0.0 <= curve.points[0].score <= 1.0

    def test_empty_text(self, generator):
        beats = [_make_beat("evidence", "")]
        curve = generator.generate(beats)
        assert curve.points[0].label == "neutral" or curve.points[0].label == "curious"

    def test_all_same_beat_type(self, generator):
        beats = [_make_beat("evidence", f"Text {i}.", i) for i in range(5)]
        curve = generator.generate(beats)
        assert len(curve.points) == 5
        # All evidence beats should have similar scores.
        scores = curve.scores
        assert max(scores) - min(scores) < 0.25


# ── Smoothing ────────────────────────────────────────────────────────


class TestSmoothing:
    def test_smoothing_preserves_spikes(self):
        gen = EmotionCurveGenerator(smoothing=0.3)
        beats = [
            _make_beat("setup", "Begin.", 0),
            _make_beat("hook", "SPIKE! Why did this happen?", 1),
            _make_beat("setup", "Continue.", 2),
        ]
        curve = gen.generate(beats)
        hook_point = [p for p in curve.points if p.beat_type == "hook"][0]
        assert hook_point.score >= 0.78, "Smoothing must not reduce spikes"

    def test_no_smoothing(self):
        gen = EmotionCurveGenerator(smoothing=0.0)
        beats = [_make_beat("evidence", f"Text {i}.", i) for i in range(5)]
        curve = gen.generate(beats)
        assert len(curve.points) == 5

    def test_heavy_smoothing_keeps_rules(self):
        gen = EmotionCurveGenerator(smoothing=0.5)
        beats = [
            _make_beat("hook", "Start.", 0),
            _make_beat("setup", "Build.", 1),
            _make_beat("reveal", "The truth.", 2),
            _make_beat("cta", "Subscribe.", 3),
        ]
        curve = gen.generate(beats)
        assert curve.points[0].score >= 0.78    # hook floor
        assert curve.points[2].score >= 0.82    # reveal floor
        assert curve.points[3].score <= 0.30    # CTA ceiling


# ── Momentum ─────────────────────────────────────────────────────────


class TestMomentum:
    def test_escalation_chain_builds(self, generator):
        beats = [
            _make_beat("setup", "Begin.", 0),
            _make_beat("escalation", "Growing danger.", 1),
            _make_beat("escalation", "Even more danger.", 2),
            _make_beat("reveal", "The truth.", 3),
        ]
        curve = generator.generate(beats)
        # Second escalation should be ≥ first.
        assert curve.points[2].score >= curve.points[1].score - 0.05

    def test_reveal_after_escalation_boosted(self, generator):
        beats = [
            _make_beat("escalation", "Building tension.", 0),
            _make_beat("reveal", "The hidden truth.", 1),
        ]
        curve = generator.generate(beats)
        reveal = curve.points[1]
        assert reveal.score >= 0.85  # Base + momentum boost


# ── Serialization ────────────────────────────────────────────────────


class TestSerialization:
    def test_point_to_dict(self, generator, typical_beats):
        curve = generator.generate(typical_beats)
        d = curve.points[0].to_dict()
        assert "score" in d
        assert "label" in d
        assert "score_breakdown" in d
        # Should be JSON-serializable.
        json.dumps(d)

    def test_curve_to_dict(self, generator, typical_beats):
        curve = generator.generate(typical_beats)
        d = curve.to_dict()
        assert "points" in d
        assert "summary" in d
        assert len(d["points"]) == 8
        json.dumps(d)


# ── Convenience function ─────────────────────────────────────────────


class TestConvenienceFunction:
    def test_generate_emotion_curve(self, typical_beats):
        curve = generate_emotion_curve(typical_beats)
        assert len(curve.points) == 8
        assert curve.points[0].score >= 0.78

    def test_with_custom_params(self, typical_beats):
        curve = generate_emotion_curve(typical_beats, hook_floor=0.85, cta_ceiling=0.20)
        assert curve.points[0].score >= 0.85
        cta = [p for p in curve.points if p.beat_type == "cta"][0]
        assert cta.score <= 0.20


# ── Integration with StoryAnalyzer ───────────────────────────────────


class TestWithStoryAnalyzer:
    def test_real_analyzer_output(self):
        """End-to-end: StoryAnalyzer → EmotionCurveGenerator."""
        from app.utils.slides.story_analyzer import StoryAnalyzer

        script = (
            "Why did Shanks sacrifice his arm for a child? "
            "Before the Great Pirate Era, Gold Roger started everything with his final words. "
            "Luffy trained with Rayleigh to grow stronger. "
            "The Marines sent three admirals to stop him. "
            "But the real secret is that Luffy carries the will of Joy Boy. "
            "Gear 5 awakened and Luffy achieved his dream of freedom. "
            "Subscribe for more analysis."
        )
        analyzer = StoryAnalyzer()
        beats = analyzer.analyze(script)
        assert len(beats) > 0

        gen = EmotionCurveGenerator()
        curve = gen.generate(beats)

        assert len(curve.points) == len(beats)
        assert curve.summary["beat_count"] == len(beats)

        # Verify all points have valid scores.
        for pt in curve.points:
            assert 0.0 <= pt.score <= 1.0
            assert pt.label
            assert pt.beat_type
