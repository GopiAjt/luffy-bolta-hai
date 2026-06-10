"""Tests for the ThumbnailMomentDetector."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.slides.thumbnail_moments import (
    DetectionResult,
    ThumbnailMoment,
    ThumbnailMomentDetector,
    detect_thumbnail_moments,
    _parse_time,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_beat(
    index: int,
    start_sec: float,
    end_sec: float,
    beat_type: str = "evidence",
    emotion: str = "curious",
    intensity: float = 0.4,
    text: str = "Regular narration text.",
):
    return {
        "start_time": f"0:{int(start_sec // 60):02d}:{start_sec % 60:05.2f}",
        "end_time": f"0:{int(end_sec // 60):02d}:{end_sec % 60:05.2f}",
        "beat_type": beat_type,
        "subtitle_text": text,
        "summary": text[:40],
        "emotion_state": {"emotion": emotion, "intensity": intensity},
    }


def _typical_storyboard():
    """~120 second storyboard with natural moment candidates."""
    return [
        _make_beat(0, 0, 5, "hook", "tension", 0.80, "Why did Shanks sacrifice his arm?"),
        _make_beat(1, 5, 12, "setup", "curious", 0.35, "Before the Great Pirate Era began."),
        _make_beat(2, 12, 20, "evidence", "curious", 0.40, "Luffy trained to grow stronger."),
        _make_beat(3, 20, 28, "escalation", "tension", 0.60, "The danger grew overwhelming."),
        _make_beat(4, 28, 38, "reveal", "revelation", 0.90, "The hidden truth is that Luffy carries Joy Boy's will."),
        _make_beat(5, 38, 48, "evidence", "curious", 0.35, "The evidence supports this theory."),
        _make_beat(6, 48, 58, "evidence", "curious", 0.30, "More details about the connection."),
        _make_beat(7, 58, 68, "escalation", "tension", 0.65, "But it gets even more impossible."),
        _make_beat(8, 68, 78, "reveal", "shock", 0.92, "Gear 5 awakened — a transformation no one expected."),
        _make_beat(9, 78, 88, "payoff", "triumph", 0.88, "Luffy achieved freedom and victory."),
        _make_beat(10, 88, 98, "warning", "fear", 0.70, "The World Government will destroy anyone."),
        _make_beat(11, 98, 110, "evidence", "curious", 0.30, "But the story continues."),
        _make_beat(12, 110, 120, "cta", "calm", 0.15, "Subscribe for more analysis."),
    ]


@pytest.fixture
def detector():
    return ThumbnailMomentDetector(min_gap=20, max_gap=40)


# ── Time parsing ────────────────────────────────────────────────────


class TestParseTime:
    def test_hms(self):
        assert _parse_time("0:01:30.50") == 90.5

    def test_ms(self):
        assert _parse_time("1:30.50") == 90.5

    def test_seconds(self):
        assert _parse_time("45.5") == 45.5

    def test_empty(self):
        assert _parse_time("") == 0.0

    def test_numeric(self):
        assert _parse_time(42.0) == 42.0


# ── Signal detection ────────────────────────────────────────────────


class TestSignalDetection:
    """Verify the three detection signal types fire correctly."""

    def test_reveal_detected(self, detector):
        beats = [_make_beat(0, 0, 5, "reveal", "revelation", 0.85, "The truth revealed.")]
        result = detector.detect(beats)
        assert len(result.moments) >= 1
        assert any("beat:reveal" in m.signals for m in result.moments)

    def test_payoff_detected(self, detector):
        beats = [_make_beat(0, 0, 5, "payoff", "triumph", 0.9, "Victory achieved.")]
        result = detector.detect(beats)
        assert len(result.moments) >= 1
        assert any("beat:payoff" in m.signals for m in result.moments)

    def test_strong_emotion_detected(self, detector):
        beats = [_make_beat(0, 0, 5, "evidence", "shock", 0.95, "This is evidence.")]
        result = detector.detect(beats)
        assert len(result.moments) >= 1
        assert any("emotion:shock" in s for m in result.moments for s in m.signals)

    def test_shocking_statement_detected(self, detector):
        beats = [_make_beat(0, 0, 5, "evidence", "curious", 0.4,
                              "The impossible truth was finally revealed.")]
        result = detector.detect(beats)
        assert len(result.moments) >= 1
        assert any("text:" in s for m in result.moments for s in m.signals)

    def test_calm_evidence_not_detected(self, detector):
        beats = [_make_beat(0, 0, 5, "evidence", "curious", 0.2,
                              "Regular narration text about the story.")]
        result = detector.detect(beats)
        # Score should be below threshold.
        assert len(result.moments) == 0

    def test_multiple_signals_stack(self, detector):
        # Reveal + shock + shocking text = high score.
        beats = [_make_beat(0, 0, 5, "reveal", "shock", 0.95,
                              "The impossible truth: betrayal of the century.")]
        result = detector.detect(beats)
        assert len(result.moments) == 1
        moment = result.moments[0]
        assert len(moment.signals) >= 2
        assert moment.score > 0.5


# ── Spacing constraints ─────────────────────────────────────────────


class TestSpacingConstraints:
    def test_min_gap_enforced(self, detector):
        """Two moments within 20s should not both be selected."""
        beats = [
            _make_beat(0, 0, 5, "reveal", "shock", 0.9, "Secret revealed!"),
            _make_beat(1, 8, 15, "reveal", "shock", 0.9, "Another secret!"),
        ]
        result = detector.detect(beats)
        # Only one should be selected (8s apart < 20s min_gap).
        assert result.moment_count == 1

    def test_well_spaced_moments_both_selected(self, detector):
        beats = [
            _make_beat(0, 0, 5, "reveal", "shock", 0.9, "Secret revealed!"),
            _make_beat(1, 25, 35, "reveal", "shock", 0.9, "Another truth revealed!"),
        ]
        result = detector.detect(beats)
        assert result.moment_count == 2

    def test_gap_fill_promotes_best_slide(self):
        """If no natural moment in a 40s window, promote best available."""
        detector = ThumbnailMomentDetector(min_gap=20, max_gap=40)
        beats = [
            _make_beat(0, 0, 5, "reveal", "shock", 0.9, "Opening reveal!"),
            # Gap: 5-45s of calm evidence beats.
            _make_beat(1, 10, 15, "evidence", "curious", 0.3, "Normal text."),
            _make_beat(2, 15, 20, "evidence", "curious", 0.3, "More normal."),
            _make_beat(3, 20, 25, "evidence", "tension", 0.5, "Building tension with danger."),
            _make_beat(4, 25, 30, "evidence", "curious", 0.3, "Still normal."),
            _make_beat(5, 30, 35, "evidence", "curious", 0.3, "Continuing."),
            _make_beat(6, 35, 45, "evidence", "curious", 0.3, "Still going."),
            _make_beat(7, 45, 55, "reveal", "revelation", 0.85, "The impossible truth!"),
        ]
        result = detector.detect(beats)
        # Should have at least 2 moments: the opening reveal and the later one.
        assert result.moment_count >= 2


class TestTypicalStoryboard:
    def test_detects_moments(self, detector):
        result = detector.detect(_typical_storyboard())
        assert result.moment_count >= 2
        assert result.moment_count <= 6  # Not too many

    def test_reveals_are_moments(self, detector):
        result = detector.detect(_typical_storyboard())
        beat_types = {m.beat_type for m in result.moments}
        assert "reveal" in beat_types or "payoff" in beat_types

    def test_moments_have_reasons(self, detector):
        result = detector.detect(_typical_storyboard())
        for m in result.moments:
            assert m.reason
            assert len(m.signals) > 0

    def test_moments_are_time_sorted(self, detector):
        result = detector.detect(_typical_storyboard())
        timestamps = [m.timestamp for m in result.moments]
        assert timestamps == sorted(timestamps)


# ── Annotation ───────────────────────────────────────────────────────


class TestAnnotation:
    def test_annotated_beats_have_field(self, detector):
        beats = _typical_storyboard()
        result = detector.detect(beats)
        for s in result.annotated_beats:
            assert "thumbnail_moment" in s

    def test_moment_slides_marked_true(self, detector):
        beats = _typical_storyboard()
        result = detector.detect(beats)
        moment_indices = set(result.moment_indices)
        for s in result.annotated_beats:
            idx = result.annotated_beats.index(s)
            if idx in moment_indices:
                assert s["thumbnail_moment"] is True

    def test_non_moment_slides_marked_false(self, detector):
        beats = _typical_storyboard()
        result = detector.detect(beats)
        moment_indices = set(result.moment_indices)
        for i, s in enumerate(result.annotated_beats):
            if i not in moment_indices:
                assert s["thumbnail_moment"] is False

    def test_original_slides_not_mutated(self, detector):
        beats = _typical_storyboard()
        originals = [dict(s) for s in beats]
        detector.detect(beats)
        for orig, slide in zip(originals, beats):
            assert "thumbnail_moment" not in slide


# ── Summary ──────────────────────────────────────────────────────────


class TestSummary:
    def test_summary_fields(self, detector):
        result = detector.detect(_typical_storyboard())
        s = result.summary
        assert "beat_count" in s
        assert "moment_count" in s
        assert "total_duration" in s
        assert "avg_gap_seconds" in s
        assert "moments_per_minute" in s
        assert "signal_distribution" in s

    def test_moments_per_minute_reasonable(self, detector):
        result = detector.detect(_typical_storyboard())
        mpm = result.summary["moments_per_minute"]
        # Should be 1-3 moments per minute for a typical storyboard.
        assert 0.5 <= mpm <= 5.0

    def test_signal_distribution(self, detector):
        result = detector.detect(_typical_storyboard())
        dist = result.summary["signal_distribution"]
        assert isinstance(dist, dict)


# ── Single slide scoring ────────────────────────────────────────────


class TestSingleSlideScoring:
    def test_high_score_for_reveal(self, detector):
        slide = _make_beat(0, 0, 5, "reveal", "shock", 0.95, "The secret truth!")
        score, signals = detector.detect_single(slide)
        assert score > 0.5
        assert len(signals) >= 2

    def test_low_score_for_calm_evidence(self, detector):
        slide = _make_beat(0, 0, 5, "evidence", "curious", 0.2, "Normal text.")
        score, signals = detector.detect_single(slide)
        assert score < 0.25


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_slides(self, detector):
        result = detector.detect([])
        assert result.moment_count == 0
        assert result.annotated_beats == []

    def test_single_slide(self, detector):
        beats = [_make_beat(0, 0, 5, "reveal", "shock", 0.9, "Reveal!")]
        result = detector.detect(beats)
        assert result.moment_count == 1

    def test_missing_fields(self, detector):
        beats = [{"start_time": "0:00:00.00", "end_time": "0:00:05.00"}]
        result = detector.detect(beats)
        assert len(result.annotated_beats) == 1

    def test_custom_thresholds(self):
        strict = ThumbnailMomentDetector(score_threshold=0.6)
        lenient = ThumbnailMomentDetector(score_threshold=0.1)
        beats = _typical_storyboard()
        assert lenient.detect(beats).moment_count >= strict.detect(beats).moment_count


# ── Serialization ────────────────────────────────────────────────────


class TestSerialization:
    def test_moment_to_dict(self, detector):
        result = detector.detect(_typical_storyboard())
        if result.moments:
            d = result.moments[0].to_dict()
            assert "thumbnail_moment" in d
            assert d["thumbnail_moment"] is True
            json.dumps(d)

    def test_result_to_dict(self, detector):
        result = detector.detect(_typical_storyboard())
        d = result.to_dict()
        assert "moments" in d
        assert "summary" in d
        json.dumps(d)


# ── Convenience function ─────────────────────────────────────────────


class TestConvenienceFunction:
    def test_detect_thumbnail_moments(self):
        result = detect_thumbnail_moments(_typical_storyboard())
        assert result.moment_count >= 2

    def test_custom_gap(self):
        result = detect_thumbnail_moments(_typical_storyboard(), min_gap=10, max_gap=30)
        assert result.moment_count >= 2
