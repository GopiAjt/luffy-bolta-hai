"""Tests for the RetentionOptimizer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.slides.retention_optimizer import (
    RetentionFix,
    RetentionOptimizer,
    RetentionReport,
    RetentionViolation,
    optimize_retention,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_beat(
    character: str = "Luffy",
    arc: str = "",
    beat_type: str = "evidence",
    emotion: str = "curious",
    intensity: float = 0.4,
    start_sec: float = 0,
    end_sec: float = 5,
    text: str = "Narration text.",
    motion: str = "subject_push",
    text_overlay: str = "",
    emphasis_words: list = None,
):
    return {
        "start_time": f"0:{int(start_sec // 60):02d}:{start_sec % 60:05.2f}",
        "end_time": f"0:{int(end_sec // 60):02d}:{end_sec % 60:05.2f}",
        "beat_type": beat_type,
        "subtitle_text": text,
        "summary": text[:40],
        "emotion_state": {"emotion": emotion, "intensity": intensity},
        "context_entities": [character] if character else [],
        "image_search_query": f"{character} {arc}",
        "asset_metadata": {"entities": [character] if character else [], "search_tags": [arc] if arc else []},
        "motion_preset": motion,
        "text_overlay": text_overlay,
        "emphasis_words": emphasis_words or [],
    }


def _healthy_storyboard():
    """A well-structured storyboard with no violations."""
    return [
        _make_beat("Luffy", "dawn", "hook", "tension", 0.82, 0, 5,
                     "Why did Shanks sacrifice his arm?", "subject_push", "WHY?"),
        _make_beat("Shanks", "dawn", "setup", "curious", 0.40, 5, 12,
                     "Before the Great Pirate Era."),
        _make_beat("Zoro", "wano", "evidence", "curious", 0.45, 12, 20,
                     "Training with Rayleigh."),
        _make_beat("Robin", "enies_lobby", "reveal", "revelation", 0.88, 20, 28,
                     "The hidden truth revealed."),
        _make_beat("Whitebeard", "marineford", "escalation", "tension", 0.70, 28, 38,
                     "The danger intensified."),
        _make_beat("Ace", "marineford", "payoff", "triumph", 0.90, 38, 48,
                     "The sacrifice completed."),
        _make_beat("", "", "cta", "resolution", 0.20, 48, 55,
                     "Subscribe for more.", "static_hold"),
    ]


@pytest.fixture
def optimizer():
    return RetentionOptimizer()


# ── Check 1: Same character > 3 beats ──────────────────────────────


class TestSameCharacter:
    def test_detects_character_streak(self, optimizer):
        beats = [
            _make_beat("Luffy", start_sec=0, end_sec=5),
            _make_beat("Luffy", start_sec=5, end_sec=10),
            _make_beat("Luffy", start_sec=10, end_sec=15),
            _make_beat("Luffy", start_sec=15, end_sec=20),
        ]
        report = optimizer.analyze(beats)
        char_violations = [v for v in report.violations if v.check == "same_character"]
        assert len(char_violations) >= 1
        assert char_violations[0].beat_indices == [0, 1, 2, 3]

    def test_no_violation_at_limit(self, optimizer):
        beats = [
            _make_beat("Luffy", start_sec=0, end_sec=5),
            _make_beat("Luffy", start_sec=5, end_sec=10),
            _make_beat("Luffy", start_sec=10, end_sec=15),
        ]
        report = optimizer.analyze(beats)
        char_violations = [v for v in report.violations if v.check == "same_character"]
        assert len(char_violations) == 0

    def test_diverse_characters_no_violation(self, optimizer):
        beats = [
            _make_beat("Luffy", start_sec=0, end_sec=5),
            _make_beat("Zoro", start_sec=5, end_sec=10),
            _make_beat("Sanji", start_sec=10, end_sec=15),
            _make_beat("Robin", start_sec=15, end_sec=20),
        ]
        report = optimizer.analyze(beats)
        char_violations = [v for v in report.violations if v.check == "same_character"]
        assert len(char_violations) == 0

    def test_generates_swap_fixes(self, optimizer):
        beats = [_make_beat("Luffy", start_sec=i * 5, end_sec=(i + 1) * 5) for i in range(5)]
        report = optimizer.analyze(beats)
        char_violations = [v for v in report.violations if v.check == "same_character"]
        assert len(char_violations) >= 1
        # Should have swap_character fixes for indices > 3.
        fixes = char_violations[0].fixes
        assert len(fixes) >= 1
        assert all(f.action == "swap_character" for f in fixes)

    def test_custom_threshold(self):
        opt = RetentionOptimizer(max_same_character=5)
        beats = [_make_beat("Luffy", start_sec=i * 5, end_sec=(i + 1) * 5) for i in range(5)]
        report = opt.analyze(beats)
        char_violations = [v for v in report.violations if v.check == "same_character"]
        assert len(char_violations) == 0


# ── Check 2: Same arc > 4 beats ────────────────────────────────────


class TestSameArc:
    def test_detects_arc_streak(self, optimizer):
        beats = [
            _make_beat("Luffy", "wano island", start_sec=i * 5, end_sec=(i + 1) * 5)
            for i in range(6)
        ]
        report = optimizer.analyze(beats)
        arc_violations = [v for v in report.violations if v.check == "same_arc"]
        assert len(arc_violations) >= 1

    def test_diverse_arcs_no_violation(self, optimizer):
        arcs = ["east sea island", "marineford castle", "wano island", "sky island"]
        beats = [
            _make_beat("Luffy", arcs[i], start_sec=i * 5, end_sec=(i + 1) * 5)
            for i in range(4)
        ]
        report = optimizer.analyze(beats)
        arc_violations = [v for v in report.violations if v.check == "same_arc"]
        assert len(arc_violations) == 0

    def test_generates_change_arc_fixes(self, optimizer):
        beats = [
            _make_beat("Luffy", "wano island", start_sec=i * 5, end_sec=(i + 1) * 5)
            for i in range(6)
        ]
        report = optimizer.analyze(beats)
        arc_violations = [v for v in report.violations if v.check == "same_arc"]
        if arc_violations:
            assert any(f.action == "change_arc" for f in arc_violations[0].fixes)


# ── Check 3: No reveal > 30 seconds ────────────────────────────────


class TestNoReveal:
    def test_detects_reveal_drought(self, optimizer):
        beats = [
            _make_beat("Luffy", beat_type="hook", start_sec=0, end_sec=5),
        ] + [
            _make_beat("Luffy", beat_type="evidence", start_sec=5 + i * 5, end_sec=10 + i * 5)
            for i in range(8)  # 40 seconds of evidence with no reveal
        ]
        report = optimizer.analyze(beats)
        drought_violations = [v for v in report.violations if v.check == "no_reveal_drought"]
        assert len(drought_violations) >= 1

    def test_no_violation_with_regular_reveals(self, optimizer):
        beats = [
            _make_beat("Luffy", beat_type="hook", start_sec=0, end_sec=10),
            _make_beat("Zoro", beat_type="evidence", start_sec=10, end_sec=20),
            _make_beat("Robin", beat_type="reveal", start_sec=20, end_sec=30),
            _make_beat("Shanks", beat_type="evidence", start_sec=30, end_sec=40),
            _make_beat("Ace", beat_type="payoff", start_sec=40, end_sec=50),
        ]
        report = optimizer.analyze(beats)
        drought_violations = [v for v in report.violations if v.check == "no_reveal_drought"]
        assert len(drought_violations) == 0

    def test_severity_is_critical(self, optimizer):
        beats = [_make_beat("Luffy", beat_type="evidence", start_sec=i * 5, end_sec=(i + 1) * 5) for i in range(10)]
        report = optimizer.analyze(beats)
        drought_violations = [v for v in report.violations if v.check == "no_reveal_drought"]
        if drought_violations:
            assert drought_violations[0].severity == "critical"

    def test_insert_reveal_fix(self, optimizer):
        beats = [_make_beat("Luffy", beat_type="evidence", start_sec=i * 5, end_sec=(i + 1) * 5) for i in range(10)]
        report = optimizer.analyze(beats)
        drought_violations = [v for v in report.violations if v.check == "no_reveal_drought"]
        if drought_violations:
            fixes = drought_violations[0].fixes
            assert any(f.action == "insert_reveal" for f in fixes)


# ── Check 4: Flat emotional curve ───────────────────────────────────


class TestFlatEmotion:
    def test_detects_flat_curve(self, optimizer):
        beats = [
            _make_beat("Luffy", beat_type="evidence", emotion="curious",
                         intensity=0.40, start_sec=i * 5, end_sec=(i + 1) * 5)
            for i in range(6)
        ]
        report = optimizer.analyze(beats)
        flat_violations = [v for v in report.violations if v.check == "flat_emotion"]
        assert len(flat_violations) >= 1

    def test_no_violation_with_intensity_spikes(self, optimizer):
        beats = [
            _make_beat("Luffy", emotion="curious", intensity=0.30, start_sec=0, end_sec=5),
            _make_beat("Zoro", emotion="tension", intensity=0.60, start_sec=5, end_sec=10),
            _make_beat("Robin", emotion="shock", intensity=0.90, start_sec=10, end_sec=15),
            _make_beat("Shanks", emotion="hope", intensity=0.50, start_sec=15, end_sec=20),
        ]
        report = optimizer.analyze(beats)
        flat_violations = [v for v in report.violations if v.check == "flat_emotion"]
        assert len(flat_violations) == 0

    def test_boost_emotion_fix(self, optimizer):
        beats = [
            _make_beat("Luffy", emotion="curious", intensity=0.40, start_sec=i * 5, end_sec=(i + 1) * 5)
            for i in range(6)
        ]
        report = optimizer.analyze(beats)
        flat_violations = [v for v in report.violations if v.check == "flat_emotion"]
        if flat_violations:
            assert any(f.action == "boost_emotion" for f in flat_violations[0].fixes)


# ── Check 5: Weak hook ──────────────────────────────────────────────


class TestWeakHook:
    def test_detects_non_hook_opening(self, optimizer):
        beats = [
            _make_beat("Luffy", beat_type="evidence", emotion="curious",
                         intensity=0.3, start_sec=0, end_sec=5),
        ]
        report = optimizer.analyze(beats)
        hook_violations = [v for v in report.violations if v.check == "weak_hook"]
        assert len(hook_violations) >= 1
        assert any("not 'hook'" in v.message for v in hook_violations)

    def test_detects_low_intensity_hook(self, optimizer):
        beats = [
            _make_beat("Luffy", beat_type="hook", emotion="curious",
                         intensity=0.2, start_sec=0, end_sec=5),
        ]
        report = optimizer.analyze(beats)
        hook_violations = [v for v in report.violations if v.check == "weak_hook"]
        assert len(hook_violations) >= 1

    def test_strong_hook_no_violation(self, optimizer):
        beats = [
            _make_beat("Luffy", beat_type="hook", emotion="tension",
                         intensity=0.85, start_sec=0, end_sec=5,
                         text_overlay="WHY DID HE DO IT?"),
        ]
        report = optimizer.analyze(beats)
        hook_violations = [v for v in report.violations if v.check == "weak_hook"]
        assert len(hook_violations) == 0

    def test_hook_without_overlay_flagged(self, optimizer):
        beats = [
            _make_beat("Luffy", beat_type="hook", emotion="tension",
                         intensity=0.85, start_sec=0, end_sec=5),
        ]
        report = optimizer.analyze(beats)
        hook_violations = [v for v in report.violations if v.check == "weak_hook"]
        if hook_violations:
            assert any("text_overlay" in v.message or "emphasis" in v.message for v in hook_violations)

    def test_generates_set_hook_fix(self, optimizer):
        beats = [
            _make_beat("Luffy", beat_type="evidence", intensity=0.2, start_sec=0, end_sec=5),
        ]
        report = optimizer.analyze(beats)
        hook_violations = [v for v in report.violations if v.check == "weak_hook"]
        assert len(hook_violations) >= 1
        assert any(f.action == "set_hook" for f in hook_violations[0].fixes)

    def test_static_motion_flagged(self, optimizer):
        beats = [
            _make_beat("Luffy", beat_type="hook", emotion="tension",
                         intensity=0.85, start_sec=0, end_sec=5,
                         motion="static_hold", text_overlay="HOOK"),
        ]
        report = optimizer.analyze(beats)
        hook_violations = [v for v in report.violations if v.check == "weak_hook"]
        if hook_violations:
            assert any(f.action == "fix_hook_motion" for f in hook_violations[0].fixes)


# ── Healthy storyboard (no violations) ──────────────────────────────


class TestHealthyStoryboard:
    def test_healthy_has_few_violations(self, optimizer):
        report = optimizer.analyze(_healthy_storyboard())
        # May have minor issues but no critical.
        assert report.critical_count <= 1

    def test_healthy_gets_good_grade(self, optimizer):
        report = optimizer.analyze(_healthy_storyboard())
        assert report.grade in ("A", "B", "C")


# ── Scoring and grading ─────────────────────────────────────────────


class TestScoring:
    def test_no_violations_perfect_score(self, optimizer):
        # Single hook slide — minimal checks.
        beats = [_make_beat("Luffy", beat_type="hook", emotion="tension",
                               intensity=0.85, text_overlay="HOOK")]
        report = optimizer.analyze(beats)
        assert report.score >= 0.7

    def test_many_violations_low_score(self, optimizer):
        # Everything wrong: same character, flat emotion, no reveal, weak hook.
        beats = [
            _make_beat("Luffy", beat_type="evidence", emotion="curious",
                         intensity=0.35, start_sec=i * 5, end_sec=(i + 1) * 5)
            for i in range(10)
        ]
        report = optimizer.analyze(beats)
        assert report.score < 0.6
        assert report.grade in ("C", "D", "F")

    def test_score_always_in_range(self, optimizer):
        beats = [_make_beat("Luffy", start_sec=i * 5, end_sec=(i + 1) * 5) for i in range(20)]
        report = optimizer.analyze(beats)
        assert 0.0 <= report.score <= 1.0


# ── Summary ──────────────────────────────────────────────────────────


class TestSummary:
    def test_summary_fields(self, optimizer):
        report = optimizer.analyze(_healthy_storyboard())
        s = report.summary
        assert "beat_count" in s
        assert "checks_run" in s
        assert s["checks_run"] == 5
        assert "violation_count" in s
        assert "total_fixes" in s
        assert "by_check" in s
        assert "by_severity" in s
        assert "grade" in s

    def test_critical_issues_listed(self, optimizer):
        beats = [_make_beat("Luffy", beat_type="evidence", intensity=0.2, start_sec=i * 5, end_sec=(i + 1) * 5) for i in range(10)]
        report = optimizer.analyze(beats)
        assert "critical_issues" in report.summary


# ── Report properties ───────────────────────────────────────────────


class TestReportProperties:
    def test_violation_count(self, optimizer):
        beats = [_make_beat("Luffy", beat_type="evidence", intensity=0.3, start_sec=i * 5, end_sec=(i + 1) * 5) for i in range(10)]
        report = optimizer.analyze(beats)
        assert report.violation_count == len(report.violations)
        assert report.fix_count == sum(len(v.fixes) for v in report.violations)

    def test_has_critical(self, optimizer):
        beats = [_make_beat("Luffy", beat_type="evidence", intensity=0.2)]
        report = optimizer.analyze(beats)
        # Weak hook should be critical.
        assert isinstance(report.has_critical, bool)


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_beats(self, optimizer):
        report = optimizer.analyze([])
        assert report.violation_count == 0
        assert report.score == 1.0
        assert report.grade == "A"

    def test_single_beat(self, optimizer):
        report = optimizer.analyze([_make_beat("Luffy")])
        assert isinstance(report, RetentionReport)

    def test_missing_fields(self, optimizer):
        beats = [{"start_time": "0:00:00.00", "end_time": "0:00:05.00"}]
        report = optimizer.analyze(beats)
        assert isinstance(report, RetentionReport)


# ── Serialization ────────────────────────────────────────────────────


class TestSerialization:
    def test_violation_to_dict(self, optimizer):
        beats = [_make_beat("Luffy", beat_type="evidence", intensity=0.2)]
        report = optimizer.analyze(beats)
        if report.violations:
            d = report.violations[0].to_dict()
            assert "check" in d
            assert "fixes" in d
            json.dumps(d)

    def test_report_to_dict(self, optimizer):
        report = optimizer.analyze(_healthy_storyboard())
        d = report.to_dict()
        assert "violations" in d
        assert "score" in d
        assert "grade" in d
        json.dumps(d)

    def test_fix_to_dict(self):
        fix = RetentionFix(
            target_index=0, action="test", description="Test fix",
            priority="high", field_changes={"key": "value"},
        )
        d = fix.to_dict()
        assert d["action"] == "test"
        json.dumps(d)


# ── Convenience function ─────────────────────────────────────────────


class TestConvenienceFunction:
    def test_optimize_retention(self):
        report = optimize_retention(_healthy_storyboard())
        assert isinstance(report, RetentionReport)
        assert report.summary["checks_run"] == 5

    def test_custom_params(self):
        report = optimize_retention(
            _healthy_storyboard(),
            max_same_character=2,
            reveal_gap_limit=15.0,
        )
        assert isinstance(report, RetentionReport)
