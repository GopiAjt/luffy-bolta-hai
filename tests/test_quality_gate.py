"""Tests for QualityGate."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.slides.quality_gate import QualityGate, evaluate_quality


@pytest.fixture
def gate():
    return QualityGate()


def _make_beat(**kwargs):
    beat = {
        "beat_type": "hook",
        "visual_role": "character",
        "motion_preset": "dramatic_push",
        "subtitle_text": "text",
        "start_time": "0:00",
        "end_time": "0:05",
    }
    beat.update(kwargs)
    return beat


def test_empty_beats(gate):
    report = gate.evaluate([])
    assert not report.passed
    assert len(report.issues) == 1
    assert report.issues[0].severity == "critical"


def test_perfect_beat(gate):
    beats = [_make_beat()]
    report = gate.evaluate(beats)
    assert report.passed
    assert len(report.issues) == 0


def test_missing_required_keys_autofix(gate):
    # Missing all required keys
    beat = _make_beat()
    del beat["beat_type"]
    del beat["visual_role"]
    del beat["motion_preset"]
    
    report = gate.evaluate([beat])
    assert report.passed  # auto-fixed, so only warnings
    assert len(report.issues) == 3
    for issue in report.issues:
        assert issue.severity == "warning"
        
    fixed = report.auto_fixed_beats[0]
    assert fixed["beat_type"] == "neutral"
    assert fixed["visual_role"] == "character"
    assert fixed["motion_preset"] == "static_hold"


def test_missing_text_critical(gate):
    beat = _make_beat()
    del beat["subtitle_text"]
    
    report = gate.evaluate([beat])
    assert not report.passed
    assert len(report.issues) == 1
    assert report.issues[0].severity == "critical"
    
    # But if it has summary instead, it passes
    beat["summary"] = "A summary"
    report2 = gate.evaluate([beat])
    assert report2.passed


def test_missing_time_warning(gate):
    beat = _make_beat()
    del beat["start_time"]
    
    report = gate.evaluate([beat])
    assert report.passed
    assert len(report.issues) == 1
    assert report.issues[0].severity == "warning"


def test_convenience_function():
    beats = [_make_beat()]
    report = evaluate_quality(beats)
    assert report.passed
