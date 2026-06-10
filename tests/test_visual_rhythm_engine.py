"""Tests for VisualRhythmEngine."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.slides.visual_rhythm_engine import VisualRhythmEngine, analyze_visual_rhythm


@pytest.fixture
def engine():
    return VisualRhythmEngine(max_consecutive_high_energy=2, max_consecutive_low_energy=3)


def _make_beat(beat_type="neutral", emotion="neutral", intensity=0.5, motion="subject_push"):
    return {
        "beat_type": beat_type,
        "emotion_state": {"emotion": emotion, "intensity": intensity},
        "motion_preset": motion,
    }


def test_empty_beats(engine):
    report = engine.analyze([])
    assert len(report.annotated_beats) == 0
    assert report.summary["beat_count"] == 0


def test_energy_computation(engine):
    high = _make_beat("climax", "rage", 1.0, "impact_shake")
    low = _make_beat("evidence", "calm", 1.0, "static_hold")
    
    e_high = engine._compute_energy(high)
    e_low = engine._compute_energy(low)
    
    assert e_high > 0.8
    assert e_low < 0.4


def test_contrast_targets(engine):
    beats = [
        _make_beat("climax", "rage", 1.0, "impact_shake"),  # High energy
        _make_beat("evidence", "calm", 1.0, "static_hold"), # Low energy
    ]
    report = engine.analyze(beats)
    assert report.annotations[0].contrast_target == "high_contrast"
    assert report.annotations[1].contrast_target == "high_contrast"


def test_fatigue_detection(engine):
    # max_consecutive_high_energy is 2
    beats = [
        _make_beat("climax", "rage", 1.0, "impact_shake"),
        _make_beat("climax", "rage", 1.0, "impact_shake"),
        _make_beat("climax", "rage", 1.0, "impact_shake"), # 3 in a row
    ]
    report = engine.analyze(beats)
    assert len(report.issues) == 1
    assert report.issues[0].issue_type == "fatigue"
    assert len(report.issues[0].beat_indices) == 3


def test_boredom_detection(engine):
    # max_consecutive_low_energy is 3
    beats = [
        _make_beat("evidence", "calm", 1.0, "static_hold"),
        _make_beat("evidence", "calm", 1.0, "static_hold"),
        _make_beat("evidence", "calm", 1.0, "static_hold"),
        _make_beat("evidence", "calm", 1.0, "static_hold"), # 4 in a row
    ]
    report = engine.analyze(beats)
    assert len(report.issues) == 1
    assert report.issues[0].issue_type == "boredom"
    assert len(report.issues[0].beat_indices) == 4


def test_convenience_function():
    beats = [_make_beat()]
    report = analyze_visual_rhythm(beats)
    assert len(report.annotated_beats) == 1
