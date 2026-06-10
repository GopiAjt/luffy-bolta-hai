"""Tests for NarrativeArcPlanner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.slides.narrative_arc_planner import NarrativeArcPlanner, plan_narrative_arc


@pytest.fixture
def planner():
    return NarrativeArcPlanner()


def _make_beat(beat_type="neutral"):
    return {"beat_type": beat_type}


def test_empty_beats(planner):
    report = planner.plan([])
    assert len(report.annotated_beats) == 0
    assert report.summary["beat_count"] == 0


def test_basic_act_structure(planner):
    # 10 beats: Act 1 (20%) -> 2 beats. Act 2 (50%) -> 5 beats. Act 3 (30%) -> 3 beats.
    beats = [_make_beat() for _ in range(10)]
    report = planner.plan(beats)
    acts = [a.act for a in report.annotations]
    assert acts == ["Act 1", "Act 1", "Act 2", "Act 2", "Act 2", "Act 2", "Act 2", "Act 3", "Act 3", "Act 3"]


def test_pacing_targets(planner):
    beats = [
        _make_beat("hook"),
        _make_beat("evidence"),
        _make_beat("escalation"),
        _make_beat("resolution"),
    ]
    report = planner.plan(beats)
    pacings = [a.pacing_target for a in report.annotations]
    assert pacings[0] == "fast"
    assert pacings[1] == "slow"
    assert pacings[2] == "fast"
    assert pacings[3] == "slow"


def test_significance(planner):
    beats = [
        _make_beat("hook"),
        _make_beat("reveal"),
        _make_beat("cta"),
        _make_beat("neutral"),
    ]
    report = planner.plan(beats)
    sigs = [a.arc_significance for a in report.annotations]
    assert sigs == ["high", "high", "medium", "low"]


def test_beat_annotation_injected(planner):
    beats = [_make_beat("hook")]
    report = planner.plan(beats)
    assert "narrative_arc" in report.annotated_beats[0]
    arc = report.annotated_beats[0]["narrative_arc"]
    assert arc["act"] == "Act 1"
    assert arc["pacing_target"] == "fast"
    assert arc["arc_significance"] == "high"


def test_convenience_function():
    beats = [_make_beat("hook")]
    report = plan_narrative_arc(beats)
    assert len(report.annotated_beats) == 1
