"""Tests for AssetNarrativePlanner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.slides.asset_narrative_planner import AssetNarrativePlanner, plan_asset_narrative


@pytest.fixture
def planner():
    return AssetNarrativePlanner(fallback_entity="Luffy")


def _make_beat(entities=None, role="character"):
    return {
        "context_entities": entities or [],
        "visual_role": role,
    }


def test_empty_beats(planner):
    report = planner.plan([])
    assert len(report.annotated_beats) == 0
    assert report.summary["beat_count"] == 0


def test_basic_assignment(planner):
    beats = [
        _make_beat(["Zoro"]),
        _make_beat(["Sanji"]),
    ]
    report = planner.plan(beats)
    assert report.assignments[0].entity == "Zoro"
    assert report.assignments[1].entity == "Sanji"


def test_location_role(planner):
    beats = [
        _make_beat(role="location"),
    ]
    report = planner.plan(beats)
    assert report.assignments[0].entity_type == "location"


def test_fallback(planner):
    beats = [
        _make_beat(["Zoro"]),
        _make_beat(), # no entities, should fallback to Zoro
    ]
    report = planner.plan(beats)
    assert report.assignments[0].entity == "Zoro"
    assert report.assignments[1].entity == "Zoro"
    assert report.assignments[1].reasoning.startswith("Fallback")


def test_default_fallback(planner):
    beats = [
        _make_beat(), # no entities, no previous
    ]
    report = planner.plan(beats)
    assert report.assignments[0].entity == "Luffy"


def test_convenience_function():
    beats = [_make_beat(["Nami"])]
    report = plan_asset_narrative(beats, fallback_entity="Usopp")
    assert report.assignments[0].entity == "Nami"
