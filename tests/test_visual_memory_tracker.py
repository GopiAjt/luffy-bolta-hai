"""Tests for VisualMemoryTracker."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.slides.visual_memory_tracker import VisualMemoryTracker, track_visual_memory


@pytest.fixture
def tracker():
    return VisualMemoryTracker(min_beats_between_reuse=3)


def _make_beat(asset_id="asset_1", beat_type="neutral"):
    return {
        "asset_metadata": {"id": asset_id},
        "beat_type": beat_type,
    }


def test_empty_beats(tracker):
    report = tracker.track([])
    assert len(report.annotated_beats) == 0
    assert report.summary["beat_count"] == 0


def test_first_appearance(tracker):
    beats = [_make_beat("a1"), _make_beat("a2")]
    report = tracker.track(beats)
    assert report.annotations[0].is_first_appearance is True
    assert report.annotations[1].is_first_appearance is True


def test_reuse_count_and_gap(tracker):
    beats = [
        _make_beat("a1"), # 0
        _make_beat("a2"), # 1
        _make_beat("a3"), # 2
        _make_beat("a1"), # 3
    ]
    report = tracker.track(beats)
    assert report.annotations[3].is_first_appearance is False
    assert report.annotations[3].reuse_count == 1
    assert report.annotations[3].beats_since_last_seen == 3


def test_premature_reuse_issue(tracker):
    beats = [
        _make_beat("a1"), # 0
        _make_beat("a2"), # 1
        _make_beat("a1"), # 2: gap is 2, min is 3 -> issue
    ]
    report = tracker.track(beats)
    assert len(report.issues) == 1
    assert report.issues[0].issue_type == "premature_reuse"
    assert report.issues[0].beat_index == 2


def test_spoiled_reveal_issue(tracker):
    beats = [
        _make_beat("a1", "neutral"), # 0
        _make_beat("a2", "neutral"), # 1
        _make_beat("a3", "neutral"), # 2
        _make_beat("a4", "neutral"), # 3
        _make_beat("a1", "reveal"),  # 4: gap is 4, but it's a reveal and already seen! -> issue
    ]
    report = tracker.track(beats)
    assert len(report.issues) == 1
    assert report.issues[0].issue_type == "spoiled_reveal"
    assert report.issues[0].beat_index == 4


def test_convenience_function():
    beats = [_make_beat()]
    report = track_visual_memory(beats)
    assert len(report.annotated_beats) == 1
