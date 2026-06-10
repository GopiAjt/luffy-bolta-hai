"""Tests for LegacyAdapter."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.slides.legacy_adapter import LegacyAdapter, adapt_to_legacy


@pytest.fixture
def adapter():
    return LegacyAdapter()


def _make_beat(**kwargs):
    beat = {
        "start_time": "10",
        "end_time": "15",
        "subtitle_text": "text",
        "beat_type": "hook",
        "visual_role": "character",
        "asset_metadata": {"search_tags": ["tag1", "tag2"]},
    }
    beat.update(kwargs)
    return beat


def test_empty_beats(adapter):
    slides = adapter.adapt([])
    assert len(slides) == 0


def test_basic_adaptation(adapter):
    beats = [_make_beat()]
    slides = adapter.adapt(beats)
    assert len(slides) == 1
    assert slides[0]["start_time"] == "10"
    assert slides[0]["end_time"] == "15"


def test_search_tags_extraction(adapter):
    beats = [_make_beat()]
    slides = adapter.adapt(beats)
    assert "search_tags" in slides[0]
    assert slides[0]["search_tags"] == ["tag1", "tag2"]


def test_duration_calculation(adapter):
    beats = [_make_beat(start_time="10.5", end_time="15.5")]
    slides = adapter.adapt(beats)
    assert "duration" in slides[0]
    assert slides[0]["duration"] == 5.0


def test_duration_calculation_fallback(adapter):
    # Non-float times
    beats = [_make_beat(start_time="00:10", end_time="00:15")]
    slides = adapter.adapt(beats)
    assert "duration" not in slides[0] # calculation fails silently


def test_convenience_function():
    beats = [_make_beat()]
    slides = adapt_to_legacy(beats)
    assert len(slides) == 1
