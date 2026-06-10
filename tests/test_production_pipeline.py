"""Tests for ProductionPipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.slides.production_pipeline import ProductionPipeline, run_pipeline


@pytest.fixture
def pipeline():
    return ProductionPipeline()


def _make_beat(**kwargs):
    beat = {
        "start_time": "10",
        "end_time": "15",
        "subtitle_text": "Sample text",
        "beat_type": "hook",
        "visual_role": "character",
        "emotion_state": {"emotion": "hope", "intensity": 0.8},
        "asset_metadata": {"search_tags": ["wano"]},
    }
    beat.update(kwargs)
    return beat


def test_empty_pipeline(pipeline):
    result = pipeline.run([])
    # Empty storyboard gets a critical error from QualityGate
    assert not result.passed_quality_gate
    assert len(result.slides) == 0


def test_basic_pipeline_run(pipeline):
    beats = [_make_beat(), _make_beat(beat_type="reveal")]
    result = pipeline.run(beats)
    
    # Should pass quality gate
    assert result.passed_quality_gate
    
    # Legacy slides are generated
    assert len(result.slides) == 2
    
    # Reports should contain all stages
    assert "narrative_arc" in result.reports
    assert "retention" in result.reports
    assert "visual_rhythm" in result.reports
    assert "asset_narrative" in result.reports
    assert "visual_memory" in result.reports
    assert "visual_diversity" in result.reports
    assert "quality_gate" in result.reports
    
    # Composition and motion should be embedded inside the output slides (since they are added to beats)
    assert "composition" in result.slides[0]
    assert "motion" in result.slides[0]
    
    # Check that legacy adapter worked
    assert "duration" in result.slides[0]
    assert result.slides[0]["duration"] == 5.0


def test_convenience_function():
    beats = [_make_beat()]
    result = run_pipeline(beats)
    assert result.passed_quality_gate
    assert len(result.slides) == 1
