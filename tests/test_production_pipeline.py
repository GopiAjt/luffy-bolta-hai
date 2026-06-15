import pytest
from app.utils.slides.production_pipeline import PipelineResult, PipelineStage, ProductionPipeline

class MockSuccessStage(PipelineStage):
    def execute(self, beats):
        for beat in beats:
            beat["processed_by_success"] = True
        return beats

class MockFailureStage(PipelineStage):
    def execute(self, beats):
        raise ValueError("Simulated stage failure")

def test_pipeline_result_dataclass():
    """Verify dataclass creation."""
    result = PipelineResult(success=True, beats=[{"id": 1}], logs=["Log 1"])
    assert result.success is True
    assert len(result.beats) == 1
    assert result.logs[0] == "Log 1"

def test_production_pipeline_success():
    """Verify pipeline successfully executes stages and logs progress."""
    pipeline = ProductionPipeline(stages=[MockSuccessStage()])
    beats = [{"id": 1}, {"id": 2}]
    
    result = pipeline.run(beats)
    
    assert result.success is True
    assert len(result.beats) == 2
    assert result.beats[0].get("processed_by_success") is True
    
    # Check logs
    assert "Starting pipeline execution" in result.logs
    assert "Executing stage: MockSuccessStage" in result.logs
    assert "Stage MockSuccessStage completed successfully" in result.logs
    assert "Pipeline execution completed successfully" in result.logs

def test_production_pipeline_failure():
    """Verify pipeline handles stage failures gracefully and logs the error."""
    pipeline = ProductionPipeline(stages=[MockSuccessStage(), MockFailureStage()])
    beats = [{"id": 1}]
    
    result = pipeline.run(beats)
    
    assert result.success is False
    assert result.beats[0].get("processed_by_success") is True  # First stage still ran
    
    # Check logs
    assert "Executing stage: MockFailureStage" in result.logs
    assert any("Simulated stage failure" in log for log in result.logs)
    assert "Pipeline execution completed successfully" not in result.logs

def test_production_pipeline_empty_stages():
    """Verify pipeline handles having no stages."""
    pipeline = ProductionPipeline()
    beats = [{"id": 1}]
    
    result = pipeline.run(beats)
    assert result.success is True
    assert len(result.beats) == 1
    assert len(result.logs) == 2  # Start and complete logs

def test_production_pipeline_register_stage():
    """Verify register_stage adds a stage."""
    pipeline = ProductionPipeline()
    pipeline.register_stage(MockSuccessStage())
    assert len(pipeline.stages) == 1
    
    result = pipeline.run([{"id": 1}])
    assert result.success is True
    assert result.beats[0].get("processed_by_success") is True

def test_production_pipeline_disable_stage():
    """Verify disable_stage skips the stage."""
    pipeline = ProductionPipeline(stages=[MockSuccessStage()])
    pipeline.disable_stage("MockSuccessStage")
    
    result = pipeline.run([{"id": 1}])
    assert result.success is True
    assert result.beats[0].get("processed_by_success") is None
    assert "Skipping disabled stage: MockSuccessStage" in result.logs

class MockReplacementStage(PipelineStage):
    def execute(self, beats):
        for beat in beats:
            beat["processed_by_replacement"] = True
        return beats

def test_production_pipeline_override_stage():
    """Verify override_stage replaces an existing stage."""
    pipeline = ProductionPipeline(stages=[MockSuccessStage()])
    pipeline.override_stage("MockSuccessStage", MockReplacementStage())
    
    result = pipeline.run([{"id": 1}])
    assert result.success is True
    assert result.beats[0].get("processed_by_success") is None
    assert result.beats[0].get("processed_by_replacement") is True
    assert "Executing stage: MockReplacementStage" in result.logs
    
def test_production_pipeline_override_stage_not_found():
    """Verify override_stage raises ValueError if stage is not found."""
    pipeline = ProductionPipeline()
    with pytest.raises(ValueError):
        pipeline.override_stage("NonExistentStage", MockReplacementStage())

from app.utils.slides.production_pipeline import EmotionCurveStage, CompositionStage, MotionStage

def test_production_pipeline_integrated_stages():
    """Verify the integrated stages process beats correctly without errors."""
    pipeline = ProductionPipeline(stages=[
        EmotionCurveStage(),
        CompositionStage(),
        MotionStage()
    ])
    
    beats = [
        {"beat_type": "hook", "text": "This is a dramatic hook!"},
        {"beat_type": "evidence", "text": "Here is some evidence."}
    ]
    
    result = pipeline.run(beats)
    
    assert result.success is True
    assert len(result.beats) == 2
    
    for beat in result.beats:
        assert "emotion_score" in beat
        assert "emotion" in beat.get("emotion_state", {})
        assert "composition" in beat
        assert "motion" in beat
        assert "motion_preset" in beat

from app.utils.slides.production_pipeline import TransitionStage, VisualMemoryStage, StoryboardStage, QualityGateStage, LegacyAdapterStage

def test_production_pipeline_full_sequence():
    """Verify the requested full execution order integrates correctly."""
    pipeline = ProductionPipeline(stages=[
        MotionStage(),
        TransitionStage(),
        StoryboardStage(),
        QualityGateStage(),
        LegacyAdapterStage()
    ])
    
    beats = [
        {"beat_type": "hook", "text": "This is a dramatic hook!", "emotion_score": 80, "duration": 2.0},
        {"beat_type": "evidence", "text": "Here is some evidence.", "emotion_score": 50, "duration": 3.0}
    ]
    
    result = pipeline.run(beats)
    assert result.success is True
    assert len(result.beats) == 2
    
    # Check that it reached LegacyAdapter format
    slide = result.beats[0]
    # Legacy fields
    assert "start_time" in slide
    assert "motion_preset" in slide
    assert "transition_in" in slide

def test_production_pipeline_fails_on_quality_error():
    """Verify that QualityGateStage raises an error and fails the pipeline on ERROR severity."""
    pipeline = ProductionPipeline(stages=[QualityGateStage()])
    
    # REVEAL longer than 8.0s triggers an ERROR in QualityGate
    beats = [
        {"beat_type": "REVEAL", "duration": 10.0, "emotion_score": 80}
    ]
    
    result = pipeline.run(beats)
    
    assert result.success is False
    assert any("QualityGate failed with 1 ERROR" in log for log in result.logs)

