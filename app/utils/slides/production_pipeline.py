from dataclasses import dataclass, field
from typing import List, Dict, Any
from app.utils.slides.emotion_curve import EmotionCurveGenerator
from app.utils.slides.composition_planner import CompositionPlanner
from app.utils.slides.motion_planner import MotionPlanner
from app.utils.slides.transition_planner import TransitionPlanner
from app.utils.slides.visual_memory_tracker import VisualMemoryTracker
from app.utils.slides.quality_gate import QualityGate
from app.utils.slides.legacy_adapter import LegacyAdapter

@dataclass
class PipelineResult:
    """The result of executing the production pipeline."""
    success: bool
    beats: List[Dict[str, Any]]
    logs: List[str] = field(default_factory=list)

class PipelineStage:
    """Interface for a stage in the production pipeline."""
    def execute(self, beats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError("PipelineStage must implement execute()")

class ProductionPipeline:
    """Executes a series of PipelineStages on a list of beats."""
    
    def __init__(self, stages: List[PipelineStage] = None):
        self.stages = stages or []
        self.disabled_stages = set()
        self.logs: List[str] = []

    def register_stage(self, stage: PipelineStage):
        """Add a stage to the end of the pipeline."""
        self.stages.append(stage)
        
    def disable_stage(self, stage_name: str):
        """Disable a stage by its class name so it won't be executed."""
        self.disabled_stages.add(stage_name)
        
    def override_stage(self, stage_name: str, new_stage: PipelineStage):
        """Replace a stage that has a specific class name with a new stage."""
        for i, stage in enumerate(self.stages):
            if stage.__class__.__name__ == stage_name:
                self.stages[i] = new_stage
                return
        raise ValueError(f"Stage '{stage_name}' not found in pipeline.")

    def run(self, beats: List[Dict[str, Any]]) -> PipelineResult:
        self.logs = ["Starting pipeline execution"]
        current_beats = list(beats)  # Work on a copy of the list
        
        for stage in self.stages:
            stage_name = stage.__class__.__name__
            
            if stage_name in self.disabled_stages:
                self.logs.append(f"Skipping disabled stage: {stage_name}")
                continue
                
            self.logs.append(f"Executing stage: {stage_name}")
            
            try:
                current_beats = stage.execute(current_beats)
                self.logs.append(f"Stage {stage_name} completed successfully")
            except Exception as e:
                self.logs.append(f"Stage {stage_name} failed: {str(e)}")
                return PipelineResult(
                    success=False,
                    beats=current_beats,
                    logs=list(self.logs)
                )
                
        self.logs.append("Pipeline execution completed successfully")
        return PipelineResult(
            success=True,
            beats=current_beats,
            logs=list(self.logs)
        )

class EmotionCurveStage(PipelineStage):
    def __init__(self, generator: EmotionCurveGenerator = None):
        self.generator = generator or EmotionCurveGenerator()

    def execute(self, beats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        curve = self.generator.generate(beats)
        for i, beat in enumerate(beats):
            if i < len(curve.points):
                pt = curve.points[i]
                beat["emotion_score"] = pt.score
                # Inject label into emotion_state
                beat.setdefault("emotion_state", {})["emotion"] = pt.label
        return beats

class CompositionStage(PipelineStage):
    def __init__(self, planner: CompositionPlanner = None):
        self.planner = planner or CompositionPlanner()

    def execute(self, beats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for beat in beats:
            composition = self.planner.plan(beat)
            beat["composition"] = composition.to_dict()
        return beats

class MotionStage(PipelineStage):
    def __init__(self, planner: MotionPlanner = None):
        self.planner = planner or MotionPlanner()

    def execute(self, beats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prev_style = ""
        for beat in beats:
            emo = beat.get("emotion_state", {}).get("emotion", "neutral")
            intent = beat.get("visual_intent", "")
            btype = beat.get("beat_type", "")
            motion_plan = self.planner.plan(
                emotion=emo, 
                visual_intent=intent, 
                beat_type=btype,
                previous_style=prev_style,
                total_beats=len(beats)
            )
            beat["motion"] = motion_plan.to_dict()
            if "motion_preset" not in beat:
                beat["motion_preset"] = motion_plan.style
            prev_style = motion_plan.style
        return beats

class TransitionStage(PipelineStage):
    def __init__(self, planner: TransitionPlanner = None):
        self.planner = planner or TransitionPlanner()

    def execute(self, beats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        total_beats = len(beats)
        for i in range(total_beats):
            curr_beat = beats[i]
            prev_beat = beats[i-1] if i > 0 else None
            plan = self.planner.plan_transition(prev_beat, curr_beat, total_beats)
            curr_beat["transition_in"] = plan.transition_type.value
            curr_beat["transition_duration"] = plan.duration
        return beats

class VisualMemoryStage(PipelineStage):
    def __init__(self, tracker: VisualMemoryTracker = None):
        self.tracker = tracker or VisualMemoryTracker()

    def execute(self, beats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        report = self.tracker.track(beats)
        return report.annotated_beats

class StoryboardStage(PipelineStage):
    def execute(self, beats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Mock/pass-through stage for now
        return beats

class QualityGateStage(PipelineStage):
    def __init__(self, gate: QualityGate = None):
        self.gate = gate or QualityGate()

    def execute(self, beats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.gate.analyze_storyboard(beats)
        report = self.gate.evaluate()
        if not report.passed:
            errors = [i.description for i in report.issues if i.severity.value == "ERROR"]
            raise RuntimeError(f"QualityGate failed with {len(errors)} ERROR(s): {errors}")
        return beats

class LegacyAdapterStage(PipelineStage):
    def __init__(self, adapter: LegacyAdapter = None):
        self.adapter = adapter or LegacyAdapter()

    def execute(self, beats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.adapter.adapt(beats)
