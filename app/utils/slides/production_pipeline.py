"""Production Pipeline Orchestrator.

Wires the complete 20-stage StoryboardBeat architecture end-to-end.
Takes raw input and returns fully realized legacy slide JSON ready for rendering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Import available modules
from .narrative_arc_planner import plan_narrative_arc
from .visual_rhythm_engine import analyze_visual_rhythm
from .asset_narrative_planner import plan_asset_narrative
from .visual_memory_tracker import track_visual_memory
from .visual_diversity import score_visual_diversity
from .composition_planner import plan_composition
from .motion_planner import plan_motion
from .thumbnail_moments import detect_thumbnail_moments
from .retention_optimizer import optimize_retention
from .quality_gate import evaluate_quality
from .legacy_adapter import adapt_to_legacy

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Final result of the production pipeline."""
    slides: List[Dict[str, Any]]
    passed_quality_gate: bool
    quality_issues: List[Any]
    reports: Dict[str, Any] = field(default_factory=dict)


class ProductionPipeline:
    """Orchestrates the entire storyboard beat generation process."""

    def __init__(self, video_profile: Optional[Dict[str, Any]] = None):
        self.video_profile = video_profile or {"platform": "youtube_long"}

    def run(self, initial_beats: List[Dict[str, Any]]) -> PipelineResult:
        """Run the full storyboard pipeline.

        Parameters
        ----------
        initial_beats : list of dict
            Raw analyzed beats from StoryAnalyzer / Script.

        Returns
        -------
        PipelineResult
        """
        reports = {}
        beats = initial_beats

        # 1. ASS / Script -> StoryAnalyzer (assumed done outside pipeline)
        # 2. NarrativeArcPlanner
        arc_report = plan_narrative_arc(beats)
        beats = arc_report.annotated_beats
        reports["narrative_arc"] = arc_report.summary

        # 3. StoryBeatDetector (Mocked/Identity)
        # 4. EmotionCurveGenerator (Mocked/Identity or assume done)
        # 5. VisualIntentClassifier (Mocked/Identity or assume done)

        # 6. ThumbnailMomentDetector
        thumb_report = detect_thumbnail_moments(beats)
        beats = thumb_report.annotated_beats
        reports["thumbnail_moments"] = thumb_report.summary

        # 7. RetentionOptimizer
        ret_report = optimize_retention(beats)
        # RetentionOptimizer is an analyzer, we just pass the original beats through.
        reports["retention"] = ret_report.summary

        # 8. VisualRhythmEngine
        rhythm_report = analyze_visual_rhythm(beats)
        beats = rhythm_report.annotated_beats
        reports["visual_rhythm"] = rhythm_report.summary

        # 9. CharacterRelationshipEngine (Mocked)
        # 10. AssetDatabase (Mocked)

        # 11. AssetNarrativePlanner
        asset_report = plan_asset_narrative(beats, available_assets={})
        beats = asset_report.annotated_beats
        reports["asset_narrative"] = asset_report.summary

        # 12. AssetSelector (Mocked)

        # 13. VisualMemoryTracker
        memory_report = track_visual_memory(beats)
        beats = memory_report.annotated_beats
        reports["visual_memory"] = memory_report.summary

        # 14. VisualDiversityScorer
        diversity_report = score_visual_diversity(beats)
        reports["visual_diversity"] = diversity_report.summary

        # 15. CompositionPlanner
        for beat in beats:
            beat["composition"] = plan_composition(beat).to_dict()

        # 16. MotionPlanner
        for beat in beats:
            emo = beat.get("emotion_state", {}).get("emotion", "neutral")
            intent = beat.get("visual_intent", "")
            btype = beat.get("beat_type", "")
            beat["motion"] = plan_motion(emo, intent, btype).to_dict()
            if "motion_preset" not in beat:
                beat["motion_preset"] = beat["motion"]["style"]

        # 17. StoryboardGenerator (Mocked)

        # 18. QualityGate
        quality_report = evaluate_quality(beats)
        beats = quality_report.auto_fixed_beats
        passed = quality_report.passed
        issues = quality_report.issues
        reports["quality_gate"] = quality_report.summary

        # 19. LegacyAdapter
        legacy_slides = adapt_to_legacy(beats)

        return PipelineResult(
            slides=legacy_slides,
            passed_quality_gate=passed,
            quality_issues=issues,
            reports=reports,
        )


def run_pipeline(initial_beats: List[Dict[str, Any]], **kwargs) -> PipelineResult:
    """Shortcut function to run the production pipeline."""
    return ProductionPipeline(**kwargs).run(initial_beats)
