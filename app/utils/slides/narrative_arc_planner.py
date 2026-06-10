"""Narrative Arc Planner — structural pacing and act boundaries.

Maps raw story beats into a classic three-act structure, assigning
pacing targets and evaluating narrative rhythm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)


@dataclass
class ArcAnnotation:
    """Narrative annotations added to a beat."""
    beat_index: int
    act: str                 # "Act 1", "Act 2", "Act 3"
    pacing_target: str       # "fast", "moderate", "slow"
    arc_significance: str    # "high", "medium", "low"


@dataclass
class NarrativeArcReport:
    """Output of the narrative arc planner."""
    annotated_beats: List[Dict[str, Any]]
    annotations: List[ArcAnnotation]
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotated_beats": self.annotated_beats,
            "annotations": [
                {
                    "beat_index": a.beat_index,
                    "act": a.act,
                    "pacing_target": a.pacing_target,
                    "arc_significance": a.arc_significance,
                }
                for a in self.annotations
            ],
            "summary": self.summary,
        }


class NarrativeArcPlanner:
    """Plans the structural pacing and act boundaries of a storyboard."""

    def __init__(self, act_1_ratio: float = 0.20, act_2_ratio: float = 0.50):
        self.act_1_ratio = act_1_ratio
        self.act_2_ratio = act_2_ratio

    def plan(self, beats: Sequence[Dict[str, Any]]) -> NarrativeArcReport:
        """Assign narrative arcs and pacing targets to all beats.

        Parameters
        ----------
        beats : list of dict
            Raw analyzed story beats.

        Returns
        -------
        NarrativeArcReport
        """
        if not beats:
            return NarrativeArcReport([], [], {"beat_count": 0})

        total = len(beats)
        act_1_end = max(1, int(total * self.act_1_ratio))
        act_2_end = max(act_1_end + 1, int(total * (self.act_1_ratio + self.act_2_ratio)))

        annotated_beats = []
        annotations = []

        for i, beat in enumerate(beats):
            beat_copy = dict(beat)
            beat_type = str(beat.get("beat_type", "neutral")).lower()

            # Determine Act.
            if i < act_1_end:
                act = "Act 1"
            elif i < act_2_end:
                act = "Act 2"
            else:
                act = "Act 3"

            # Determine Pacing Target.
            if beat_type in ("hook", "escalation", "payoff", "climax", "action"):
                pacing = "fast"
            elif beat_type in ("evidence", "explanation", "context", "resolution"):
                pacing = "slow"
            else:
                if act == "Act 1" and i == 0:
                    pacing = "fast"
                elif act == "Act 3":
                    pacing = "fast" if i < total - 1 else "slow"
                else:
                    pacing = "moderate"

            # Determine Significance.
            if beat_type in ("hook", "reveal", "payoff", "reversal"):
                sig = "high"
            elif beat_type in ("escalation", "evidence", "cta"):
                sig = "medium"
            else:
                sig = "low"

            ann = ArcAnnotation(
                beat_index=i,
                act=act,
                pacing_target=pacing,
                arc_significance=sig,
            )
            annotations.append(ann)

            # Inject annotation into the beat copy.
            beat_copy["narrative_arc"] = {
                "act": act,
                "pacing_target": pacing,
                "arc_significance": sig,
            }
            annotated_beats.append(beat_copy)

        summary = {
            "beat_count": total,
            "act_1_count": act_1_end,
            "act_2_count": act_2_end - act_1_end,
            "act_3_count": total - act_2_end,
            "high_significance_beats": sum(1 for a in annotations if a.arc_significance == "high"),
        }

        return NarrativeArcReport(annotated_beats, annotations, summary)


def plan_narrative_arc(beats: Sequence[Dict[str, Any]], **kwargs) -> NarrativeArcReport:
    """Shortcut function to plan narrative arc."""
    return NarrativeArcPlanner(**kwargs).plan(beats)
