"""Visual Rhythm Engine — controls pacing, energy, and contrast.

Analyzes the sequence of beats to ensure a dynamic visual rhythm,
alternating high and low energy to prevent viewer fatigue or boredom.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)


@dataclass
class RhythmAnnotation:
    """Rhythm metrics and targets for a single beat."""
    beat_index: int
    energy: float              # [0.0, 1.0]
    tempo: str                 # "fast", "moderate", "slow"
    contrast_target: str       # "high_contrast", "smooth_blend"


@dataclass
class RhythmIssue:
    """A detected pacing issue in the sequence."""
    issue_type: str            # "fatigue", "boredom", "jarring_transition"
    beat_indices: List[int]
    description: str


@dataclass
class VisualRhythmReport:
    """Output of the visual rhythm engine."""
    annotated_beats: List[Dict[str, Any]]
    annotations: List[RhythmAnnotation]
    issues: List[RhythmIssue]
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotated_beats": self.annotated_beats,
            "annotations": [
                {
                    "beat_index": a.beat_index,
                    "energy": round(a.energy, 3),
                    "tempo": a.tempo,
                    "contrast_target": a.contrast_target,
                }
                for a in self.annotations
            ],
            "issues": [
                {
                    "issue_type": i.issue_type,
                    "beat_indices": i.beat_indices,
                    "description": i.description,
                }
                for i in self.issues
            ],
            "summary": self.summary,
        }


class VisualRhythmEngine:
    """Manages beat-to-beat visual energy transitions."""

    def __init__(self, max_consecutive_high_energy: int = 3, max_consecutive_low_energy: int = 4):
        self.max_high = max_consecutive_high_energy
        self.max_low = max_consecutive_low_energy

    def analyze(self, beats: Sequence[Dict[str, Any]]) -> VisualRhythmReport:
        """Analyze and annotate rhythm for all beats.

        Parameters
        ----------
        beats : list of dict
            Story beats.

        Returns
        -------
        VisualRhythmReport
        """
        if not beats:
            return VisualRhythmReport([], [], [], {"beat_count": 0})

        annotations = []
        annotated_beats = []

        # Pass 1: Compute raw energy per beat.
        for i, beat in enumerate(beats):
            energy = self._compute_energy(beat)
            tempo = "fast" if energy > 0.7 else "slow" if energy < 0.4 else "moderate"
            
            ann = RhythmAnnotation(
                beat_index=i,
                energy=energy,
                tempo=tempo,
                contrast_target="smooth_blend",  # Default, adjusted in Pass 2
            )
            annotations.append(ann)

        # Pass 2: Compute contrast targets based on adjacent energy shifts.
        for i in range(len(annotations) - 1):
            curr = annotations[i]
            nxt = annotations[i + 1]
            shift = abs(nxt.energy - curr.energy)
            if shift > 0.4:
                curr.contrast_target = "high_contrast"
                nxt.contrast_target = "high_contrast"

        # Pass 3: Detect pacing issues.
        issues = self._detect_issues(annotations)

        # Pass 4: Apply annotations.
        for i, beat in enumerate(beats):
            beat_copy = dict(beat)
            ann = annotations[i]
            beat_copy["visual_rhythm"] = {
                "energy": round(ann.energy, 3),
                "tempo": ann.tempo,
                "contrast_target": ann.contrast_target,
            }
            annotated_beats.append(beat_copy)

        summary = {
            "beat_count": len(beats),
            "avg_energy": sum(a.energy for a in annotations) / len(annotations),
            "high_contrast_transitions": sum(1 for a in annotations if a.contrast_target == "high_contrast"),
            "issue_count": len(issues),
        }

        return VisualRhythmReport(annotated_beats, annotations, issues, summary)

    def _compute_energy(self, beat: Dict[str, Any]) -> float:
        """Compute the visual energy of a single beat."""
        base = 0.5
        
        # Beat type contribution
        bt = str(beat.get("beat_type", "")).lower()
        if bt in ("hook", "climax", "payoff", "action"):
            base += 0.2
        elif bt in ("evidence", "context", "resolution"):
            base -= 0.15

        # Emotion contribution
        emo = beat.get("emotion_state")
        if isinstance(emo, dict):
            intensity = float(emo.get("intensity", 0.5))
            emotion = str(emo.get("emotion", "")).lower()
            if emotion in ("fear", "rage", "shock", "triumph"):
                base += (intensity * 0.3)
            elif emotion in ("grief", "calm", "sadness"):
                base -= (intensity * 0.2)

        # Motion preset contribution
        motion = str(beat.get("motion_preset", "")).lower()
        if motion in ("impact_shake", "dramatic_push", "climax_zoom"):
            base += 0.15
        elif motion in ("static_hold", "grief_hold"):
            base -= 0.2

        return max(0.0, min(1.0, base))

    def _detect_issues(self, annotations: List[RhythmAnnotation]) -> List[RhythmIssue]:
        """Find fatigue or boredom patterns."""
        issues = []
        
        high_streak = 0
        high_start = 0
        
        low_streak = 0
        low_start = 0

        for i, ann in enumerate(annotations):
            if ann.energy > 0.75:
                if high_streak == 0:
                    high_start = i
                high_streak += 1
                low_streak = 0
            elif ann.energy < 0.35:
                if low_streak == 0:
                    low_start = i
                low_streak += 1
                high_streak = 0
            else:
                high_streak = 0
                low_streak = 0

            if high_streak > self.max_high:
                indices = list(range(high_start, i + 1))
                issues.append(RhythmIssue(
                    issue_type="fatigue",
                    beat_indices=indices,
                    description=f"Too many high energy beats ({high_streak}). Viewers may experience fatigue.",
                ))
                high_streak = 0  # reset to avoid overlapping issues
                
            if low_streak > self.max_low:
                indices = list(range(low_start, i + 1))
                issues.append(RhythmIssue(
                    issue_type="boredom",
                    beat_indices=indices,
                    description=f"Too many low energy beats ({low_streak}). Viewers may get bored.",
                ))
                low_streak = 0

        return issues


def analyze_visual_rhythm(beats: Sequence[Dict[str, Any]], **kwargs) -> VisualRhythmReport:
    """Shortcut function to analyze visual rhythm."""
    return VisualRhythmEngine(**kwargs).analyze(beats)
