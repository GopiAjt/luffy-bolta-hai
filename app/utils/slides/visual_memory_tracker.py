"""Visual Memory Tracker — tracks viewer exposure to assets.

Monitors what assets have been shown to the viewer, flags premature reuses,
and tracks callbacks and visual memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)


@dataclass
class MemoryAnnotation:
    """Memory tracking for a single beat."""
    beat_index: int
    asset_id: str
    is_first_appearance: bool
    reuse_count: int
    beats_since_last_seen: int


@dataclass
class MemoryIssue:
    """A detected visual memory issue."""
    issue_type: str
    beat_index: int
    description: str


@dataclass
class VisualMemoryReport:
    """Output of the visual memory tracker."""
    annotated_beats: List[Dict[str, Any]]
    annotations: List[MemoryAnnotation]
    issues: List[MemoryIssue]
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotated_beats": self.annotated_beats,
            "annotations": [
                {
                    "beat_index": a.beat_index,
                    "asset_id": a.asset_id,
                    "is_first_appearance": a.is_first_appearance,
                    "reuse_count": a.reuse_count,
                    "beats_since_last_seen": a.beats_since_last_seen,
                }
                for a in self.annotations
            ],
            "issues": [
                {
                    "issue_type": i.issue_type,
                    "beat_index": i.beat_index,
                    "description": i.description,
                }
                for i in self.issues
            ],
            "summary": self.summary,
        }


class VisualMemoryTracker:
    """Tracks what the viewer has already seen."""

    def __init__(self, min_beats_between_reuse: int = 5):
        self.min_gap = min_beats_between_reuse

    def track(self, beats: Sequence[Dict[str, Any]]) -> VisualMemoryReport:
        """Track asset exposure across beats.

        Parameters
        ----------
        beats : list of dict
            Story beats with selected assets.

        Returns
        -------
        VisualMemoryReport
        """
        if not beats:
            return VisualMemoryReport([], [], [], {"beat_count": 0})

        annotated_beats = []
        annotations = []
        issues = []
        
        seen_assets = {}  # asset_id -> list of beat indices

        for i, beat in enumerate(beats):
            beat_copy = dict(beat)
            
            # Asset is determined by asset metadata or image query.
            asset_meta = beat.get("asset_metadata") or {}
            asset_id = asset_meta.get("id") or beat.get("image_search_query") or f"unknown_{i}"

            history = seen_assets.get(asset_id, [])
            
            is_first = len(history) == 0
            reuse_count = len(history)
            gap = (i - history[-1]) if history else -1
            
            ann = MemoryAnnotation(
                beat_index=i,
                asset_id=asset_id,
                is_first_appearance=is_first,
                reuse_count=reuse_count,
                beats_since_last_seen=gap,
            )
            annotations.append(ann)
            
            seen_assets.setdefault(asset_id, []).append(i)
            
            # Detect Issues
            beat_type = str(beat.get("beat_type", "")).lower()
            if beat_type == "reveal" and not is_first:
                issues.append(MemoryIssue(
                    issue_type="spoiled_reveal",
                    beat_index=i,
                    description=f"Beat {i} is a reveal, but asset '{asset_id}' was already seen at beat {history[0]}.",
                ))
                
            if gap > 0 and gap < self.min_gap:
                issues.append(MemoryIssue(
                    issue_type="premature_reuse",
                    beat_index=i,
                    description=f"Asset '{asset_id}' reused too quickly (gap of {gap} beats, minimum is {self.min_gap}).",
                ))

            # Inject into beat
            beat_copy["visual_memory"] = {
                "is_first_appearance": is_first,
                "reuse_count": reuse_count,
                "beats_since_last_seen": gap,
            }
            annotated_beats.append(beat_copy)

        summary = {
            "beat_count": len(beats),
            "unique_assets": len(seen_assets),
            "total_reuses": sum(len(h) - 1 for h in seen_assets.values() if len(h) > 1),
            "issue_count": len(issues),
        }

        return VisualMemoryReport(annotated_beats, annotations, issues, summary)


def track_visual_memory(beats: Sequence[Dict[str, Any]], **kwargs) -> VisualMemoryReport:
    """Shortcut function to track visual memory."""
    return VisualMemoryTracker(**kwargs).track(beats)
