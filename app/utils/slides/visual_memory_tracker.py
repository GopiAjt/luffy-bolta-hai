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


# --- NEW LOGIC ADDED FOR V2 MEMORY TRACKING ---

@dataclass
class AssetMemoryRecord:
    asset_id: str
    first_seen: int
    last_seen: int
    times_used: int

class VisualMemoryTracker:
    """Tracks what the viewer has already seen."""

    def __init__(self, min_beats_between_reuse: int = 5):
        self.min_gap = min_beats_between_reuse
        self.memory: Dict[str, AssetMemoryRecord] = {}

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

    # --- NEW METHODS ---

    def register_asset(self, asset_id: str, current_time: int):
        """Register that an asset was used at a specific time (e.g. slide index)."""
        if asset_id not in self.memory:
            self.memory[asset_id] = AssetMemoryRecord(
                asset_id=asset_id,
                first_seen=current_time,
                last_seen=current_time,
                times_used=1
            )
        else:
            record = self.memory[asset_id]
            record.last_seen = current_time
            record.times_used += 1

    def asset_seen_recently(self, asset_id: str, current_time: int, threshold: int = 3) -> bool:
        """Return True if the asset was seen within the threshold time."""
        if asset_id not in self.memory:
            return False
        return (current_time - self.memory[asset_id].last_seen) <= threshold

    def reuse_score(self, asset_id: str, current_time: int) -> float:
        """
        Calculate a score (0.0 to 1.0) indicating how good it is to reuse this asset.
        Higher score = better candidate for reuse.
        """
        if asset_id not in self.memory:
            return 1.0  # New assets are perfect candidates
            
        record = self.memory[asset_id]
        
        # Base penalty for being used multiple times
        usage_penalty = 1.0 / (record.times_used + 1)
        
        # Time since last seen (longer = better to reuse)
        time_diff = max(0, current_time - record.last_seen)
        recency_bonus = min(time_diff / 5.0, 1.0)
        
        # Blend the two factors
        score = (usage_penalty * 0.4) + (recency_bonus * 0.6)
        return min(max(score, 0.0), 1.0)
        
    def is_asset_allowed(self, asset_id: str, current_time: int, is_callback: bool = False) -> bool:
        """
        Returns True if the asset is allowed to be used.
        Rule: Same asset not allowed within 3 beats unless it is explicitly a callback.
        """
        if is_callback:
            return True
        return not self.asset_seen_recently(asset_id, current_time, threshold=3)

    def asset_fatigue_score(self, asset_id: str, current_time: int) -> float:
        """
        Calculate fatigue score (0.0 to 1.0). Higher score means viewer is tired of it.
        """
        if asset_id not in self.memory:
            return 0.0
            
        record = self.memory[asset_id]
        
        # Base fatigue increases with usage, caps at 5 uses (fatigue = 1.0)
        base_fatigue = min(record.times_used / 5.0, 1.0)
        
        # Viewer "forgets" slightly if it hasn't been seen recently
        time_diff = max(0, current_time - record.last_seen)
        recovery = min(time_diff / 10.0, 0.5)  # Can recover up to 50% fatigue
        
        return max(0.0, base_fatigue - recovery)

    def should_allow_callback(self, asset_id: str, current_time: int, beat_type: str = "") -> bool:
        """
        Returns True if the asset is a valid candidate for a callback moment.
        Rules:
        - Must have been seen before.
        - Must have a significant gap since last seen (e.g., >= 8 beats).
        - OR beat_type is 'PAYOFF' or 'FINALE' and it has been seen before.
        """
        if asset_id not in self.memory:
            return False
            
        record = self.memory[asset_id]
        time_diff = current_time - record.last_seen
        
        if beat_type.upper() in ("PAYOFF", "FINALE"):
            return True
            
        if time_diff >= 8:
            return True
            
        return False


def track_visual_memory(beats: Sequence[Dict[str, Any]], **kwargs) -> VisualMemoryReport:
    """Shortcut function to track visual memory."""
    return VisualMemoryTracker(**kwargs).track(beats)
