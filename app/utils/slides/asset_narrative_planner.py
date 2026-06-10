"""Asset Narrative Planner — plans visual storytelling.

Decides which characters and locations to feature on each beat
to maintain narrative coherence and visual storytelling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)


@dataclass
class AssetAssignment:
    """An asset assigned to a beat."""
    beat_index: int
    entity: str
    entity_type: str           # "character", "location", "object"
    reasoning: str


@dataclass
class AssetNarrativeReport:
    """Output of the asset narrative planner."""
    annotated_beats: List[Dict[str, Any]]
    assignments: List[AssetAssignment]
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotated_beats": self.annotated_beats,
            "assignments": [
                {
                    "beat_index": a.beat_index,
                    "entity": a.entity,
                    "entity_type": a.entity_type,
                    "reasoning": a.reasoning,
                }
                for a in self.assignments
            ],
            "summary": self.summary,
        }


class AssetNarrativePlanner:
    """Plans narrative asset usage across the storyboard."""

    def __init__(self, fallback_entity: str = "protagonist"):
        self.fallback_entity = fallback_entity

    def plan(
        self,
        beats: Sequence[Dict[str, Any]],
        available_assets: Dict[str, Any] = None,
    ) -> AssetNarrativeReport:
        """Assign primary narrative assets to beats.

        Parameters
        ----------
        beats : list of dict
            Story beats.
        available_assets : dict, optional
            Pool of available characters, locations, etc.

        Returns
        -------
        AssetNarrativeReport
        """
        if not beats:
            return AssetNarrativeReport([], [], {"beat_count": 0})

        annotated_beats = []
        assignments = []
        
        last_entity = None
        last_type = None

        for i, beat in enumerate(beats):
            beat_copy = dict(beat)
            
            # Extract basic context
            text = (beat.get("subtitle_text", "") + " " + beat.get("summary", "")).lower()
            role = beat.get("visual_role", "character").lower()
            entities = beat.get("context_entities", [])
            
            # Try to pick an entity
            if entities:
                entity = entities[0]
                entity_type = "character" if role == "character" else "object"
                reasoning = f"Derived from context_entities: {entity}"
            elif role == "location":
                entity = "Current Arc/Location"
                entity_type = "location"
                reasoning = "Visual role is location"
            else:
                # Fallback to previous or default
                entity = last_entity or self.fallback_entity
                entity_type = last_type or "character"
                reasoning = f"Fallback to {entity} (carryover or default)"

            ann = AssetAssignment(
                beat_index=i,
                entity=entity,
                entity_type=entity_type,
                reasoning=reasoning,
            )
            assignments.append(ann)
            
            last_entity = entity
            last_type = entity_type

            # Inject into beat
            beat_copy["narrative_asset"] = {
                "entity": entity,
                "entity_type": entity_type,
                "reasoning": reasoning,
            }
            annotated_beats.append(beat_copy)

        summary = {
            "beat_count": len(beats),
            "unique_entities": len(set(a.entity for a in assignments)),
            "character_beats": sum(1 for a in assignments if a.entity_type == "character"),
            "location_beats": sum(1 for a in assignments if a.entity_type == "location"),
        }

        return AssetNarrativeReport(annotated_beats, assignments, summary)


def plan_asset_narrative(
    beats: Sequence[Dict[str, Any]],
    available_assets: Dict[str, Any] = None,
    **kwargs,
) -> AssetNarrativeReport:
    """Shortcut function to plan asset narrative."""
    return AssetNarrativePlanner(**kwargs).plan(beats, available_assets)
