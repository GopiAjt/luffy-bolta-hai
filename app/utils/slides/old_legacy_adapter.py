"""Legacy Adapter — bridges new storyboard pipeline to legacy renderer.

Converts the rich StoryboardBeat format back into the flat slide JSON
expected by `image_slides.py` and `generate_slideshow.py`.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)


class LegacyAdapter:
    """Adapts new StoryboardBeats to legacy slides."""

    def __init__(self):
        # Keys to strictly preserve for legacy renderer
        self.legacy_keys = [
            "start_time",
            "end_time",
            "duration",
            "subtitle_text",
            "summary",
            "image_search_query",
            "search_tags",
            "image_url",
            "fallback_image_url",
        ]

    def adapt(self, beats: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert a sequence of beats back to legacy slides.

        Parameters
        ----------
        beats : list of dict
            Fully annotated StoryboardBeats.

        Returns
        -------
        list of dict
            Legacy slides ready for `generate_slideshow.py`.
        """
        if not beats:
            return []

        slides = []
        for beat in beats:
            slide = copy.deepcopy(beat)
            
            # Ensure "search_tags" exists if asset_metadata has it
            if "search_tags" not in slide:
                asset_meta = slide.get("asset_metadata", {})
                if "search_tags" in asset_meta:
                    slide["search_tags"] = asset_meta["search_tags"]
            
            # Legacy renderer might expect 'duration' if not present
            if "duration" not in slide and "start_time" in slide and "end_time" in slide:
                try:
                    # Simple conversion if time is in seconds
                    start = float(slide["start_time"])
                    end = float(slide["end_time"])
                    slide["duration"] = max(0.0, end - start)
                except ValueError:
                    pass

            slides.append(slide)

        return slides


def adapt_to_legacy(beats: Sequence[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """Shortcut function to adapt beats to legacy slides."""
    return LegacyAdapter(**kwargs).adapt(beats)
