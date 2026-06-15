from typing import List, Dict, Any

class LegacyAdapter:
    """Adapts the rich new StoryboardBeat into the legacy JSON format."""
    
    def adapt(self, beats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        legacy_slides = []
        for beat in beats:
            # Extract standard fields
            start_time = beat.get("start_time", 0.0)
            end_time = beat.get("end_time", 0.0)
            summary = beat.get("summary", "")
            image_search_query = beat.get("image_search_query", "")
            
            # Extract motion_preset (from 'motion_preset' or 'motion.style')
            motion_preset = beat.get("motion_preset")
            if not motion_preset and "motion" in beat:
                motion_preset = beat["motion"].get("style")
            if not motion_preset:
                motion_preset = "none"
                
            # Extract transition_in (from 'transition_in' or 'transition_type')
            transition_in = beat.get("transition_in")
            if not transition_in:
                transition_in = beat.get("transition_type", "crossfade")
                
            slide = {
                "start_time": start_time,
                "end_time": end_time,
                "summary": summary,
                "image_search_query": image_search_query,
                "motion_preset": motion_preset,
                "transition_in": transition_in
            }
            legacy_slides.append(slide)
            
        return legacy_slides

def adapt_to_legacy(beats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return LegacyAdapter().adapt(beats)
