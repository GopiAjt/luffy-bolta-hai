from typing import List, Dict, Any

class LegacyAdapter:
    """Adapts the rich new StoryboardBeat into the legacy JSON format.
    
    The adapter preserves all fields from the modern pipeline while ensuring
    the core legacy fields are always present with sensible defaults.
    """
    
    # Fields that must always be present with defaults
    _REQUIRED_FIELDS = {
        "start_time": 0.0,
        "end_time": 0.0,
        "summary": "",
        "image_search_query": "",
        "motion_preset": "none",
        "transition_in": "crossfade",
    }
    
    def adapt(self, beats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        legacy_slides = []
        for beat in beats:
            # Start with a full copy of all modern fields
            slide = dict(beat)
            
            # Ensure required legacy fields exist
            for field, default in self._REQUIRED_FIELDS.items():
                if field not in slide or not slide[field]:
                    slide[field] = default
            
            # Resolve motion_preset from nested motion dict if needed
            if not slide.get("motion_preset") or slide["motion_preset"] == "none":
                if "motion" in beat and isinstance(beat["motion"], dict):
                    slide["motion_preset"] = beat["motion"].get("style", "none")
                    
            # Resolve transition_in from transition_type if needed
            if not slide.get("transition_in") or slide["transition_in"] == "crossfade":
                if beat.get("transition_type"):
                    slide["transition_in"] = beat["transition_type"]
                
            legacy_slides.append(slide)
            
        return legacy_slides

def adapt_to_legacy(beats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return LegacyAdapter().adapt(beats)
