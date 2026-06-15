import pytest
from app.utils.slides.legacy_adapter import LegacyAdapter, adapt_to_legacy

def test_legacy_adapter_extracts_required_fields():
    adapter = LegacyAdapter()
    
    beats = [
        {
            "start_time": 1.5,
            "end_time": 4.5,
            "summary": "Luffy punches",
            "image_search_query": "Luffy gear 5",
            "motion_preset": "pan_left",
            "transition_in": "zoom_dissolve",
            "ignored_field": "This should not be in the output"
        }
    ]
    
    slides = adapter.adapt(beats)
    assert len(slides) == 1
    slide = slides[0]
    
    assert slide["start_time"] == 1.5
    assert slide["end_time"] == 4.5
    assert slide["summary"] == "Luffy punches"
    assert slide["image_search_query"] == "Luffy gear 5"
    assert slide["motion_preset"] == "pan_left"
    assert slide["transition_in"] == "zoom_dissolve"
    assert "ignored_field" not in slide

def test_legacy_adapter_handles_fallback_fields():
    adapter = LegacyAdapter()
    
    beats = [
        {
            # Missing start/end time, missing transition_in
            "summary": "Fallback test",
            "image_search_query": "Test",
            "motion": {"style": "slow_push"},
            "transition_type": "fade_eased"
        }
    ]
    
    slides = adapt_to_legacy(beats)
    assert len(slides) == 1
    slide = slides[0]
    
    # Defaults
    assert slide["start_time"] == 0.0
    assert slide["end_time"] == 0.0
    
    # Fallback from motion.style
    assert slide["motion_preset"] == "slow_push"
    
    # Fallback from transition_type
    assert slide["transition_in"] == "fade_eased"
