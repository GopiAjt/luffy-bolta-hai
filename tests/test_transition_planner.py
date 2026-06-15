import pytest
from app.utils.slides.transition_planner import TransitionType, TransitionPlan, TransitionPlanner

def test_transition_type_enum():
    """Verify enum serialization and members."""
    # Test valid members
    assert TransitionType.crossfade.value == "crossfade"
    assert TransitionType.glitch_cut.value == "glitch_cut"
    
    # Test string serialization (since it inherits from str, enum.Enum)
    assert str(TransitionType.zoom_dissolve) == "TransitionType.zoom_dissolve"
    assert TransitionType.zoom_dissolve == "zoom_dissolve"

def test_transition_plan_dataclass():
    """Verify dataclass construction."""
    plan = TransitionPlan(
        transition_type=TransitionType.whip_pan_left,
        duration=0.5,
        intensity=1.2,
        reason="Action beat requires high energy transition"
    )
    
    assert plan.transition_type == TransitionType.whip_pan_left
    assert plan.duration == 0.5
    assert plan.intensity == 1.2
    assert plan.reason == "Action beat requires high energy transition"

def test_transition_planner_initialization():
    """Verify planner initialization."""
    planner = TransitionPlanner()
    assert planner is not None
    assert isinstance(planner, TransitionPlanner)

def test_plan_transition_rules():
    """Verify transition rules mapping."""
    planner = TransitionPlanner()
    
    # Test HOOK
    plan = planner.plan_transition({}, {"beat_type": "hook"})
    assert plan.transition_type == TransitionType.zoom_dissolve
    
    # Test REVEAL
    plan = planner.plan_transition({}, {"beat_type": "reveal"})
    assert plan.transition_type == TransitionType.iris_wipe
    
    # Test TWIST
    plan = planner.plan_transition({}, {"beat_type": "twist"})
    assert plan.transition_type == TransitionType.glitch_cut
    
    # Test EMOTIONAL
    plan = planner.plan_transition({}, {"beat_type": "emotional"})
    assert plan.transition_type == TransitionType.fade_eased
    
    # Test PAYOFF
    plan = planner.plan_transition({}, {"beat_type": "payoff"})
    assert plan.transition_type == TransitionType.zoom_dissolve
    
    # Test Default
    plan = planner.plan_transition({}, {"beat_type": "unknown"})
    assert plan.transition_type == TransitionType.crossfade
    
    # Test missing beat type
    plan = planner.plan_transition({}, {})
    assert plan.transition_type == TransitionType.crossfade

def test_emotion_score_scaling():
    """Verify duration and intensity scale inversely with emotion_score."""
    planner = TransitionPlanner()
    
    # Range 0-20
    plan = planner.plan_transition({}, {"beat_type": "unknown", "emotion_score": 10})
    assert plan.intensity == 0.3
    assert plan.duration == 1.0
    
    # Range 20-50
    plan = planner.plan_transition({}, {"beat_type": "unknown", "emotion_score": 40})
    assert plan.intensity == 0.5
    assert plan.duration == 0.8
    
    # Range 50-80
    plan = planner.plan_transition({}, {"beat_type": "unknown", "emotion_score": 60})
    assert plan.intensity == 0.7
    assert plan.duration == 0.5
    
    # Range 80-100
    plan = planner.plan_transition({}, {"beat_type": "unknown", "emotion_score": 90})
    assert plan.intensity == 1.0
    assert plan.duration == 0.3
    
    # Missing emotion_score defaults to 0
    plan = planner.plan_transition({}, {"beat_type": "unknown"})
    assert plan.intensity == 0.3
    assert plan.duration == 1.0

def test_transition_history_tracker():
    """Verify transitions don't repeat more than twice and fallback is used."""
    planner = TransitionPlanner()
    
    # 1st HOOK: zoom_dissolve
    plan1 = planner.plan_transition({}, {"beat_type": "hook"})
    assert plan1.transition_type == TransitionType.zoom_dissolve
    
    # 2nd HOOK: zoom_dissolve (Allowed to repeat once)
    plan2 = planner.plan_transition({}, {"beat_type": "hook"})
    assert plan2.transition_type == TransitionType.zoom_dissolve
    
    # 3rd HOOK: Should trigger violation and fallback to next choice (whip_pan_right)
    plan3 = planner.plan_transition({}, {"beat_type": "hook"})
    assert plan3.transition_type == TransitionType.whip_pan_right
    
    # 4th HOOK: fade_eased is different from zoom_dissolve, so it's not a violation
    # But wait, the 4th HOOK will ask for zoom_dissolve again.
    # The tracker history is now [zoom_dissolve, zoom_dissolve, fade_eased].
    # So asking for zoom_dissolve is fine, no violation.
    plan4 = planner.plan_transition({}, {"beat_type": "hook"})
    assert plan4.transition_type == TransitionType.zoom_dissolve
