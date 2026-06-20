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
    
    # Test HOOK - now falls into Default (crossfade/fade_eased)
    plan = TransitionPlanner().plan_transition({}, {"beat_type": "hook"})
    assert plan.transition_type in (TransitionType.crossfade, TransitionType.fade_eased)
    
    # Test REVEAL
    plan = TransitionPlanner().plan_transition({}, {"beat_type": "reveal"})
    assert plan.transition_type in (TransitionType.iris_wipe, TransitionType.zoom_dissolve)
    
    # Test TWIST
    plan = TransitionPlanner().plan_transition({}, {"beat_type": "twist"})
    assert plan.transition_type == TransitionType.glitch_cut
    
    # Test EMOTIONAL
    plan = TransitionPlanner().plan_transition({}, {"beat_type": "emotional"})
    assert plan.transition_type in (TransitionType.fade_eased, TransitionType.crossfade)
    
    # Test PAYOFF
    plan = TransitionPlanner().plan_transition({}, {"beat_type": "payoff"})
    assert plan.transition_type in (TransitionType.zoom_dissolve, TransitionType.iris_wipe)
    
    # Test Default
    plan = TransitionPlanner().plan_transition({}, {"beat_type": "unknown"})
    assert plan.transition_type in (TransitionType.crossfade, TransitionType.fade_eased)
    
    # Test missing beat type
    plan = TransitionPlanner().plan_transition({}, {})
    assert plan.transition_type in (TransitionType.crossfade, TransitionType.fade_eased)

def test_emotion_score_scaling():
    """Verify duration and intensity scale based on the new formula and clean_pro modifiers."""
    planner = TransitionPlanner(visual_style="clean_pro")
    
    # Formula: base_intensity = 0.5 + score/200, base_duration = 1.0 - score/200
    # Modifiers for clean_pro: int_mod = 0.7, dur_mod = 1.2
    
    # Score 10: base_int=0.55 (final=0.385->0.39), base_dur=0.95 (final=1.14)
    plan = planner.plan_transition({}, {"beat_type": "unknown", "emotion_score": 10})
    assert plan.intensity == 0.39
    assert plan.duration == 1.14
    
    # Score 40: base_int=0.7 (final=0.49), base_dur=0.8 (final=0.96)
    plan = planner.plan_transition({}, {"beat_type": "unknown", "emotion_score": 40})
    assert plan.intensity == 0.49
    assert plan.duration == 0.96
    
    # Score 60: base_int=0.8 (final=0.56), base_dur=0.7 (final=0.84)
    plan = planner.plan_transition({}, {"beat_type": "unknown", "emotion_score": 60})
    assert plan.intensity == 0.56
    assert plan.duration == 0.84
    
    # Score 90: base_int=0.95 (final=0.665->0.67), base_dur=0.55 (final=0.66)
    plan = planner.plan_transition({}, {"beat_type": "unknown", "emotion_score": 90})
    assert plan.intensity == 0.66
    assert plan.duration == 0.66
    
    # Missing defaults to 50: base_int=0.75 (final=0.525->0.52), base_dur=0.75 (final=0.9)
    plan = planner.plan_transition({}, {"beat_type": "unknown"})
    assert plan.intensity == 0.52
    assert plan.duration == 0.9

def test_transition_history_tracker():
    """Verify transitions don't repeat too often."""
    planner = TransitionPlanner(visual_style="action")
    
    # 1st ACTION beat
    plan1 = planner.plan_transition({}, {"beat_type": "action"})
    first_choice = plan1.transition_type
    
    # Repeated ACTION beats should force the planner to pick different transitions
    # due to diversity penalties.
    seen = {first_choice}
    for _ in range(5):
        plan = planner.plan_transition({}, {"beat_type": "action"})
        seen.add(plan.transition_type)
        
    assert len(seen) > 2  # Should have picked at least a few different ones
