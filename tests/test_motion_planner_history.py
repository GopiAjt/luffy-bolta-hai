from app.utils.slides.motion_planner import MotionPlanner

def test_motion_planner_history_tracker_consecutive():
    """Verify reveal_zoom does not repeat more than twice consecutively."""
    planner = MotionPlanner()
    
    # 1st time
    style1, _ = planner._resolve_style("neutral", "", "reveal", "", total_beats=10)
    assert style1 == "impact_zoom" or style1 == "slow_push" or style1 == "reveal_zoom"  # Usually map resolves to something else if overrides are hit.
    
    # Actually, let's force the tracker to think reveal_zoom was used twice
    planner.tracker.add_motion("reveal_zoom")
    planner.tracker.add_motion("reveal_zoom")
    
    assert planner.tracker.is_violation("reveal_zoom", total_beats=10) is True

def test_motion_planner_history_tracker_frequency():
    """Verify reveal_zoom cannot exceed 30% of total slides."""
    planner = MotionPlanner()
    
    # total_beats = 10, max 30% = 3. Wait, int(10*0.30) = 3, max(2, 3) = 3. 
    # So 3 is the violation threshold (meaning at count=3, it returns True).
    
    planner.tracker.add_motion("reveal_zoom")
    planner.tracker.add_motion("reveal_zoom")
    
    # Wait, consecutive rule will make is_violation("reveal_zoom") True here anyway.
    # Let's break consecutive by adding a different one
    planner.tracker.add_motion("slow_push")
    
    # Current counts: reveal_zoom: 2, slow_push: 1.
    assert planner.tracker.is_violation("reveal_zoom", total_beats=10) is False
    
    planner.tracker.add_motion("reveal_zoom")
    # Current counts: reveal_zoom: 3. This hits the >= 3 threshold.
    assert planner.tracker.is_violation("reveal_zoom", total_beats=10) is True
