import pytest
from app.utils.slides.visual_memory_tracker import AssetMemoryRecord, VisualMemoryTracker

def test_asset_memory_record_creation():
    """Verify dataclass creation."""
    record = AssetMemoryRecord(
        asset_id="img_123",
        first_seen=0,
        last_seen=5,
        times_used=2
    )
    assert record.asset_id == "img_123"
    assert record.first_seen == 0
    assert record.last_seen == 5
    assert record.times_used == 2

def test_tracker_register_asset():
    """Verify registering a new and existing asset."""
    tracker = VisualMemoryTracker()
    
    # Register new
    tracker.register_asset("img_1", current_time=0)
    assert "img_1" in tracker.memory
    assert tracker.memory["img_1"].times_used == 1
    assert tracker.memory["img_1"].first_seen == 0
    
    # Register existing
    tracker.register_asset("img_1", current_time=5)
    assert tracker.memory["img_1"].times_used == 2
    assert tracker.memory["img_1"].last_seen == 5
    assert tracker.memory["img_1"].first_seen == 0

def test_tracker_asset_seen_recently():
    """Verify recently seen logic."""
    tracker = VisualMemoryTracker()
    tracker.register_asset("img_1", current_time=10)
    
    # Seen recently (within threshold 3)
    assert tracker.asset_seen_recently("img_1", current_time=12, threshold=3) is True
    
    # Not seen recently (outside threshold)
    assert tracker.asset_seen_recently("img_1", current_time=15, threshold=3) is False
    
    # Never seen
    assert tracker.asset_seen_recently("unknown", current_time=10) is False

def test_tracker_reuse_score():
    """Verify reuse score logic penalizes over-use and rewards recency."""
    tracker = VisualMemoryTracker()
    
    # New asset gets perfect score
    assert tracker.reuse_score("new_img", 0) == 1.0
    
    # Register an asset
    tracker.register_asset("img_1", current_time=0)
    
    # Immediate reuse should have low score
    score_immediate = tracker.reuse_score("img_1", current_time=0)
    
    # Reuse later should have higher score due to recency bonus
    score_later = tracker.reuse_score("img_1", current_time=5)
    assert score_later > score_immediate
    
    # Register it again (over-use)
    tracker.register_asset("img_1", current_time=5)
    
    # Score right after second use should be lower than right after first use
    score_after_second = tracker.reuse_score("img_1", current_time=5)
    assert score_after_second < score_immediate

def test_tracker_is_asset_allowed():
    """Verify rules for allowing asset use based on 3-beat gap or callback."""
    tracker = VisualMemoryTracker()
    
    # New asset allowed
    assert tracker.is_asset_allowed("img_1", 0) is True
    
    # Register it
    tracker.register_asset("img_1", 0)
    
    # Not allowed within 3 beats
    assert tracker.is_asset_allowed("img_1", 1) is False
    assert tracker.is_asset_allowed("img_1", 2) is False
    assert tracker.is_asset_allowed("img_1", 3) is False
    
    # Allowed after 3 beats (i.e. at beat 4, gap is 4 > 3)
    assert tracker.is_asset_allowed("img_1", 4) is True
    
    # BUT allowed within 3 beats if is_callback is True
    assert tracker.is_asset_allowed("img_1", 2, is_callback=True) is True

def test_tracker_asset_fatigue_score():
    """Verify fatigue score calculation."""
    tracker = VisualMemoryTracker()
    
    # New asset has zero fatigue
    assert tracker.asset_fatigue_score("img_1", 0) == 0.0
    
    # 1 use -> base fatigue 0.2
    tracker.register_asset("img_1", 0)
    assert tracker.asset_fatigue_score("img_1", 0) == 0.2
    
    # Time recovery (at t=2, recovery=0.2, fatigue drops to 0.0)
    assert tracker.asset_fatigue_score("img_1", 2) == 0.0
    
    # Multiple uses increase fatigue
    tracker.register_asset("img_1", 5) # 2 uses (0.4)
    tracker.register_asset("img_1", 10) # 3 uses (0.6)
    tracker.register_asset("img_1", 15) # 4 uses (0.8)
    tracker.register_asset("img_1", 20) # 5 uses (1.0 max base)
    
    assert tracker.asset_fatigue_score("img_1", 20) == 1.0
    
    # Long time passes -> recovery maxes out at 0.5
    assert tracker.asset_fatigue_score("img_1", 30) == 0.5

def test_tracker_should_allow_callback():
    """Verify rules for allowing callback moments."""
    tracker = VisualMemoryTracker()
    
    # New asset cannot be a callback
    assert tracker.should_allow_callback("roger_img", 0) is False
    
    # Register it
    tracker.register_asset("roger_img", 0)
    
    # Short gap is not a callback
    assert tracker.should_allow_callback("roger_img", 4) is False
    
    # Long gap (>= 8) is a callback
    assert tracker.should_allow_callback("roger_img", 8) is True
    assert tracker.should_allow_callback("roger_img", 10) is True
    
    # Short gap IS a callback if beat_type is PAYOFF or FINALE
    assert tracker.should_allow_callback("roger_img", 4, beat_type="PAYOFF") is True
    assert tracker.should_allow_callback("roger_img", 2, beat_type="finale") is True
