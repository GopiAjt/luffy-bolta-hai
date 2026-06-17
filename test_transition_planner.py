from app.utils.slides.transition_planner import TransitionPlanner, TransitionType

def test_semantic_planner():
    # Test Clean Pro
    planner_clean = TransitionPlanner(visual_style="clean_pro")
    
    beat1 = {"beat_type": "HOOK", "emotion_state": {"emotion": "excitement"}}
    beat2 = {"beat_type": "REVEAL", "emotion_state": {"emotion": "surprise"}}
    beat3 = {"beat_type": "TWIST", "emotion_state": {"emotion": "shock"}}
    beat4 = {"beat_type": "EMOTIONAL", "emotion_state": {"emotion": "sadness"}}
    beat5 = {"beat_type": "ACTION", "emotion_state": {"emotion": "rage"}}
    
    beats = [beat1, beat2, beat3, beat4, beat5]
    
    print("=== CLEAN PRO ===")
    prev = None
    for b in beats:
        plan = planner_clean.plan_transition(prev, b)
        print(f"{b['beat_type']:10} -> {plan.transition_type:20} (Intensity: {plan.intensity}, Duration: {plan.duration}) Reason: {plan.reason}")
        prev = b
        
    # Test Action
    planner_action = TransitionPlanner(visual_style="action")
    print("\n=== ACTION ===")
    prev = None
    for b in beats:
        plan = planner_action.plan_transition(prev, b)
        print(f"{b['beat_type']:10} -> {plan.transition_type:20} (Intensity: {plan.intensity}, Duration: {plan.duration}) Reason: {plan.reason}")
        prev = b

if __name__ == "__main__":
    test_semantic_planner()
