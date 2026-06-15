import enum
from dataclasses import dataclass
from typing import Optional

class TransitionType(str, enum.Enum):
    crossfade = "crossfade"
    fade_eased = "fade_eased"
    zoom_dissolve = "zoom_dissolve"
    whip_pan_left = "whip_pan_left"
    whip_pan_right = "whip_pan_right"
    motion_slide_left = "motion_slide_left"
    motion_slide_right = "motion_slide_right"
    iris_wipe = "iris_wipe"
    radial_wipe = "radial_wipe"
    glitch_cut = "glitch_cut"
    cube_rotation = "cube_rotation"

@dataclass
class TransitionPlan:
    transition_type: TransitionType
    duration: float
    intensity: float
    reason: str

class TransitionHistoryTracker:
    def __init__(self):
        self.history = []
        self.counts = {}
        
    def add_transition(self, transition: TransitionType):
        self.history.append(transition)
        self.counts[transition] = self.counts.get(transition, 0) + 1
        if len(self.history) > 10:
            self.history.pop(0)
            
    def is_violation(self, transition: TransitionType, total_beats: int = 10) -> bool:
        if len(self.history) >= 2:
            if self.history[-1] == transition and self.history[-2] == transition:
                return True
                
        # Frequency limit: fade_eased max 30%
        if transition == TransitionType.fade_eased:
            if self.counts.get(transition, 0) >= max(2, int(total_beats * 0.30)):
                return True
                
        return False

class TransitionPlanner:
    def __init__(self):
        self.tracker = TransitionHistoryTracker()
        
    def get_fallback(self, transition_type: TransitionType) -> TransitionType:
        """Return a compatible fallback if a transition is used too many times."""
        fallbacks = {
            TransitionType.zoom_dissolve: TransitionType.fade_eased,
            TransitionType.iris_wipe: TransitionType.radial_wipe,
            TransitionType.glitch_cut: TransitionType.whip_pan_right,
            TransitionType.fade_eased: TransitionType.crossfade,
            TransitionType.crossfade: TransitionType.zoom_dissolve,
            TransitionType.cube_rotation: TransitionType.zoom_dissolve,
            TransitionType.whip_pan_right: TransitionType.whip_pan_left,
            TransitionType.whip_pan_left: TransitionType.whip_pan_right,
            TransitionType.motion_slide_left: TransitionType.motion_slide_right,
            TransitionType.motion_slide_right: TransitionType.motion_slide_left,
        }
        return fallbacks.get(transition_type, TransitionType.crossfade)

    def plan_transition(self, current_beat: dict, next_beat: dict, total_beats: int = 10) -> TransitionPlan:
        """Plan the transition between two beats."""
        beat_type = next_beat.get("beat_type", "").upper() if next_beat else ""
        emotion = str(next_beat.get("emotion_state", {}).get("emotion", "neutral")).lower() if next_beat else "neutral"
        story_phase = str(next_beat.get("story_phase", "middle")).lower() if next_beat else "middle"

        choices = []
        reason = ""

        if beat_type == "HOOK" or story_phase == "beginning":
            choices = [TransitionType.zoom_dissolve, TransitionType.whip_pan_right]
            reason = f"{beat_type or 'Hook'} beat in {story_phase} requires high impact"
        elif beat_type == "REVEAL":
            choices = [TransitionType.iris_wipe, TransitionType.radial_wipe, TransitionType.cube_rotation]
            reason = "Reveal beat uses focus or rotation transitions"
        elif beat_type == "TWIST" or emotion in ["shock", "fear", "surprise", "tension"]:
            choices = [TransitionType.glitch_cut, TransitionType.zoom_dissolve]
            reason = "Twist/Shocking beat requires jarring cut"
        elif beat_type == "ACTION" or emotion in ["excitement", "rage", "anger"]:
            choices = [TransitionType.whip_pan_right, TransitionType.whip_pan_left, TransitionType.motion_slide_left, TransitionType.motion_slide_right]
            reason = "Action beat uses fast directional motion"
        elif beat_type in ["EMOTIONAL", "EVIDENCE", "SETUP"] or emotion in ["sadness", "joy", "calm", "hope"]:
            choices = [TransitionType.fade_eased, TransitionType.crossfade]
            reason = "Emotional/Evidence beat uses smooth fade"
        elif beat_type == "PAYOFF" or story_phase == "end":
            choices = [TransitionType.zoom_dissolve, TransitionType.iris_wipe]
            reason = "Payoff beat uses cinematic closure"
        else:
            choices = [TransitionType.crossfade]
            reason = f"Default crossfade for {beat_type or 'unknown'} beat"
        # Scale intensity and duration based on emotion_score
        emotion_score = next_beat.get("emotion_score", 0) if next_beat else 0
        if emotion_score <= 20:
            intensity = 0.3
            duration = 1.0
        elif emotion_score <= 50:
            intensity = 0.5
            duration = 0.8
        elif emotion_score <= 80:
            intensity = 0.7
            duration = 0.5
        else:
            intensity = 1.0
            duration = 0.3
            
        # Check history for consecutive repetitions
        transition_type = None
        for choice in choices:
            if not self.tracker.is_violation(choice, total_beats):
                transition_type = choice
                break
                
        if transition_type is None:
            # If all choices violate history, use fallback on the first choice
            transition_type = self.get_fallback(choices[0])
            reason += " (Fallback due to repetition limit)"
            
        self.tracker.add_transition(transition_type)
            
        return TransitionPlan(
            transition_type=transition_type,
            duration=duration,
            intensity=intensity,
            reason=reason
        )
