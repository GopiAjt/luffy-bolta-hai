import enum
import random
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

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
        self.history: List[TransitionType] = []
        self.counts: Dict[TransitionType, int] = {}
        
    def add_transition(self, transition: TransitionType):
        self.history.append(transition)
        self.counts[transition] = self.counts.get(transition, 0) + 1
        if len(self.history) > 10:
            self.history.pop(0)
            
    def is_violation(self, transition: TransitionType) -> bool:
        # Never repeat the same transition more than twice consecutively
        if len(self.history) >= 2:
            if self.history[-1] == transition and self.history[-2] == transition:
                return True
        return False

class TransitionPlanner:
    def __init__(self, visual_style: str = "clean_pro"):
        self.visual_style = visual_style.lower()
        self.tracker = TransitionHistoryTracker()
        
    def _get_semantic_candidates(self, beat_type: str, emotion: str) -> Dict[TransitionType, int]:
        """Map narrative context to transition choices with base semantic scores."""
        candidates = {}
        
        # Base mappings based on narrative beat
        if beat_type == "REVEAL":
            candidates[TransitionType.zoom_dissolve] = 50
            candidates[TransitionType.iris_wipe] = 50
            candidates[TransitionType.radial_wipe] = 40
        elif beat_type in ("SETUP", "MYSTERY"):
            candidates[TransitionType.iris_wipe] = 50
            candidates[TransitionType.radial_wipe] = 50
            candidates[TransitionType.crossfade] = 30
        elif beat_type == "TWIST" or emotion in ("shock", "fear", "surprise"):
            candidates[TransitionType.glitch_cut] = 60
            candidates[TransitionType.zoom_dissolve] = 40
            candidates[TransitionType.whip_pan_right] = 30
        elif beat_type == "ESCALATION" or emotion in ("excitement", "rage", "anger"):
            candidates[TransitionType.whip_pan_left] = 50
            candidates[TransitionType.whip_pan_right] = 50
            candidates[TransitionType.motion_slide_left] = 40
            candidates[TransitionType.motion_slide_right] = 40
            candidates[TransitionType.cube_rotation] = 30
        elif beat_type == "ACTION":
            candidates[TransitionType.whip_pan_left] = 50
            candidates[TransitionType.whip_pan_right] = 50
            candidates[TransitionType.motion_slide_left] = 40
            candidates[TransitionType.motion_slide_right] = 40
            candidates[TransitionType.glitch_cut] = 35
        elif beat_type == "EMOTIONAL" or emotion in ("sadness", "joy", "hope"):
            candidates[TransitionType.crossfade] = 50
            candidates[TransitionType.fade_eased] = 40
        elif beat_type == "PAYOFF":
            candidates[TransitionType.zoom_dissolve] = 50
            candidates[TransitionType.iris_wipe] = 40
            candidates[TransitionType.fade_eased] = 30
        elif beat_type == "RESOLUTION":
            candidates[TransitionType.fade_eased] = 55
            candidates[TransitionType.crossfade] = 45
        else: # Default/Evidence/Hook
            candidates[TransitionType.crossfade] = 40
            candidates[TransitionType.fade_eased] = 40
            candidates[TransitionType.motion_slide_right] = 20
            
        return candidates

    def _get_style_weights(self) -> Dict[TransitionType, int]:
        """Return bonus scores based on the visual style's preferred transitions."""
        weights = {}
        
        if self.visual_style == "action" or self.visual_style == "yonko_hype":
            weights[TransitionType.whip_pan_left] = 30
            weights[TransitionType.whip_pan_right] = 30
            weights[TransitionType.glitch_cut] = 30
        elif self.visual_style == "dark_lore" or self.visual_style == "void_century":
            weights[TransitionType.iris_wipe] = 30
            weights[TransitionType.radial_wipe] = 30
            weights[TransitionType.zoom_dissolve] = 20
        elif self.visual_style == "emotional":
            weights[TransitionType.crossfade] = 30
            weights[TransitionType.fade_eased] = 30
        elif self.visual_style == "clean_pro":
            # Clean pro tolerates all but favors clean dissolves and fades
            weights[TransitionType.fade_eased] = 20
            weights[TransitionType.crossfade] = 20
            weights[TransitionType.zoom_dissolve] = 20
            weights[TransitionType.motion_slide_left] = 10
            weights[TransitionType.motion_slide_right] = 10
            
        return weights

    def _calculate_diversity_score(self, transition: TransitionType) -> int:
        """Reward transitions that haven't been used recently."""
        score = 30
        # Penalty for being used at all
        count = self.tracker.counts.get(transition, 0)
        score -= min(count * 5, 20)
        
        # Heavy penalty for being used immediately prior
        if self.tracker.history and self.tracker.history[-1] == transition:
            score -= 15
            
        return max(0, score)

    def _get_style_modifiers(self) -> Tuple[float, float]:
        """Return (intensity_mod, duration_mod) based on style."""
        if self.visual_style == "action" or self.visual_style == "yonko_hype":
            return 1.5, 0.7  # Stronger motion blur, faster cuts
        elif self.visual_style == "clean_pro":
            return 0.7, 1.2  # Lower blur, shorter/softer motions, longer fades
        elif self.visual_style == "emotional":
            return 0.8, 1.3  # Soft, lingering transitions
        return 1.0, 1.0

    def plan_transition(self, current_beat: dict, next_beat: dict, total_beats: int = 10) -> TransitionPlan:
        beat_type = next_beat.get("beat_type", "").upper() if next_beat else ""
        emotion = str(next_beat.get("emotion_state", {}).get("emotion", "neutral")).lower() if next_beat else "neutral"
        pacing_intent = (next_beat.get("pacing_intent") or "standard").strip().lower() if next_beat else "standard"

        # 1. Get semantic base candidates
        candidates = self._get_semantic_candidates(beat_type, emotion)
        
        # 2. Add style weights
        style_weights = self._get_style_weights()
        for t, weight in style_weights.items():
            if t in candidates:
                candidates[t] += weight
            else:
                # Allow style to introduce transitions even if not semantically mapped, but at lower priority
                candidates[t] = weight

        # 2b. Pacing intent bonuses
        if pacing_intent == "rapid_montage":
            for fast_t in (TransitionType.whip_pan_left, TransitionType.whip_pan_right, 
                           TransitionType.motion_slide_left, TransitionType.motion_slide_right,
                           TransitionType.glitch_cut):
                candidates[fast_t] = candidates.get(fast_t, 0) + 25
        elif pacing_intent == "hold_frame":
            for slow_t in (TransitionType.fade_eased, TransitionType.crossfade):
                candidates[slow_t] = candidates.get(slow_t, 0) + 30
        elif pacing_intent == "dramatic_pause":
            candidates[TransitionType.fade_eased] = candidates.get(TransitionType.fade_eased, 0) + 35
                
        # 3. Add diversity scores and filter violations
        scored_candidates = []
        for t, base_score in candidates.items():
            if self.tracker.is_violation(t):
                continue
            diversity = self._calculate_diversity_score(t)
            total_score = base_score + diversity
            scored_candidates.append((total_score, t))
            
        # 4. Select the best transition
        if not scored_candidates:
            # Absolute fallback
            selected_type = TransitionType.crossfade
            reason = "Fallback (all choices violated rules)"
        else:
            # Sort by highest score
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Add some slight randomness among top contenders to avoid determinism if scores are close
            top_score = scored_candidates[0][0]
            top_contenders = [t for score, t in scored_candidates if score >= top_score - 15]
            selected_type = random.choice(top_contenders)
            
            reason = f"Semantic match for {beat_type}/{emotion}/{pacing_intent} under {self.visual_style} style"

        # 5. Calculate base intensity/duration based on emotion
        emotion_score = next_beat.get("emotion_score", 50) if next_beat else 50
        base_intensity = 0.5 + (emotion_score / 200.0) # 0.5 to 1.0
        base_duration = 1.0 - (emotion_score / 200.0)  # 1.0 to 0.5
        
        # 6. Apply style modifiers
        int_mod, dur_mod = self._get_style_modifiers()
        final_intensity = min(1.0, max(0.1, base_intensity * int_mod))
        final_duration = min(2.0, max(0.2, base_duration * dur_mod))

        # Record in history
        self.tracker.add_transition(selected_type)

        return TransitionPlan(
            transition_type=selected_type,
            duration=round(final_duration, 2),
            intensity=round(final_intensity, 2),
            reason=reason
        )
