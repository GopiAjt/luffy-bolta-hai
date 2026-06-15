import enum
from dataclasses import dataclass, field
from typing import List

class Severity(str, enum.Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

@dataclass
class QualityIssue:
    severity: Severity
    component: str
    description: str

@dataclass
class QualityReport:
    passed: bool
    issues: List[QualityIssue] = field(default_factory=list)

class QualityGate:
    """Evaluates the quality of a generated slide or video to determine if it meets standards."""
    
    def __init__(self):
        self.issues: List[QualityIssue] = []

    def add_issue(self, severity: Severity, component: str, description: str):
        """Register a new quality issue."""
        self.issues.append(QualityIssue(
            severity=severity,
            component=component,
            description=description
        ))

    def evaluate(self) -> QualityReport:
        """
        Evaluate all collected issues.
        Returns a QualityReport that passes ONLY if there are no ERROR severity issues.
        """
        passed = True
        for issue in self.issues:
            if issue.severity == Severity.ERROR:
                passed = False
                break
                
        # Return a copy of the issues list to avoid external modification
        return QualityReport(passed=passed, issues=list(self.issues))

    def analyze_storyboard(self, beats: List[dict]):
        """
        Analyze a sequence of beats for common quality issues like consecutive repetitions.
        """
        low_energy_streak = 0
        
        # Check individual beat properties
        for i, beat in enumerate(beats):
            beat_type = str(beat.get("beat_type", "")).upper()
            duration = beat.get("duration")
            if duration is None:
                start = beat.get("start_time", 0)
                end = beat.get("end_time", 0)
                duration = end - start
                
            emotion_score = beat.get("emotion_score", 0)
            
            # Check Reveal duration
            if beat_type == "REVEAL" and duration > 8.0:
                self.add_issue(
                    severity=Severity.ERROR,
                    component=f"Beat {i}",
                    description=f"REVEAL beat is too long ({duration}s). Maximum allowed is 8s."
                )
                
            # Check Hook strength
            if beat_type == "HOOK" and emotion_score < 70:
                self.add_issue(
                    severity=Severity.WARNING,
                    component=f"Beat {i}",
                    description=f"HOOK beat is too weak (emotion_score={emotion_score}). Should be >= 70."
                )
                
            # Track low-energy streaks
            if emotion_score < 30:
                low_energy_streak += 1
                if low_energy_streak > 2:
                    self.add_issue(
                        severity=Severity.WARNING,
                        component=f"Beat {i}",
                        description=f"Too many consecutive low-energy beats (streak of {low_energy_streak})."
                    )
            else:
                low_energy_streak = 0

        # Check consecutive repetitions
        for i in range(1, len(beats)):
            prev_beat = beats[i-1]
            curr_beat = beats[i]
            
            # Check repeated assets
            curr_asset = curr_beat.get("asset_id") or curr_beat.get("image_search_query")
            prev_asset = prev_beat.get("asset_id") or prev_beat.get("image_search_query")
            if curr_asset and curr_asset == prev_asset:
                self.add_issue(
                    severity=Severity.WARNING,
                    component=f"Beat {i}",
                    description=f"Consecutive repeated asset detected: {curr_asset}"
                )
                
            # Check repeated transitions
            curr_trans = curr_beat.get("transition_type")
            prev_trans = prev_beat.get("transition_type")
            if curr_trans and curr_trans == prev_trans:
                self.add_issue(
                    severity=Severity.INFO,
                    component=f"Beat {i}",
                    description=f"Consecutive repeated transition detected: {curr_trans}"
                )
                
            # Check repeated motion presets
            curr_motion = curr_beat.get("motion_preset")
            prev_motion = prev_beat.get("motion_preset")
            if curr_motion and curr_motion == prev_motion:
                self.add_issue(
                    severity=Severity.INFO,
                    component=f"Beat {i}",
                    description=f"Consecutive repeated motion preset detected: {curr_motion}"
                )
