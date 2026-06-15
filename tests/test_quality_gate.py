import pytest
from app.utils.slides.quality_gate import Severity, QualityIssue, QualityReport, QualityGate

def test_severity_enum():
    """Verify enum serialization."""
    assert Severity.INFO.value == "INFO"
    assert Severity.WARNING.value == "WARNING"
    assert Severity.ERROR.value == "ERROR"

def test_quality_issue_dataclass():
    """Verify dataclass construction."""
    issue = QualityIssue(
        severity=Severity.WARNING,
        component="ImageLoader",
        description="Image is low resolution"
    )
    assert issue.severity == Severity.WARNING
    assert issue.component == "ImageLoader"
    assert issue.description == "Image is low resolution"

def test_quality_gate_initialization():
    """Verify gate starts empty."""
    gate = QualityGate()
    assert len(gate.issues) == 0

def test_quality_gate_add_issue():
    """Verify adding issues to the gate."""
    gate = QualityGate()
    gate.add_issue(Severity.INFO, "Slide 1", "Contrast is slightly low")
    
    assert len(gate.issues) == 1
    assert gate.issues[0].severity == Severity.INFO
    assert gate.issues[0].component == "Slide 1"

def test_quality_gate_evaluate_pass():
    """Verify evaluate() passes if there are no ERRORs."""
    gate = QualityGate()
    gate.add_issue(Severity.INFO, "Audio", "Slight noise")
    gate.add_issue(Severity.WARNING, "Video", "Frame drop")
    
    report = gate.evaluate()
    assert report.passed is True
    assert len(report.issues) == 2

def test_quality_gate_evaluate_fail():
    """Verify evaluate() fails if there is at least one ERROR."""
    gate = QualityGate()
    gate.add_issue(Severity.INFO, "Audio", "Slight noise")
    gate.add_issue(Severity.ERROR, "Render", "Rendering failed to allocate memory")
    
    report = gate.evaluate()
    assert report.passed is False
    assert len(report.issues) == 2

def test_quality_gate_analyze_storyboard():
    """Verify consecutive repetitions are caught and logged as issues."""
    gate = QualityGate()
    
    beats = [
        {
            "asset_id": "img_A",
            "transition_type": "fade",
            "motion_preset": "pan_left",
            "emotion_score": 50
        },
        {
            "asset_id": "img_A",  # Repeated asset -> WARNING
            "transition_type": "fade",  # Repeated transition -> INFO
            "motion_preset": "pan_left",  # Repeated motion -> INFO
            "emotion_score": 50
        },
        {
            "asset_id": "img_B",
            "transition_type": "zoom",
            "motion_preset": "pan_right",
            "emotion_score": 50
        }
    ]
    
    gate.analyze_storyboard(beats)
    
    # We expect 3 issues from the transition between beat 0 and beat 1.
    assert len(gate.issues) == 3
    
    warnings = [i for i in gate.issues if i.severity == Severity.WARNING]
    infos = [i for i in gate.issues if i.severity == Severity.INFO]
    
    assert len(warnings) == 1
    assert "repeated asset" in warnings[0].description
    
    assert len(infos) == 2
    descriptions = [i.description for i in infos]
    assert any("repeated transition" in d for d in descriptions)
    assert any("repeated motion" in d for d in descriptions)

def test_quality_gate_advanced_checks():
    """Verify checks for REVEAL duration, HOOK strength, and low-energy streaks."""
    gate = QualityGate()
    
    beats = [
        # Beat 0: Weak HOOK -> WARNING
        {
            "beat_type": "HOOK",
            "emotion_score": 50,  # Below 70
            "duration": 3.0
        },
        # Beat 1: Too long REVEAL -> ERROR
        {
            "beat_type": "REVEAL",
            "emotion_score": 80,
            "duration": 10.0  # Above 8.0
        },
        # Beat 2: Low energy (streak 1)
        {
            "beat_type": "NEUTRAL",
            "emotion_score": 20
        },
        # Beat 3: Low energy (streak 2)
        {
            "beat_type": "NEUTRAL",
            "emotion_score": 20
        },
        # Beat 4: Low energy (streak 3 -> WARNING)
        {
            "beat_type": "NEUTRAL",
            "emotion_score": 20
        }
    ]
    
    gate.analyze_storyboard(beats)
    
    # Expect: 1 ERROR (reveal), 2 WARNINGs (weak hook, low energy streak)
    assert len(gate.issues) == 3
    
    errors = [i for i in gate.issues if i.severity == Severity.ERROR]
    warnings = [i for i in gate.issues if i.severity == Severity.WARNING]
    
    assert len(errors) == 1
    assert "REVEAL beat is too long" in errors[0].description
    
    assert len(warnings) == 2
    descriptions = [w.description for w in warnings]
    assert any("HOOK beat is too weak" in d for d in descriptions)
    assert any("Too many consecutive low-energy beats" in d for d in descriptions)
