"""Quality Gate — final validation for the storyboard.

Validates the complete storyboard against production rules, ensuring
no missing data, valid annotations, and correct formatting before rendering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)


@dataclass
class QualityIssue:
    """An issue caught by the quality gate."""
    beat_index: int
    severity: str        # "critical", "warning"
    description: str


@dataclass
class QualityGateReport:
    """Output of the quality gate."""
    passed: bool
    issues: List[QualityIssue]
    auto_fixed_beats: List[Dict[str, Any]]
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [
                {
                    "beat_index": i.beat_index,
                    "severity": i.severity,
                    "description": i.description,
                }
                for i in self.issues
            ],
            "auto_fixed_beats": self.auto_fixed_beats,
            "summary": self.summary,
        }


class QualityGate:
    """Validates and applies auto-fixes to the complete storyboard."""

    def __init__(self):
        self.required_keys = ["beat_type", "visual_role", "motion_preset"]

    def evaluate(self, beats: Sequence[Dict[str, Any]]) -> QualityGateReport:
        """Run quality checks on the storyboard.

        Parameters
        ----------
        beats : list of dict
            The fully annotated storyboard beats.

        Returns
        -------
        QualityGateReport
        """
        if not beats:
            return QualityGateReport(
                passed=False,
                issues=[QualityIssue(-1, "critical", "Storyboard is empty.")],
                auto_fixed_beats=[],
                summary={"beat_count": 0},
            )

        issues = []
        fixed_beats = []

        for i, beat in enumerate(beats):
            beat_copy = dict(beat)
            fixed = False

            # 1. Missing required keys
            for key in self.required_keys:
                if not beat_copy.get(key):
                    # Auto-fix if possible
                    if key == "beat_type":
                        beat_copy[key] = "neutral"
                        fixed = True
                        issues.append(QualityIssue(i, "warning", "Missing beat_type. Auto-fixed to 'neutral'."))
                    elif key == "visual_role":
                        beat_copy[key] = "character"
                        fixed = True
                        issues.append(QualityIssue(i, "warning", "Missing visual_role. Auto-fixed to 'character'."))
                    elif key == "motion_preset":
                        beat_copy[key] = "static_hold"
                        fixed = True
                        issues.append(QualityIssue(i, "warning", "Missing motion_preset. Auto-fixed to 'static_hold'."))
                    else:
                        issues.append(QualityIssue(i, "critical", f"Missing required key: {key}"))

            # 2. Text checks
            text = beat_copy.get("subtitle_text", "")
            summary_txt = beat_copy.get("summary", "")
            if not text and not summary_txt:
                issues.append(QualityIssue(i, "critical", "Beat has no subtitle_text and no summary."))

            # 3. Time checks
            start = beat_copy.get("start_time")
            end = beat_copy.get("end_time")
            if start is None or end is None:
                issues.append(QualityIssue(i, "warning", "Missing start_time or end_time."))

            fixed_beats.append(beat_copy)

        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0

        summary = {
            "beat_count": len(beats),
            "passed": passed,
            "critical_issues": critical_count,
            "warnings": len(issues) - critical_count,
        }

        return QualityGateReport(passed, issues, fixed_beats, summary)


def evaluate_quality(beats: Sequence[Dict[str, Any]], **kwargs) -> QualityGateReport:
    """Shortcut function for the quality gate."""
    return QualityGate(**kwargs).evaluate(beats)
