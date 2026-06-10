"""Retention optimizer — second-pass storyboard analyzer.

Runs five retention-killing pattern checks against a full storyboard
and generates actionable fixes for each violation.

Checks
------
1. **Same character > 3 beats** — character fatigue
2. **Same arc > 4 beats** — location fatigue
3. **No reveal > 30 seconds** — information drought
4. **No emotional increase** — flat intensity curve
5. **Weak hook** — opening beat lacks punch

Each violation produces a ``RetentionFix`` with a specific,
actionable recommendation and the beat indices affected.

Example::

    >>> optimizer = RetentionOptimizer()
    >>> report = optimizer.analyze(beats)
    >>> for v in report.violations:
    ...     print(f"[{v.check}] severity={v.severity} beats={v.beat_indices}")
    ...     for f in v.fixes:
    ...         print(f"  → {f}")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class RetentionFix:
    """A single actionable fix for a retention violation."""

    target_index: int          # beat to modify (-1 for general)
    action: str                # what to do (e.g., "swap_character", "insert_reveal")
    description: str           # human-readable fix description
    priority: str              # "critical", "high", "medium", "low"
    field_changes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_index": self.target_index,
            "action": self.action,
            "description": self.description,
            "priority": self.priority,
            "field_changes": self.field_changes,
        }


@dataclass
class RetentionViolation:
    """A detected retention-killing pattern."""

    check: str                 # check name
    severity: str              # "critical", "high", "medium", "low"
    message: str               # human-readable description
    beat_indices: List[int]   # affected beats
    fixes: List[RetentionFix]  # recommended fixes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
            "beat_indices": self.beat_indices,
            "fixes": [f.to_dict() for f in self.fixes],
        }


@dataclass
class RetentionReport:
    """Full retention optimization report."""

    violations: List[RetentionViolation]
    score: float               # overall retention health [0, 1]
    grade: str                 # letter grade
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violations": [v.to_dict() for v in self.violations],
            "score": round(self.score, 3),
            "grade": self.grade,
            "summary": self.summary,
        }

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    @property
    def critical_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "critical")

    @property
    def has_critical(self) -> bool:
        return self.critical_count > 0

    @property
    def fix_count(self) -> int:
        return sum(len(v.fixes) for v in self.violations)


# ── Time parsing ────────────────────────────────────────────────────

def _parse_time(ts) -> float:
    if not ts:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    ts = str(ts).strip()
    parts = ts.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(ts)
    except (ValueError, IndexError):
        return 0.0


# ── Feature extractors (shared with visual_diversity) ────────────────

_NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> str:
    return _NORM_RE.sub("", (text or "").lower())


def _extract_character(beat: Dict) -> str:
    entities = beat.get("context_entities") or []
    if entities:
        return _norm(entities[0])
    meta_entities = (beat.get("asset_metadata") or {}).get("entities") or []
    if meta_entities:
        return _norm(meta_entities[0])
    query = beat.get("image_search_query", "")
    words = [w for w in query.split() if len(w) > 2]
    return _norm(words[0]) if words else ""


def _extract_arc(beat: Dict) -> str:
    text = beat.get("subtitle_text", "") + " " + beat.get("summary", "")
    query = beat.get("image_search_query", "")
    try:
        from app.utils.assets.broll_rules import ARC_LOCATION_TERMS
        combined = (text + " " + query).lower()
        for term in ARC_LOCATION_TERMS:
            if term in combined:
                return _norm(term)
    except ImportError:
        pass
    tags = (beat.get("asset_metadata") or {}).get("search_tags") or []
    for tag in tags:
        if any(loc in tag.lower() for loc in ["island", "village", "castle", "sea", "ship", "tower"]):
            return _norm(tag)
    return ""


def _extract_emotion(beat: Dict) -> Tuple[str, float]:
    es = beat.get("emotion_state") or {}
    if isinstance(es, dict):
        return (es.get("emotion") or "").strip().lower(), float(es.get("intensity") or 0.0)
    return "", 0.0


# ── Alternative visual roles for fix suggestions ────────────────────

_ALT_VISUAL_ROLES = [
    "evidence", "symbol", "location", "comparison",
    "timeline", "object", "quote_card", "section_card",
]

_ALT_BEAT_TYPES = [
    "mystery", "contradiction", "reveal", "evidence", "escalation",
]


# ── RetentionOptimizer ──────────────────────────────────────────────


class RetentionOptimizer:
    """Second-pass storyboard analyzer that finds retention killers.

    Parameters
    ----------
    max_same_character : int
        Max consecutive beats with the same character (default 3).
    max_same_arc : int
        Max consecutive beats in the same arc (default 4).
    reveal_gap_limit : float
        Max seconds without a reveal/payoff beat (default 30).
    hook_min_intensity : float
        Minimum emotion intensity for the hook to be considered strong (default 0.6).
    """

    def __init__(
        self,
        max_same_character: int = 3,
        max_same_arc: int = 4,
        reveal_gap_limit: float = 30.0,
        hook_min_intensity: float = 0.6,
    ):
        self.max_same_character = max(1, max_same_character)
        self.max_same_arc = max(1, max_same_arc)
        self.reveal_gap_limit = max(10.0, reveal_gap_limit)
        self.hook_min_intensity = hook_min_intensity

    def analyze(self, beats: Sequence[Dict[str, Any]]) -> RetentionReport:
        """Run all retention checks against a storyboard.

        Parameters
        ----------
        beats : list of dict
            Full storyboard beats.

        Returns
        -------
        RetentionReport
        """
        if not beats:
            return RetentionReport(
                violations=[], score=1.0, grade="A",
                summary={"beat_count": 0, "checks_run": 0},
            )

        violations: List[RetentionViolation] = []

        # Run all five checks.
        violations.extend(self._check_same_character(beats))
        violations.extend(self._check_same_arc(beats))
        violations.extend(self._check_no_reveal(beats))
        violations.extend(self._check_flat_emotion(beats))
        violations.extend(self._check_weak_hook(beats))

        # Compute overall score.
        score = self._compute_score(violations, beats)
        grade = self._grade(score)
        summary = self._build_summary(violations, beats, score)

        return RetentionReport(
            violations=violations,
            score=score,
            grade=grade,
            summary=summary,
        )

    # ── Check 1: Same character > 3 beats ──────────────────────────

    def _check_same_character(self, beats: Sequence[Dict]) -> List[RetentionViolation]:
        violations = []
        chars = [_extract_character(s) for s in beats]
        runs = self._find_runs(chars)

        for char, start, length in runs:
            if length > self.max_same_character and char:
                indices = list(range(start, start + length))
                fixes = []

                # Suggest swapping characters starting at position max+1.
                for fix_idx in indices[self.max_same_character:]:
                    alt_role = _ALT_VISUAL_ROLES[fix_idx % len(_ALT_VISUAL_ROLES)]
                    fixes.append(RetentionFix(
                        target_index=fix_idx,
                        action="swap_character",
                        description=(
                            f"Slide {fix_idx}: switch from '{char}' to a different character, "
                            f"or use visual_role='{alt_role}' (object/symbol/location shot)"
                        ),
                        priority="high",
                        field_changes={"visual_role": alt_role},
                    ))

                violations.append(RetentionViolation(
                    check="same_character",
                    severity="high" if length > self.max_same_character + 1 else "medium",
                    message=(
                        f"Character '{char}' appears in {length} consecutive beats "
                        f"(indices {start}–{start + length - 1}). "
                        f"Limit is {self.max_same_character}."
                    ),
                    beat_indices=indices,
                    fixes=fixes,
                ))

        return violations

    # ── Check 2: Same arc > 4 beats ────────────────────────────────

    def _check_same_arc(self, beats: Sequence[Dict]) -> List[RetentionViolation]:
        violations = []
        arcs = [_extract_arc(s) for s in beats]
        runs = self._find_runs(arcs)

        for arc, start, length in runs:
            if length > self.max_same_arc and arc:
                indices = list(range(start, start + length))
                fixes = []

                for fix_idx in indices[self.max_same_arc:]:
                    fixes.append(RetentionFix(
                        target_index=fix_idx,
                        action="change_arc",
                        description=(
                            f"Slide {fix_idx}: break away from arc '{arc}' — "
                            f"use a flashback, comparison, or cross-cut to a different arc"
                        ),
                        priority="medium",
                        field_changes={"visual_role": "comparison"},
                    ))

                violations.append(RetentionViolation(
                    check="same_arc",
                    severity="medium" if length <= self.max_same_arc + 2 else "high",
                    message=(
                        f"Arc '{arc}' stays for {length} consecutive beats "
                        f"(indices {start}–{start + length - 1}). "
                        f"Limit is {self.max_same_arc}."
                    ),
                    beat_indices=indices,
                    fixes=fixes,
                ))

        return violations

    # ── Check 3: No reveal > 30 seconds ─────────────────────────────

    def _check_no_reveal(self, beats: Sequence[Dict]) -> List[RetentionViolation]:
        violations = []
        reveal_beats = {"reveal", "payoff"}

        # Build timeline of reveal timestamps.
        last_reveal_time = 0.0
        last_reveal_idx = -1
        drought_starts: List[Tuple[int, float, float]] = []  # (start_idx, start_time, gap)

        for i, entry in enumerate(beats):
            bt = (entry.get("beat_type") or "").strip().lower()
            ts = _parse_time(entry.get("start_time", "0"))

            if bt in reveal_beats:
                gap = ts - last_reveal_time
                if gap > self.reveal_gap_limit and i > 0:
                    drought_starts.append((last_reveal_idx + 1, last_reveal_time, gap))
                last_reveal_time = ts
                last_reveal_idx = i

        # Check trailing gap (after last reveal to end).
        if beats:
            end_time = _parse_time(beats[-1].get("end_time", "0"))
            trailing_gap = end_time - last_reveal_time
            if trailing_gap > self.reveal_gap_limit and last_reveal_idx < len(beats) - 1:
                drought_starts.append((last_reveal_idx + 1, last_reveal_time, trailing_gap))

        for start_idx, start_time, gap in drought_starts:
            # Find beats in the drought window.
            drought_indices = []
            for i, beat in enumerate(beats):
                ts = _parse_time(beat.get("start_time", "0"))
                if ts >= start_time and ts < start_time + gap:
                    drought_indices.append(i)

            if not drought_indices:
                continue

            # Suggest inserting a reveal/mystery beat at the midpoint.
            mid_idx = drought_indices[len(drought_indices) // 2]
            alt_beat = _ALT_BEAT_TYPES[mid_idx % len(_ALT_BEAT_TYPES)]

            fixes = [
                RetentionFix(
                    target_index=mid_idx,
                    action="insert_reveal",
                    description=(
                        f"Slide {mid_idx}: convert to beat_type='{alt_beat}' — "
                        f"add a revelation, mystery, or contradiction to break the {gap:.0f}s information drought"
                    ),
                    priority="critical",
                    field_changes={
                        "beat_type": alt_beat,
                        "motion_preset": "reveal_zoom",
                    },
                ),
            ]

            # Additional suggestion: add emphasis words.
            fixes.append(RetentionFix(
                target_index=mid_idx,
                action="add_emphasis",
                description=(
                    f"Slide {mid_idx}: add emphasis_words and text_overlay "
                    f"to create a visual hook mid-sequence"
                ),
                priority="high",
                field_changes={"text_overlay": "auto_generate"},
            ))

            violations.append(RetentionViolation(
                check="no_reveal_drought",
                severity="critical",
                message=(
                    f"No reveal/payoff for {gap:.0f}s "
                    f"(beats {drought_indices[0]}–{drought_indices[-1]}). "
                    f"Limit is {self.reveal_gap_limit:.0f}s."
                ),
                beat_indices=drought_indices,
                fixes=fixes,
            ))

        return violations

    # ── Check 4: No emotional increase (flat curve) ─────────────────

    def _check_flat_emotion(self, beats: Sequence[Dict]) -> List[RetentionViolation]:
        violations = []
        if len(beats) < 4:
            return violations

        intensities = [_extract_emotion(s)[1] for s in beats]

        # Check for flat windows of 4+ beats with no increase.
        window_size = 4
        for start in range(len(intensities) - window_size + 1):
            window = intensities[start:start + window_size]

            # Flat = no single increase of > 0.10 in the window.
            max_increase = 0.0
            for i in range(1, len(window)):
                increase = window[i] - window[i - 1]
                max_increase = max(max_increase, increase)

            if max_increase <= 0.10 and max(window) - min(window) < 0.15:
                indices = list(range(start, start + window_size))

                # Suggest boosting emotion at the midpoint.
                boost_idx = indices[len(indices) // 2]
                current_emo, current_int = _extract_emotion(beats[boost_idx])

                fixes = [
                    RetentionFix(
                        target_index=boost_idx,
                        action="boost_emotion",
                        description=(
                            f"Slide {boost_idx}: increase emotional intensity "
                            f"(currently '{current_emo}' at {current_int:.2f}). "
                            f"Add a tension spike, mystery, or revelation."
                        ),
                        priority="high",
                        field_changes={
                            "emotion_state": {"emotion": "tension", "intensity": 0.75},
                            "beat_type": "escalation",
                        },
                    ),
                    RetentionFix(
                        target_index=boost_idx,
                        action="change_motion",
                        description=(
                            f"Slide {boost_idx}: switch motion_preset to 'reveal_zoom' "
                            f"or 'dramatic_push' to create visual energy"
                        ),
                        priority="medium",
                        field_changes={"motion_preset": "reveal_zoom"},
                    ),
                ]

                violations.append(RetentionViolation(
                    check="flat_emotion",
                    severity="medium",
                    message=(
                        f"Flat emotional curve across beats {start}–{start + window_size - 1} "
                        f"(intensity range: {min(window):.2f}–{max(window):.2f}, "
                        f"max increase: {max_increase:.2f}). "
                        f"Viewers disengage when intensity stays flat."
                    ),
                    beat_indices=indices,
                    fixes=fixes,
                ))
                # Skip overlapping windows — advance past this violation.
                break

        return violations

    # ── Check 5: Weak hook ──────────────────────────────────────────

    def _check_weak_hook(self, beats: Sequence[Dict]) -> List[RetentionViolation]:
        violations = []
        if not beats:
            return violations

        first = beats[0]
        beat = (first.get("beat_type") or "").strip().lower()
        emotion, intensity = _extract_emotion(first)
        text = first.get("subtitle_text", "") + " " + first.get("summary", "")

        problems: List[str] = []
        fixes: List[RetentionFix] = []

        # Check 1: First beat should be a hook.
        if beat != "hook":
            problems.append(f"First beat is beat_type='{beat}', not 'hook'")
            fixes.append(RetentionFix(
                target_index=0,
                action="set_hook",
                description=(
                    "Slide 0: set beat_type='hook' — the opening must grab attention"
                ),
                priority="critical",
                field_changes={"beat_type": "hook"},
            ))

        # Check 2: Hook intensity should be strong.
        if intensity < self.hook_min_intensity:
            problems.append(
                f"Hook emotion intensity is {intensity:.2f} "
                f"(minimum: {self.hook_min_intensity:.2f})"
            )
            fixes.append(RetentionFix(
                target_index=0,
                action="boost_hook_intensity",
                description=(
                    f"Slide 0: boost emotion intensity from {intensity:.2f} to ≥{self.hook_min_intensity} — "
                    f"use a provocative question, shocking statement, or strong visual"
                ),
                priority="high",
                field_changes={
                    "emotion_state": {"emotion": "tension", "intensity": 0.82},
                },
            ))

        # Check 3: Hook should have a text overlay or emphasis words.
        if not first.get("text_overlay") and not first.get("emphasis_words"):
            problems.append("Hook has no text_overlay or emphasis_words")
            fixes.append(RetentionFix(
                target_index=0,
                action="add_hook_overlay",
                description=(
                    "Slide 0: add a text_overlay with a provocative question or bold statement"
                ),
                priority="high",
                field_changes={"text_overlay": "auto_generate"},
            ))

        # Check 4: Hook motion should be dynamic.
        motion = (first.get("motion_preset") or "").strip().lower()
        if motion in ("static_hold", "hold_still", "evidence_hold", ""):
            problems.append(f"Hook motion_preset='{motion}' is too static")
            fixes.append(RetentionFix(
                target_index=0,
                action="fix_hook_motion",
                description=(
                    "Slide 0: set motion_preset='subject_push' or 'reveal_zoom' — "
                    "the hook needs dynamic camera movement"
                ),
                priority="medium",
                field_changes={"motion_preset": "subject_push"},
            ))

        if problems:
            violations.append(RetentionViolation(
                check="weak_hook",
                severity="critical" if beat != "hook" or intensity < 0.3 else "high",
                message="Weak hook: " + "; ".join(problems),
                beat_indices=[0],
                fixes=fixes,
            ))

        return violations

    # ── Helper: find consecutive runs ────────────────────────────────

    @staticmethod
    def _find_runs(values: List[str]) -> List[Tuple[str, int, int]]:
        """Find consecutive runs of the same value.

        Returns list of (value, start_index, run_length).
        """
        if not values:
            return []

        runs: List[Tuple[str, int, int]] = []
        current_val = values[0]
        start = 0
        length = 1

        for i in range(1, len(values)):
            if values[i] == current_val and current_val:
                length += 1
            else:
                if length > 1 and current_val:
                    runs.append((current_val, start, length))
                current_val = values[i]
                start = i
                length = 1

        if length > 1 and current_val:
            runs.append((current_val, start, length))

        return runs

    # ── Scoring ──────────────────────────────────────────────────────

    def _compute_score(
        self,
        violations: List[RetentionViolation],
        beats: Sequence[Dict],
    ) -> float:
        """Compute overall retention health score.

        Starts at 1.0 and deducts per violation severity.
        """
        score = 1.0
        deductions = {
            "critical": 0.20,
            "high": 0.12,
            "medium": 0.08,
            "low": 0.04,
        }
        for v in violations:
            score -= deductions.get(v.severity, 0.05)

        return max(0.0, min(1.0, score))

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 0.90:
            return "A"
        if score >= 0.75:
            return "B"
        if score >= 0.60:
            return "C"
        if score >= 0.40:
            return "D"
        return "F"

    def _build_summary(
        self,
        violations: List[RetentionViolation],
        beats: Sequence[Dict],
        score: float,
    ) -> Dict[str, Any]:
        by_check: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for v in violations:
            by_check[v.check] = by_check.get(v.check, 0) + 1
            by_severity[v.severity] = by_severity.get(v.severity, 0) + 1

        total_fixes = sum(len(v.fixes) for v in violations)

        return {
            "beat_count": len(beats),
            "checks_run": 5,
            "violation_count": len(violations),
            "total_fixes": total_fixes,
            "by_check": by_check,
            "by_severity": by_severity,
            "score": round(score, 3),
            "grade": self._grade(score),
            "critical_issues": [v.message for v in violations if v.severity == "critical"],
        }


# ── Convenience function ─────────────────────────────────────────────


def optimize_retention(
    beats: Sequence[Dict[str, Any]],
    **kwargs,
) -> RetentionReport:
    """Shortcut: analyze storyboard retention in one call."""
    return RetentionOptimizer(**kwargs).analyze(beats)
