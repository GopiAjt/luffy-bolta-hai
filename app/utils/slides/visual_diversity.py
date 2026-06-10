"""Visual diversity scorer for storyboard beats.

Penalizes repetition across four dimensions and rejects storyboard
sections that exceed a repetition threshold.

Penalty dimensions
------------------
1. **Same character** — consecutive beats featuring the same character
2. **Same arc** — consecutive beats set in the same arc/location
3. **Same composition** — consecutive beats using the same visual_role + layout_mode
4. **Same emotion** — consecutive beats with the same emotion label

A section's *repetition score* (0 = fully diverse, 1 = fully repetitive) is
the weighted sum of per-dimension penalties.  Sections above the rejection
threshold (default **0.7**) are flagged for replacement.

Example::

    >>> scorer = VisualDiversityScorer()
    >>> report = scorer.score(beats)
    >>> for s in report.sections:
    ...     print(f"[{s.index}] rep={s.repetition_score:.2f} rejected={s.rejected}")
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class SectionScore:
    """Diversity score for a single beat/section."""

    index: int
    repetition_score: float     # 0.0 (diverse) → 1.0 (fully repetitive)
    rejected: bool              # True when repetition_score > threshold
    penalties: Dict[str, float] # per-dimension penalty values
    notes: List[str]            # human-readable penalty explanations
    suggestions: List[str]      # actionable fix suggestions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "repetition_score": round(self.repetition_score, 3),
            "rejected": self.rejected,
            "penalties": {k: round(v, 3) for k, v in self.penalties.items()},
            "notes": self.notes,
            "suggestions": self.suggestions,
        }


@dataclass
class DiversityReport:
    """Full diversity report across all beats."""

    sections: List[SectionScore]
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sections": [s.to_dict() for s in self.sections],
            "summary": self.summary,
        }

    @property
    def rejected_indices(self) -> List[int]:
        return [s.index for s in self.sections if s.rejected]

    @property
    def rejected_count(self) -> int:
        return sum(1 for s in self.sections if s.rejected)

    @property
    def mean_repetition(self) -> float:
        if not self.sections:
            return 0.0
        return sum(s.repetition_score for s in self.sections) / len(self.sections)


# ── Field extractors ────────────────────────────────────────────────

_NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> str:
    return _NORM_RE.sub("", (text or "").lower())


def _extract_character(beat: Dict) -> str:
    """Extract the primary character from a beat."""
    # Try context_entities first (most reliable).
    entities = beat.get("context_entities") or []
    if entities:
        return _norm(entities[0])

    # Fall back to asset_metadata entities.
    asset_meta = beat.get("asset_metadata") or {}
    meta_entities = asset_meta.get("entities") or []
    if meta_entities:
        return _norm(meta_entities[0])

    # Fall back to query.
    query = beat.get("image_search_query", "")
    if query:
        # Take the first significant word.
        words = [w for w in query.split() if len(w) > 2]
        if words:
            return _norm(words[0])

    return ""


def _extract_arc(beat: Dict) -> str:
    """Extract the arc/location context from a beat."""
    # Try beat analysis arc.
    text = beat.get("subtitle_text", "") + " " + beat.get("summary", "")
    query = beat.get("image_search_query", "")

    # Check for known arc location terms.
    try:
        from app.utils.assets.broll_rules import ARC_LOCATION_TERMS
        combined = (text + " " + query).lower()
        for term in ARC_LOCATION_TERMS:
            if term in combined:
                return _norm(term)
    except ImportError:
        pass

    # Fall back to search tags that look like locations.
    tags = (beat.get("asset_metadata") or {}).get("search_tags") or []
    for tag in tags:
        if any(loc in tag.lower() for loc in ["island", "village", "castle", "sea", "ship", "tower"]):
            return _norm(tag)

    return ""


def _extract_composition(beat: Dict) -> str:
    """Extract the composition signature: visual_role + layout_mode."""
    role = (beat.get("visual_role") or "character").strip().lower()
    layout = (beat.get("layout_mode") or "safe_subject").strip().lower()
    return f"{role}_{layout}"


def _extract_emotion(beat: Dict) -> str:
    """Extract the emotion label from a beat."""
    emotion_state = beat.get("emotion_state") or {}
    if isinstance(emotion_state, dict):
        label = emotion_state.get("emotion", "")
        if label:
            return _norm(label)

    # Fall back to text-based detection.
    return ""


# ── Suggestion generators ───────────────────────────────────────────

_CHARACTER_SUGGESTIONS = [
    "Switch to a different character's perspective",
    "Use a location or object shot instead of character close-up",
    "Show a symbol or abstract visual related to the narration",
    "Use a wide establishing shot or environment",
]

_ARC_SUGGESTIONS = [
    "Transition to a flashback or flash-forward in a different arc",
    "Use a comparison/contrast visual from another location",
    "Show a map, timeline, or abstract representation",
]

_COMPOSITION_SUGGESTIONS = [
    "Switch visual_role (e.g., character → evidence → symbol)",
    "Change layout_mode (e.g., safe_subject → title_card → bleed)",
    "Use a split-screen or comparison layout",
    "Switch to a quote_card or section_card for variety",
]

_EMOTION_SUGGESTIONS = [
    "Introduce tonal contrast — shift to a different emotional register",
    "Use visual pacing: insert a calm setup beat between intense moments",
    "Vary the color palette to signal emotional change",
]


# ── VisualDiversityScorer ───────────────────────────────────────────


class VisualDiversityScorer:
    """Scores visual diversity and rejects repetitive storyboard sections.

    Parameters
    ----------
    rejection_threshold : float
        Sections with a repetition score above this are rejected.
        Default: 0.7.
    lookback : int
        Number of preceding beats to compare against.
        Default: 3 (checks the previous 3 beats).
    weights : dict, optional
        Per-dimension weights. Default:
        ``{"character": 0.30, "arc": 0.25, "composition": 0.25, "emotion": 0.20}``
    """

    def __init__(
        self,
        rejection_threshold: float = 0.7,
        lookback: int = 3,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.rejection_threshold = rejection_threshold
        self.lookback = max(1, lookback)
        self.weights = weights or {
            "character": 0.30,
            "arc": 0.25,
            "composition": 0.25,
            "emotion": 0.20,
        }

    def score(self, beats: Sequence[Dict[str, Any]]) -> DiversityReport:
        """Score all beats and return a full diversity report.

        Parameters
        ----------
        beats : list of dict
            Storyboard beats (as produced by the image_slides pipeline).

        Returns
        -------
        DiversityReport
            Per-section scores, rejected indices, and overall summary.
        """
        if not beats:
            return DiversityReport(sections=[], summary={"beat_count": 0})

        # Pre-extract features for all beats.
        features = [self._extract_features(s) for s in beats]

        sections: List[SectionScore] = []
        for i, beat in enumerate(beats):
            section = self._score_section(i, features)
            sections.append(section)

        summary = self._build_summary(sections, features)
        return DiversityReport(sections=sections, summary=summary)

    def score_single(
        self,
        beat: Dict[str, Any],
        previous_beats: Sequence[Dict[str, Any]],
    ) -> SectionScore:
        """Score a single beat against its predecessors.

        Useful for real-time scoring during beat generation.
        """
        features = [self._extract_features(s) for s in previous_beats]
        current = self._extract_features(beat)
        features.append(current)
        return self._score_section(len(features) - 1, features)

    # ── Internal methods ─────────────────────────────────────────────

    def _extract_features(self, beat: Dict) -> Dict[str, str]:
        """Extract the four diversity dimensions from a beat."""
        return {
            "character": _extract_character(beat),
            "arc": _extract_arc(beat),
            "composition": _extract_composition(beat),
            "emotion": _extract_emotion(beat),
        }

    def _score_section(
        self,
        index: int,
        all_features: List[Dict[str, str]],
    ) -> SectionScore:
        """Compute the repetition score for a single section."""
        if index == 0:
            return SectionScore(
                index=0,
                repetition_score=0.0,
                rejected=False,
                penalties={"character": 0.0, "arc": 0.0, "composition": 0.0, "emotion": 0.0},
                notes=["Opening beat — no repetition possible"],
                suggestions=[],
            )

        current = all_features[index]
        lookback_start = max(0, index - self.lookback)
        recent = all_features[lookback_start:index]

        penalties: Dict[str, float] = {}
        notes: List[str] = []
        suggestions: List[str] = []

        # --- Character penalty ---
        char_penalty = self._compute_repeat_penalty(
            current["character"], [f["character"] for f in recent], "character"
        )
        penalties["character"] = char_penalty
        if char_penalty > 0:
            notes.append(
                f"Same character '{current['character']}' repeated in {int(char_penalty * len(recent))}/{len(recent)} recent beats"
            )
            suggestions.append(_CHARACTER_SUGGESTIONS[index % len(_CHARACTER_SUGGESTIONS)])

        # --- Arc penalty ---
        arc_penalty = self._compute_repeat_penalty(
            current["arc"], [f["arc"] for f in recent], "arc"
        )
        penalties["arc"] = arc_penalty
        if arc_penalty > 0:
            notes.append(
                f"Same arc/location '{current['arc']}' repeated"
            )
            suggestions.append(_ARC_SUGGESTIONS[index % len(_ARC_SUGGESTIONS)])

        # --- Composition penalty ---
        comp_penalty = self._compute_repeat_penalty(
            current["composition"], [f["composition"] for f in recent], "composition"
        )
        penalties["composition"] = comp_penalty
        if comp_penalty > 0:
            notes.append(
                f"Same composition '{current['composition']}' repeated"
            )
            suggestions.append(_COMPOSITION_SUGGESTIONS[index % len(_COMPOSITION_SUGGESTIONS)])

        # --- Emotion penalty ---
        emo_penalty = self._compute_repeat_penalty(
            current["emotion"], [f["emotion"] for f in recent], "emotion"
        )
        penalties["emotion"] = emo_penalty
        if emo_penalty > 0:
            notes.append(
                f"Same emotion '{current['emotion']}' repeated"
            )
            suggestions.append(_EMOTION_SUGGESTIONS[index % len(_EMOTION_SUGGESTIONS)])

        if not notes:
            notes.append("Good visual diversity — no repetition detected")

        # Weighted sum.
        repetition_score = sum(
            self.weights.get(dim, 0.25) * pen
            for dim, pen in penalties.items()
        )
        repetition_score = round(max(0.0, min(1.0, repetition_score)), 3)
        rejected = repetition_score > self.rejection_threshold

        if rejected:
            suggestions.insert(0, "⚠ REJECTED — replace this section to reduce visual monotony")

        return SectionScore(
            index=index,
            repetition_score=repetition_score,
            rejected=rejected,
            penalties=penalties,
            notes=notes,
            suggestions=suggestions,
        )

    def _compute_repeat_penalty(
        self,
        current_val: str,
        recent_vals: List[str],
        dimension: str,
    ) -> float:
        """Compute how much a value repeats in recent context.

        Returns a penalty in [0, 1]:
        - 0.0 = no repetition
        - 1.0 = identical to all recent beats
        """
        if not current_val or not recent_vals:
            return 0.0

        # Count how many recent beats have the same value.
        matches = sum(1 for v in recent_vals if v and v == current_val)
        if matches == 0:
            return 0.0

        # Base penalty: fraction of recent beats that match.
        base = matches / len(recent_vals)

        # Consecutive-run bonus: penalize harder for unbroken streaks.
        consecutive = 0
        for v in reversed(recent_vals):
            if v and v == current_val:
                consecutive += 1
            else:
                break
        run_bonus = min(0.25, 0.10 * consecutive)

        return min(1.0, base + run_bonus)

    def _build_summary(
        self,
        sections: List[SectionScore],
        features: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Aggregate stats for the full storyboard."""
        if not sections:
            return {"beat_count": 0}

        rejected = [s for s in sections if s.rejected]
        scores = [s.repetition_score for s in sections]

        # Character distribution.
        char_counts = Counter(f["character"] for f in features if f["character"])
        # Composition distribution.
        comp_counts = Counter(f["composition"] for f in features if f["composition"])
        # Arc distribution.
        arc_counts = Counter(f["arc"] for f in features if f["arc"])

        # Most overused character.
        most_used_char = char_counts.most_common(1)[0] if char_counts else ("none", 0)
        # Most overused composition.
        most_used_comp = comp_counts.most_common(1)[0] if comp_counts else ("none", 0)

        return {
            "beat_count": len(sections),
            "rejected_count": len(rejected),
            "rejected_indices": [s.index for s in rejected],
            "mean_repetition": round(sum(scores) / len(scores), 3),
            "max_repetition": round(max(scores), 3),
            "min_repetition": round(min(scores), 3),
            "character_distribution": dict(char_counts.most_common(5)),
            "composition_distribution": dict(comp_counts.most_common(5)),
            "arc_distribution": dict(arc_counts.most_common(5)),
            "most_used_character": most_used_char[0],
            "most_used_character_count": most_used_char[1],
            "most_used_composition": most_used_comp[0],
            "most_used_composition_count": most_used_comp[1],
            "diversity_grade": self._grade(sum(scores) / len(scores)),
        }

    @staticmethod
    def _grade(mean_repetition: float) -> str:
        """Assign a letter grade based on mean repetition."""
        if mean_repetition <= 0.15:
            return "A"   # Excellent diversity
        if mean_repetition <= 0.30:
            return "B"   # Good diversity
        if mean_repetition <= 0.50:
            return "C"   # Moderate repetition
        if mean_repetition <= 0.70:
            return "D"   # Excessive repetition
        return "F"       # Severe — needs rework


# ── Convenience function ─────────────────────────────────────────────


def score_visual_diversity(
    beats: Sequence[Dict[str, Any]],
    rejection_threshold: float = 0.7,
    **kwargs,
) -> DiversityReport:
    """Shortcut: create a scorer and produce the report in one call."""
    return VisualDiversityScorer(
        rejection_threshold=rejection_threshold, **kwargs
    ).score(beats)
