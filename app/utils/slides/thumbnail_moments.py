"""Thumbnail moment detector — finds retention-spike candidates.

Production editors create retention spikes every 20–40 seconds.
This module scans storyboard beats and marks the strongest moments as
``thumbnail_moment = True``.

Detection signals
-----------------
1. **Reveals** — beat_type == "reveal" or "payoff"
2. **Strongest emotions** — high emotion intensity (grief, shock, rage, triumph)
3. **Shocking statements** — text containing shock/twist/impossible/secret keywords

Spacing rule: moments are spaced ≥ ``min_gap`` seconds apart (default 20s)
and ≤ ``max_gap`` seconds (default 40s). If no natural moment exists in a
gap window, the strongest available beat is promoted.

Example::

    >>> detector = ThumbnailMomentDetector()
    >>> result = detector.detect(beats)
    >>> for m in result.moments:
    ...     print(f"[{m.index}] {m.timestamp}s  score={m.score:.2f}  {m.reason}")
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class ThumbnailMoment:
    """A detected thumbnail / retention-spike moment."""

    index: int                 # beat index
    timestamp: float           # seconds from start
    score: float               # moment strength [0, 1]
    reason: str                # why this was selected
    beat_type: str             # the beat's beat type
    emotion: str               # detected emotion
    text_snippet: str          # first 80 chars of text
    signals: List[str]         # which detection signals fired

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": round(self.timestamp, 2),
            "score": round(self.score, 3),
            "reason": self.reason,
            "beat_type": self.beat_type,
            "emotion": self.emotion,
            "text_snippet": self.text_snippet,
            "signals": self.signals,
            "thumbnail_moment": True,
        }


@dataclass
class DetectionResult:
    """Full result of thumbnail moment detection across a storyboard."""

    moments: List[ThumbnailMoment]
    annotated_beats: List[Dict]   # original beats with thumbnail_moment added
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "moments": [m.to_dict() for m in self.moments],
            "summary": self.summary,
        }

    @property
    def moment_count(self) -> int:
        return len(self.moments)

    @property
    def moment_indices(self) -> List[int]:
        return [m.index for m in self.moments]


# ── Detection patterns ──────────────────────────────────────────────

# Shocking statement keywords (high-signal text patterns).
_SHOCK_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
    (re.compile(r"\b(impossible|unbelievable|no way|can't be|inconceivable)\b", re.I),
     "shocking_statement", 0.30),
    (re.compile(r"\b(secret|hidden|truth|concealed|suppressed|covered up)\b", re.I),
     "hidden_truth", 0.25),
    (re.compile(r"\b(reveal|revealed|unveil|unmasked|exposed|discover)\b", re.I),
     "revelation", 0.28),
    (re.compile(r"\b(twist|plot twist|actually|turns out|real reason|in reality)\b", re.I),
     "narrative_twist", 0.32),
    (re.compile(r"\b(betray|betrayed|betrayal|traitor|double.cross)\b", re.I),
     "betrayal", 0.30),
    (re.compile(r"\b(death|died|killed|murdered|sacrifice|sacrificed)\b", re.I),
     "death_sacrifice", 0.28),
    (re.compile(r"\b(destroy|destruction|annihilat|obliterat|wipe out)\b", re.I),
     "destruction", 0.22),
    (re.compile(r"\b(transform|awaken|awakening|evolve|gear.5|nika)\b", re.I),
     "transformation", 0.30),
    (re.compile(r"\b(war|battle|final|ultimate|legendary|ancient)\b", re.I),
     "epic_scale", 0.15),
    (re.compile(r"\b(never|no one|only|first|last|greatest|strongest)\b", re.I),
     "superlative", 0.12),
]

# High-arousal emotions that signal thumbnail moments.
_STRONG_EMOTIONS = {
    "grief": 0.28,
    "shock": 0.35,
    "rage": 0.25,
    "fear": 0.22,
    "triumph": 0.30,
    "revelation": 0.32,
    "tension": 0.18,
    "hope": 0.15,
    "dread": 0.20,
    "excitement": 0.22,
}

# Beat types that inherently signal a moment.
_MOMENT_BEAT_TYPES = {
    "reveal": 0.35,
    "payoff": 0.30,
    "hook": 0.20,
    "warning": 0.15,
    "contradiction": 0.12,
}


# ── Time parsing ────────────────────────────────────────────────────

def _parse_time(ts: str) -> float:
    """Parse 'H:MM:SS.ff' or 'MM:SS.ff' or seconds to float."""
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


# ── ThumbnailMomentDetector ─────────────────────────────────────────


class ThumbnailMomentDetector:
    """Detects thumbnail-worthy moments in a storyboard.

    Parameters
    ----------
    min_gap : float
        Minimum seconds between moments (default 20).
    max_gap : float
        Maximum seconds between moments — if no natural moment exists
        in a window, the best available beat is promoted (default 40).
    score_threshold : float
        Minimum moment score to be considered (default 0.25).
    """

    def __init__(
        self,
        min_gap: float = 20.0,
        max_gap: float = 40.0,
        score_threshold: float = 0.25,
    ):
        self.min_gap = max(5.0, min_gap)
        self.max_gap = max(self.min_gap + 5.0, max_gap)
        self.score_threshold = score_threshold

    def detect(self, beats: Sequence[Dict[str, Any]]) -> DetectionResult:
        """Detect thumbnail moments across all beats.

        Parameters
        ----------
        beats : list of dict
            Storyboard beats with ``start_time``, ``end_time``,
            ``beat_type``, ``emotion_state``, ``subtitle_text``, etc.

        Returns
        -------
        DetectionResult
            Detected moments, annotated beats, and summary.
        """
        if not beats:
            return DetectionResult(
                moments=[], annotated_beats=[], summary={"beat_count": 0}
            )

        # 1. Score every beat as a potential moment.
        candidates = [self._score_beat(i, beat) for i, beat in enumerate(beats)]

        # 2. Select moments with spacing constraints.
        selected = self._select_with_spacing(candidates, beats)

        # 3. Annotate original beats.
        annotated = self._annotate_beats(beats, selected)

        # 4. Build summary.
        summary = self._build_summary(selected, beats)

        return DetectionResult(
            moments=selected,
            annotated_beats=annotated,
            summary=summary,
        )

    def detect_single(
        self,
        beat: Dict[str, Any],
        index: int = 0,
    ) -> Tuple[float, List[str]]:
        """Score a single beat as a thumbnail moment candidate.

        Returns (score, signals) — useful for real-time scoring.
        """
        candidate = self._score_beat(index, beat)
        return candidate["score"], candidate["signals"]

    # ── Internal scoring ─────────────────────────────────────────────

    def _score_beat(self, index: int, beat: Dict) -> Dict:
        """Compute a thumbnail-moment score for a single beat."""
        score = 0.0
        signals: List[str] = []
        text = " ".join(filter(None, [
            beat.get("subtitle_text", ""),
            beat.get("summary", ""),
            beat.get("viewer_focus", ""),
        ]))
        beat_type = (beat.get("beat_type") or "").strip().lower()
        emotion_state = beat.get("emotion_state") or {}
        emotion = ""
        intensity = 0.0
        if isinstance(emotion_state, dict):
            emotion = (emotion_state.get("emotion") or "").strip().lower()
            intensity = float(emotion_state.get("intensity") or 0.0)
        elif isinstance(emotion_state, str):
            emotion = emotion_state.strip().lower()

        timestamp = _parse_time(beat.get("start_time", "0"))

        # Signal 1: Reveal / payoff beat type.
        beat_bonus = _MOMENT_BEAT_TYPES.get(beat_type, 0.0)
        if beat_bonus > 0:
            score += beat_bonus
            signals.append(f"beat:{beat_type}")

        # Signal 2: Strong emotion.
        emo_bonus = _STRONG_EMOTIONS.get(emotion, 0.0)
        if emo_bonus > 0:
            # Scale by intensity if available.
            emo_bonus *= max(0.5, min(1.5, intensity / 0.5)) if intensity > 0 else 1.0
            score += emo_bonus
            signals.append(f"emotion:{emotion}(i={intensity:.2f})")

        # Signal 3: Shocking statement keywords.
        for pattern, label, bonus in _SHOCK_PATTERNS:
            if text and pattern.search(text):
                score += bonus
                signals.append(f"text:{label}")
                break  # Only the strongest text signal.

        # Signal 4: High emotion intensity regardless of label.
        if intensity >= 0.75 and not any(s.startswith("emotion:") for s in signals):
            score += 0.15
            signals.append(f"high_intensity:{intensity:.2f}")

        # Signal 5: Emphasis words or text overlay (editorial emphasis).
        if beat.get("emphasis_words"):
            score += 0.08
            signals.append("has_emphasis_words")
        if beat.get("text_overlay"):
            score += 0.05
            signals.append("has_text_overlay")

        return {
            "index": index,
            "timestamp": timestamp,
            "score": min(1.0, score),
            "signals": signals,
            "beat_type": beat_type,
            "emotion": emotion,
            "text": text[:80] if text else "",
        }

    # ── Spacing-constrained selection ────────────────────────────────

    def _select_with_spacing(
        self,
        candidates: List[Dict],
        beats: Sequence[Dict],
    ) -> List[ThumbnailMoment]:
        """Select moments with 20-40s spacing constraints.

        Algorithm:
        1. Sort candidates by score (descending).
        2. Greedily select the highest-scoring candidates that respect min_gap.
        3. If any window > max_gap has no moment, promote the best
           available beat in that gap.
        """
        # Phase 1: Greedy selection of strong candidates.
        sorted_candidates = sorted(candidates, key=lambda c: -c["score"])
        selected_indices: set = set()
        selected_timestamps: List[float] = []

        for cand in sorted_candidates:
            if cand["score"] < self.score_threshold:
                continue
            ts = cand["timestamp"]
            # Check min_gap against all already-selected moments.
            if all(abs(ts - st) >= self.min_gap for st in selected_timestamps):
                selected_indices.add(cand["index"])
                selected_timestamps.append(ts)

        # Phase 2: Fill gaps > max_gap.
        if candidates:
            total_duration = max(c["timestamp"] for c in candidates) if candidates else 0
            # Walk through time in max_gap steps.
            window_start = 0.0
            while window_start < total_duration:
                window_end = window_start + self.max_gap
                # Check if there's a moment in [window_start, window_end].
                has_moment = any(
                    window_start <= cand["timestamp"] < window_end
                    for cand in candidates
                    if cand["index"] in selected_indices
                )
                if not has_moment:
                    # Find the best unselected candidate in this window.
                    window_candidates = [
                        c for c in candidates
                        if window_start <= c["timestamp"] < window_end
                        and c["index"] not in selected_indices
                    ]
                    if window_candidates:
                        best = max(window_candidates, key=lambda c: c["score"])
                        # Check min_gap before promoting.
                        ts = best["timestamp"]
                        if all(abs(ts - st) >= self.min_gap * 0.7 for st in selected_timestamps):
                            selected_indices.add(best["index"])
                            selected_timestamps.append(ts)
                            logger.debug(
                                "Promoted beat %d at %.1fs to fill gap [%.1f-%.1f]",
                                best["index"], ts, window_start, window_end,
                            )
                window_start += self.max_gap

        # Build final moment list, sorted by timestamp.
        moments = []
        for cand in sorted(candidates, key=lambda c: c["timestamp"]):
            if cand["index"] in selected_indices:
                reason_parts = []
                if any(s.startswith("beat:") for s in cand["signals"]):
                    reason_parts.append(f"{cand['beat_type']} beat")
                if any(s.startswith("emotion:") for s in cand["signals"]):
                    reason_parts.append(f"strong {cand['emotion']}")
                if any(s.startswith("text:") for s in cand["signals"]):
                    reason_parts.append("shocking statement")
                if not reason_parts:
                    reason_parts.append("gap-fill promotion")
                reason = " + ".join(reason_parts)

                moments.append(ThumbnailMoment(
                    index=cand["index"],
                    timestamp=cand["timestamp"],
                    score=cand["score"],
                    reason=reason,
                    beat_type=cand["beat_type"],
                    emotion=cand["emotion"],
                    text_snippet=cand["text"],
                    signals=cand["signals"],
                ))

        return moments

    # ── Annotation ───────────────────────────────────────────────────

    def _annotate_beats(
        self,
        beats: Sequence[Dict],
        moments: List[ThumbnailMoment],
    ) -> List[Dict]:
        """Return copies of beats with ``thumbnail_moment`` field added."""
        moment_indices = {m.index for m in moments}
        annotated = []
        for i, beat in enumerate(beats):
            copy = dict(beat)
            copy["thumbnail_moment"] = i in moment_indices
            annotated.append(copy)
        return annotated

    # ── Summary ──────────────────────────────────────────────────────

    def _build_summary(
        self,
        moments: List[ThumbnailMoment],
        beats: Sequence[Dict],
    ) -> Dict[str, Any]:
        if not beats:
            return {"beat_count": 0}

        total_duration = 0.0
        for s in beats:
            end = _parse_time(s.get("end_time", "0"))
            if end > total_duration:
                total_duration = end

        gaps: List[float] = []
        sorted_moments = sorted(moments, key=lambda m: m.timestamp)
        for i in range(1, len(sorted_moments)):
            gaps.append(sorted_moments[i].timestamp - sorted_moments[i - 1].timestamp)

        return {
            "beat_count": len(beats),
            "moment_count": len(moments),
            "moment_indices": [m.index for m in moments],
            "total_duration": round(total_duration, 2),
            "avg_gap_seconds": round(sum(gaps) / len(gaps), 1) if gaps else 0.0,
            "min_gap_seconds": round(min(gaps), 1) if gaps else 0.0,
            "max_gap_seconds": round(max(gaps), 1) if gaps else 0.0,
            "moments_per_minute": round(
                len(moments) / max(1.0, total_duration / 60), 1
            ) if total_duration > 0 else 0.0,
            "signal_distribution": self._signal_distribution(moments),
        }

    @staticmethod
    def _signal_distribution(moments: List[ThumbnailMoment]) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for m in moments:
            for sig in m.signals:
                category = sig.split(":")[0]
                dist[category] = dist.get(category, 0) + 1
        return dist


# ── Convenience function ─────────────────────────────────────────────


def detect_thumbnail_moments(
    beats: Sequence[Dict[str, Any]],
    min_gap: float = 20.0,
    max_gap: float = 40.0,
    **kwargs,
) -> DetectionResult:
    """Shortcut: detect thumbnail moments in one call."""
    return ThumbnailMomentDetector(
        min_gap=min_gap, max_gap=max_gap, **kwargs
    ).detect(beats)
