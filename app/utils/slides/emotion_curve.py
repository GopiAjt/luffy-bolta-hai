"""Emotion curve generator for story beats.

Converts a sequence of story beats (from ``StoryAnalyzer``) into a smooth
emotional intensity curve with per-beat scores, labels, and narrative arc
metadata.

Curve rules
-----------
- **Hooks spike** — opening beat punches high to grab attention.
- **Reveals spike** — truth-drops create sharp emotional peaks.
- **Payoffs peak** — the argument climax reaches the global maximum.
- **CTAs drop** — closing call-to-action deflates to calm resolution.
- **Setup / evidence** builds gradually.
- **Escalation** ramps between setup and reveal.
- **Mystery / contradiction** holds elevated tension.
- **Warning** sustains high dread.

Example::

    >>> from app.utils.slides.story_analyzer import StoryAnalyzer
    >>> from app.utils.slides.emotion_curve import EmotionCurveGenerator
    >>>
    >>> beats = StoryAnalyzer().analyze("Why did Shanks bet his arm? ...")
    >>> curve = EmotionCurveGenerator().generate(beats)
    >>> for pt in curve.points:
    ...     print(f"{pt.index:2d}  {pt.beat_type:12s}  {pt.score:.2f}  {pt.label}")
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class EmotionPoint:
    """A single point on the emotion curve."""

    index: int
    beat_type: str
    text: str
    score: float              # overall emotional intensity [0, 1]
    label: str                # human-readable emotion name
    valence: float            # positive/negative sentiment [-1, 1]
    energy: float             # arousal / kinetic energy [0, 1]
    curve_position: float     # normalized position in story [0, 1]
    spike: bool = False       # True if this beat is a spike point
    score_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for key in ("score", "valence", "energy", "curve_position"):
            d[key] = round(d[key], 3)
        d["score_breakdown"] = {k: round(v, 3) for k, v in d["score_breakdown"].items()}
        return d


@dataclass
class EmotionCurve:
    """The complete emotion curve for a sequence of beats."""

    points: List[EmotionPoint]
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "points": [p.to_dict() for p in self.points],
            "summary": self.summary,
        }

    @property
    def scores(self) -> List[float]:
        return [p.score for p in self.points]

    @property
    def labels(self) -> List[str]:
        return [p.label for p in self.points]

    @property
    def peak_index(self) -> int:
        if not self.points:
            return 0
        return max(range(len(self.points)), key=lambda i: self.points[i].score)


# ── Beat-type intensity profiles ────────────────────────────────────

# Base intensity and valence by beat type.
# (base_score, valence, energy, is_spike)
_BEAT_PROFILES: Dict[str, Tuple[float, float, float, bool]] = {
    "hook":          (0.82, 0.10, 0.88, True),    # spikes
    "setup":         (0.38, 0.05, 0.35, False),    # builds gently
    "mystery":       (0.62, -0.05, 0.65, False),   # elevated tension
    "contradiction": (0.68, -0.15, 0.72, False),   # jarring contrast
    "evidence":      (0.45, 0.08, 0.42, False),    # steady build
    "escalation":    (0.72, -0.10, 0.78, False),   # ramping up
    "reveal":        (0.88, 0.15, 0.90, True),     # spikes
    "payoff":        (0.95, 0.40, 0.85, True),     # peaks (global max)
    "warning":       (0.78, -0.45, 0.82, False),   # sustained dread
    "cta":           (0.25, 0.20, 0.18, False),    # drops to calm
}

# ── Text-level emotion detection ────────────────────────────────────

_EMOTION_PATTERNS: List[Tuple[re.Pattern, str, float, float]] = [
    # (pattern, label, valence_shift, intensity_boost)
    (re.compile(r"\b(death|died|kill|murder|pain|sacrifice|betrayal|cry|tears|alone|weak|suffer)\b", re.I),
     "grief", -0.55, 0.20),
    (re.compile(r"\b(danger|threat|villain|war|fight|destroy|monster|evil|terror|dread|doom)\b", re.I),
     "tension", -0.35, 0.15),
    (re.compile(r"\b(secret|hidden|truth|mystery|impossible|twist|clue|unknown|why)\b", re.I),
     "intrigue", -0.05, 0.12),
    (re.compile(r"\b(dream|freedom|joy|hope|promise|future|win|victory|dawn|liberation)\b", re.I),
     "hope", 0.45, 0.10),
    (re.compile(r"\b(fear|afraid|terrified|helpless|panic|haunted|trauma)\b", re.I),
     "fear", -0.50, 0.18),
    (re.compile(r"\b(rage|fury|furious|anger|wrath|vengeance)\b", re.I),
     "rage", -0.30, 0.22),
    (re.compile(r"\b(love|bond|protect|save|friend|family|brother|crew|nakama)\b", re.I),
     "bond", 0.35, 0.08),
    (re.compile(r"\b(shock|impossible|unbelievable|no way|can't be)\b", re.I),
     "shock", -0.10, 0.25),
    (re.compile(r"\b(reveal|truth|actually|turns out|real reason|the answer)\b", re.I),
     "revelation", 0.15, 0.18),
]


# ── Position modifiers ──────────────────────────────────────────────

def _position_modifier(index: int, total: int) -> float:
    """Slight intensity boost based on narrative position.

    The story naturally builds: middle beats get a small ramp,
    late beats (pre-CTA) get the highest modifier.
    """
    if total <= 1:
        return 0.0
    progress = index / max(1, total - 1)
    # Bell-shaped boost peaking around 70-80% of the story.
    return 0.08 * math.sin(progress * math.pi * 0.9)


def _momentum_modifier(
    index: int,
    beat_type: str,
    prev_score: float,
    prev_beat: str,
) -> float:
    """Momentum: consecutive escalation builds, CTA after peak drops harder."""
    mod = 0.0
    # Escalation chains build.
    if beat_type in ("escalation", "evidence") and prev_beat in ("escalation", "evidence", "setup"):
        mod += 0.06
    # Reveal after escalation gets extra punch.
    if beat_type == "reveal" and prev_beat in ("escalation", "mystery", "contradiction"):
        mod += 0.08
    # Payoff after reveal gets the climax boost.
    if beat_type == "payoff" and prev_beat in ("reveal", "escalation", "warning"):
        mod += 0.06
    # CTA drops harder if previous was intense.
    if beat_type == "cta" and prev_score > 0.7:
        mod -= 0.10
    return mod


# ── EmotionCurveGenerator ───────────────────────────────────────────


class EmotionCurveGenerator:
    """Generate emotion intensity curves from story beats.

    Input: list of beat dicts from ``StoryAnalyzer.analyze()``
    (must have ``index``, ``beat_type``, ``text``).

    Output: ``EmotionCurve`` with per-beat ``EmotionPoint`` values and
    a summary with global stats.
    """

    def __init__(
        self,
        smoothing: float = 0.15,
        hook_floor: float = 0.78,
        reveal_floor: float = 0.82,
        payoff_floor: float = 0.90,
        cta_ceiling: float = 0.30,
    ):
        self.smoothing = max(0.0, min(1.0, smoothing))
        self.hook_floor = hook_floor
        self.reveal_floor = reveal_floor
        self.payoff_floor = payoff_floor
        self.cta_ceiling = cta_ceiling

    def generate(self, beats: Sequence[Dict[str, Any]]) -> EmotionCurve:
        """Generate the emotion curve from story beats.

        Parameters
        ----------
        beats : list of dict
            Each dict must have ``beat_type`` and ``text``.
            Optionally ``index`` and ``confidence``.

        Returns
        -------
        EmotionCurve
        """
        if not beats:
            return EmotionCurve(points=[], summary={"beat_count": 0})

        total = len(beats)
        raw_points: List[EmotionPoint] = []

        prev_score = 0.0
        prev_beat = ""

        for i, beat in enumerate(beats):
            beat_type = (beat.get("beat_type") or "evidence").strip().lower()
            text = beat.get("text", "")
            beat_confidence = float(beat.get("confidence", 0.5))

            # 1. Base profile from beat type.
            base_score, base_valence, base_energy, is_spike = _BEAT_PROFILES.get(
                beat_type, (0.45, 0.0, 0.42, False)
            )

            # 2. Text-level emotion detection.
            label, text_valence_shift, text_intensity_boost = self._detect_emotion(text)

            # 3. Position modifier (narrative arc shaping).
            pos_mod = _position_modifier(i, total)

            # 4. Momentum modifier (beat-to-beat flow).
            mom_mod = _momentum_modifier(i, beat_type, prev_score, prev_beat)

            # 5. Confidence weighting (higher confidence beats are more definitive).
            conf_mod = (beat_confidence - 0.5) * 0.08

            # Composite score.
            raw_score = base_score + text_intensity_boost + pos_mod + mom_mod + conf_mod
            valence = base_valence + text_valence_shift * 0.5
            energy = base_energy + text_intensity_boost * 0.5

            # 6. Enforce spike / drop rules.
            raw_score = self._apply_rules(beat_type, raw_score)

            # Clamp.
            score = max(0.0, min(1.0, raw_score))
            valence = max(-1.0, min(1.0, valence))
            energy = max(0.0, min(1.0, energy))
            curve_position = round(i / max(1, total - 1), 3) if total > 1 else 0.5

            point = EmotionPoint(
                index=i,
                beat_type=beat_type,
                text=text[:120],
                score=score,
                label=label,
                valence=valence,
                energy=energy,
                curve_position=curve_position,
                spike=is_spike,
                score_breakdown={
                    "base": base_score,
                    "text_boost": text_intensity_boost,
                    "position": pos_mod,
                    "momentum": mom_mod,
                    "confidence": conf_mod,
                },
            )
            raw_points.append(point)
            prev_score = score
            prev_beat = beat_type

        # 7. Optional smoothing pass (mild EMA to soften jagged transitions).
        if self.smoothing > 0 and len(raw_points) > 2:
            raw_points = self._smooth(raw_points)

        # 8. Build summary.
        summary = self._build_summary(raw_points)

        return EmotionCurve(points=raw_points, summary=summary)

    # ── Internal methods ─────────────────────────────────────────────

    def _detect_emotion(self, text: str) -> Tuple[str, float, float]:
        """Detect the dominant emotion from text content.

        Returns (label, valence_shift, intensity_boost).
        """
        if not text:
            return "neutral", 0.0, 0.0

        best_label = "curious"
        best_valence = 0.0
        best_boost = 0.0
        best_priority = -1

        for priority, (pattern, label, valence, boost) in enumerate(_EMOTION_PATTERNS):
            if pattern.search(text):
                if boost > best_boost or (boost == best_boost and priority < best_priority):
                    best_label = label
                    best_valence = valence
                    best_boost = boost
                    best_priority = priority

        return best_label, best_valence, best_boost

    def _apply_rules(self, beat_type: str, score: float) -> float:
        """Enforce the user's spike/peak/drop rules.

        - hooks spike  → floor at hook_floor
        - reveals spike → floor at reveal_floor
        - payoffs peak → floor at payoff_floor (global max target)
        - CTA drops   → ceiling at cta_ceiling
        """
        if beat_type == "hook":
            score = max(score, self.hook_floor)
        elif beat_type == "reveal":
            score = max(score, self.reveal_floor)
        elif beat_type == "payoff":
            score = max(score, self.payoff_floor)
        elif beat_type == "cta":
            score = min(score, self.cta_ceiling)
        return score

    def _smooth(self, points: List[EmotionPoint]) -> List[EmotionPoint]:
        """Apply mild exponential moving average to soften jagged jumps.

        Spike points (hook/reveal/payoff) are exempt from smoothing
        to preserve their sharp character.
        """
        alpha = self.smoothing
        smoothed = list(points)
        prev = smoothed[0].score

        for i in range(1, len(smoothed)):
            pt = smoothed[i]
            if pt.spike:
                # Don't smooth spikes — they should stay sharp.
                prev = pt.score
                continue
            new_score = alpha * prev + (1 - alpha) * pt.score
            # Re-apply rules after smoothing (CTA must stay low, etc.).
            new_score = self._apply_rules(pt.beat_type, new_score)
            object.__setattr__(pt, "score", max(0.0, min(1.0, new_score)))
            prev = pt.score

        return smoothed

    def _build_summary(self, points: List[EmotionPoint]) -> Dict[str, Any]:
        """Aggregate stats for the full curve."""
        if not points:
            return {"beat_count": 0}

        scores = [p.score for p in points]
        peak_idx = max(range(len(scores)), key=lambda i: scores[i])
        spike_indices = [i for i, p in enumerate(points) if p.spike]

        # Detect the dominant emotional arc shape.
        arc_shape = self._classify_arc_shape(points)

        return {
            "beat_count": len(points),
            "mean_intensity": round(sum(scores) / len(scores), 3),
            "peak_intensity": round(max(scores), 3),
            "peak_index": peak_idx,
            "peak_beat_type": points[peak_idx].beat_type,
            "trough_intensity": round(min(scores), 3),
            "intensity_range": round(max(scores) - min(scores), 3),
            "spike_count": len(spike_indices),
            "spike_indices": spike_indices,
            "arc_shape": arc_shape,
            "dominant_emotion": self._dominant_emotion(points),
            "valence_trend": self._valence_trend(points),
        }

    def _classify_arc_shape(self, points: List[EmotionPoint]) -> str:
        """Classify the overall emotional arc shape."""
        if len(points) < 3:
            return "flat"

        scores = [p.score for p in points]
        n = len(scores)
        peak_pos = scores.index(max(scores)) / max(1, n - 1)

        # Check for rising action pattern.
        first_third = sum(scores[:n // 3]) / max(1, n // 3)
        last_third = sum(scores[2 * n // 3:]) / max(1, n - 2 * n // 3)
        middle_third = sum(scores[n // 3:2 * n // 3]) / max(1, n // 3)

        if peak_pos >= 0.7:
            return "rising_climax"       # Builds to late peak (classic essay)
        if peak_pos <= 0.2:
            return "front_loaded"        # Hook dominates
        if middle_third > first_third and middle_third > last_third:
            return "mountain"            # Peak in middle
        if first_third > middle_third and last_third > middle_third:
            return "valley"              # Dips in middle
        if max(scores) - min(scores) < 0.15:
            return "flat"                # Low dynamic range
        return "wave"                    # Multiple ups and downs

    def _dominant_emotion(self, points: List[EmotionPoint]) -> str:
        """Most frequently occurring emotion label."""
        counts: Dict[str, int] = {}
        for p in points:
            counts[p.label] = counts.get(p.label, 0) + 1
        if not counts:
            return "neutral"
        return max(counts, key=lambda k: counts[k])

    def _valence_trend(self, points: List[EmotionPoint]) -> str:
        """Overall valence direction: positive, negative, or mixed."""
        if len(points) < 2:
            return "neutral"
        avg_valence = sum(p.valence for p in points) / len(points)
        if avg_valence > 0.15:
            return "positive"
        if avg_valence < -0.15:
            return "negative"
        return "mixed"


# ── Convenience function ─────────────────────────────────────────────


def generate_emotion_curve(
    beats: Sequence[Dict[str, Any]],
    **kwargs,
) -> EmotionCurve:
    """Shortcut: create a generator and produce the curve in one call."""
    return EmotionCurveGenerator(**kwargs).generate(beats)
