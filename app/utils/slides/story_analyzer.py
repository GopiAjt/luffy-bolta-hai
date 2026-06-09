"""Rule-based story beat analysis for slide planning."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


STORY_BEAT_TYPES = (
    "hook",
    "setup",
    "mystery",
    "contradiction",
    "evidence",
    "escalation",
    "reveal",
    "payoff",
    "warning",
    "cta",
)


@dataclass(frozen=True)
class BeatRule:
    beat_type: str
    pattern: re.Pattern
    weight: float
    reason: str


class StoryAnalyzer:
    """Analyze a narration script into story beats with confidence scores.

    The analyzer is intentionally deterministic: it combines position priors
    with narration keyword/pattern matches, then returns one best beat type per
    script segment plus a confidence map for every supported beat type.
    """

    beat_types: Sequence[str] = STORY_BEAT_TYPES

    def __init__(self, min_segment_words: int = 7, max_segment_words: int = 16):
        self.min_segment_words = max(1, int(min_segment_words))
        self.max_segment_words = max(self.min_segment_words, int(max_segment_words))

    def analyze(self, script: str) -> List[Dict]:
        """Return ordered story beats for a script.

        Each item contains:
        - index
        - beat_type
        - text
        - confidence
        - confidence_scores
        - matched_signals
        """
        segments = self.segment_script(script)
        total = len(segments)
        if not total:
            return []

        beats: List[Dict] = []
        for index, text in enumerate(segments):
            scores, signals = self.score_segment(text, index, total)
            beat_type, confidence = self._best_beat(scores, index, total)
            beats.append(
                {
                    "index": index,
                    "beat_type": beat_type,
                    "text": text,
                    "confidence": confidence,
                    "confidence_scores": scores,
                    "matched_signals": signals.get(beat_type, []),
                }
            )
        return beats

    def segment_script(self, script: str) -> List[str]:
        """Split script into compact story units."""
        text = re.sub(r"\s+", " ", script or "").strip()
        if not text:
            return []

        sentences = [
            item.strip()
            for item in re.split(r"(?<=[.!?])\s+", text)
            if item.strip()
        ]
        if not sentences:
            return [text]

        segments: List[str] = []
        current: List[str] = []
        current_words = 0
        for sentence in sentences:
            words = len(sentence.split())
            should_flush = (
                current
                and current_words >= self.min_segment_words
                and current_words + words > self.max_segment_words
            )
            if should_flush:
                segments.append(" ".join(current).strip())
                current = []
                current_words = 0

            current.append(sentence)
            current_words += words

            if current_words >= self.max_segment_words:
                segments.append(" ".join(current).strip())
                current = []
                current_words = 0

        if current:
            if segments and current_words < self.min_segment_words:
                segments[-1] = f"{segments[-1]} {' '.join(current)}".strip()
            else:
                segments.append(" ".join(current).strip())
        return segments

    def score_segment(self, text: str, index: int, total: int) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """Return confidence scores and matched signals for a segment."""
        scores = {beat_type: 0.04 for beat_type in self.beat_types}
        signals: Dict[str, List[str]] = {beat_type: [] for beat_type in self.beat_types}

        for beat_type, value, reason in self._position_priors(index, total):
            scores[beat_type] += value
            signals[beat_type].append(reason)

        for rule in BEAT_RULES:
            if rule.pattern.search(text):
                scores[rule.beat_type] += rule.weight
                signals[rule.beat_type].append(rule.reason)

        if re.search(r"\bnot\b.{0,45}\bbut\b|\bbut\b.{0,45}\bactually\b", text, re.I):
            scores["contradiction"] += 0.22
            signals["contradiction"].append("not/but or but/actually contrast")

        if "?" in text:
            scores["mystery"] += 0.16
            signals["mystery"].append("question mark")
            if index == 0:
                scores["hook"] += 0.18
                signals["hook"].append("opening question")

        escalation_terms = re.findall(
            r"\b(then|even|worse|bigger|more|until|suddenly|stakes|consequence|now|not only)\b",
            text,
            re.I,
        )
        if len(escalation_terms) >= 2:
            scores["escalation"] += 0.14
            signals["escalation"].append("multiple escalation signals")

        return self._normalize_scores(scores), signals

    def _position_priors(self, index: int, total: int) -> List[Tuple[str, float, str]]:
        if total <= 1:
            return [("hook", 0.18, "single-segment opening"), ("payoff", 0.12, "single-segment close")]

        progress = index / max(1, total - 1)
        priors: List[Tuple[str, float, str]] = []
        if index == 0:
            priors.append(("hook", 0.34, "opening segment"))
        if 0.05 <= progress <= 0.35:
            priors.append(("setup", 0.18, "early setup position"))
        if 0.20 <= progress <= 0.70:
            priors.append(("evidence", 0.12, "middle evidence position"))
        if 0.35 <= progress <= 0.78:
            priors.append(("escalation", 0.10, "middle escalation position"))
        if 0.55 <= progress <= 0.88:
            priors.append(("reveal", 0.13, "late reveal position"))
        if progress >= 0.70:
            priors.append(("payoff", 0.15, "closing payoff position"))
            priors.append(("warning", 0.08, "late warning position"))
        if index == total - 1:
            priors.append(("cta", 0.22, "final segment"))
        return priors

    def _best_beat(self, scores: Dict[str, float], index: int, total: int) -> Tuple[str, float]:
        priority = self._tie_break_priority(index, total)
        beat_type = max(scores, key=lambda key: (scores[key], -priority.index(key)))
        return beat_type, scores[beat_type]

    def _tie_break_priority(self, index: int, total: int) -> List[str]:
        if index == 0:
            return ["hook", "mystery", "contradiction", "setup", "evidence", "warning", "reveal", "escalation", "payoff", "cta"]
        if index == total - 1:
            return ["cta", "payoff", "warning", "reveal", "escalation", "evidence", "contradiction", "mystery", "setup", "hook"]
        return ["reveal", "contradiction", "evidence", "mystery", "escalation", "warning", "payoff", "setup", "hook", "cta"]

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        return {
            beat_type: round(max(0.0, min(1.0, score)), 2)
            for beat_type, score in scores.items()
        }


BEAT_RULES: Tuple[BeatRule, ...] = (
    BeatRule("hook", re.compile(r"\b(what if|why|nobody|most fans|hidden truth|secret truth|this changes everything|you missed)\b", re.I), 0.34, "hook phrase"),
    BeatRule("hook", re.compile(r"\b(chapter|episode)\s+\w+", re.I), 0.12, "canon opener"),
    BeatRule("setup", re.compile(r"\b(starts?|begins?|first|before|context|inside|at the start|sets up)\b", re.I), 0.28, "setup phrase"),
    BeatRule("mystery", re.compile(r"\b(mystery|secret|hidden|clue|unknown|missing|impossible|question|why|nobody knows)\b", re.I), 0.34, "mystery signal"),
    BeatRule("contradiction", re.compile(r"\b(but|however|yet|unlike|instead|opposite|contradiction|doesn'?t fit|should not)\b", re.I), 0.34, "contrast signal"),
    BeatRule("evidence", re.compile(r"\b(chapter|episode|panel|scene|canon|shows?|says?|proves?|because|arc|page|flashback)\b", re.I), 0.34, "evidence signal"),
    BeatRule("escalation", re.compile(r"\b(then|even|worse|bigger|more|until|suddenly|stakes|consequence|now|not only)\b", re.I), 0.28, "escalation signal"),
    BeatRule("reveal", re.compile(r"\b(reveal|truth|actually|real reason|turns out|we realize|means|the answer|that is why)\b", re.I), 0.36, "reveal signal"),
    BeatRule("payoff", re.compile(r"\b(payoff|this means|therefore|so the point|takeaway|changes everything|in the end|finally)\b", re.I), 0.36, "payoff signal"),
    BeatRule("warning", re.compile(r"\b(warning|danger|threat|careful|unless|risk|terrifying|doomed|if this is true|bad news)\b", re.I), 0.34, "warning signal"),
    BeatRule("cta", re.compile(r"\b(follow|subscribe|comment|tell me|share|what do you think|prove me wrong|like and subscribe)\b", re.I), 0.52, "CTA phrase"),
)
