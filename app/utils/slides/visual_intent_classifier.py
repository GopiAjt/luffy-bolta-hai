"""Visual intent classification for story beats."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


VISUAL_INTENTS = (
    "DREAM",
    "FEAR",
    "WARNING",
    "MYSTERY",
    "CONTRADICTION",
    "EVIDENCE",
    "ESCALATION",
    "REVEAL",
    "PAYOFF",
    "CTA",
    "CHARACTER",
    "LOCATION",
    "SYMBOL",
)


@dataclass(frozen=True)
class IntentRule:
    intent: str
    pattern: re.Pattern
    weight: float
    signal: str


class VisualIntentClassifier:
    """Classify the visual intent of a slide beat.

    Input is the story beat type, local subtitle text, and detected entities.
    Output is a deterministic intent object with confidence and full scores.
    """

    intents: Sequence[str] = VISUAL_INTENTS

    def classify(
        self,
        beat_type: str,
        subtitle_text: str,
        entities: Sequence[str] | None = None,
    ) -> Dict:
        entities = list(entities or [])
        text = self._context_text(beat_type, subtitle_text, entities)
        scores = {intent: 0.03 for intent in self.intents}
        signals: Dict[str, List[str]] = {intent: [] for intent in self.intents}

        for intent, value, signal in self._beat_priors(beat_type):
            scores[intent] += value
            signals[intent].append(signal)

        for rule in INTENT_RULES:
            if rule.pattern.search(text):
                scores[rule.intent] += rule.weight
                signals[rule.intent].append(rule.signal)

        if entities:
            scores["CHARACTER"] += min(0.18, 0.06 * len(entities))
            signals["CHARACTER"].append("named entities present")
        if self._has_place_entity(entities) or LOCATION_RE.search(text):
            scores["LOCATION"] += 0.22
            signals["LOCATION"].append("location/entity anchor")
        if self._has_symbol_entity(entities) or SYMBOL_RE.search(text):
            scores["SYMBOL"] += 0.20
            signals["SYMBOL"].append("symbol/object anchor")

        scores = self._normalize_scores(scores)
        intent = self._best_intent(scores, beat_type)
        return {
            "intent": intent,
            "confidence": scores[intent],
            "confidence_scores": scores,
            "matched_signals": signals.get(intent, []),
            "entities": entities,
        }

    def _context_text(self, beat_type: str, subtitle_text: str, entities: Sequence[str]) -> str:
        parts = [subtitle_text or "", " ".join(entities)]
        return " ".join(part for part in parts if part).strip()

    def _beat_priors(self, beat_type: str) -> List[Tuple[str, float, str]]:
        beat = (beat_type or "").strip().lower()
        priors = {
            "hook": [("MYSTERY", 0.16, "hook curiosity prior")],
            "setup": [("CHARACTER", 0.10, "setup anchor prior")],
            "mystery": [("MYSTERY", 0.30, "story beat prior")],
            "contradiction": [("CONTRADICTION", 0.30, "story beat prior")],
            "evidence": [("EVIDENCE", 0.30, "story beat prior")],
            "escalation": [("ESCALATION", 0.30, "story beat prior")],
            "reveal": [("REVEAL", 0.30, "story beat prior")],
            "payoff": [("PAYOFF", 0.30, "story beat prior")],
            "warning": [("WARNING", 0.34, "story beat prior")],
            "cta": [("CTA", 0.40, "story beat prior")],
        }
        return priors.get(beat, [])

    def _best_intent(self, scores: Dict[str, float], beat_type: str) -> str:
        priority = self._tie_break_priority(beat_type)
        return max(scores, key=lambda intent: (scores[intent], -priority.index(intent)))

    def _tie_break_priority(self, beat_type: str) -> List[str]:
        beat = (beat_type or "").strip().lower()
        if beat == "warning":
            return ["WARNING", "FEAR", "MYSTERY", "ESCALATION", "REVEAL", "EVIDENCE", "CONTRADICTION", "DREAM", "PAYOFF", "CTA", "CHARACTER", "LOCATION", "SYMBOL"]
        if beat == "cta":
            return ["CTA", "PAYOFF", "WARNING", "DREAM", "CHARACTER", "LOCATION", "SYMBOL", "MYSTERY", "FEAR", "EVIDENCE", "REVEAL", "ESCALATION", "CONTRADICTION"]
        return ["WARNING", "FEAR", "DREAM", "REVEAL", "MYSTERY", "CONTRADICTION", "EVIDENCE", "ESCALATION", "PAYOFF", "CTA", "SYMBOL", "LOCATION", "CHARACTER"]

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        return {
            intent: round(max(0.0, min(1.0, score)), 2)
            for intent, score in scores.items()
        }

    def _has_place_entity(self, entities: Sequence[str]) -> bool:
        return any(LOCATION_RE.search(entity or "") for entity in entities)

    def _has_symbol_entity(self, entities: Sequence[str]) -> bool:
        return any(SYMBOL_RE.search(entity or "") for entity in entities)


LOCATION_RE = re.compile(
    r"\b(elbaf|wano|egghead|ohara|jaya|mary geoise|marineford|sabaody|baratie|island|village|kingdom|room)\b",
    re.I,
)
SYMBOL_RE = re.compile(
    r"\b(nika|joy boy|poneglyph|world government|five elders|flag|logo|throne|crown|fruit|sword|straw hat)\b",
    re.I,
)


INTENT_RULES: Tuple[IntentRule, ...] = (
    IntentRule("DREAM", re.compile(r"\b(dream|dreams|ambition|promise|future|freedom|pirate king|goal|hope)\b", re.I), 0.50, "dream/ambition language"),
    IntentRule("FEAR", re.compile(r"\b(fear|afraid|terrified|weak|helpless|trauma|panic|haunted|loss|pain)\b", re.I), 0.52, "fear/vulnerability language"),
    IntentRule("WARNING", re.compile(r"\b(monster|warning|danger|threat|risk|doomed|bad news|terrifying|evil|villain|destroy)\b", re.I), 0.48, "warning/threat language"),
    IntentRule("MYSTERY", re.compile(r"\b(mystery|secret|hidden|clue|unknown|question|why|impossible|missing)\b", re.I), 0.36, "mystery language"),
    IntentRule("CONTRADICTION", re.compile(r"\b(but|however|yet|unlike|instead|opposite|contradiction|doesn'?t fit|should not)\b", re.I), 0.36, "contrast language"),
    IntentRule("EVIDENCE", re.compile(r"\b(evidence|chapter|episode|panel|scene|canon|proves?|shows?|because|flashback)\b", re.I), 0.36, "proof/canon language"),
    IntentRule("ESCALATION", re.compile(r"\b(then|even|worse|bigger|stakes|consequence|until|suddenly|not only)\b", re.I), 0.34, "escalation language"),
    IntentRule("REVEAL", re.compile(r"\b(reveal|truth|actually|turns out|real reason|we realize|means|answer)\b", re.I), 0.38, "reveal language"),
    IntentRule("PAYOFF", re.compile(r"\b(payoff|this means|therefore|takeaway|changes everything|in the end|finally)\b", re.I), 0.38, "payoff language"),
    IntentRule("CTA", re.compile(r"\b(follow|subscribe|comment|tell me|share|what do you think|prove me wrong)\b", re.I), 0.52, "CTA language"),
)
