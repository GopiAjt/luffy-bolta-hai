"""Auto B-roll source routing for slide image downloads."""

from __future__ import annotations

import re
from typing import Dict, List, Literal

BrollIntent = Literal["fandom_first", "oparchive_first", "branding", "scene_panel"]

META_TERMS = frozenset(
    {
        "oda",
        "eiichiro",
        "interview",
        "author",
        "creator",
        "weekly shonen",
        "shonen jump",
        "chapter 1",
        "chapter one",
        "manga panel",
        "panel",
        "screenshot",
        "flashback",
    }
)

BRANDING_TERMS = frozenset({"logo", "outro", "grand line map", "map", "subscribe", "follow"})

ARC_LOCATION_TERMS = frozenset(
    {
        "egghead",
        "marineford",
        "wano",
        "thriller bark",
        "dressrosa",
        "elbaf",
        "laugh tale",
        "laughtale",
        "whole cake",
        "enies lobby",
        "ohara",
        "sabaody",
        "impel down",
        "skypiea",
        "water 7",
        "reverie",
        "mary geoise",
        "mariejois",
        "onigashima",
    }
)

CHARACTER_HINTS = frozenset(
    {
        "luffy",
        "zoro",
        "sanji",
        "nami",
        "robin",
        "chopper",
        "usopp",
        "franky",
        "brook",
        "jinbe",
        "shanks",
        "roger",
        "gol d",
        "blackbeard",
        "teach",
        "whitebeard",
        "ace",
        "sabo",
        "law",
        "kid",
        "kuma",
        "vegapunk",
        "imu",
        "gorosei",
        "mihawk",
    }
)


def _blob(query: str, summary: str, entities: List[str]) -> str:
    return f"{query} {summary} {' '.join(entities)}".lower()


def classify_broll_intent(
    query: str,
    summary: str = "",
    context_entities: List[str] = None,
) -> BrollIntent:
    """
    Decide which image backends to try first for a slide.

    - fandom_first: Oda, interviews, chapter panels, scene screenshots
    - oparchive_first: named characters and islands (catalog art)
    - branding: logo / map / CTA slides
    - scene_panel: multi-entity comparison or arc war scenes
    """
    text = _blob(query, summary, context_entities or [])
    if any(term in text for term in BRANDING_TERMS):
        return "branding"
    if any(term in text for term in META_TERMS):
        return "fandom_first"
    if " arc" in text or " war" in text or " vs " in text:
        return "scene_panel"
    if any(term in text for term in ARC_LOCATION_TERMS):
        # Comparison slides mention two places — prefer scene stills
        hits = sum(1 for term in ARC_LOCATION_TERMS if term in text)
        if hits >= 2 or "longer than" in text or "compared" in text:
            return "scene_panel"
        return "oparchive_first"
    if context_entities:
        return "oparchive_first"
    if any(re.search(rf"\b{re.escape(name)}\b", text) for name in CHARACTER_HINTS):
        return "oparchive_first"
    if len((query or "").split()) >= 3:
        return "scene_panel"
    return "oparchive_first"


def broll_source_order(intent: BrollIntent) -> List[str]:
    """Return download backend order for an intent."""
    orders: Dict[BrollIntent, List[str]] = {
        "fandom_first": ["fandom", "oparchive", "cse"],
        "oparchive_first": ["oparchive", "fandom", "cse"],
        "branding": ["fandom", "cse", "oparchive"],
        "scene_panel": ["fandom", "oparchive", "cse"],
    }
    return orders.get(intent, ["oparchive", "fandom", "cse"])
