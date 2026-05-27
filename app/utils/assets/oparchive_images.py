"""Fetch character, devil fruit, and island images from oparchive.com."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import requests

logger = logging.getLogger(__name__)

OPARCHIVE_BASE_URL = os.getenv("OPARCHIVE_BASE_URL", "https://oparchive.com").rstrip("/")
OPARCHIVE_CACHE_DIR = Path(
    os.getenv(
        "OPARCHIVE_CACHE_DIR",
        Path(__file__).resolve().parent.parent / "output" / "cache" / "oparchive",
    )
)
OPARCHIVE_CACHE_TTL_SECONDS = int(os.getenv("OPARCHIVE_CACHE_TTL_SECONDS", str(24 * 3600)))
OPARCHIVE_ENABLED = os.getenv("OPARCHIVE_ENABLED", "true").lower() not in {"0", "false", "no"}
REQUEST_DELAY = float(os.getenv("OPARCHIVE_REQUEST_DELAY", "0.35"))
MIN_OPARCHIVE_SCORE = int(os.getenv("OPARCHIVE_MIN_SCORE", "10"))

DATA_ENDPOINTS = {
    "characters": "/data/characters.json",
    "devil_fruits": "/data/devil_fruits.json",
    "islands": "/data/islands.json",
}

# Map slide / Gemini queries to exact OPArchive catalog names
ISLAND_QUERY_ALIASES: Dict[str, str] = {
    "egghead": "Egghead",
    "egghead arc": "Egghead",
    "marineford": "Marineford",
    "marineford arc": "Marineford",
    "marineford war": "Marineford",
    "summit war": "Marineford",
    "egghead island": "Egghead",
    "laugh tale": "Laugh Tale",
    "laughtale": "Laugh Tale",
    "elbaf": "Elbaf",
    "wano": "Wano",
    "wano country": "Wano",
    "thriller bark": "Thriller Bark",
    "dressrosa": "Dressrosa",
    "whole cake": "Whole Cake Island",
    "enies lobby": "Enies Lobby",
    "ohara": "Ohara",
    "mary geoise": "Mary Geoise",
    "mariejois": "Mary Geoise",
    "sabaody": "Sabaody Archipelago",
    "grand line": "Grand Line",
}

CHARACTER_QUERY_ALIASES: Dict[str, str] = {
    "luffy": "Monkey D. Luffy",
    "monkey d luffy": "Monkey D. Luffy",
    "zoro": "Roronoa Zoro",
    "roronoa zoro": "Roronoa Zoro",
    "sanji": "Sanji",
    "nami": "Nami",
    "robin": "Nico Robin",
    "chopper": "Tony Tony Chopper",
    "usopp": "Usopp",
    "shanks": "Shanks",
    "ace": "Portgas D. Ace",
    "sabo": "Sabo",
    "roger": "Gol D. Roger",
    "gol d roger": "Gol D. Roger",
    "blackbeard": "Marshall D. Teach",
    "teach": "Marshall D. Teach",
    "whitebeard": "Edward Newgate",
    "kuma": "Bartholomew Kuma",
    "law": "Trafalgar D. Water Law",
    "kid": "Eustass Kid",
    "mihawk": "Dracule Mihawk",
    "vegapunk": "Vegapunk",
    "imu": "Imu",
    "joy boy": "Joy Boy",
    "gorosei": "Five Elders",
    "five elders": "Five Elders",
}

# Real people / meta topics — not in OPArchive character list; skip weak fuzzy matches
NON_CATALOG_QUERY_TERMS = frozenset(
    {
        "oda",
        "eiichiro",
        "interview",
        "author",
        "creator",
        "logo",
        "map",
        "chapter 1",
        "manga panel",
    }
)


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def _absolute_image_url(relative_path: str) -> str:
    if not relative_path:
        return ""
    if relative_path.startswith(("http://", "https://")):
        return relative_path
    return f"{OPARCHIVE_BASE_URL}/{relative_path.lstrip('/')}"


def _cache_path(dataset: str) -> Path:
    OPARCHIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return OPARCHIVE_CACHE_DIR / f"{dataset}.json"


def _load_dataset(dataset: str) -> List[Dict]:
    cache_file = _cache_path(dataset)
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < OPARCHIVE_CACHE_TTL_SECONDS:
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Invalid oparchive cache for %s; refetching", dataset)

    endpoint = DATA_ENDPOINTS.get(dataset)
    if not endpoint:
        return []

    url = f"{OPARCHIVE_BASE_URL}{endpoint}"
    try:
        time.sleep(REQUEST_DELAY)
        response = requests.get(url, timeout=30, headers={"User-Agent": "luffy-bolta-hai/1.0"})
        response.raise_for_status()
        data = response.json()
        cache_file.write_text(json.dumps(data), encoding="utf-8")
        logger.info("Cached oparchive dataset %s (%s records)", dataset, len(data))
        return data
    except Exception as exc:
        logger.warning("Failed to fetch oparchive %s: %s", dataset, exc)
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return []


@lru_cache(maxsize=1)
def _character_index() -> Dict[str, Dict]:
    index: Dict[str, Dict] = {}
    for entry in _load_dataset("characters"):
        name = entry.get("name", "")
        if not name:
            continue
        index[_normalize_key(name)] = entry
    return index


@lru_cache(maxsize=1)
def _devil_fruit_index() -> Dict[str, Dict]:
    index: Dict[str, Dict] = {}
    for entry in _load_dataset("devil_fruits"):
        for field in ("name", "english", "user"):
            key = _normalize_key(entry.get(field, ""))
            if key and key not in index:
                index[key] = entry
    return index


@lru_cache(maxsize=1)
def _island_index() -> Dict[str, Dict]:
    index: Dict[str, Dict] = {}
    for entry in _load_dataset("islands"):
        name = entry.get("name", "")
        if name:
            index[_normalize_key(name)] = entry
    return index


def _entry_to_image(entry: Dict, source: str, title_field: str = "name") -> Optional[Dict]:
    image_path = entry.get("image") or entry.get("image_pre")
    url = _absolute_image_url(image_path)
    if not url:
        return None
    title = entry.get(title_field) or entry.get("name") or source
    return {
        "title": title,
        "url": url,
        "width": 800,
        "height": 800,
        "description": entry.get("description", "") or entry.get("affiliation", ""),
        "source": source,
        "mime": "image/webp",
    }


def _score_entry(entry: Dict, query: str, entities: Sequence[str]) -> int:
    name = entry.get("name", "")
    name_key = _normalize_key(name)
    haystack = " ".join(
        filter(
            None,
            [
                name,
                entry.get("english", ""),
                entry.get("user", ""),
                entry.get("affiliation", ""),
                entry.get("description", ""),
                entry.get("location", ""),
            ],
        )
    ).lower()

    score = 0
    query_norm = _normalize_key(query)
    if name_key and name_key == query_norm:
        score += 80
    elif name_key and len(query_norm) >= 5 and query_norm in name_key:
        score += 40

    query_words = [
        w
        for w in re.findall(r"\b[\w'-]+\b", (query or "").lower())
        if len(w) > 2 and w not in {"one", "piece", "arc", "chapter"}
    ]
    name_tokens = set(re.findall(r"\b[\w'-]+\b", name.lower()))
    matched_name_tokens = sum(1 for w in query_words if w in name_tokens)
    if matched_name_tokens:
        score += matched_name_tokens * 12

    for word in query_words:
        if re.search(r"\b" + re.escape(word) + r"\b", haystack):
            score += 2

    for entity in entities:
        entity_key = _normalize_key(entity)
        if entity_key and entity_key == name_key:
            score += 50
        elif entity_key and len(entity_key) >= 5 and entity_key in name_key:
            score += 25

    if entry.get("status") == "Unknown" and score < MIN_OPARCHIVE_SCORE:
        score -= 3
    return score


def _find_best_entries(
    index: Dict[str, Dict],
    query: str,
    entities: Sequence[str],
    limit: int,
) -> List[Dict]:
    scored: List[Tuple[int, Dict]] = []
    for entry in index.values():
        score = _score_entry(entry, query, entities)
        if score >= MIN_OPARCHIVE_SCORE:
            scored.append((score, entry))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in scored[:limit]]


def fetch_oparchive_images(
    query: str,
    context_entities: Optional[Sequence[str]] = None,
    max_results: int = 5,
    used_url_hashes: Optional[Set[str]] = None,
) -> List[Dict]:
    """
    Return ranked oparchive image candidates compatible with image_slides download loop.
    """
    if not OPARCHIVE_ENABLED:
        return []

    entities = list(context_entities or [])
    combined_lower = f"{query} {' '.join(entities)}".lower()
    if any(term in combined_lower for term in NON_CATALOG_QUERY_TERMS):
        logger.debug("OPArchive skipped for non-catalog query: %s", query)
        return []

    results: List[Dict] = []
    seen_urls = set()

    def _add(entry: Dict, source: str, title_field: str = "name", bonus: int = 0) -> None:
        image = _entry_to_image(entry, source, title_field=title_field)
        if not image or image["url"] in seen_urls:
            return
        seen_urls.add(image["url"])
        image["score"] = _score_entry(entry, query, entities) + bonus
        results.append(image)

    blob = f"{query} {' '.join(entities)}".lower()
    for phrase, island_name in sorted(ISLAND_QUERY_ALIASES.items(), key=lambda x: -len(x[0])):
        if phrase in blob:
            entry = _island_index().get(_normalize_key(island_name))
            if entry:
                _add(entry, "oparchive_island", bonus=30)
    for phrase, char_name in sorted(CHARACTER_QUERY_ALIASES.items(), key=lambda x: -len(x[0])):
        if re.search(r"\b" + re.escape(phrase) + r"\b", blob):
            entry = _character_index().get(_normalize_key(char_name))
            if entry:
                _add(entry, "oparchive_character", bonus=30)

    for entity in entities:
        char = _character_index().get(_normalize_key(entity))
        if char:
            _add(char, "oparchive_character", bonus=20)
        island = _island_index().get(_normalize_key(entity))
        if island:
            _add(island, "oparchive_island", bonus=20)

    if len(results) < max_results:
        for entry in _find_best_entries(_island_index(), query, entities, max(2, max_results // 2)):
            _add(entry, "oparchive_island")

    if len(results) < max_results:
        for entry in _find_best_entries(_character_index(), query, entities, max_results):
            _add(entry, "oparchive_character")

    fruit_terms = ["fruit", "devil", "gomu", "yami", "gura", "nika", "hito"]
    if any(term in combined_lower for term in fruit_terms):
        for entry in _find_best_entries(_devil_fruit_index(), query, entities, max_results):
            _add(entry, "oparchive_devil_fruit", title_field="english")

    used = used_url_hashes or set()

    def _sort_key(img: Dict) -> tuple:
        url_hash = ""
        if img.get("url"):
            clean = img["url"].split("?")[0].split("#")[0]
            url_hash = hashlib.md5(clean.encode("utf-8")).hexdigest()
        reuse_penalty = 1 if url_hash in used else 0
        type_rank = 0 if "character" in img.get("source", "") else 1
        return (-img.get("score", 0), reuse_penalty, type_rank, img.get("title", ""))

    results.sort(key=_sort_key)
    trimmed = results[:max_results]
    if trimmed:
        logger.info(
            "OPArchive candidates for %r: %s",
            query,
            ", ".join(f"{img['title']}({img.get('score')})" for img in trimmed[:3]),
        )
    return trimmed
