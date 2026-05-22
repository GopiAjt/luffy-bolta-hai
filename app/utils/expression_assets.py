"""Resolve character expression PNGs from Vivre Card renders or fallback overlays."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Map Gemini expression labels to common Vivre Card filename tokens
EXPRESSION_FILENAME_TOKENS: Dict[str, List[str]] = {
    "neutral": ["neutral", "normal", "default", "base", "serious face"],
    "happy": ["happy", "smile", "grin", "laugh", "joy"],
    "angry": ["angry", "rage", "mad", "furious", "annoyed"],
    "surprised": ["surprised", "shock", "shocked", "wow"],
    "sad": ["sad", "cry", "crying", "tear", "depressed"],
    "smirking": ["smirk", "smirking", "sly"],
    "confident": ["confident", "cool", "smug"],
    "serious": ["serious", "stern", "focused"],
    "worried": ["worried", "concern", "nervous", "anxious"],
    "intense": ["intense", "determined", "battle", "fight"],
    "excited": ["excited", "hype", "energetic", "pumped"],
    "embarrassed": ["embarrassed", "blush", "awkward"],
}

CHARACTER_ALIASES: Dict[str, List[str]] = {
    "luffy": ["luffy", "monkey d luffy", "monkey d. luffy", "mugiwara"],
    "zoro": ["zoro", "roronoa zoro"],
    "sanji": ["sanji", "vinsmoke sanji"],
    "nami": ["nami"],
    "usopp": ["usopp", "sogeking"],
    "chopper": ["chopper", "tony tony chopper"],
    "robin": ["robin", "nico robin"],
    "franky": ["franky"],
    "brook": ["brook"],
    "jinbe": ["jinbe", "jimbei"],
    "shanks": ["shanks"],
    "ace": ["ace", "portgas d ace"],
    "sabo": ["sabo"],
    "law": ["law", "trafalgar law"],
    "kid": ["kid", "eustass kid"],
    "blackbeard": ["blackbeard", "teach", "marshall d teach"],
    "whitebeard": ["whitebeard", "edward newgate"],
    "kuma": ["kuma", "bartholomew kuma"],
}


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def _vivre_card_dir() -> Optional[Path]:
    raw = os.getenv("VIVRE_CARD_ASSETS_DIR", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.exists() else None


def _index_path() -> Optional[Path]:
    root = _vivre_card_dir()
    if not root:
        return None
    return root / ".vivre_index.json"


def _scan_png_assets(root: Path) -> List[Dict]:
    records = []
    skip_dirs = {".git", "__pycache__", "node_modules"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
        for filename in filenames:
            if not filename.lower().endswith(".png"):
                continue
            full_path = Path(dirpath) / filename
            stem = Path(filename).stem
            parent = Path(dirpath).name
            records.append(
                {
                    "path": str(full_path.resolve()),
                    "filename": filename.lower(),
                    "stem": stem.lower(),
                    "parent": parent.lower(),
                    "relative": str(full_path.relative_to(root)).lower(),
                }
            )
    return records


def build_vivre_card_index(force: bool = False) -> List[Dict]:
    """Scan VIVRE_CARD_ASSETS_DIR and write .vivre_index.json for fast lookups."""
    root = _vivre_card_dir()
    if not root:
        logger.warning("VIVRE_CARD_ASSETS_DIR is not set or does not exist")
        return []

    index_file = _index_path()
    if not force and index_file and index_file.exists():
        try:
            return json.loads(index_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    records = _scan_png_assets(root)
    if index_file:
        index_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
    logger.info("Indexed %s Vivre Card PNGs under %s", len(records), root)
    return records


def _load_index() -> List[Dict]:
    index_file = _index_path()
    if index_file and index_file.exists():
        try:
            data = json.loads(index_file.read_text(encoding="utf-8"))
            if data:
                return data
        except json.JSONDecodeError:
            logger.warning("Rebuilding invalid Vivre Card index")
    return build_vivre_card_index(force=True)


def _character_match_score(record: Dict, character: str) -> int:
    aliases = CHARACTER_ALIASES.get(_normalize(character), [_normalize(character)])
    score = 0
    blob = f"{record['stem']} {record['parent']} {record['relative']}"
    for alias in aliases:
        alias_norm = _normalize(alias)
        if not alias_norm:
            continue
        if alias_norm in _normalize(blob):
            score += 10
        if alias.replace(" ", "") in blob.replace(" ", ""):
            score += 6
    return score


def _expression_match_score(record: Dict, expression: str) -> int:
    tokens = EXPRESSION_FILENAME_TOKENS.get(expression.lower(), [expression.lower()])
    score = 0
    blob = f"{record['stem']} {record['filename']}"
    for token in tokens:
        if token in blob:
            score += 8
    return score


def resolve_expression_image(
    character: Optional[str],
    expression: str,
    fallback_dir: Optional[Path] = None,
) -> Optional[str]:
    """
    Pick the best PNG for a character + expression.
    Order: Vivre Card index -> generic fallback_dir/{expression}.png
    """
    expression = (expression or "neutral").strip().lower()
    character = (character or os.getenv("NARRATOR_CHARACTER", "luffy")).strip().lower()

    index = _load_index()
    if index:
        ranked: List[Tuple[int, str]] = []
        for record in index:
            char_score = _character_match_score(record, character)
            if char_score <= 0:
                continue
            expr_score = _expression_match_score(record, expression)
            total = char_score + expr_score
            if total > 0:
                ranked.append((total, record["path"]))
        if ranked:
            ranked.sort(key=lambda item: item[0], reverse=True)
            best_score, best_path = ranked[0]
            logger.debug(
                "Vivre Card match for %s/%s: %s (score=%s)",
                character,
                expression,
                best_path,
                best_score,
            )
            return best_path

    if fallback_dir:
        generic = Path(fallback_dir) / f"{expression}.png"
        if generic.exists():
            return str(generic.resolve())
        # Last resort: any png in fallback dir
        pngs = sorted(Path(fallback_dir).glob("*.png"))
        if pngs:
            return str(pngs[0].resolve())

    return None
