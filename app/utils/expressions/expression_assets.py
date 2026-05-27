"""Resolve character expression PNGs from Vivre Card renders or fallback overlays."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from urllib.parse import quote, unquote
from typing import Dict, List, Literal, Optional, Sequence, Tuple

VivreAssetKind = Literal["character", "symbol", "location"]

from app.config import EXPRESSIONS_DIR, USE_STATIC_EXPRESSIONS_ONLY, VIVRE_CARD_ASSETS_DIR

# Filenames in app/static/expressions/ (without .png)
STATIC_EXPRESSION_LABELS = frozenset(
    {
        "neutral",
        "serious",
        "happy",
        "excited",
        "angry",
        "surprised",
        "sad",
        "worried",
        "smirking",
        "confident",
        "intense",
        "embarrassed",
    }
)

# Map Gemini / ASS labels to a static expression file
EXPRESSION_LABEL_ALIASES: Dict[str, str] = {
    "emotion": "neutral",
    "emotional": "sad",
    "calm": "neutral",
    "hype": "excited",
    "shock": "surprised",
    "fear": "worried",
    "annoyed": "angry",
    "determined": "intense",
    "battle": "intense",
    "laugh": "happy",
    "smile": "happy",
}

logger = logging.getLogger(__name__)

# Map Gemini expression labels to optional filename hints (most Vivre files are one pose per character)
EXPRESSION_FILENAME_TOKENS: Dict[str, List[str]] = {
    "neutral": ["neutral", "normal", "default", "base"],
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
    "ace": ["ace", "portgas d ace", "portgas d. ace"],
    "sabo": ["sabo"],
    "law": ["law", "trafalgar law", "trafalgar d. water law"],
    "kid": ["kid", "eustass kid"],
    "blackbeard": ["blackbeard", "teach", "marshall d teach", "marshall d. teach"],
    "whitebeard": ["whitebeard", "edward newgate"],
    "kuma": ["kuma", "bartholomew kuma"],
    "doflamingo": ["doflamingo", "donquixote doflamingo"],
    "kaido": ["kaido"],
    "big mom": ["big mom", "charlotte linlin"],
    "mihawk": ["mihawk", "dracule mihawk"],
}

# Pirate crews / emblems in Symbols & Jolly Rogers (filename stems)
SYMBOL_QUERY_ALIASES: Dict[str, List[str]] = {
    "straw hat": ["straw hat pirates", "straw hat", "mugiwara"],
    "whitebeard": ["whitebeard pirates", "whitebeard"],
    "blackbeard": ["blackbeard pirates", "blackbeard", "marshall d teach"],
    "heart": ["heart pirates", "trafalgar law", "law"],
    "kid": ["kid pirates", "eustass kid"],
    "beasts": ["beasts pirates", "kaido", "beast pirates"],
    "big mom": ["big mom pirates", "charlotte linlin", "whole cake"],
    "red hair": ["red hair pirates", "shanks"],
    "revolutionary": ["revolutionary army"],
    "marines": ["marines", "marine", "navy"],
    "baroque works": ["baroque works", "croco"],
    "donquixote": ["donquixote family", "doflamingo"],
    "sun": ["sun pirates", "fish-man pirates", "jimbei"],
    "thriller bark": ["thriller bark", "thriller bark pirates"],
    "impel down": ["impel down"],
    "arabasta": ["arabasta", "alabasta"],
    "kozuki": ["kozuki family", "wano"],
}

# Islands / places in misc/
LOCATION_QUERY_ALIASES: Dict[str, List[str]] = {
    "skypiea": ["skypiea", "shandia", "shandora", "god's temple", "upper yard"],
    "wano": ["wano", "ono", "flower capital"],
    "dressrosa": ["dressrosa", "dress rosa"],
    "whole cake": ["whole cake island", "whole cake"],
    "egghead": ["egghead", "vegapunk"],
    "ohara": ["ohara"],
    "water 7": ["water 7", "water seven", "galley-la"],
    "marineford": ["marineford", "summit war"],
    "sabaody": ["sabaody", "archipelago"],
    "enies lobby": ["enies lobby"],
    "thriller bark": ["thriller bark"],
    "punk hazard": ["punk hazard"],
    "zou": ["zou", "mokomo"],
    "elbaf": ["elbaf", "elbash"],
    "laugh tale": ["laugh tale", "laughtale"],
    "mary geoise": ["mary geoise", "mariejois", "holy land"],
    "loguetown": ["loguetown", "logue town"],
    "baratie": ["baratie"],
    "arlong": ["arlong park"],
    "orange town": ["orange town"],
    "syrup village": ["syrup village"],
    "drum island": ["drum", "sakura kingdom"],
}

BRANDING_TERMS = frozenset(
    {"logo", "outro", "subscribe", "follow", "grand line map", "map", "cta", "end card"}
)
DEFAULT_BRANDING_SYMBOL = "straw hat pirates"

# Deprioritize alternate Vivre variants unless the query hints match
_VARIANT_HINTS = re.compile(
    r"\b(child|pre-timeskip|post-timeskip|hatless|falling|fake|jr\.?|young|old|"
    r"thriller bark|wano|egghead|sabaody|marineford)\b",
    re.IGNORECASE,
)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def _vivre_card_dir() -> Optional[Path]:
    """Resolved Vivre Card root (env override, then config default)."""
    raw = os.getenv("VIVRE_CARD_ASSETS_DIR", "").strip()
    candidates = []
    if raw:
        candidates.append(Path(raw).expanduser())
    configured = Path(VIVRE_CARD_ASSETS_DIR).expanduser()
    if configured not in candidates:
        candidates.append(configured)

    for path in candidates:
        if path.is_dir():
            return path.resolve()
    return None


def _index_path() -> Optional[Path]:
    root = _vivre_card_dir()
    if not root:
        return None
    return root / ".vivre_index.json"


def _asset_kind_from_parent(parent: str) -> VivreAssetKind:
    p = (parent or "").lower()
    if p == "characters":
        return "character"
    if "symbol" in p or "jolly" in p:
        return "symbol"
    if p == "misc":
        return "location"
    return "character"


def _parse_vivre_stem(stem: str) -> Tuple[str, str]:
    """
    Parse Vivre filename stems like 'Monkey D. Luffy (Thriller Bark)'.
    Returns (primary_name, parenthetical_hint).
    """
    text = (stem or "").strip()
    if "(" not in text:
        return text, ""
    idx = text.index("(")
    primary = text[:idx].strip()
    hint = text[idx + 1 :].strip().rstrip(")").strip()
    return primary, hint


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
            primary_name, variant_hint = _parse_vivre_stem(stem)
            asset_kind = _asset_kind_from_parent(parent)
            relative = str(full_path.relative_to(root))
            records.append(
                {
                    "path": str(full_path.resolve()),
                    "filename": filename.lower(),
                    "stem": stem.lower(),
                    "primary_name": primary_name.lower(),
                    "variant_hint": variant_hint.lower(),
                    "parent": parent.lower(),
                    "relative": relative,
                    "relative_key": relative.replace("\\", "/").lower(),
                    "asset_kind": asset_kind,
                    "is_character": asset_kind == "character",
                }
            )
    return records


def build_vivre_card_index(force: bool = False) -> List[Dict]:
    """Scan Vivre Card assets and write .vivre_index.json for fast lookups."""
    root = _vivre_card_dir()
    if not root:
        logger.warning(
            "Vivre Card directory not found (set VIVRE_CARD_ASSETS_DIR or add app/data/vivre-card)"
        )
        return []

    index_file = root / ".vivre_index.json"
    if not force and index_file.exists():
        try:
            cached = json.loads(index_file.read_text(encoding="utf-8"))
            if cached and isinstance(cached, list) and not _index_needs_rebuild(cached):
                return cached
            if cached:
                logger.info("Rebuilding Vivre Card index (format or path fix)")
        except json.JSONDecodeError:
            logger.warning("Invalid Vivre Card index; rebuilding")

    records = _scan_png_assets(root)
    index_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
    logger.info("Indexed %s Vivre Card PNGs under %s", len(records), root)
    return records


def ensure_vivre_card_index(force: bool = False) -> int:
    """
    Build or refresh the index when missing or older than asset files.
    Returns number of indexed PNGs.
    """
    root = _vivre_card_dir()
    if not root:
        return 0

    index_file = root / ".vivre_index.json"
    if not force and index_file.exists():
        try:
            data = json.loads(index_file.read_text(encoding="utf-8"))
            if isinstance(data, list) and _index_needs_rebuild(data):
                force = True
            elif not force:
                index_mtime = index_file.stat().st_mtime
                newest_asset = max(
                    (p.stat().st_mtime for p in root.rglob("*.png")),
                    default=index_mtime,
                )
                if newest_asset <= index_mtime:
                    return len(data)
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    return len(build_vivre_card_index(force=True))


def vivre_card_status() -> Dict:
    """Summary for health checks and debugging."""
    root = _vivre_card_dir()
    if not root:
        return {
            "enabled": False,
            "path": str(VIVRE_CARD_ASSETS_DIR),
            "indexed": 0,
            "characters": 0,
        }
    records = _load_index()
    by_kind = {
        "characters": sum(1 for r in records if r.get("asset_kind") == "character"),
        "symbols": sum(1 for r in records if r.get("asset_kind") == "symbol"),
        "locations": sum(1 for r in records if r.get("asset_kind") == "location"),
    }
    return {
        "enabled": True,
        "path": str(root),
        "indexed": len(records),
        "characters": by_kind["characters"],
        "symbols": by_kind["symbols"],
        "locations": by_kind["locations"],
        "by_kind": by_kind,
        "index_file": str(_index_path()) if _index_path() else None,
    }


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


def classify_vivre_intent(text: str) -> VivreAssetKind:
    """Guess whether a query/subtitle wants a character, crew symbol, or location still."""
    blob = (text or "").lower()
    if any(term in blob for term in BRANDING_TERMS):
        return "symbol"
    if "jolly roger" in blob or " jolly " in blob or "flag" in blob:
        return "symbol"
    if "pirates" in blob or " pirate" in blob or "crew" in blob or " fleet" in blob:
        return "symbol"
    if "emblem" in blob or "kingdom" in blob and "goa" in blob:
        return "symbol"
    for _key, aliases in SYMBOL_QUERY_ALIASES.items():
        if any(alias in blob for alias in aliases[:2]):
            if "pirates" in blob or "crew" in blob or "flag" in blob or "logo" in blob:
                return "symbol"
    for _key, aliases in LOCATION_QUERY_ALIASES.items():
        if any(alias in blob for alias in aliases):
            return "location"
    if any(
        hint in blob
        for hint in (
            "island",
            "village",
            "town",
            "kingdom",
            "arc",
            "marineford",
            "ohara",
            "baratie",
            "harbor",
            "port",
        )
    ):
        return "location"
    return "character"


def _query_blob(text: str) -> Tuple[str, str]:
    """Return (normalized, lower) search blobs for matching."""
    lower = (text or "").lower()
    return _normalize(lower), lower


def _score_symbol_record(record: Dict, query_norm: str, query_lower: str) -> int:
    if record.get("asset_kind") != "symbol":
        return 0
    blob_norm = _normalize(
        f"{record.get('primary_name', '')} {record.get('variant_hint', '')} {record.get('stem', '')}"
    )
    blob_lower = f"{record.get('primary_name', '')} {record.get('variant_hint', '')} {record.get('stem', '')}"

    if any(term in query_lower for term in BRANDING_TERMS):
        if DEFAULT_BRANDING_SYMBOL.replace(" ", "") in blob_norm.replace(" ", ""):
            return 28
        if "straw hat" in blob_lower:
            return 22

    score = 0
    for _key, aliases in SYMBOL_QUERY_ALIASES.items():
        for alias in aliases:
            alias_norm = _normalize(alias)
            if alias_norm and alias_norm in query_norm:
                if alias_norm in blob_norm or alias.lower() in blob_lower:
                    score = max(score, 22)
                elif alias_norm[:6] in blob_norm:
                    score = max(score, 14)

    for token in query_lower.replace(",", " ").split():
        token_norm = _normalize(token)
        if len(token_norm) < 4:
            continue
        if token_norm in blob_norm:
            score = max(score, 12)

    if "transparent" in record.get("variant_hint", "") or "transparent" in record.get("stem", ""):
        score -= 2
    return score


def _score_location_record(record: Dict, query_norm: str, query_lower: str) -> int:
    if record.get("asset_kind") != "location":
        return 0
    blob_norm = _normalize(
        f"{record.get('primary_name', '')} {record.get('variant_hint', '')} {record.get('stem', '')}"
    )
    blob_lower = f"{record.get('primary_name', '')} {record.get('variant_hint', '')} {record.get('stem', '')}"

    score = 0
    for _key, aliases in LOCATION_QUERY_ALIASES.items():
        for alias in aliases:
            alias_norm = _normalize(alias)
            if alias_norm in query_norm or alias.lower() in query_lower:
                if alias_norm in blob_norm or alias.lower() in blob_lower:
                    score = max(score, 24)
                elif alias_norm[:5] in blob_norm:
                    score = max(score, 14)

    for token in query_lower.replace(",", " ").split():
        token_norm = _normalize(token)
        if len(token_norm) < 4:
            continue
        if token_norm in blob_norm:
            score = max(score, 16)

    if "zoomed" in record.get("variant_hint", "") or "zoomed" in record.get("stem", ""):
        score -= 3
    return score


def _rank_vivre_records(
    query: str,
    asset_kinds: Optional[Sequence[VivreAssetKind]] = None,
    min_score: int = 10,
    limit: int = 5,
) -> List[Tuple[int, Dict]]:
    query_norm, query_lower = _query_blob(query)
    if not query_norm and not query_lower.strip():
        return []

    ranked: List[Tuple[int, Dict]] = []
    for record in _load_index():
        kind = record.get("asset_kind", "character")
        if asset_kinds and kind not in asset_kinds:
            continue
        if kind == "character":
            continue
        if kind == "symbol":
            score = _score_symbol_record(record, query_norm, query_lower)
        else:
            score = _score_location_record(record, query_norm, query_lower)
        if score >= min_score:
            ranked.append((score, record))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:limit]


def suggest_vivre_assets(query: str, limit: int = 5) -> List[Dict]:
    """Return ranked Vivre PNG suggestions (symbols + locations) for a slide search query."""
    intent = classify_vivre_intent(query)
    kinds: Sequence[VivreAssetKind]
    if intent == "symbol":
        kinds = ("symbol", "location")
    elif intent == "location":
        kinds = ("location", "symbol")
    else:
        kinds = ("character", "symbol", "location")

    query_norm, query_lower = _query_blob(query)
    ranked: List[Tuple[int, Dict]] = []

    for record in _load_index():
        kind = record.get("asset_kind", "character")
        if kind not in kinds:
            continue
        if kind == "character":
            score = _character_match_score(record, query_lower.split()[0] if query_lower else "")
            for alias_list in CHARACTER_ALIASES.values():
                for alias in alias_list:
                    if _normalize(alias) in query_norm:
                        score = max(score, _character_match_score(record, alias))
        elif kind == "symbol":
            score = _score_symbol_record(record, query_norm, query_lower)
        else:
            score = _score_location_record(record, query_norm, query_lower)
        if score >= 8:
            ranked.append((score, record))

    ranked.sort(key=lambda item: item[0], reverse=True)
    results = []
    for score, record in ranked[:limit]:
        results.append(
            {
                "path": record["path"],
                "relative": record["relative"],
                "label": Path(record["path"]).stem,
                "asset_kind": record.get("asset_kind"),
                "score": score,
                "preview_url": (
                    f"/api/v1/vivre-cards/preview?relative="
                    f"{quote(record['relative'], safe='')}"
                ),
            }
        )
    return results


def resolve_vivre_asset_for_query(
    query: str,
    prefer_kinds: Optional[Sequence[VivreAssetKind]] = None,
) -> Optional[str]:
    """Best single Vivre PNG for a slide search query (symbol/location/character)."""
    ensure_vivre_card_index()
    intent = classify_vivre_intent(query)
    kinds = prefer_kinds or (
        ("symbol",) if intent == "symbol" else ("location",) if intent == "location" else ("character",)
    )

    query_norm, query_lower = _query_blob(query)
    best_score = 0
    best_path: Optional[str] = None

    for record in _load_index():
        kind = record.get("asset_kind", "character")
        if kind not in kinds:
            continue
        if kind == "character":
            for char_key, aliases in CHARACTER_ALIASES.items():
                if any(a.lower() in query_lower for a in aliases) or char_key in query_lower:
                    score = _character_match_score(record, char_key)
                    break
            else:
                score = 0
                for token in query_lower.split():
                    score = max(score, _character_match_score(record, token))
        elif kind == "symbol":
            score = _score_symbol_record(record, query_norm, query_lower)
        else:
            score = _score_location_record(record, query_norm, query_lower)

        if score > best_score:
            best_score = score
            best_path = record["path"]

    if best_path and best_score >= 10:
        return best_path
    return None


def _index_needs_rebuild(records: List[Dict]) -> bool:
    """Rebuild when index used lowercase-only paths (broken on Linux)."""
    if not records:
        return True
    sample = records[0]
    if "relative_key" not in sample:
        return True
    rel = sample.get("relative", "")
    # Legacy entries: "characters/monkey d. luffy.png" (all lower, wrong on Linux)
    return bool(rel) and rel == rel.lower() and rel.startswith(("characters/", "misc/", "symbols"))


def vivre_asset_path_from_relative(relative: str) -> Optional[Path]:
    """Safe resolve of an indexed relative path under the Vivre root."""
    root = _vivre_card_dir()
    if not root or not relative:
        return None
    rel = unquote(relative).replace("\\", "/").lstrip("/")
    if ".." in rel.split("/"):
        return None

    candidate = (root / rel).resolve()
    if candidate.is_file():
        try:
            candidate.relative_to(root.resolve())
            return candidate
        except ValueError:
            pass

    rel_key = rel.lower()
    for record in _load_index():
        if record.get("relative_key") == rel_key:
            path = Path(record["path"])
            if path.is_file():
                return path
        stored_rel = (record.get("relative") or "").replace("\\", "/")
        if stored_rel.lower() == rel_key:
            path = Path(record["path"])
            if path.is_file():
                return path
    return None


def _character_match_score(record: Dict, character: str) -> int:
    """Score how well a Vivre PNG matches a requested character name."""
    char_key = _normalize(character)
    aliases = CHARACTER_ALIASES.get(char_key, [character])
    if char_key and char_key not in [_normalize(a) for a in aliases]:
        aliases = list(aliases) + [character]

    blob_parts = [
        record.get("primary_name", ""),
        record.get("variant_hint", ""),
        record.get("stem", ""),
        record.get("parent", ""),
    ]
    blob_norm = _normalize(" ".join(blob_parts))
    blob_lower = " ".join(blob_parts).lower()

    score = 0
    for alias in aliases:
        alias_norm = _normalize(alias)
        if not alias_norm:
            continue
        if alias_norm == _normalize(record.get("primary_name", "")):
            score = max(score, 24)
        elif alias_norm in blob_norm:
            score = max(score, 16)
        elif alias.lower() in blob_lower:
            score = max(score, 12)

    if record.get("asset_kind") != "character":
        return 0

    variant_hint = record.get("variant_hint") or ""
    if variant_hint:
        score -= 4
        if _VARIANT_HINTS.search(variant_hint):
            score -= 2

    return score


def _expression_match_score(record: Dict, expression: str) -> int:
    """Bonus when filename hints match emotion (rare in this asset pack)."""
    tokens = EXPRESSION_FILENAME_TOKENS.get(expression.lower(), [expression.lower()])
    blob = f"{record.get('stem', '')} {record.get('variant_hint', '')} {record.get('filename', '')}"
    score = 0
    for token in tokens:
        if token in blob:
            score += 6
    return score


def normalize_expression_label(expression: str) -> str:
    """Map any expression label to a file stem in app/static/expressions/."""
    label = (expression or "neutral").strip().lower()
    if label in STATIC_EXPRESSION_LABELS:
        return label
    if label in EXPRESSION_LABEL_ALIASES:
        return EXPRESSION_LABEL_ALIASES[label]
    for token in re.split(r"[^a-z]+", label):
        if token in STATIC_EXPRESSION_LABELS:
            return token
        if token in EXPRESSION_LABEL_ALIASES:
            return EXPRESSION_LABEL_ALIASES[token]
    return "neutral"


def list_static_expression_files(directory: Optional[Path] = None) -> List[str]:
    """Return available expression PNG stems under static/expressions."""
    root = directory or EXPRESSIONS_DIR
    if not root.is_dir():
        return []
    return sorted(
        {
            path.stem.lower()
            for path in root.glob("*.png")
            if path.stem.lower() in STATIC_EXPRESSION_LABELS or path.stem.lower() in EXPRESSION_LABEL_ALIASES
        }
    ) or sorted(p.stem.lower() for p in root.glob("*.png"))


def resolve_static_expression_image(
    expression: str,
    expressions_dir: Optional[Path] = None,
) -> Optional[str]:
    """
    Resolve overlay PNG from app/static/expressions/{emotion}.png only.
    Character and subtitle context are ignored (one art set per emotion).
    """
    root = Path(expressions_dir) if expressions_dir else EXPRESSIONS_DIR
    if not root.is_dir():
        logger.error("Expressions directory not found: %s", root)
        return None

    label = normalize_expression_label(expression)
    candidates = [root / f"{label}.png"]
    for path in root.glob("*.png"):
        if path.stem.lower() == label:
            candidates.insert(0, path)

    for path in candidates:
        if path.is_file():
            return str(path.resolve())

    neutral = root / "neutral.png"
    if neutral.is_file():
        logger.warning(
            "Expression '%s' (mapped to '%s') not found in %s; using neutral.png",
            expression,
            label,
            root,
        )
        return str(neutral.resolve())

    pngs = sorted(root.glob("*.png"))
    if pngs:
        logger.warning("No neutral.png in %s; using %s", root, pngs[0].name)
        return str(pngs[0].resolve())
    return None


def resolve_expression_image(
    character: Optional[str],
    expression: str,
    fallback_dir: Optional[Path] = None,
    context_text: Optional[str] = None,
) -> Optional[str]:
    """
    Pick the expression overlay PNG.

    Default (EXPRESSION_ASSETS_SOURCE=static): only app/static/expressions/{emotion}.png.
    Optional legacy mode: set EXPRESSION_ASSETS_SOURCE=vivre to use Vivre Card characters.
    """
    if USE_STATIC_EXPRESSIONS_ONLY:
        path = resolve_static_expression_image(
            expression,
            expressions_dir=fallback_dir or EXPRESSIONS_DIR,
        )
        if path:
            logger.info(
                "Static expression overlay: %s -> %s",
                normalize_expression_label(expression),
                os.path.basename(path),
            )
        return path

    # Legacy Vivre Card path (image slides still use Vivre via suggest_vivre_assets)
    expression_label = (expression or "neutral").strip().lower()
    character = (character or os.getenv("NARRATOR_CHARACTER", "luffy")).strip().lower()
    context_text = context_text or ""

    ensure_vivre_card_index()

    combined = f"{context_text} {character} {expression_label}".strip()
    intent = classify_vivre_intent(combined)

    if intent in {"symbol", "location"}:
        asset_path = resolve_vivre_asset_for_query(
            combined,
            prefer_kinds=("symbol", "location") if intent == "symbol" else ("location", "symbol"),
        )
        if asset_path:
            return asset_path

    index = _load_index()
    if index:
        ranked: List[Tuple[int, str]] = []
        for record in index:
            if record.get("asset_kind") != "character":
                continue
            char_score = _character_match_score(record, character)
            if char_score <= 0:
                continue
            expr_score = _expression_match_score(record, expression_label)
            ranked.append((char_score + expr_score, record["path"]))
        if ranked:
            ranked.sort(key=lambda item: item[0], reverse=True)
            return ranked[0][1]

    if context_text:
        fallback_asset = resolve_vivre_asset_for_query(context_text)
        if fallback_asset:
            return fallback_asset

    return resolve_static_expression_image(expression, expressions_dir=fallback_dir)
