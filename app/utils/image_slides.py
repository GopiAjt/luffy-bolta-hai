import re
import json
import requests
from typing import List, Dict, Optional, Set, Tuple
import os
from dotenv import load_dotenv
import logging
import time
import shutil
import hashlib
import urllib.parse
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageOps

from app.utils.oparchive_images import fetch_oparchive_images
from app.utils.image_slides_llm import call_image_slides_llm
from app.utils.broll_rules import classify_broll_intent, broll_source_order

# Set up logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Load environment variables from .env
load_dotenv()

# Fandom API settings
FANDOM_API_URL = "https://onepiece.fandom.com/api.php"
FANDOM_IMAGE_LIMIT = 5  # Max images to fetch per query
REQUEST_DELAY = 0.5  # Delay between API requests to respect rate limits

# Preferred domains for image sourcing (searched first with site-restricted queries)
PRIORITY_DOMAINS: List[str] = [
    "onepiece.fandom.com",  # Highest priority - we'll use the API directly for this
    "www.cbr.com",
    "cbr.com",
    "thelibraryofohara.com",
    "www.thelibraryofohara.com",
    "www.screenrant.com",
    "screenrant.com",
    "www.gamerant.com",
    "gamerant.com",
]

ONE_PIECE_RELEVANCE_TERMS = [
    "one piece", "luffy", "zoro", "sanji", "nami", "usopp", "chopper",
    "robin", "franky", "brook", "jinbe", "straw hat", "mugiwara",
    "blackbeard", "whitebeard", "teach", "edward newgate", "gorosei",
    "shanks", "mihawk", "kaido", "big mom", "ace", "sabo", "marineford",
    "wano", "egghead", "void century", "devil fruit", "yami yami",
    "gura gura", "nika", "joy boy", "imu", "world government"
]

CANONICAL_ENTITIES = [
    "Luffy", "Zoro", "Sanji", "Nami", "Usopp", "Chopper", "Robin", "Franky",
    "Brook", "Jinbe", "Shanks", "Mihawk", "Blackbeard", "Marshall D. Teach",
    "Whitebeard", "Ace", "Sabo", "Dragon", "Garp", "Koby", "Law", "Kid",
    "Bonney", "Kuma", "Vegapunk", "Imu", "Five Elders", "Gorosei",
    "Joy Boy", "Nika", "Mary Geoise", "Wano", "Egghead", "Elbaf",
    "Marineford", "Dressrosa", "Whole Cake Island", "Thriller Bark",
    "Enies Lobby", "Ohara", "Void Century", "Poneglyph", "Sunny Ship",
    "Grand Line", "Red Line", "World Government", "Revolutionary Army",
]

ENTITY_ALIASES = {
    "straw hat": "Luffy",
    "straw hats": "Straw Hat Pirates",
    "teach": "Marshall D. Teach",
    "blackbeard": "Marshall D. Teach",
    "gorosei": "Five Elders",
    "five elders": "Five Elders",
    "nika": "Nika",
    "joyboy": "Joy Boy",
    "mary geoise": "Mary Geoise",
    "mariejois": "Mary Geoise",
    "elbaf": "Elbaf",
    "egghead": "Egghead",
    "wano": "Wano",
}

FANDOM_PAGE_ALIASES = {
    "blackbeard": [
        "Marshall D. Teach",
        "Blackbeard Pirates",
        "Yami Yami no Mi",
        "Gura Gura no Mi",
    ],
    "teach": ["Marshall D. Teach", "Blackbeard Pirates"],
    "yami yami": ["Yami Yami no Mi", "Marshall D. Teach"],
    "gura gura": ["Gura Gura no Mi", "Edward Newgate", "Marshall D. Teach"],
    "whitebeard": ["Edward Newgate", "Whitebeard Pirates", "Gura Gura no Mi"],
    "gorosei": [
        "Five Elders",
        "Jaygarcia Saturn",
        "Shepherd Ju Peter",
        "Topman Warcury",
        "Marcus Mars",
        "Ethanbaron V. Nusjuro",
    ],
    "five elders": ["Five Elders"],
    "marineford": ["Marineford", "Marineford Arc", "Summit War of Marineford"],
    "void century": ["Void Century", "Joy Boy", "Poneglyph"],
    "grand line": ["Grand Line", "World", "Log Pose"],
    "one piece logo": ["One Piece", "One Piece (Manga)", "One Piece Wiki"],
}

GOOGLE_CSE_DISABLED = False


def parse_ass_dialogues(ass_path: str, min_words=3) -> List[Dict[str, str]]:
    """Parse .ass file and extract dialogues with timestamps.
    
    Args:
        ass_path: Path to the .ass subtitle file
        min_words: Minimum number of words to consider a line complete (lines with fewer words will be grouped)
    """
    raw_dialogues = []
    
    with open(ass_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    
    # Find the [Events] section
    events_start = False
    for line in lines:
        line = line.strip()
        if line.lower() == '[events]':
            events_start = True
            continue
        if not events_start:
            continue
            
        # Parse dialogue lines (format: Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text)
        if line.lower().startswith('dialogue:'):
            parts = line.split(',', 9)  # Split into max 10 parts
            if len(parts) >= 10:
                start_time = parts[1].strip()
                end_time = parts[2].strip()
                text = parts[9].strip()
                # Remove override tags and special formatting
                text = re.sub(r'\{.*?\}', '', text)  # Remove {\an8} and similar
                text = re.sub(r'\\[nNh]', ' ', text)  # Replace newlines with spaces
                text = re.sub(r'\\N', ' ', text)  # Replace \N with space
                text = re.sub(r'\\n', ' ', text)  # Replace \n with space
                text = re.sub(r'\\h', ' ', text)  # Replace \h with space
                text = re.sub(r'\\[^ ]*', '', text)  # Remove other backslash commands
                text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
                
                if text:  # Only add non-empty dialogues
                    raw_dialogues.append({
                        'start': start_time,
                        'end': end_time,
                        'text': text,
                        'word_count': len(text.split())
                    })
    
    # Group short lines with adjacent lines
    grouped_dialogues = []
    i = 0
    n = len(raw_dialogues)
    
    while i < n:
        current = raw_dialogues[i].copy()
        
        # If current line is short, try to group with next lines
        if current['word_count'] < min_words and i < n - 1:
            j = i + 1
            # Keep adding next lines until we have enough words or reach end
            while j < n and len(current['text'].split()) < min_words:
                next_dialogue = raw_dialogues[j]
                current['text'] += ' ' + next_dialogue['text']
                current['end'] = next_dialogue['end']  # Update end time to the last grouped line
                current['word_count'] = len(current['text'].split())
                j += 1
            i = j  # Skip the lines we've grouped
        else:
            i += 1
            
        grouped_dialogues.append({
            'start': current['start'],
            'end': current['end'],
            'text': current['text']
        })
    
    return grouped_dialogues


def group_dialogues(dialogues: List[Dict]) -> List[Dict]:
    """Return the original dialogues without any grouping.
    
    This function preserves the original subtitle lines and their timestamps
    exactly as they appear in the input file.
    """
    if not dialogues:
        return []
    
    # Just ensure no overlaps in timestamps
    for i in range(len(dialogues) - 1):
        current_end = parse_time(dialogues[i]['end'])
        next_start = parse_time(dialogues[i + 1]['start'])
        
        # If current end is after next start, adjust it to avoid overlap
        if current_end > next_start:
            # Set current end to be just before next start (1ms before)
            adjusted_end = next_start - 0.001
            dialogues[i]['end'] = format_time(adjusted_end)
    
    # Return the original dialogues with only the necessary adjustments
    return [
        {
            'start': d['start'],
            'end': d['end'],
            'text': d['text'].strip()
        }
        for d in dialogues
        if d['text'].strip()  # Skip empty lines
    ]

def parse_time(t):
    """Converts '0:00:06.28' to seconds."""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def format_time(seconds):
    """Converts seconds to 'H:MM:SS.ms' format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def filter_irrelevant_summaries(slides):
    # List of phrases that indicate a slide should be merged with the previous
    skip_phrases = [
        "get this.", "it's both.", "trick question.", "but here's the question.",
        "but that's not all.", "trick question", "get this", "it's both", "but here's the question", "but that's not all"
    ]
    filtered = []
    for slide in slides:
        summary = slide.get("summary", "").strip().lower()
        # If summary is in skip_phrases, merge interval with previous
        if any(phrase in summary for phrase in skip_phrases):
            if filtered:
                # Extend the end_time of the previous slide
                filtered[-1]["end_time"] = slide["end_time"]
        else:
            filtered.append(slide)
    return filtered


def fill_gaps_in_slides(slides: List[Dict], total_duration: float) -> List[Dict]:
    filled_slides = []
    current_time = 0.0

    for slide in slides:
        slide_start = parse_time(slide["start_time"])
        slide_end = parse_time(slide["end_time"])

        # If there's a gap before the current slide, fill it
        if slide_start > current_time:
            gap_duration = slide_start - current_time
            # Create a blank slide for the gap
            filled_slides.append({
                "start_time": format_time(current_time),
                "end_time": format_time(slide_start),
                "summary": "Silence/Transition",
                "image_search_query": "One Piece scenery calm"  # Generic query for silent parts
            })

        filled_slides.append(slide)
        current_time = slide_end

    # Ensure the last slide extends to the total_duration
    if current_time < total_duration:
        filled_slides.append({
            "start_time": format_time(current_time),
            "end_time": format_time(total_duration),
            "summary": "Ending/Outro",
            "image_search_query": "One Piece outro credits"  # Generic query for the end
        })

    return filled_slides


def _extract_context_entities(text: str, limit: int = 8) -> List[str]:
    text_lower = (text or "").lower()
    entities = []

    for alias, canonical in ENTITY_ALIASES.items():
        if alias in text_lower and canonical not in entities:
            entities.append(canonical)

    for entity in CANONICAL_ENTITIES:
        pattern = r"\b" + re.escape(entity.lower()) + r"\b"
        if re.search(pattern, text_lower) and entity not in entities:
            entities.append(entity)

    return entities[:limit]


def _dedupe_query_words(query: str) -> str:
    words = query.split()
    seen = set()
    unique = []
    for word in words:
        key = word.lower()
        if key not in seen:
            unique.append(word)
            seen.add(key)
    return " ".join(unique)


def _meaningful_query_text(query: str) -> str:
    """Keep middle initials (Monkey D. Luffy) when building search phrases."""
    words = query.split()
    merged: List[str] = []
    for word in words:
        if re.fullmatch(r"[A-Za-z]\.?", word) and merged:
            merged[-1] = f"{merged[-1]} {word}"
        else:
            merged.append(word)
    return " ".join(merged)


def _query_without_one_piece(query: str) -> str:
    words = _meaningful_query_text(query).split()
    return " ".join(word for word in words if word.lower() not in {"one", "piece"})


def _sanitize_image_query(query: str, summary: str, context_entities: List[str]) -> str:
    """Normalize Gemini queries without re-prefixing on every pipeline pass."""
    raw = (query or "").strip()
    combined_lower = f"{raw} {summary}".lower()

    # Preserve canonical transition / branding queries.
    for term in ("one piece logo", "grand line map", "sunny ship", "one piece outro"):
        if term in combined_lower or term in raw.lower():
            return "One Piece Logo" if "logo" in term or "outro" in term else raw or "One Piece Logo"

    query = re.sub(
        r"\b(one piece anime|anime screenshot|manga panel|fan art|wallpaper|dramatic|cinematic|close up)\b",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    query = re.sub(r"[^A-Za-z0-9 .'-]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()

    # Idempotent: do not stack script-wide anchors if they are already present.
    if context_entities:
        anchor = " ".join(context_entities[:2])
        if anchor and query.lower().startswith(anchor.lower()):
            return _dedupe_query_words(query)

    combined = f"{query} {summary}".lower()
    chosen_entities = [
        entity
        for entity in context_entities
        if re.search(r"\b" + re.escape(entity.lower()) + r"\b", combined)
        or any(
            re.search(r"\b" + re.escape(alias) + r"\b", combined) and canonical == entity
            for alias, canonical in ENTITY_ALIASES.items()
        )
    ]

    query_without_one_piece = _query_without_one_piece(query)

    if chosen_entities:
        anchored = " ".join(chosen_entities[:2])
        if query_without_one_piece:
            query_lower = query_without_one_piece.lower()
            anchored_lower = anchored.lower()
            if query_lower == anchored_lower or query_lower.startswith(anchored_lower + " "):
                return _dedupe_query_words(query_without_one_piece)
            missing = [
                entity
                for entity in chosen_entities[:2]
                if entity.lower() not in query_lower
            ]
            if missing:
                return _dedupe_query_words(f"{' '.join(missing)} {query_without_one_piece}".strip())
            return _dedupe_query_words(query_without_one_piece)
        return anchored

    if query_without_one_piece:
        # Keep multi-word proper nouns intact (Eiichiro Oda, Gol D. Roger) for Fandom/OPArchive.
        if len(query_without_one_piece.split()) >= 2:
            return _dedupe_query_words(query_without_one_piece)
        return _dedupe_query_words(f"One Piece {query_without_one_piece}".strip())
    return "One Piece Logo"


def _slide_context_entities(slide: Dict) -> List[str]:
    """Per-slide entities for image matching (avoid script-wide anchor bleed)."""
    blob = " ".join(
        filter(
            None,
            [
                slide.get("summary", ""),
                slide.get("image_search_query", ""),
            ],
        )
    )
    return _extract_context_entities(blob, limit=6)


_SLIDE_REQUIRED_KEYS = ("start_time", "end_time", "summary", "image_search_query")


def _is_slide_dict(value: object) -> bool:
    return isinstance(value, dict) and all(key in value for key in _SLIDE_REQUIRED_KEYS)


def _normalize_parsed_slides(parsed: object) -> List[Dict]:
    """Coerce LLM JSON into a list of slide dicts."""
    if isinstance(parsed, list):
        slides = [item for item in parsed if _is_slide_dict(item)]
        if slides:
            return slides
        raise ValueError("LLM JSON array contained no valid slide objects")

    if _is_slide_dict(parsed):
        return [parsed]

    if isinstance(parsed, dict):
        for key in ("slides", "segments", "items", "data"):
            nested = parsed.get(key)
            if isinstance(nested, list):
                return _normalize_parsed_slides(nested)

    raise ValueError(
        "LLM must return a JSON array of slides, not a single object or other shape"
    )


def _extract_json_array_candidates(text: str) -> List[str]:
    """Find JSON array substrings when the model wraps or truncates output."""
    candidates = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    start = stripped.find("[")
    while start != -1:
        depth = 0
        for idx in range(start, len(stripped)):
            char = stripped[idx]
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    candidates.append(stripped[start : idx + 1])
                    break
        start = stripped.find("[", start + 1)
    return candidates


def parse_image_slides_llm_response(response_text: str) -> List[Dict]:
    """Parse and normalize slide JSON from any supported LLM backend."""
    json_str = (response_text or "").strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    elif json_str.startswith("```"):
        json_str = json_str[3:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    json_str = json_str.strip()

    errors = []
    for candidate in _extract_json_array_candidates(json_str):
        try:
            parsed = json.loads(candidate)
            return _normalize_parsed_slides(parsed)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            errors.append(str(exc))

    # Last resort: single object without array wrapper
    try:
        parsed = json.loads(json_str)
        return _normalize_parsed_slides(parsed)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        errors.append(str(exc))

    detail = "; ".join(errors[:3])
    raise ValueError(f"Could not parse image slides JSON from LLM response: {detail}")


def _image_slides_rules_block(context_entities: List[str]) -> str:
    entities_line = (
        ", ".join(context_entities)
        if context_entities
        else "None detected; use One Piece Logo or Grand Line Map for transitions"
    )
    return (
        "CRITICAL RULES:\n"
        "1. Every query MUST reference at least one canonical One Piece entity.\n"
        "2. Use OPArchive-friendly names: exact character (Monkey D. Luffy, Gol D. Roger) or island (Egghead, Marineford, Laugh Tale) — not 'Egghead Arc'.\n"
        "3. Fandom works for Oda interviews, chapter panels, and scene screenshots (Eiichiro Oda, Luffy Chapter 1).\n"
        "4. No adjectives like 'dramatic' or 'cinematic'.\n"
        "5. Transitions → 'One Piece Logo' or 'Grand Line Map'.\n\n"
        f"SCRIPT-WIDE ENTITIES: {entities_line}\n"
        "Match each subtitle; do not invent unrelated characters.\n\n"
        "MERGE short fragments (<4 words) with neighbors. MAX segment 3.5s; split longer ones.\n"
        "Timestamps must be continuous within this batch.\n\n"
        "OUTPUT: ONLY a JSON array. Each item: start_time, end_time, summary, image_search_query "
        "(2–5 words, canonical entity names).\n"
    )


def _build_image_slides_full_prompt(
    timestamped_dialogues: List[Dict],
    context_entities: List[str],
) -> str:
    raw_subtitles = "\n".join(
        f"{d['start']} - {d['end']}: {d['text']}" for d in timestamped_dialogues
    )
    return (
        "You are an expert video content designer specializing in One Piece. "
        "Map timestamped subtitles to image search queries as a JSON array.\n\n"
        f"{_image_slides_rules_block(context_entities)}\n"
        "EXAMPLE:\n"
        '[{"start_time":"0:00:03.10","end_time":"0:00:06.60","summary":"Shanks at Mary Geoise",'
        '"image_search_query":"Shanks Mary Geoise"}]\n\n'
        "Subtitles:\n"
        f"{raw_subtitles}\n"
    )


def _script_context_summary(dialogues: List[Dict]) -> Dict:
    full_text = " ".join(d.get("text", "") for d in dialogues)
    entities = _extract_context_entities(full_text)
    return {
        "text": full_text[:3000],
        "entities": entities,
        "fallback_query": " ".join(entities[:2]) if entities else "One Piece Logo",
    }


def generate_gemini_image_slides(ass_path: str, out_path: str, total_duration: float, image_dir: str = None) -> str:
    # Read and group all dialogues as before
    dialogues = parse_ass_dialogues(ass_path)
    grouped_dialogues = group_dialogues(dialogues)
    
    # Store the original timestamps
    timestamped_dialogues = [
        {
            'start': d['start'],
            'end': d['end'],
            'text': d['text'].strip()
        }
        for d in grouped_dialogues
    ]
    
    # Prepare the prompt for Gemini with raw subtitles and timestamps
    raw_subtitles = "\n".join(f"{d['start']} - {d['end']}: {d['text']}" for d in timestamped_dialogues)
    script_context = _script_context_summary(timestamped_dialogues)
    context_entities = script_context["entities"]
    logger.info(f"Raw subtitles with timestamps:\n{raw_subtitles}")
    
    try:
        prompt = _build_image_slides_full_prompt(timestamped_dialogues, context_entities)
        response_text = call_image_slides_llm(prompt)
        if not response_text:
            raise ValueError("Empty LLM response for image slides")

        logger.info(f"Raw Gemini response:\n{response_text}")
        gemini_slides = parse_image_slides_llm_response(response_text)
        if len(gemini_slides) < 2 and len(timestamped_dialogues) > 3:
            raise ValueError(
                f"Gemini returned only {len(gemini_slides)} slide(s) for "
                f"{len(timestamped_dialogues)} subtitle lines — response likely truncated."
            )
        logger.info(f"Successfully parsed {len(gemini_slides)} slides from Gemini response")

        final_slides = []

        for gemini_slide in gemini_slides:
            slide_entities = _extract_context_entities(
                f"{gemini_slide.get('summary', '')} {gemini_slide.get('image_search_query', '')}",
                limit=6,
            )
            slide = {
                'start_time': gemini_slide['start_time'],
                'end_time': gemini_slide['end_time'],
                'summary': gemini_slide['summary'],
                'image_search_query': _sanitize_image_query(
                    gemini_slide['image_search_query'],
                    gemini_slide['summary'],
                    slide_entities or context_entities,
                ),
                'context_entities': slide_entities or context_entities,
            }
            if image_dir:
                os.makedirs(image_dir, exist_ok=True)
                slide_hash = hashlib.md5(slide['image_search_query'].encode()).hexdigest()[:8]
                slide['image_path'] = os.path.join(
                    image_dir, f"slide_{len(final_slides)+1:03d}_{slide_hash}.jpg"
                )
            final_slides.append(slide)

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(final_slides, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully generated {len(final_slides)} slides at {out_path}")

        if image_dir:
            download_images_for_slides(out_path, image_dir)

        return out_path

    except (TimeoutError, ConnectionError, RuntimeError, ValueError) as e:
        logger.error(f"Error in generate_gemini_image_slides: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_gemini_image_slides: {e}")
        raise


def generate_image_slides(ass_path: str, out_path: str, total_duration: float, image_dir: str = None) -> str:
    """
    Backward-compatible alias for generate_gemini_image_slides.
    """
    return generate_gemini_image_slides(ass_path, out_path, total_duration, image_dir)


@lru_cache(maxsize=100)
def _google_image_search_impl(query: str, api_key: str, cse_id: str, num_results: int = 10) -> List[Dict]:
    """Internal implementation of Google image search with caching.
    
    Args:
        query: Search query string
        api_key: Google API key
        cse_id: Custom Search Engine ID
        num_results: Maximum number of results to return
        
    Returns:
        List of image search results
    """
    url = "https://www.googleapis.com/customsearch/v1"
    
    params = {
        "q": query,
        "cx": cse_id,
        "key": api_key,
        "searchType": "image",
        "num": num_results,
        "safe": "active"
    }
    
    global GOOGLE_CSE_DISABLED
    if GOOGLE_CSE_DISABLED:
        logger.debug("Skipping Google CSE query because the API is rate-limited/forbidden for this run.")
        return []

    try:
        logger.debug(f"Executing search: {query}")
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        results = resp.json()
        return results.get("items", [])
    except requests.exceptions.RequestException as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        detail = f"HTTP {status}" if status else e.__class__.__name__
        logger.error(f"Google Image Search API error for query '{query}': {detail}")
        if status in (403, 429):
            GOOGLE_CSE_DISABLED = True
            logger.warning(
                "Disabling Google CSE for the rest of this image download run "
                "because it returned %s.", detail
            )
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response for query '{query}': {str(e)}")
        return []

def google_image_search(query: str, api_key: Optional[str] = None, cse_id: Optional[str] = None, num_results: int = 10) -> List[Dict]:
    """Search for images using Google Custom Search API with caching and retry logic.
    
    Args:
        query: Search query string
        api_key: Google API key (defaults to GOOGLE_API_KEY env var)
        cse_id: Custom Search Engine ID (defaults to GOOGLE_SEARCH_ENGINE_ID env var)
        num_results: Maximum number of results to return
        
    Returns:
        List of image search results
    """
    # Prefer a dedicated CSE API key if provided; fall back to GOOGLE_API_KEY
    api_key = api_key or os.getenv("GOOGLE_CSE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    cse_id = cse_id or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    
    if not api_key or not cse_id:
        logger.error("Missing required API key or CSE ID")
        return []
    
    # Try with the exact query first
    items = _google_image_search_impl(query, api_key, cse_id, num_results)
    
    # If no results, try a broader search by removing any special characters
    if not items and "site:" not in query and any(c in query for c in ':"'):
        simplified_query = re.sub(r'[:"]', '', query)
        logger.debug(f"No results, trying simplified query: {simplified_query}")
        items = _google_image_search_impl(simplified_query, api_key, cse_id, num_results)
    
    return items


def _get_image_hash(image_url: str) -> str:
    """Generate a hash for an image URL to detect duplicates."""
    # Remove query parameters and fragments
    clean_url = image_url.split('?')[0].split('#')[0]
    # Use MD5 hash of the URL as the image identifier
    return hashlib.md5(clean_url.encode('utf-8')).hexdigest()

def _is_duplicate_image(item: Dict, downloaded_hashes: Set[str]) -> Tuple[bool, str]:
    """Check if an image is a duplicate based on its URL hash."""
    image_url = item.get('link')
    if not image_url:
        return True, ""
        
    image_hash = _get_image_hash(image_url)
    return (image_hash in downloaded_hashes, image_hash)

def download_catalog_image(
    url: str,
    save_path: str,
    timeout: int = 10,
    min_width: int = 200,
    min_height: int = 150,
    upscale_to: Tuple[int, int] = (1080, 1920),
) -> bool:
    """
    Download OPArchive / catalog art (often small WebP cards) and upscale for vertical video.
    """
    temp_path = f"{save_path}.catalog"
    if not download_image(url, temp_path, timeout=timeout, min_width=min_width, min_height=min_height):
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False
    try:
        target_w, target_h = upscale_to
        from app.utils.image_composition import compose_vertical_subject

        with Image.open(temp_path) as image:
            image = ImageOps.exif_transpose(image.convert("RGB"))
            composed = compose_vertical_subject(image, (target_w, target_h))
            composed.save(save_path, "JPEG", quality=92, optimize=True)
        os.remove(temp_path)
        return True
    except Exception as exc:
        logger.warning("Catalog upscale failed for %s: %s", url, exc)
        if os.path.exists(temp_path):
            try:
                os.rename(temp_path, save_path)
                return True
            except OSError:
                os.remove(temp_path)
        return False


def download_image(url: str, save_path: str, timeout: int = 10, min_width: int = 360, min_height: int = 260) -> bool:
    """Download an image from a URL to a given path with improved error handling.
    
    Args:
        url: Image URL to download
        save_path: Local path to save the image
        timeout: Request timeout in seconds
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    # Validate inputs
    if not url or not isinstance(url, str):
        logger.error(f"Invalid URL: {url}")
        return False
        
    # Clean up the URL
    url = url.strip()
    
    # Add protocol if missing
    if not url.startswith(('http://', 'https://')):
        url = f'https://{url}'
    
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Download the image with a user agent header
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        logger.debug(f"Downloading image from: {url}")
        with requests.get(url, stream=True, timeout=timeout, headers=headers) as resp:
            resp.raise_for_status()
            
            # Check content type
            content_type = resp.headers.get('content-type', '').lower()
            if 'image' not in content_type:
                logger.error(f"URL does not point to an image: {url} (Content-Type: {content_type})")
                return False
                
            # Skip GIF files
            if 'gif' in content_type:
                logger.info(f"Skipping GIF file: {url}")
                return False
                
            temp_path = f"{save_path}.download"

            # Download in chunks to handle large files
            with open(temp_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)

            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                logger.error(f"Downloaded file is empty: {url}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return False

            try:
                with Image.open(temp_path) as image:
                    image = ImageOps.exif_transpose(image)
                    width, height = image.size
                    if width < min_width or height < min_height:
                        logger.info(f"Skipping small image {width}x{height}: {url}")
                        os.remove(temp_path)
                        return False

                    aspect_ratio = width / max(height, 1)
                    if aspect_ratio < 0.35 or aspect_ratio > 3.0:
                        logger.info(f"Skipping awkward aspect ratio {aspect_ratio:.2f}: {url}")
                        os.remove(temp_path)
                        return False

                    if image.mode not in ("RGB", "L"):
                        image = image.convert("RGB")
                    elif image.mode == "L":
                        image = image.convert("RGB")
                    image.save(save_path, "JPEG", quality=92, optimize=True)
            except Exception as exc:
                logger.warning(f"Downloaded URL was not a usable still image: {url} ({exc})")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return False

            if os.path.exists(temp_path):
                os.remove(temp_path)

            file_size = os.path.getsize(save_path)
                
            logger.info(f"Successfully downloaded {os.path.basename(save_path)} ({file_size/1024:.1f} KB)")
            return True
            
    except requests.exceptions.Timeout:
        logger.error(f"Request timed out after {timeout} seconds: {url}")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error {e.response.status_code} for URL: {url}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {url}: {str(e)}")
    except IOError as e:
        logger.error(f"I/O error saving to {save_path}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error downloading {url}: {str(e)}")
        logger.debug(f"Error details:", exc_info=True)
    
    # Clean up partially downloaded file if it exists
    if os.path.exists(save_path):
        try:
            os.remove(save_path)
        except OSError as e:
            logger.warning(f"Failed to clean up partially downloaded file {save_path}: {str(e)}")
    
    return False


def _generate_fallback_queries(base_query: str) -> List[str]:
    """Generate fallback search queries if the original query doesn't yield results."""
    # Remove any special characters and split into words
    words = re.findall(r'\b\w+\b', base_query.lower())
    
    # Common One Piece related terms to try in fallback queries
    fallbacks = [
        f"One Piece {base_query}",  # Original with One Piece prefix
        base_query,  # Original query as is
    ]
    
    # Add character names if detected
    characters = [w for w in words if w.lower() in [
        'luffy', 'zoro', 'sanji', 'nami', 'usopp', 'chopper', 
        'robin', 'franky', 'brook', 'jinbe', 'mihawk', 'shanks',
        'kaido', 'big mom', 'blackbeard', 'ace', 'sabo'
    ]]
    
    if characters:
        fallbacks.append(f"One Piece {' '.join(characters)}")
    
    # Add variations with different combinations
    if len(words) > 1:
        fallbacks.extend([
            f"One Piece {words[0]} {words[-1]}",  # First and last word
            f"One Piece {words[0]}",  # Just first word
        ])
    
    # Add a generic fallback if nothing else works
    fallbacks.append("One Piece")
    
    # Remove duplicates while preserving order
    seen = set()
    return [f for f in fallbacks if not (f in seen or seen.add(f))]


def _unique_queries(*query_groups: List[str]) -> List[str]:
    """Return unique non-empty queries while preserving order."""
    queries = []
    seen = set()
    for group in query_groups:
        for query in group:
            normalized = query.strip()
            if normalized and normalized not in seen:
                queries.append(normalized)
                seen.add(normalized)
    return queries


def _fandom_page_aliases(query: str) -> List[str]:
    """Return known One Piece Wiki page titles related to a search query."""
    query_lower = query.lower()
    aliases = []
    for term, page_titles in FANDOM_PAGE_ALIASES.items():
        if term in query_lower:
            aliases.extend(page_titles)
    return _unique_queries(aliases)


def _image_relevance_score(img: Dict, query: str) -> int:
    """Score a Fandom image using title/description/query overlap."""
    haystack = " ".join([
        img.get("title", ""),
        img.get("description", ""),
        img.get("url", ""),
    ]).lower()
    words = [w for w in re.findall(r'\b[\w-]+\b', query.lower()) if len(w) > 2]
    score = sum(3 for word in words if word in haystack)

    for term in ONE_PIECE_RELEVANCE_TERMS:
        if term in haystack:
            score += 1

    noisy_terms = ["logo", "icon", "symbol", "wiki", "placeholder", "question mark"]
    score -= sum(4 for term in noisy_terms if term in haystack)
    score += min(img.get("width", 0), 2000) // 500
    return score


def _search_item_relevance_score(item: Dict, query: str, summary: str = "") -> int:
    haystack = " ".join([
        item.get("title", ""),
        item.get("displayLink", ""),
        item.get("snippet", ""),
        item.get("htmlSnippet", ""),
        item.get("image", {}).get("contextLink", "") if isinstance(item.get("image"), dict) else "",
    ]).lower()
    query_words = _query_keywords(f"{query} {summary}")
    score = sum(4 for word in query_words if word in haystack)

    if any(term in haystack for term in ONE_PIECE_RELEVANCE_TERMS):
        score += 8
    if "onepiece.fandom.com" in haystack:
        score += 8
    elif any(domain in haystack for domain in PRIORITY_DOMAINS):
        score += 4

    noisy_terms = ["cosplay", "toy", "figure", "tattoo", "shirt", "poster", "fanart", "wallpaper", "logo"]
    score -= sum(5 for term in noisy_terms if term in haystack)

    image_meta = item.get("image", {}) if isinstance(item.get("image"), dict) else {}
    width = int(image_meta.get("width", 0) or 0)
    height = int(image_meta.get("height", 0) or 0)
    if width >= 640 and height >= 360:
        score += 4
    if width and height:
        aspect_ratio = width / max(height, 1)
        if 0.55 <= aspect_ratio <= 2.2:
            score += 2
        else:
            score -= 3

    return score


def _rank_search_items(items: List[Dict], query: str, summary: str = "") -> List[Dict]:
    return sorted(
        items,
        key=lambda item: _search_item_relevance_score(item, query, summary),
        reverse=True,
    )


def _query_keywords(query: str) -> Set[str]:
    """Extract useful words for comparing slide image queries."""
    stopwords = {"one", "piece", "the", "and", "for", "with", "from", "logo", "image"}
    return {
        word
        for word in re.findall(r'\b[\w-]+\b', query.lower())
        if len(word) > 2 and word not in stopwords
    }


def _best_previous_image(query: str, previous_images: List[Dict]) -> Optional[str]:
    """Pick the most relevant prior downloaded image for this query."""
    if not previous_images:
        return None

    query_words = _query_keywords(query)
    best = max(
        previous_images,
        key=lambda item: (
            len(query_words & item.get("keywords", set())),
            item.get("index", -1),
        ),
    )
    return best.get("path")

def search_fandom_pages(query: str, limit: int = 5) -> List[Dict]:
    """Search for pages on One Piece Fandom."""
    searches = [
        f'intitle:"{query}"',
        query,
    ]
    pages = []
    seen_titles = set()

    for search in searches:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": search,
            "format": "json",
            "srlimit": limit,
            "srwhat": "text"
        }

        try:
            response = requests.get(FANDOM_API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            for page in data.get("query", {}).get("search", []):
                title = page.get("title")
                if title and title not in seen_titles:
                    pages.append(page)
                    seen_titles.add(title)
        except Exception as e:
            logger.error(f"Fandom search error for '{query}': {str(e)}")

        if len(pages) >= limit:
            break

    return pages[:limit]


def get_fandom_page_title(page_title: str) -> Optional[str]:
    """Return the canonical title if a Fandom page exists."""
    params = {
        "action": "query",
        "titles": page_title,
        "format": "json",
        "redirects": 1,
    }

    try:
        response = requests.get(FANDOM_API_URL, params=params, timeout=10)
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        for page in pages.values():
            if "missing" not in page and page.get("title"):
                return page["title"]
    except Exception as e:
        logger.debug(f"Fandom page lookup error for '{page_title}': {str(e)}")
    return None

def get_page_images(page_title: str) -> List[Dict]:
    """Get all images from a specific Fandom page."""
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "images",
        "format": "json",
        "imlimit": 50
    }
    
    try:
        response = requests.get(FANDOM_API_URL, params=params, timeout=10)
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        return [img["title"] for page in pages.values() 
                for img in page.get("images", []) 
                if img["title"].lower().endswith(('.jpg', '.jpeg', '.png'))]
    except Exception as e:
        logger.error(f"Error getting images for page '{page_title}': {str(e)}")
        return []

def get_image_info(image_titles: List[str]) -> List[Dict]:
    """Get full URLs and metadata for a list of image titles."""
    if not image_titles:
        return []
    
    # Process in batches to avoid URL length limits
    batch_size = 20
    all_images = []
    
    for i in range(0, len(image_titles), batch_size):
        batch = image_titles[i:i + batch_size]
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": "800",
            "format": "json"
        }
        
        try:
            time.sleep(REQUEST_DELAY)
            response = requests.get(FANDOM_API_URL, params=params, timeout=15)
            response.raise_for_status()
            pages = response.json().get("query", {}).get("pages", {})
            
            for page in pages.values():
                if "imageinfo" in page:
                    info = page["imageinfo"][0]
                    ext_meta = info.get("extmetadata", {})
                    all_images.append({
                        "title": urllib.parse.unquote(page["title"].replace("File:", "")),
                        "url": info["url"],
                        "width": info.get("width", 0),
                        "height": info.get("height", 0),
                        "size": info.get("size", 0),
                        "mime": info.get("mime", ""),
                        "description": ext_meta.get("ImageDescription", {}).get("*", "") if ext_meta else "",
                        "source": "fandom"
                    })
        except Exception as e:
            logger.error(f"Error fetching image info: {str(e)}")
            continue
            
    return all_images

def fetch_fandom_images(query: str, max_results: int = 5) -> List[Dict]:
    """Fetch relevant images from One Piece Fandom for a given query."""
    logger.debug(f"[Fandom] Searching for: {query}")
    
    # 1. Search for relevant pages, plus known canonical page aliases.
    page_titles = []
    for alias in _fandom_page_aliases(query):
        canonical_title = get_fandom_page_title(alias)
        if canonical_title:
            page_titles.append(canonical_title)

    pages = search_fandom_pages(query)
    page_titles.extend(page["title"] for page in pages if page.get("title"))
    page_titles = _unique_queries(page_titles)

    if not page_titles:
        logger.debug(f"[Fandom] No pages found for: {query}")
        return []
        
    # 2. Get images from top pages
    all_images = []
    for page_title in page_titles[:6]:  # Limit to top pages to avoid too many requests
        logger.debug(f"[Fandom] Getting images for page: {page_title}")
        
        image_titles = get_page_images(page_title)
        if not image_titles:
            continue
            
        # 3. Get image URLs and metadata
        images = get_image_info(image_titles[:20])  # Limit to 20 images per page
        all_images.extend(images)
        
        # Early exit if we have enough results
        if len(all_images) >= max_results * 2:  # Get extra for filtering
            break
    
    # 4. Filter and sort images
    filtered = []
    seen = set()
    
    # Prefer query-relevant larger images with common One Piece aspect ratios.
    for img in sorted(all_images, key=lambda x: (_image_relevance_score(x, query), x.get("width", 0)), reverse=True):
        if len(filtered) >= max_results:
            break
            
        # Basic deduplication
        img_key = img["url"].split("/")[-1].split("?")[0]
        if img_key in seen:
            continue
            
        # Filter criteria
        width = img.get("width", 0)
        height = img.get("height", 1)
        aspect_ratio = width / height if height > 0 else 0
        
        # Common One Piece image aspect ratios (16:9, 3:4, 1:1)
        if not (0.7 <= aspect_ratio <= 2.0):
            continue
            
        # Minimum size check
        if width < 400 or height < 300:
            continue
            
        filtered.append(img)
        seen.add(img_key)
    
    logger.info(f"[Fandom] Found {len(filtered)} suitable images for: {query}")
    return filtered[:max_results]

def download_images_for_slides(json_path: str, out_dir: str) -> None:
    """Download images for slides based on search queries in a JSON file.
    
    Args:
        json_path: Path to the JSON file containing slide data
        out_dir: Directory to save downloaded images
    """
    global GOOGLE_CSE_DISABLED
    GOOGLE_CSE_DISABLED = False

    # Clear the output directory before downloading new images
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    logger.info(f"Downloading images to {out_dir} from {json_path}")

    # Early credential check to avoid repeated error spam
    _cse_api_key = os.getenv("GOOGLE_CSE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    _cse_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    google_cse_available = bool(_cse_api_key and _cse_id)
    if not google_cse_available:
        logger.warning(
            "Google Custom Search credentials missing. Fandom images and reuse fallbacks will still be attempted."
        )
    
    with open(json_path, "r", encoding="utf-8") as f:
        slides = json.load(f)
    
    # Track downloaded image hashes to avoid duplicates
    downloaded_hashes = set()
    
    last_downloaded_path = None
    previous_images = []

    for idx, slide in enumerate(slides):
        search_query = slide.get("image_search_query")
        if not search_query:
            logger.warning(f"[Slide {idx+1}] No image_search_query")
            continue
        # Queries are sanitized when slides are generated; avoid double-prefixing here.
        search_query = _dedupe_query_words(search_query.strip())

        logger.info(f"[Slide {idx+1}] Processing query: {search_query}")

        # Track if we've successfully downloaded an image
        downloaded = False

        # Prepare the list of queries: try site-restricted searches on preferred domains first
        fallback_queries = _generate_fallback_queries(search_query)
        candidate_queries = _unique_queries([search_query], fallback_queries)
        fandom_queries = [
            q for q in candidate_queries
            if q.lower() != "one piece" or search_query.lower() == "one piece" or "logo" in search_query.lower()
        ]

        slide_entities = _slide_context_entities(slide)
        broll_intent = classify_broll_intent(
            search_query,
            slide.get("summary", ""),
            slide_entities,
        )
        slide["broll_intent"] = broll_intent
        logger.info(
            f"[Slide {idx+1}] Auto B-roll intent={broll_intent} "
            f"(order: {' -> '.join(broll_source_order(broll_intent))})"
        )

        # OPArchive first when intent prefers catalog art (characters / islands)
        if not downloaded and broll_intent == "oparchive_first":
            logger.info(f"[Slide {idx+1}] Trying OPArchive for: {search_query}")
            oparchive_images = fetch_oparchive_images(
                search_query,
                slide_entities,
                max_results=8,
                used_url_hashes=downloaded_hashes,
            )
            for img in oparchive_images:
                if downloaded:
                    break
                try:
                    image_hash = _get_image_hash(img["url"])
                    if image_hash in downloaded_hashes:
                        logger.info(
                            f"[Slide {idx+1}] Skipping already-used OPArchive image: {img.get('title')}"
                        )
                        continue
                    filename = f"slide_{idx+1:03d}_oparchive_{image_hash[:8]}.jpg"
                    save_path = os.path.join(out_dir, filename)
                    if download_catalog_image(img["url"], save_path):
                        downloaded_hashes.add(image_hash)
                        slide["image_path"] = save_path
                        slide["image_source"] = img.get("source", "oparchive")
                        slide["image_title"] = img.get("title", "")
                        slide["image_score"] = img.get("score", _image_relevance_score(img, search_query))
                        last_downloaded_path = save_path
                        previous_images.append({
                            "path": save_path,
                            "keywords": _query_keywords(search_query),
                            "index": idx,
                        })
                        downloaded = True
                        logger.info(
                            f"[Slide {idx+1}] Downloaded from OPArchive: {os.path.basename(save_path)}"
                        )
                        break
                except Exception as e:
                    logger.warning(f"[Slide {idx+1}] OPArchive image error: {e}")
                    continue
            if not downloaded and not oparchive_images:
                logger.info(f"[Slide {idx+1}] No OPArchive matches for: {search_query}")

        # Fandom — panels, Oda, interviews, arc screenshots (often first for scene/meta)
        if not downloaded:
            duplicate_fandom_count = 0
            for fandom_query in fandom_queries:
                if downloaded:
                    break
                logger.info(f"[Slide {idx+1}] Trying Fandom API for: {fandom_query}")
                fandom_images = fetch_fandom_images(fandom_query, max_results=10)

                for img in fandom_images:
                    if downloaded:
                        break

                    try:
                        image_hash = _get_image_hash(img["url"])

                        if image_hash in downloaded_hashes:
                            duplicate_fandom_count += 1
                            logger.debug(
                                f"[Slide {idx+1}] Skipping duplicate Fandom image: "
                                f"{img.get('title') or img.get('url')}"
                            )
                            continue

                        filename = f"slide_{idx+1:03d}_fandom_{image_hash[:8]}.jpg"
                        save_path = os.path.join(out_dir, filename)

                        if download_image(img["url"], save_path):
                            downloaded_hashes.add(image_hash)
                            slide["image_path"] = save_path
                            slide["image_source"] = "fandom"
                            slide["image_title"] = img.get("title", "")
                            slide["image_score"] = _image_relevance_score(img, fandom_query)
                            last_downloaded_path = save_path
                            previous_images.append({
                                "path": save_path,
                                "keywords": _query_keywords(search_query),
                                "index": idx,
                            })
                            downloaded = True
                            logger.info(f"[Slide {idx+1}] Downloaded from Fandom: {os.path.basename(save_path)}")
                            break

                    except Exception as e:
                        logger.warning(f"[Slide {idx+1}] Error processing Fandom image: {str(e)}")
                        continue

            if not downloaded and duplicate_fandom_count:
                logger.info(
                    f"[Slide {idx+1}] Fandom only returned duplicate images "
                    f"({duplicate_fandom_count} skipped); trying other sources."
                )

        # OPArchive after Fandom when intent is scene/meta/branding
        if not downloaded and broll_intent != "oparchive_first":
            logger.info(f"[Slide {idx+1}] Trying OPArchive (after Fandom) for: {search_query}")
            oparchive_images = fetch_oparchive_images(
                search_query,
                slide_entities,
                max_results=8,
                used_url_hashes=downloaded_hashes,
            )
            for img in oparchive_images:
                if downloaded:
                    break
                try:
                    image_hash = _get_image_hash(img["url"])
                    if image_hash in downloaded_hashes:
                        continue
                    filename = f"slide_{idx+1:03d}_oparchive_{image_hash[:8]}.jpg"
                    save_path = os.path.join(out_dir, filename)
                    if download_catalog_image(img["url"], save_path):
                        downloaded_hashes.add(image_hash)
                        slide["image_path"] = save_path
                        slide["image_source"] = img.get("source", "oparchive")
                        slide["image_title"] = img.get("title", "")
                        slide["image_score"] = img.get("score", _image_relevance_score(img, search_query))
                        last_downloaded_path = save_path
                        previous_images.append({
                            "path": save_path,
                            "keywords": _query_keywords(search_query),
                            "index": idx,
                        })
                        downloaded = True
                        logger.info(
                            f"[Slide {idx+1}] Downloaded from OPArchive: {os.path.basename(save_path)}"
                        )
                        break
                except Exception as e:
                    logger.warning(f"[Slide {idx+1}] OPArchive image error: {e}")
                    continue

        # Google CSE — priority domains, then generic
        if not downloaded and google_cse_available:
            combined_site_filter = "(" + " OR ".join([f"site:{d}" for d in PRIORITY_DOMAINS[1:]]) + ")"  # Skip fandom.com as we already tried it
            for q in candidate_queries:
                if downloaded:
                    break
                    
                site_query = f"{combined_site_filter} {q}"
                logger.debug(f"[Slide {idx+1}] Trying Google CSE with priority sites: {site_query}")
                items = google_image_search(site_query, num_results=10)

                if not items:
                    logger.debug(f"[Slide {idx+1}] No results for: {site_query}")
                    continue

                logger.info(f"[Slide {idx+1}] Found {len(items)} results across priority sites")

                for item in _rank_search_items(items, q, slide.get("summary", "")):
                    if downloaded:
                        break

                    item_link = item.get('link', '')
                    item_title = item.get('title', '').lower()
                    item_mime = item.get('mime', '')

                    if not item_link or not item_mime.startswith('image/'):
                        continue

                    is_onepiece = any(term in item_title for term in ONE_PIECE_RELEVANCE_TERMS)

                    if not is_onepiece:
                        logger.debug(f"[Slide {idx+1}] Skipping non-One Piece image: {item_title}")
                        continue

                    is_duplicate, image_hash = _is_duplicate_image(item, downloaded_hashes)
                    if is_duplicate:
                        logger.debug(f"[Slide {idx+1}] Skipping duplicate image: {item_title}")
                        continue

                    filename = f"slide_{idx+1:03d}_{image_hash[:8]}.jpg"
                    save_path = os.path.join(out_dir, filename)

                    if download_image(item_link, save_path):
                        downloaded_hashes.add(image_hash)
                        slide["image_path"] = save_path
                        slide["image_source"] = "priority_search"
                        slide["image_score"] = _search_item_relevance_score(item, q, slide.get("summary", ""))
                        last_downloaded_path = save_path
                        previous_images.append({
                            "path": save_path,
                            "keywords": _query_keywords(search_query),
                            "index": idx,
                        })
                        downloaded = True
                        logger.info(f"[Slide {idx+1}] Downloaded from priority sites: {os.path.basename(save_path)}")
                        break

        # 3) If still not downloaded, fall back to generic searches as a last resort
        if not downloaded and google_cse_available:
            for query in candidate_queries:
                if downloaded:
                    break

                logger.debug(f"[Slide {idx+1}] Trying generic search: {query}")
                items = google_image_search(query, num_results=10)

                if not items:
                    logger.debug(f"[Slide {idx+1}] No results for: {query}")
                    continue

                logger.info(f"[Slide {idx+1}] Found {len(items)} results for generic query")

                # Process search results
                for item in _rank_search_items(items, query, slide.get("summary", "")):
                    if downloaded:
                        break

                    item_link = item.get('link', '')
                    item_title = item.get('title', '').lower()
                    item_mime = item.get('mime', '')

                    # Skip if no link or invalid MIME type
                    if not item_link or not item_mime.startswith('image/'):
                        continue

                    # Check for One Piece relevance
                    is_onepiece = any(term in item_title for term in ONE_PIECE_RELEVANCE_TERMS)

                    if not is_onepiece:
                        logger.debug(f"[Slide {idx+1}] Skipping non-One Piece image: {item_title}")
                        continue

                    # Check for duplicates
                    is_duplicate, image_hash = _is_duplicate_image(item, downloaded_hashes)
                    if is_duplicate:
                        logger.debug(f"[Slide {idx+1}] Skipping duplicate image: {item_title}")
                        continue

                    # Try to download the image
                    filename = f"slide_{idx+1:03d}_{image_hash[:8]}.jpg"
                    save_path = os.path.join(out_dir, filename)

                    if download_image(item_link, save_path):
                        downloaded_hashes.add(image_hash)
                        slide["image_path"] = save_path
                        slide["image_source"] = "generic_search"
                        slide["image_score"] = _search_item_relevance_score(item, query, slide.get("summary", ""))
                        last_downloaded_path = save_path
                        previous_images.append({
                            "path": save_path,
                            "keywords": _query_keywords(search_query),
                            "index": idx,
                        })
                        downloaded = True
                        logger.info(f"[Slide {idx+1}] Successfully downloaded: {os.path.basename(save_path)}")
                        break

        fallback_image_path = _best_previous_image(search_query, previous_images) or last_downloaded_path
        if not downloaded and fallback_image_path:
            slide["image_path"] = fallback_image_path
            slide["image_source"] = "reused_previous"
            logger.warning(
                f"[Slide {idx+1}] No image downloaded. Reusing closest previous image to preserve timing: "
                f"{os.path.basename(fallback_image_path)}"
            )
    
    # Save the updated slides with image paths
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(slides, f, indent=2, ensure_ascii=False)
    
    # Log download statistics. A slide can have an image_path because it reused
    # a previous image, so keep that separate from real unique downloads.
    assigned_count = sum(1 for slide in slides if "image_path" in slide)
    reused_count = sum(1 for slide in slides if slide.get("image_source") == "reused_previous")
    unique_image_paths = {
        slide["image_path"]
        for slide in slides
        if slide.get("image_path") and slide.get("image_source") != "reused_previous"
    }
    logger.info(
        "Image assignment complete. %s/%s slides have image paths; "
        "%s unique images downloaded; %s slides reused a previous image.",
        assigned_count,
        len(slides),
        len(unique_image_paths),
        reused_count,
    )

    # Clean up any temporary attributes we added
    for slide in slides:
        for attr in ['_priority']:
            if attr in slide:
                del slide[attr]
    
    return assigned_count
