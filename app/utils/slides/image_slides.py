import re
import json
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv
import logging

from app.utils.slides.image_slides_llm import call_image_slides_llm

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

CANONICAL_ENTITIES = [
    "Luffy", "Zoro", "Sanji", "Nami", "Usopp", "Chopper", "Robin", "Franky",
    "Brook", "Jinbe", "Shanks", "Mihawk", "Blackbeard", "Marshall D. Teach",
    "Whitebeard", "Ace", "Sabo", "Dragon", "Garp", "Koby", "Law", "Kid",
    "Bonney", "Kuma", "Kuina", "King", "Vegapunk", "Imu", "Five Elders", "Gorosei",
    "Joy Boy", "Nika", "Mary Geoise", "Wano", "Egghead", "Elbaf",
    "Marineford", "Dressrosa", "Whole Cake Island", "Thriller Bark",
    "Baratie", "Sabaody", "Onigashima", "Kuraigana Island",
    "Shimotsuki Village", "Enies Lobby", "Ohara", "Void Century", "Poneglyph", "Sunny Ship",
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
    "caribou": "Caribou",
    "shirahoshi": "Shirahoshi",
    "poseidon": "Poseidon",
    "pluton": "Pluton",
    "fishman island": "Fishman Island",
    "shimotsuki": "Shimotsuki Village",
    "sabaody archipelago": "Sabaody",
    "bartholomew kuma": "Kuma",
    "dracule mihawk": "Mihawk",
    "enma": "Enma",
}

for _extra_entity in ("Caribou", "Shirahoshi", "Pluton", "Poseidon", "Fishman Island", "Enma"):
    if _extra_entity not in CANONICAL_ENTITIES:
        CANONICAL_ENTITIES.append(_extra_entity)

IMAGE_SLIDE_MAX_SECONDS = float(os.getenv("IMAGE_SLIDE_MAX_SECONDS", "5.5"))
IMAGE_SLIDE_MIN_SECONDS = float(os.getenv("IMAGE_SLIDE_MIN_SECONDS", "1.6"))


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

    standalone_places = {
        "egghead", "wano", "skypiea", "marineford", "dressrosa", "ohara",
        "enies lobby", "whole cake island", "punk hazard", "elbaf", "zou",
    }
    if raw.lower() in standalone_places:
        return raw.title() if raw.islower() else raw

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


_SLIDE_REQUIRED_KEYS = ("start_time", "end_time", "summary")
_VISUAL_SOURCE_ASSET = "asset_search"
_VISUAL_SOURCE_AI = "ai_generate"

# Queries that Vivre / manual search cannot satisfy — route to AI generation
_BAD_ASSET_QUERY_RE = re.compile(
    r"\b("
    r"model\s+\w+|hito\s+hito|gear\s+v\b|laboratory|battlefield|manga\s+panel|"
    r"stretching|aura|thermodynamic|kinetic|infographic|diagram|explosion|"
    r"internal\s+friction|comparison\s+chart|theories"
    r")\b",
    re.IGNORECASE,
)

_AI_VISUAL_HINTS_RE = re.compile(
    r"\b("
    r"thermodynamic|kinetic|friction|pressure|elastic|detonat|explosion|"
    r"internal\s+blast|diagram|infographic|two\s+devil\s+fruit\s+rule|"
    r"vs\s+normal|energy\s+expansion|mythical\s+zoan\s+property|abstract"
    r")\b",
    re.IGNORECASE,
)

_ASSET_ONLY_HINTS_RE = re.compile(
    r"\b(one piece logo|grand line map|follow|subscribe|outro)\b",
    re.IGNORECASE,
)


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
        "CRITICAL — each slide matches the SPOKEN lines in [start_time, end_time]. Faceless theory video (voiceover + B-roll).\n\n"
        "CONTEXT-AWARE PACING:\n"
        "- Cut on idea changes, not only character names: hook, cause, consequence, example, rebuttal, payoff.\n"
        "- Prefer 2-5 seconds per slide. Never hold one visual across a new canon event, new character, or new emotional turn.\n"
        "- Long narration sections must become multiple visual beats with different scene-search queries.\n"
        "- For trauma/psychology narration, DO NOT invent symbolic AI art first. Map the feeling to the closest canon scene: Kuina stairs, Mihawk Baratie, Zoro nothing happened, Kuma Sabaody, Enma Onigashima, Zoro bowing to Mihawk.\n"
        "- Avoid generic repeats like only 'Zoro training' or 'Zoro scars'; each slide needs a new focal action or object.\n\n"
        "For EVERY slide set visual_source to ONE of:\n"
        f'  - "{_VISUAL_SOURCE_ASSET}": use existing art (Vivre pack / upload). Provide image_search_query only.\n'
        f'  - "{_VISUAL_SOURCE_AI}": LAST RESORT only. Provide ai_image_prompt only when no searchable canon scene exists.\n\n'
        "USE asset_search WHEN:\n"
        "- A named character appears (Monkey D. Luffy, Jabra, Marshall D. Teach, Five Elders, etc.)\n"
        "- A known place in the asset pack (Egghead, Enies Lobby, Marineford, Skypiea, Mary Geoise)\n"
        "- Pirate crew / flag / logo (Straw Hat Pirates, Blackbeard Pirates, One Piece Logo)\n"
        "- A specific manga chapter beat tied to a named character or arc location\n\n"
        "SEARCH QUERY STYLE FOR asset_search:\n"
        "- Use specific manga/anime scene phrases, not abstract emotions.\n"
        "- Good: 'Zoro Kuina Shimotsuki stairs', 'Zoro Mihawk Baratie cross knife', 'Zoro nothing happened Thriller Bark', 'Kuma Sabaody Straw Hats vanish', 'Zoro Enma King Onigashima', 'Zoro bows to Mihawk Kuraigana'.\n"
        "- Bad: 'Zoro trauma', 'fear of weakness', 'fragility of life', 'Zoro aesthetic', 'sad anime boy'.\n\n"
        "USE ai_generate WHEN (do NOT use vague image_search_query):\n"
        "- Abstract science (thermodynamics, kinetic energy, friction, pressure, elastic limit)\n"
        "- Diagrams, comparisons, 'two devil fruits rule', internal explosion metaphor\n"
        "- Impossible or non-catalog queries (Hito Hito Model Nika, Gear V aura, laboratory explosion)\n"
        "- Generic 'manga panels' or 'tell me your theories' — use a concrete AI scene instead\n"
        "- Same hero pose would repeat for the 3rd+ time and no better canon scene exists\n\n"
        "FIELD RULES:\n"
        "- summary: short beat from subtitles (what the viewer should understand).\n"
        "- image_search_query: ONLY if visual_source=asset_search. 3–9 words, searchable canon scene phrase. "
        "Never repeat the same query twice. No jargon filenames.\n"
        "- ai_image_prompt: ONLY if visual_source=ai_generate. Generate ONE cinematic image prompt for an image model.\n"
        "Rules:\n"
        "* Start EXACTLY with: "
        "Vertical 9:16 One Piece anime style illustration, no text, no watermark.\n"
        "* Anchor with: "
        "Visualize this narration beat, without showing words: '<spoken subtitle text>'.\n"
        "* Resolve pronouns using main context.\n"
        "* Generate the SINGLE decisive cinematic moment (freeze-frame), not a generic setting.\n"
        "* Use: SUBJECT + ACTION + SYMBOL + ENVIRONMENT + LIGHTING.\n"
        "* Prefer ACTION + CONSEQUENCE: what happens, what changed, what object proves it.\n"
        "* Prefer dynamic verbs: walking away, dropping, tearing, revealing, confronting.\n"
        "* Avoid passive scenes: standing, looking, sitting.\n"
        "* Show characters from behind/silhouette where possible.\n"
        "* One clear focal subject only.\n"
        "* Faceless YouTube B-roll only.\n"
        "* Not a poster, not collage, not multiple panels.\n"
        "* Highly detailed anime art.\n"
        "- If asset_search, set ai_image_prompt to empty string \"\".\n"
        "- If ai_generate, set image_search_query to empty string \"\".\n\n"
        f"SCRIPT-WIDE ENTITIES (context only): {entities_line}\n"
        "Max 2 slides with Monkey D. Luffy in image_search_query; use Teach/Jabra/diagrams for variety.\n"
        "TIMING: merge short fragments; max ~3.5s per slide; continuous timestamps.\n\n"
        "OUTPUT: ONLY a JSON array. Each object MUST include:\n"
        "start_time, end_time, summary, visual_source, image_search_query, ai_image_prompt\n"
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
        "GOOD EXAMPLE (canon-scene-first, AI only when unavoidable):\n"
        '[{"start_time":"0:00:00.03","end_time":"0:00:03.08",'
        '"summary":"Zoro trains from fear of weakness",'
        f'"visual_source":"{_VISUAL_SOURCE_ASSET}",'
        '"image_search_query":"Roronoa Zoro training dojo",'
        '"ai_image_prompt":""},'
        '{"start_time":"0:00:03.60","end_time":"0:00:07.80",'
        '"summary":"Kuina death creates Zoro fear",'
        f'"visual_source":"{_VISUAL_SOURCE_ASSET}",'
        '"image_search_query":"Zoro Kuina Shimotsuki stairs",'
        '"ai_image_prompt":""},'
        '{"start_time":"0:00:09.00","end_time":"0:00:11.84",'
        '"summary":"Mihawk humiliates Zoro at Baratie",'
        f'"visual_source":"{_VISUAL_SOURCE_ASSET}",'
        '"image_search_query":"Zoro Mihawk Baratie cross knife",'
        '"ai_image_prompt":""},'
        '{"start_time":"0:00:31.49","end_time":"0:00:32.43",'
        '"summary":"Follow CTA",'
        f'"visual_source":"{_VISUAL_SOURCE_ASSET}",'
        '"image_search_query":"One Piece Logo",'
        '"ai_image_prompt":""}]\n\n'
        "BAD: abstract emotion queries like 'Zoro trauma'; AI prompts for searchable scenes; five generic Zoro asset slides in a row.\n\n"
        "Subtitles:\n"
        f"{raw_subtitles}\n"
    )


# Ordered (pattern, preferred Vivre query) — subtitle-driven fallbacks when LLM is generic
_SUBTITLE_QUERY_HOOKS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bkuina\b|shimotsuki|flight of stairs|dojo", re.I), "Zoro Kuina Shimotsuki stairs"),
    (re.compile(r"\bmihawk\b|baratie|cross knife|tiny knife", re.I), "Zoro Mihawk Baratie cross knife"),
    (re.compile(r"\bnothing happened\b|thriller bark|luffy'?s pain|red bubble|pool of blood", re.I), "Zoro nothing happened Thriller Bark"),
    (re.compile(r"\bkuma\b|sabaody|vanish|erases|separation", re.I), "Kuma Sabaody Straw Hats vanish"),
    (re.compile(r"\bking\b|onigashima|enma|conflagration|armament haki", re.I), "Zoro Enma King Onigashima"),
    (re.compile(r"\bkuraigana\b|bows to mihawk|begging.*train|mihawk.*train", re.I), "Zoro bows to Mihawk Kuraigana"),
    (re.compile(r"\bzoro\b.*\btrain|\btraining\b.*\bzoro\b", re.I), "Roronoa Zoro training"),
    (re.compile(r"\bzoro\b.*\bscar|\bscar\b.*\bzoro\b", re.I), "Roronoa Zoro scars"),
    (re.compile(r"\bzoro\b.*\bcrew|\bprotect\b.*\bcrew|\bstraw hat pirates\b", re.I), "Zoro Straw Hat Pirates"),
    (re.compile(r"\bjabra\b|chapter\s*385", re.I), "Jabra Enies Lobby"),
    (re.compile(r"\begghead\b", re.I), "Egghead"),
    (re.compile(r"\benies lobby\b", re.I), "Enies Lobby"),
    (re.compile(r"\bgear\s*5\b|\bnika\b|awakened", re.I), "Monkey D. Luffy"),
    (re.compile(r"\bgomu|rubber|devil fruit", re.I), "Monkey D. Luffy"),
    (re.compile(r"\bblackbeard\b|marshall d\.?\s*teach\b|two devil fruits\b", re.I), "Marshall D. Teach"),
    (re.compile(r"\bwhitebeard\b", re.I), "Edward Newgate"),
    (re.compile(r"\bmarineford\b", re.I), "Marineford"),
    (re.compile(r"\bwano\b", re.I), "Wano"),
    (re.compile(r"\bskypiea\b", re.I), "Skypiea"),
    (re.compile(r"\bshanks\b", re.I), "Shanks"),
    (re.compile(r"\broger\b", re.I), "Gol D. Roger"),
    (re.compile(r"\bfollow\b|subscribe\b|outro\b", re.I), "One Piece Logo"),
    (re.compile(r"\bcaribou\b", re.I), "Caribou"),
    (re.compile(r"\bshirahoshi\b|\bposeidon\b", re.I), "Shirahoshi"),
    (re.compile(r"\bpluton\b", re.I), "Pluton"),
    (re.compile(r"\bfishman island\b|fish-man island\b", re.I), "Fishman Island"),
]


def _collect_subtitle_text_in_range(
    dialogues: List[Dict],
    start_time: str,
    end_time: str,
) -> str:
    """Concatenate subtitle lines overlapping a slide window."""
    win_start = parse_time(start_time)
    win_end = parse_time(end_time)
    parts: List[str] = []
    for line in dialogues:
        line_start = parse_time(line["start"])
        line_end = parse_time(line["end"])
        if line_end <= win_start or line_start >= win_end:
            continue
        text = (line.get("text") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _subtitle_lines_in_range(
    dialogues: List[Dict],
    start_time: str,
    end_time: str,
) -> List[Dict]:
    """Return subtitle lines overlapping a slide window."""
    win_start = parse_time(start_time)
    win_end = parse_time(end_time)
    lines: List[Dict] = []
    for line in dialogues:
        line_start = parse_time(line["start"])
        line_end = parse_time(line["end"])
        if line_end <= win_start or line_start >= win_end:
            continue
        text = (line.get("text") or "").strip()
        if text:
            lines.append(
                {
                    "start": line["start"],
                    "end": line["end"],
                    "text": text,
                }
            )
    return lines


def _split_subtitle_lines_into_visual_beats(lines: List[Dict]) -> List[List[Dict]]:
    """Chunk subtitle lines into visual beats with an upper duration bound."""
    if not lines:
        return []

    beats: List[List[Dict]] = []
    current: List[Dict] = []
    for line in lines:
        candidate = [*current, line]
        start = parse_time(candidate[0]["start"])
        end = parse_time(candidate[-1]["end"])
        duration = end - start
        current_text = " ".join(item["text"] for item in candidate)
        sentence_boundary = bool(re.search(r"[.!?]$", (line.get("text") or "").strip()))

        if current and duration > IMAGE_SLIDE_MAX_SECONDS:
            beats.append(current)
            current = [line]
            continue

        current = candidate
        current_duration = parse_time(current[-1]["end"]) - parse_time(current[0]["start"])
        if current_duration >= IMAGE_SLIDE_MIN_SECONDS and sentence_boundary:
            beats.append(current)
            current = []
        elif current_duration >= IMAGE_SLIDE_MAX_SECONDS * 0.8 and re.search(
            r"\bbut|when|because|then|only this time|even during|ultimately\b",
            current_text,
            re.I,
        ):
            beats.append(current)
            current = []

    if current:
        if beats and parse_time(current[-1]["end"]) - parse_time(current[0]["start"]) < IMAGE_SLIDE_MIN_SECONDS:
            beats[-1].extend(current)
        else:
            beats.append(current)
    return beats


def _summarize_visual_beat(text: str, fallback: str) -> str:
    """Short context-aware summary from the actual spoken beat."""
    cleaned = _clean_prompt_text(text, max_len=150)
    if not cleaned:
        return fallback

    entity_match = re.search(
        r"\b(Zoro|Kuina|Mihawk|Kuma|Sanji|Luffy|King|Enma|Sabaody|Baratie|Thriller Bark|Onigashima|Kuraigana)\b",
        cleaned,
        re.I,
    )
    subject = entity_match.group(0) if entity_match else ""
    lower = cleaned.lower()
    if "stairs" in lower or "kuina" in lower:
        return "Kuina's death breaks Zoro's sense of safety"
    if "mihawk" in lower or "cross knife" in lower:
        return "Mihawk exposes the gap in Zoro's strength"
    if "nothing happened" in lower or "pain" in lower:
        return "Zoro hides unbearable pain at Thriller Bark"
    if "sabaody" in lower or "vanish" in lower or "erases" in lower:
        return "Zoro watches the crew disappear at Sabaody"
    if "enma" in lower or "king" in lower:
        return "Zoro risks being drained by Enma against King"
    if "bows" in lower or "begging" in lower or "train" in lower and "mihawk" in lower:
        return "Zoro sacrifices pride to train under Mihawk"
    if subject:
        return f"{subject}: {cleaned[:90].rstrip(' ,.;')}"
    return cleaned[:110].rstrip(" ,.;")


def _split_long_slides_by_dialogues(
    slides: List[Dict],
    dialogues: List[Dict],
    context_entities: List[str],
) -> List[Dict]:
    """Split overlong LLM slides into smaller subtitle-grounded visual beats."""
    split_slides: List[Dict] = []
    for slide in slides:
        start = parse_time(slide["start_time"])
        end = parse_time(slide["end_time"])
        if end - start <= IMAGE_SLIDE_MAX_SECONDS:
            split_slides.append(slide)
            continue

        lines = _subtitle_lines_in_range(dialogues, slide["start_time"], slide["end_time"])
        beats = _split_subtitle_lines_into_visual_beats(lines)
        if len(beats) <= 1:
            split_slides.append(slide)
            continue

        logger.info(
            "Splitting long image slide %.2f-%.2fs into %s visual beats",
            start,
            end,
            len(beats),
        )
        for beat in beats:
            subtitle_text = " ".join(line["text"] for line in beat).strip()
            summary = _summarize_visual_beat(subtitle_text, slide.get("summary", ""))
            child = dict(slide)
            child["start_time"] = beat[0]["start"]
            child["end_time"] = beat[-1]["end"]
            child["summary"] = summary
            child["subtitle_text"] = subtitle_text
            child["context_entities"] = _ai_prompt_context_entities(
                subtitle_text,
                summary,
                slide.get("context_entities") or context_entities,
            )
            inferred_query = _infer_query_from_subtitle_text(subtitle_text, child["context_entities"])
            if inferred_query:
                child["image_search_query"] = inferred_query
            if _needs_ai_visual(subtitle_text, summary, child.get("image_search_query", ""), 0):
                child["visual_source"] = _VISUAL_SOURCE_AI
                child["image_search_query"] = ""
                child["ai_image_prompt"] = _build_ai_image_prompt(
                    summary,
                    subtitle_text,
                    child["context_entities"],
                )
            else:
                child["ai_image_prompt"] = ""
            split_slides.append(child)

    return split_slides


def _normalize_visual_source(raw: Optional[str]) -> str:
    value = (raw or "").strip().lower().replace("-", "_")
    if value in {"ai", "ai_generate", "generate", "ai_generation", "ai_image"}:
        return _VISUAL_SOURCE_AI
    if value in {"asset", "search", "asset_search", "vivre", "upload", "catalog"}:
        return _VISUAL_SOURCE_ASSET
    return _VISUAL_SOURCE_ASSET


def _clean_prompt_text(value: str, max_len: int = 260) -> str:
    text = re.sub(r"\s+", " ", (value or "").strip())
    return text[:max_len].rstrip(" ,.;")


def _ai_prompt_context_entities(subtitle_text: str, summary: str, context_entities: List[str]) -> List[str]:
    """Prefer entities from the spoken beat, then fall back to slide/script context."""
    beat_entities = _extract_context_entities(f"{subtitle_text} {summary}", limit=3)
    merged = []
    for entity in [*beat_entities, *(context_entities or [])]:
        if entity and entity not in merged:
            merged.append(entity)
    return merged[:3]


def _build_ai_image_prompt(summary: str, subtitle_text: str, context_entities: List[str]) -> str:
    """Template prompt for external AI image tools (Gemini Imagen, DALL·E, etc.)."""
    line = _clean_prompt_text(subtitle_text or summary or "One Piece theory scene")
    scene = _clean_prompt_text(summary or subtitle_text or "One Piece theory scene", max_len=180)
    cast_entities = _ai_prompt_context_entities(subtitle_text, summary, context_entities)
    cast = ", ".join(cast_entities) if cast_entities else "One Piece characters"
    return (
        "Vertical 9:16 One Piece anime style illustration, no text, no watermark. "
        f"Visualize this narration beat, without showing words: '{line}'. "
        f"Main context: {cast}. "
        f"{scene}. "
        "One clear cinematic scene, faceless theory video B-roll, dramatic lighting, readable on mobile."
    )


def _anchor_ai_image_prompt(
    ai_prompt: str,
    summary: str,
    subtitle_text: str,
    context_entities: List[str],
) -> str:
    """Ensure Gemini-provided AI prompts stay tied to the exact spoken beat."""
    base_prompt = (ai_prompt or "").strip()
    if not base_prompt:
        return _build_ai_image_prompt(summary, subtitle_text, context_entities)

    prefix = "Vertical 9:16 One Piece anime style illustration, no text, no watermark."
    if base_prompt.lower().startswith(prefix.lower()):
        scene = base_prompt[len(prefix):].strip()
    else:
        scene = base_prompt
    scene = re.sub(r"^faceless (youtube )?theory video b-roll\.?\s*", "", scene, flags=re.I)
    scene = re.sub(
        r"^visualize this narration beat,\s*without showing words:\s*'[^']*'\.?\s*",
        "",
        scene,
        flags=re.I,
    )
    scene = re.sub(r"^main context:\s*[^.]+\.?\s*", "", scene, flags=re.I)
    if "visualize this narration beat" in base_prompt.lower() and "main context:" in base_prompt.lower():
        return base_prompt

    line = _clean_prompt_text(subtitle_text or summary or "One Piece theory scene")
    cast_entities = _ai_prompt_context_entities(subtitle_text, summary, context_entities)
    cast = ", ".join(cast_entities) if cast_entities else "One Piece characters"
    return (
        f"{prefix} "
        f"Visualize this narration beat, without showing words: '{line}'. "
        f"Main context: {cast}. "
        f"{scene}"
    ).strip()


def _needs_ai_visual(
    subtitle_text: str,
    summary: str,
    image_search_query: str,
    luffy_asset_count: int,
) -> bool:
    """Heuristic: catalog/Vivre cannot satisfy this beat — use AI generation."""
    blob = f"{subtitle_text} {summary} {image_search_query}".strip()
    if not blob:
        return False
    if _ASSET_ONLY_HINTS_RE.search(blob):
        return False
    if _BAD_ASSET_QUERY_RE.search(image_search_query):
        return True
    if _AI_VISUAL_HINTS_RE.search(blob):
        return True
    if re.search(r"\bmanga\s+panel", blob, re.I):
        return True
    if re.search(r"\btell me your theories\b", blob, re.I):
        return True
    if re.search(r"\bblueprint\b|\bancient weapon\b|\bsplit screen\b", blob, re.I):
        return True
    if re.search(r"\bpluton\b", blob, re.I) and not re.search(
        r"\bshirahoshi\b|\bposeidon\b", blob, re.I
    ):
        return True
    if luffy_asset_count >= 2 and re.search(r"\bluffy\b", blob, re.I) and not re.search(
        r"\bjabra\b|blackbeard|teach|egghead", blob, re.I
    ):
        return True
    return False


def _llm_chose_ai_slide(slide: Dict) -> bool:
    """Trust Gemini when it already returned a full AI prompt."""
    source = _normalize_visual_source(slide.get("visual_source"))
    prompt = (slide.get("ai_image_prompt") or "").strip()
    return source == _VISUAL_SOURCE_AI and len(prompt) >= 24


def _apply_visual_source_plan(
    slide: Dict,
    subtitle_text: str,
    context_entities: List[str],
    luffy_asset_count: int,
) -> Dict:
    """Finalize visual_source, image_search_query, and ai_image_prompt per slide."""
    slide = dict(slide)
    summary = slide.get("summary") or ""
    query = (slide.get("image_search_query") or "").strip()
    ai_prompt = (slide.get("ai_image_prompt") or "").strip()
    source = _normalize_visual_source(slide.get("visual_source"))
    inferred_query = _infer_query_from_subtitle_text(
        f"{subtitle_text} {summary}",
        context_entities,
    )

    if _llm_chose_ai_slide(slide):
        if inferred_query and not _needs_ai_visual(subtitle_text, summary, inferred_query, luffy_asset_count):
            slide["visual_source"] = _VISUAL_SOURCE_ASSET
            slide["image_search_query"] = _dedupe_query_words(inferred_query)
            slide["ai_image_prompt"] = ""
            return slide
        slide["visual_source"] = _VISUAL_SOURCE_AI
        slide["ai_image_prompt"] = _anchor_ai_image_prompt(
            ai_prompt,
            summary,
            subtitle_text,
            context_entities,
        )
        slide["image_search_query"] = ""
        return slide

    if _needs_ai_visual(subtitle_text, summary, query, luffy_asset_count):
        source = _VISUAL_SOURCE_AI
    elif query and not _BAD_ASSET_QUERY_RE.search(query):
        source = _VISUAL_SOURCE_ASSET
    elif _ASSET_ONLY_HINTS_RE.search(f"{subtitle_text} {summary}"):
        source = _VISUAL_SOURCE_ASSET
        query = query or "One Piece Logo"

    if source == _VISUAL_SOURCE_AI:
        ai_prompt = _anchor_ai_image_prompt(
            ai_prompt,
            summary,
            subtitle_text,
            context_entities,
        )
        slide["visual_source"] = _VISUAL_SOURCE_AI
        slide["ai_image_prompt"] = ai_prompt
        slide["image_search_query"] = ""
    else:
        slide["visual_source"] = _VISUAL_SOURCE_ASSET
        slide["image_search_query"] = _dedupe_query_words(
            query or _infer_query_from_subtitle_text(subtitle_text, context_entities) or "One Piece Logo"
        )
        slide["ai_image_prompt"] = ""

    return slide


def _infer_query_from_subtitle_text(text: str, context_entities: List[str]) -> Optional[str]:
    """Pick a Vivre-friendly query from spoken words in this slide window."""
    if not text:
        return None
    lower = text.lower()
    for pattern, query in _SUBTITLE_QUERY_HOOKS:
        if pattern.search(lower):
            return query
    entities = _extract_context_entities(text, limit=3)
    if entities:
        return _sanitize_image_query(" ".join(entities[:2]), text, entities)
    if context_entities:
        return _sanitize_image_query(context_entities[0], text, context_entities)
    return None


def _refine_slides_from_subtitles(
    slides: List[Dict],
    dialogues: List[Dict],
    context_entities: List[str],
) -> List[Dict]:
    """
    Attach subtitle_text per slide, align queries to spoken content, dedupe repeats.
    """
    used_queries: Dict[str, int] = {}
    luffy_asset_count = 0
    refined: List[Dict] = []

    for slide in slides:
        start = slide["start_time"]
        end = slide["end_time"]
        subtitle_text = _collect_subtitle_text_in_range(dialogues, start, end)
        slide = dict(slide)
        slide["subtitle_text"] = subtitle_text

        summary = slide.get("summary") or ""
        query = slide.get("image_search_query") or ""
        combined = f"{subtitle_text} {summary}".strip()
        preserve_ai = _llm_chose_ai_slide(slide)

        inferred = _infer_query_from_subtitle_text(subtitle_text, context_entities)
        if inferred:
            query_norm = _dedupe_query_words(query).lower()
            inferred_norm = _dedupe_query_words(inferred).lower()
            # Replace when LLM query is generic repeat or missing the speaker/subject of the line
            luffy_heavy = "luffy" in query_norm and "luffy" not in subtitle_text.lower()
            jabra_line = "jabra" in subtitle_text.lower() and "jabra" not in query_norm
            teach_line = "blackbeard" in subtitle_text.lower() and "teach" not in query_norm
            egghead_line = "egghead" in subtitle_text.lower() and "egghead" not in query_norm
            if luffy_heavy or jabra_line or teach_line or egghead_line:
                query = inferred
            elif not query or len(query.split()) < 2:
                query = inferred

        if not preserve_ai:
            if not query and combined:
                query = _infer_query_from_subtitle_text(combined, context_entities) or query

            query = _sanitize_image_query(
                query or "One Piece Logo",
                combined or summary,
                _slide_context_entities(slide) or context_entities,
            )

        # Force unique queries: suffix beat hint when duplicate (asset slides only)
        base_key = (query or "").lower()
        if not preserve_ai and base_key in used_queries:
            beat_hint = _extract_context_entities(subtitle_text, limit=1)
            if beat_hint and beat_hint[0].lower() not in base_key:
                query = _sanitize_image_query(
                    f"{beat_hint[0]} {query}",
                    combined,
                    context_entities,
                )
            elif "egghead" in subtitle_text.lower() and "egghead" not in base_key:
                query = "Egghead"
            elif "jabra" in subtitle_text.lower():
                query = "Jabra Enies Lobby"
            elif "blackbeard" in subtitle_text.lower() or "unique" in subtitle_text.lower():
                query = "Marshall D. Teach Blackbeard Pirates"
            else:
                used_queries[base_key] += 1
                query = f"{query} {used_queries[base_key]}"

        if not preserve_ai:
            used_queries[base_key] = used_queries.get(base_key, 0) + 1
            slide["image_search_query"] = _dedupe_query_words(query)
        if subtitle_text and len(summary) < 12:
            slide["summary"] = summary or subtitle_text[:80]

        slide = _apply_visual_source_plan(
            slide,
            subtitle_text,
            slide.get("context_entities") or context_entities,
            luffy_asset_count,
        )
        if slide.get("visual_source") == _VISUAL_SOURCE_ASSET:
            q_lower = (slide.get("image_search_query") or "").lower()
            if "luffy" in q_lower:
                luffy_asset_count += 1
        refined.append(slide)

    return refined


def _script_context_summary(dialogues: List[Dict]) -> Dict:
    full_text = " ".join(d.get("text", "") for d in dialogues)
    entities = _extract_context_entities(full_text)
    return {
        "text": full_text[:3000],
        "entities": entities,
        "fallback_query": " ".join(entities[:2]) if entities else "One Piece Logo",
    }


def generate_gemini_image_slides(ass_path: str, out_path: str, total_duration: float) -> str:
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
                f"{gemini_slide.get('summary', '')} {gemini_slide.get('image_search_query', '')} "
                f"{gemini_slide.get('ai_image_prompt', '')}",
                limit=6,
            )
            raw_query = (gemini_slide.get("image_search_query") or "").strip()
            if raw_query:
                raw_query = _sanitize_image_query(
                    raw_query,
                    gemini_slide["summary"],
                    slide_entities or context_entities,
                )
            slide = {
                "start_time": gemini_slide["start_time"],
                "end_time": gemini_slide["end_time"],
                "summary": gemini_slide["summary"],
                "visual_source": _normalize_visual_source(gemini_slide.get("visual_source")),
                "image_search_query": raw_query,
                "ai_image_prompt": (gemini_slide.get("ai_image_prompt") or "").strip(),
                "context_entities": slide_entities or context_entities,
            }
            final_slides.append(slide)

        final_slides = _split_long_slides_by_dialogues(
            final_slides,
            timestamped_dialogues,
            context_entities,
        )
        final_slides = _refine_slides_from_subtitles(
            final_slides,
            timestamped_dialogues,
            context_entities,
        )

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(final_slides, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully generated {len(final_slides)} slides at {out_path}")
        return out_path

    except (TimeoutError, ConnectionError, RuntimeError, ValueError) as e:
        logger.error(f"Error in generate_gemini_image_slides: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_gemini_image_slides: {e}")
        raise


def generate_image_slides(ass_path: str, out_path: str, total_duration: float) -> str:
    """Generate slide timing + search terms JSON (images uploaded separately)."""
    return generate_gemini_image_slides(ass_path, out_path, total_duration)
