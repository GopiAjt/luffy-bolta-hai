import re
import json
import requests
from typing import List, Dict, Optional, Set, Tuple
import os
from dotenv import load_dotenv
import google.generativeai as genai
import logging
import time
import shutil
import hashlib
import urllib.parse
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageOps

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
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash-lite")

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


def _sanitize_image_query(query: str, summary: str, context_entities: List[str]) -> str:
    query = re.sub(r"\b(one piece anime|anime screenshot|manga panel|fan art|wallpaper|dramatic|cinematic|close up)\b", "", query or "", flags=re.IGNORECASE)
    query = re.sub(r"[^A-Za-z0-9 .'-]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()

    combined = f"{query} {summary}".lower()
    chosen_entities = [
        entity for entity in context_entities
        if entity.lower() in combined or any(alias in combined and canonical == entity for alias, canonical in ENTITY_ALIASES.items())
    ]
    if not chosen_entities:
        chosen_entities = context_entities[:2]

    query_words = [word for word in query.split() if len(word) > 2]
    query_without_one_piece = " ".join(word for word in query_words if word.lower() not in {"one", "piece"})
    if chosen_entities:
        anchored = " ".join(chosen_entities[:2])
        if query_without_one_piece and query_without_one_piece.lower() not in anchored.lower():
            return f"{anchored} {query_without_one_piece}".strip()
        return anchored
    if query_without_one_piece:
        return f"One Piece {query_without_one_piece}".strip()
    return "One Piece Logo"


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
    
    prompt = (
        "You are an expert video content designer specializing in One Piece. "
        "Your job is to map timestamped subtitle lines to visually engaging image search queries, "
        "structured in JSON format.\n\n"

        "CRITICAL RULES:\n"
        "1. Every query MUST reference at least one canonical One Piece entity (character, arc, island, or crew).\n"
        "2. You are optimizing for the One Piece Fandom Wiki (MediaWiki API) — not a general image search engine.\n"
        "3. Use **short, canonical names** that directly match Fandom pages: characters (Luffy, Zoro, Shanks), arcs (Marineford, Dressrosa), or places (Mary Geoise, Wano Country).\n"
        "4. Do NOT use adjectives like 'dramatic', 'close up', or cinematic words — they reduce Fandom search accuracy.\n"
        "5. The `image_search_query` should ideally look like: 'Shanks Mary Geoise', 'Zoro Thriller Bark', 'Luffy Marineford', etc.\n"
        "6. When possible, combine character + location or character + event for higher precision.\n"
        "7. If multiple entities fit, choose the most specific combination (e.g., 'Shanks Reverie', not just 'Shanks').\n"
        "8. Detect if the subtitle talks about power, event, or location.\n"
        "9. Use appropriate query forms:\n"
        "  * Power-related → 'Character + Power term'\n"
        "  * Event-related → 'Character + Arc'\n"
        "  * Neutral → 'Character'\n"
        "  * Transition → 'Grand Line Map' or 'One Piece Logo'\n"
        "10. NEVER include the phrase 'One Piece anime' in Fandom queries.\n\n"

        "SCRIPT-WIDE CONTEXT:\n"
        f"- Detected core entities: {', '.join(context_entities) if context_entities else 'None detected; use One Piece Logo or Grand Line Map for transitions'}.\n"
        "- Every visual must match the local subtitle AND stay consistent with these core entities.\n"
        "- Do not introduce unrelated characters or arcs just because they are popular.\n"
        "- If the subtitle is a generic hook, use the most relevant detected core entity instead of a random famous character.\n\n"

        "MERGE RULES (fragments):\n"
        "- If a subtitle contains fewer than 4 words or is a fragment, merge it with adjacent subtitles "
        "to form a meaningful sentence.\n"
        "- Use the start_time of the first and end_time of the last merged fragment.\n\n"

        "TIMESTAMP RULES:\n"
        "- Maintain continuous timestamps (no gaps, no overlaps).\n"
        "- If two subtitles overlap, merge them.\n"
        "- If there's a gap, adjust previous end_time to match the next start_time.\n\n"

        "SPLITTING RULES:\n"
        "- MAX_DURATION = 3.5 seconds per segment.\n"
        "- If a merged segment exceeds MAX_DURATION, split evenly into subsegments (<=3.5s each).\n"
        "- Append '(part X/N)' in summary if split.\n\n"

        "SEGMENT GUIDELINES:\n"
        "- Normally map each subtitle line to one segment (after merging short fragments).\n"
        "- Duration should be 1–4 seconds (never exceed 5s).\n"
        "- Avoid merging more than 3 subtitles unless it's one continuous sentence.\n\n"

        "VISUAL VARIETY:\n"
        "- Rotate between characters, arcs, and settings (avoid using the same entity 3+ times in a row).\n"
        "- If a scene is ambiguous, default to the primary speaker or most relevant location.\n\n"

        "SILENCE HANDLING:\n"
        "- If there’s silence before/after subtitles, fill with neutral transition slides using canonical entities "
        "like 'One Piece Logo', 'Grand Line Map', or 'Sunny Ship'.\n\n"

        "OUTPUT FORMAT (STRICT):\n"
        "- Return ONLY a JSON array (no code fences or text).\n"
        "- Each element must have exactly these keys:\n"
        "  * start_time (string)\n"
        "  * end_time (string)\n"
        "  * summary (short paraphrase)\n"
        "  * image_search_query (short Fandom-optimized query — canonical names only)\n"
        "- Timestamps must be continuous: slides[i].end_time == slides[i+1].start_time.\n\n"

        "EXAMPLES:\n"
        "[\n"
        "  {\n"
        "    \"start_time\": \"0:00:03.10\",\n"
        "    \"end_time\": \"0:00:06.60\",\n"
        "    \"summary\": \"Shanks walks into Mary Geoise\",\n"
        "    \"image_search_query\": \"Shanks Mary Geoise\"\n"
        "  },\n"
        "  {\n"
        "    \"start_time\": \"0:00:06.60\",\n"
        "    \"end_time\": \"0:00:09.80\",\n"
        "    \"summary\": \"Zoro absorbs Luffy's pain (part 1/2)\",\n"
        "    \"image_search_query\": \"Zoro Luffy Thriller Bark\"\n"
        "  }\n"
        "]\n\n"

        "Now analyze the following timestamped subtitles, apply the fragment-merge and split logic, "
        "ensure timestamps are continuous, and return ONLY a JSON array following the above rules:\n\n"
        f"{raw_subtitles}"
    )


    try:
        response = model.generate_content(prompt)
        if not response.text:
            raise ValueError("Empty response from Gemini")
        
        # Log the raw response for debugging
        logger.info(f"Raw Gemini response:\n{response.text}")
            
        # Parse the JSON response
        try:
            # Clean up the response to ensure valid JSON
            json_str = response.text.strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:]
            if json_str.endswith('```'):
                json_str = json_str[:-3]
                
            gemini_slides = json.loads(json_str)
            logger.info(f"Successfully parsed {len(gemini_slides)} slides from Gemini response")
            
            # Create final slides using Gemini's grouping
            final_slides = []
            
            # For each Gemini slide (which may cover multiple original subtitles)
            for gemini_slide in gemini_slides:
                # Create one slide per Gemini group with its own timing
                slide = {
                    'start_time': gemini_slide['start_time'],
                    'end_time': gemini_slide['end_time'],
                    'summary': gemini_slide['summary'],
                    'image_search_query': _sanitize_image_query(
                        gemini_slide['image_search_query'],
                        gemini_slide['summary'],
                        context_entities,
                    ),
                    'context_entities': context_entities,
                }
                
                # Add image path if image_dir is provided
                if image_dir:
                    os.makedirs(image_dir, exist_ok=True)
                    slide_hash = hashlib.md5(slide['image_search_query'].encode()).hexdigest()[:8]
                    slide['image_path'] = os.path.join(image_dir, f"slide_{len(final_slides)+1:03d}_{slide_hash}.jpg")
                
                final_slides.append(slide)
            
            # Save the slides to a JSON file
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(final_slides, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Successfully generated {len(final_slides)} slides at {out_path}")
            
            # Download images if image_dir is provided
            if image_dir:
                download_images_for_slides(out_path, image_dir)
                
            return out_path
            
        except (json.JSONDecodeError, _json.JSONDecodeError) as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            logger.error(f"Response text: {response.text}")
            raise
            
    except Exception as e:
        logger.error(f"Error in generate_gemini_image_slides: {str(e)}")
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
        search_query = _sanitize_image_query(
            search_query,
            slide.get("summary", ""),
            slide.get("context_entities", []),
        )
        slide["image_search_query"] = search_query

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

        # 1) First try One Piece Fandom API for high-quality, relevant images
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
        
        # 2) Fall back to Google CSE with priority domains if Fandom didn't work
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
