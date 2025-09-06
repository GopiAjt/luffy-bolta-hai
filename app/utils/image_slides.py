import re
import json
import requests
from typing import List, Dict, Optional, Set, Tuple, Any
import os
from dotenv import load_dotenv
import google.generativeai as genai
import logging
import time
import shutil
import hashlib
from functools import lru_cache
from collections import defaultdict
import nltk
from nltk.corpus import stopwords

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

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
model = genai.GenerativeModel("gemini-2.0-flash")


def parse_ass_dialogues(ass_path: str) -> List[Dict]:
    dialogues = []
    dialogue_re = re.compile(
        r"Dialogue: [^,]*,([^,]*),([^,]*),[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,(.*)")
    with open(ass_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('Dialogue:'):
                match = dialogue_re.match(line)
                if match:
                    start, end, text = match.groups()
                    text = re.sub(
                        r'{.*?}', '', text).replace('\n', ' ').strip()
                    dialogues.append(
                        {'start': start.strip(), 'end': end.strip(), 'text': text})
    return dialogues


def group_dialogues(dialogues: List[Dict]) -> List[Dict]:
    groups = []
    current = None
    for dlg in dialogues:
        if not current:
            current = {'start': dlg['start'],
                       'end': dlg['end'], 'text': dlg['text']}
        else:
            if re.search(r'[.!?]$', current['text']) or (dlg['text'].lower().startswith('but') or dlg['text'].lower().startswith('and')):
                groups.append(current)
                current = {'start': dlg['start'],
                           'end': dlg['end'], 'text': dlg['text']}
            else:
                current['end'] = dlg['end']
                current['text'] += ' ' + dlg['text']
    if current:
        groups.append(current)
    return groups

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


def _detect_emotional_tone(text: str) -> str:
    """Detect the emotional tone of the given text."""
    text_lower = text.lower()
    emotional_indicators = {
        'happy': ['laugh', 'happy', 'joy', 'excite', 'celebrate', 'smile'],
        'sad': ['cry', 'tear', 'sad', 'grief', 'loss', 'death', 'die'],
        'angry': ['angry', 'rage', 'fury', 'yell', 'scream', 'fight', 'battle'],
        'shocked': ['shock', 'surprise', 'reveal', 'betray', 'betrayal', 'secret']
    }
    
    for emotion, indicators in emotional_indicators.items():
        if any(indicator in text_lower for indicator in indicators):
            return emotion
    return 'neutral'

def _is_transition_segment(text: str) -> bool:
    """Check if the segment is likely a transition or silence."""
    text_lower = text.lower()
    transition_indicators = {
        '...', '--', 'silence', 'pause', 'beat', 'moment', 'scene',
        'cut to', 'fade', 'meanwhile', 'later', 'after', 'before'
    }
    
    # Check for very short text or transition indicators
    if len(text_lower.split()) < 3:
        return True
        
    return any(indicator in text_lower for indicator in transition_indicators)

def generate_gemini_image_slides(ass_path: str, out_path: str, total_duration: float, image_dir: str = None) -> str:
    # Read and group all dialogues
    dialogues = parse_ass_dialogues(ass_path)
    grouped_dialogues = group_dialogues(dialogues)
    
    # Initialize context tracking
    context = {
        'characters': set(),
        'locations': set(),
        'current_arc': None,
        'recent_events': [],
        'emotional_tone': 'neutral',
        'previous_segment': None
    }
    
    # Prepare segments with context
    segments_with_context = []
    for i, d in enumerate(grouped_dialogues):
        is_transition = _is_transition_segment(d['text'])
        emotional_tone = _detect_emotional_tone(d['text'])
        
        # Update context
        context['emotional_tone'] = emotional_tone
        context['previous_segment'] = d['text']
        
        # Extract potential characters and locations
        key_terms = _extract_key_terms(d['text'])
        characters = [t for t in key_terms if _is_character_name(t)]
        locations = [t for t in key_terms if _is_location(t)]
        
        if characters:
            context['characters'].update(characters[:2])  # Keep only top 2 characters
        
        if locations:
            context['locations'].update(locations[:1])  # Keep only the main location
        
        segments_with_context.append({
            'start': d['start'],
            'end': d['end'],
            'text': d['text'],
            'is_transition': is_transition,
            'emotional_tone': emotional_tone,
            'characters': characters,
            'locations': locations,
            'context': context.copy()  # Store a snapshot of the current context
        })
    
    # Prepare the prompt for Gemini with context
    all_segments = "\n".join(
        f"- {d['start']} to {d['end']}: {d['text']} "
        f"[Context: {'Transition' if d['is_transition'] else 'Content'}, "
        f"Tone: {d['emotional_tone']}, "
        f"Chars: {', '.join(d['characters']) if d['characters'] else 'None'}, "
        f"Loc: {', '.join(d['locations']) if d['locations'] else 'None'}]"
        for d in segments_with_context
    )
    
    # Prepare context summary for the prompt
    context_summary = (
        "CONTEXT SUMMARY:\n"
        f"- Current emotional tone: {context['emotional_tone']}\n"
        f"- Recent characters: {', '.join(context['characters']) if context['characters'] else 'None'}\n"
        f"- Recent locations: {', '.join(context['locations']) if context['locations'] else 'None'}\n"
    )
    
    prompt = (
        "You are an expert video content designer specializing in One Piece. "
        "Your task is to map subtitle segments to suggested Google image search queries, structured in JSON format.\n\n"
        "CONTEXT AND GUIDELINES:\n"
        "1. Always include 'One Piece' in every image_search_query\n"
        "2. Priority order for query terms: characters > actions > locations > objects\n"
        "3. For multi-character scenes, include up to 2 main characters\n"
        "4. For fight/action scenes, include both characters and the action\n\n"
        "HANDLING DIFFERENT SCENE TYPES (based on context tags):\n\n"
        "For TRANSITION segments (marked as [Context: Transition]):\n"
        "- If between related scenes: 'One Piece [character/location] transition'\n"
        "- If after an intense scene: 'One Piece [character] reaction [emotion]'\n"
        "- If introducing a new location: 'One Piece [location] scenery'\n"
        "- For emotional transitions: 'One Piece [character] [emotion] moment'\n\n"
        "For ACTION scenes (contains fight/battle/attack terms):\n"
        "- Format: 'One Piece [character1] [action] [character2/object] [location]'\n"
        "- Example: 'One Piece Luffy punches Kaido Onigashima'\n"
        "- Include power-ups/transformations: 'One Piece Luffy Gear 5 vs Kaido'\n\n"
        "For DIALOGUE scenes (conversation-focused):\n"
        "- Focus on character expressions and relationships\n"
        "- Example: 'One Piece Luffy and Law talking Wano serious'\n"
        "- Include emotional tone: 'One Piece Nami crying emotional scene'\n\n"
        f"{context_summary}\n"
        "SEGMENT ANALYSIS:\n"
        "Each segment is annotated with context in square brackets:\n"
        "- [Context: Transition/Content] - Whether this is a transition or content segment\n"
        "- [Tone: emotion] - The emotional tone of the segment\n"
        "- [Chars: character1, character2] - Characters mentioned in the segment\n"
        "- [Loc: location] - Locations mentioned in the segment\n\n"
        f"Here are the grouped subtitle segments with context for a short video:\n{all_segments}\n\n"
        "OUTPUT FORMAT (JSON array of objects):\n"
        "[\n"
        "  {\n"
        "    \"start_time\": \"0:00:00\",\n"
        "    \"end_time\": \"0:00:10\",\n"
        "    \"summary\": \"Concise description of scene\",\n"
        "    \"image_search_query\": \"One Piece [relevant_terms]\"\n"
        "  }\n"
        "]\n\n"
        "IMPORTANT NOTES:\n"
        "1. For transition segments, use the context to determine the best image (reaction, location, etc.)\n"
        "2. Maintain character and location context between segments when appropriate\n"
        "3. For emotional scenes, include the emotion in the query\n"
        "4. For action scenes, be specific about the action and participants\n"
        "5. For dialogue scenes, focus on character expressions and relationships\n\n"
        "Now generate the JSON array for the segments above. "
        "Respond ONLY with valid JSON. Do not include any extra commentary or explanations."
    )

    try:
        # Import json at the function level to avoid issues
        import json as _json
        
        # Generate content with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                if not response.text:
                    raise ValueError("Empty response from Gemini API")
                
                logger.debug(f"Gemini raw response (attempt {attempt + 1}): {response.text}")
                text = response.text.strip()
                
                # Clean up the response if it's wrapped in markdown code blocks
                if '```' in text:
                    # Extract content between the first and last ```
                    start_idx = text.find('```')
                    end_idx = text.rfind('```')
                    if start_idx != -1 and end_idx != -1 and start_idx != end_idx:
                        text = text[start_idx + 3:end_idx].strip()
                        # Remove 'json' prefix if present
                        if text.startswith('json'):
                            text = text[4:].strip()
                
                # Try to find the first { and last } to extract JSON
                first_brace = text.find('{')
                last_brace = text.rfind('}')
                
                if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
                    json_str = text[first_brace:last_brace + 1]
                    logger.debug(f"Extracted JSON string: {json_str[:200]}...")  # Log first 200 chars
                    
                    # Try to parse the JSON
                    try:
                        slides = _json.loads(json_str)
                    except _json.JSONDecodeError as je:
                        # If parsing fails, try to find array start/end
                        first_bracket = text.find('[')
                        last_bracket = text.rfind(']')
                        if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
                            json_str = text[first_bracket:last_bracket + 1]
                            logger.debug(f"Trying array extraction: {json_str[:200]}...")
                            slides = _json.loads(json_str)
                        else:
                            raise
                else:
                    raise ValueError("Could not find valid JSON in response")
                
                # If we get here, parsing was successful
                break
                
            except (_json.JSONDecodeError, ValueError) as e:
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(f"Failed to parse Gemini response after {max_retries} attempts")
                    raise
                logger.warning(f"JSON parse error on attempt {attempt + 1}, retrying...")
                time.sleep(1)  # Wait before retrying
        
        # Validate the structure
        if not isinstance(slides, list):
            raise ValueError(f"Expected a list of slides, got {type(slides).__name__}")
            
        for i, slide in enumerate(slides):
            if not isinstance(slide, dict):
                raise ValueError(f"Slide {i} is not a dictionary")
                
            required_keys = ["start_time", "end_time", "summary", "image_search_query"]
            for key in required_keys:
                if key not in slide:
                    raise ValueError(f"Slide {i} is missing required key: {key}")
                if not isinstance(slide[key], str):
                    raise ValueError(f"Slide {i} has invalid type for {key}: {type(slide[key]).__name__}")
        
        # Fill any gaps in the timeline
        slides = fill_gaps_in_slides(slides, total_duration)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        
        # Save the slides to the output file with atomic write
        temp_path = f"{out_path}.tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            _json.dump(slides, f, indent=2, ensure_ascii=False)
        
        # Atomic rename to prevent partial writes
        if os.path.exists(out_path):
            os.remove(out_path)
        os.rename(temp_path, out_path)
            
        logger.info(f"Successfully generated {len(slides)} slides at {out_path}")
        
        # Download images if image_dir is provided
        if image_dir:
            download_images_for_slides(out_path, image_dir)
            
        return out_path
        
    except Exception as e:
        logger.error(f"Error generating image slides: {str(e)}")
        logger.debug("Error details:", exc_info=True)
        
        # If we have a partial response, try to save it for debugging
        if 'slides' in locals() and slides:
            debug_path = f"{out_path}.error.json"
            try:
                with open(debug_path, 'w', encoding='utf-8') as f:
                    _json.dump({
                        'error': str(e),
                        'partial_slides': slides[:10]  # Only save first 10 slides to avoid huge files
                    }, f, indent=2)
                logger.info(f"Saved partial response to {debug_path} for debugging")
            except Exception as save_error:
                logger.error(f"Failed to save partial response: {str(save_error)}")
        
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
    
    try:
        logger.debug(f"Executing search: {query}")
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        results = resp.json()
        return results.get("items", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Google Image Search API error for query '{query}': {str(e)}")
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
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    cse_id = cse_id or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    
    if not api_key or not cse_id:
        logger.error("Missing required API key or CSE ID")
        return []
    
    # Try with the exact query first
    items = _google_image_search_impl(query, api_key, cse_id, num_results)
    
    # If no results, try a broader search by removing any special characters
    if not items and any(c in query for c in ':"'):
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

def download_image(url: str, save_path: str, timeout: int = 10) -> bool:
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
                
            # Get file extension from content type or URL
            ext = None
            if 'jpeg' in content_type or 'jpg' in content_type:
                ext = '.jpg'
            elif 'png' in content_type:
                ext = '.png'
            elif 'gif' in content_type:
                logger.info(f"Skipping GIF file (content-type): {url}")
                return False
            else:
                # Try to get extension from URL
                url_path = url.split('?')[0]  # Remove query params
                url_ext = os.path.splitext(url_path)[1].lower()
                if url_ext in ['.jpg', '.jpeg', '.png', '.webp']:
                    ext = url_ext
                elif url_ext == '.gif':
                    logger.info(f"Skipping GIF file (URL extension): {url}")
                    return False
                else:
                    ext = '.jpg'  # Default to jpg if unknown
            
            # Update save_path with correct extension if needed
            if ext and not save_path.lower().endswith(ext):
                save_path = os.path.splitext(save_path)[0] + ext
            
            # Download in chunks to handle large files
            with open(save_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
            
            # Verify the file was saved properly
            if not os.path.exists(save_path):
                logger.error(f"Failed to save image to {save_path}")
                return False
                
            file_size = os.path.getsize(save_path)
            if file_size == 0:
                logger.error(f"Downloaded file is empty: {save_path}")
                os.remove(save_path)
                return False
                
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


def _extract_key_terms(text: str) -> List[str]:
    """Extract key terms from text, removing common words and duplicates."""
    # Custom stop words for One Piece context
    custom_stopwords = set(stopwords.words('english')) - {'one', 'piece', 'straw', 'hat', 'devil', 'fruit'}
    custom_stopwords.update({
        'would', 'could', 'should', 'might', 'must', 'may', 'shall',
        'also', 'even', 'still', 'much', 'many', 'every', 'very', 'really',
        'just', 'like', 'get', 'go', 'see', 'make', 'know', 'take', 'use',
        'come', 'think', 'look', 'want', 'need', 'way', 'thing', 'things',
        'something', 'anything', 'nothing', 'everything', 'someone', 'anyone',
        'no one', 'everyone', 'somebody', 'anybody', 'nobody', 'everybody',
        'somewhere', 'anywhere', 'nowhere', 'everywhere', 'some', 'any',
        'no', 'none', 'all', 'both', 'each', 'few', 'more', 'most', 'other',
        'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'will',
        'can', 'may', 'might', 'must', 'shall', 'should', 'ought', 'dare',
        'need', 'used', 'using', 'one', 'two', 'three', 'four', 'five',
        'first', 'second', 'third', 'fourth', 'fifth', 'another', 'next',
        'last', 'previous', 'early', 'later', 'ago', 'before', 'after',
        'since', 'until', 'when', 'while', 'during', 'before', 'after',
        'until', 'till', 'by', 'for', 'from', 'in', 'into', 'of', 'on',
        'to', 'with', 'without', 'about', 'against', 'between', 'through',
        'under', 'over', 'again', 'further', 'then', 'once', 'here', 'there',
        'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very'
    })
    
    # Tokenize and clean the text
    words = nltk.word_tokenize(text.lower())
    words = [w for w in words if w.isalnum() and w not in custom_stopwords and len(w) > 2]
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(words))

def _is_character_name(term: str) -> bool:
    """Check if a term is a One Piece character name."""
    one_piece_characters = {
        'luffy', 'zoro', 'nami', 'usopp', 'sanji', 'chopper', 'robin', 'franky',
        'brook', 'jinbe', 'ace', 'sabo', 'shanks', 'buggy', 'mihawk', 'crocodile',
        'doflamingo', 'kaido', 'big mom', 'blackbeard', 'whitebeard', 'law', 'kidd',
        'yamato', 'carrot', 'vivi', 'bonney', 'kuma', 'ivankov', 'dragon', 'garp',
        'sengoku', 'akainu', 'kizaru', 'fujitora', 'ryokugyu', 'smoker', 'tashigi'
    }
    return term.lower() in one_piece_characters

def _is_location(term: str) -> bool:
    """Check if a term is a One Piece location."""
    one_piece_locations = {
        'east blue', 'grand line', 'new world', 'mariejois', 'mary geoise', 'reverie',
        'alabasta', 'skypiea', 'water 7', 'enies lobby', 'thriller bark', 'sabaody',
        'fishman island', 'dressrosa', 'whole cake', 'wano', 'onigashima', 'elbaf',
        'marinford', 'impel down', 'amazon lily', 'punk hazard', 'zou', 'wci', 'wano'
    }
    return term.lower() in one_piece_locations

def _is_ability(term: str) -> bool:
    """Check if a term is a One Piece ability or power."""
    one_piece_abilities = {
        'haki', 'haoshoku', 'busoshoku', 'kenbunshoku', 'armament', 'observation',
        'conqueror', 'conquerors', 'devil fruit', 'logia', 'zoan', 'paramecia',
        'awakening', 'tekkai', 'sorcery', 'mink', 'sulong', 'voice', 'roger',
        'poseidon', 'pluton', 'uranus', 'ancient weapon', 'poneglyph', 'road'
    }
    return term.lower() in one_piece_abilities

def _generate_contextual_queries(slide: Dict, context: Dict) -> List[str]:
    """Generate contextual search queries for a slide with improved fallbacks."""
    def clean_query(query: str) -> str:
        """Clean and format search query."""
        return ' '.join(query.split())  # Remove extra whitespace
    
    queries = []
    summary = slide.get('summary', '').lower()
    key_terms = _extract_key_terms(summary)
    
    # Check if this is a silence/transition slide
    is_transition = any(term in summary.lower() for term in ['silence', 'transition', '...', '--', 'beat', 'pause'])
    
    # Extract entities with more detailed analysis
    characters = [t for t in key_terms if _is_character_name(t)]
    locations = [t for t in key_terms if _is_location(t)]
    abilities = [t for t in key_terms if _is_ability(t)]
    actions = [t for t in key_terms if t.endswith(('ing', 'ed')) and t not in abilities]
    
    # Update context with new information
    context['characters'].update(characters)
    context['locations'].update(locations)
    context['abilities'].update(abilities)
    
    # For transition/silence slides, generate more relevant queries
    if is_transition:
        # 1. Try to maintain context from previous/next slides
        if context.get('characters'):
            chars = list(context['characters'])[:2]  # Get up to 2 recent characters
            queries.extend([
                clean_query(f"One Piece {' '.join(chars)} scene"),
                clean_query(f"One Piece {' '.join(chars)} moment"),
                clean_query(f"One Piece {' '.join(chars)} reaction"),
                clean_query(f"One Piece {' '.join(chars)} expression")
            ])
        
        # 2. Use location if available
        if context.get('locations'):
            loc = next(iter(context['locations']))
            queries.extend([
                clean_query(f"One Piece {loc} scenery"),
                clean_query(f"One Piece {loc} background"),
                clean_query(f"One Piece {loc} location")
            ])
        
        # 3. Add emotional context if available
        emotion = context.get('emotional_tone', 'neutral')
        if emotion != 'neutral':
            queries.extend([
                clean_query(f"One Piece {emotion} moment"),
                clean_query(f"One Piece {emotion} scene")
            ])
    
    # Regular content processing (for non-transition slides)
    if not is_transition or not queries:  # If no transition queries were generated
        # 1. Most specific queries (character + action + location)
        if characters and actions and locations:
            queries.append(clean_query(
                f"One Piece {' '.join(characters)} {' '.join(actions)} at {' '.join(locations)}"
            ))
        
        # 2. Character + Action/Ability
        for action in actions + abilities:
            if characters:
                for char in characters:
                    queries.append(clean_query(f"One Piece {char} {action}"))
        
        # 3. Location + Event
        if locations and (actions or abilities):
            queries.append(clean_query(
                f"One Piece {summary} at {' '.join(locations)}"
            ))
        
        # 4. Character + Summary (if not too long)
        if characters and len(summary) < 50:
            queries.append(clean_query(f"One Piece {' '.join(characters)} {summary}"))
        
        # 5. Context from previous slides
        if context.get('characters') and (actions or abilities):
            for action in actions + abilities:
                queries.append(clean_query(
                    f"One Piece {' '.join(context['characters'])} {action}"
                ))
    
    # Fallback combinations (for both transition and regular slides)
    if len(key_terms) >= 2:
        # Try different term combinations
        for i in range(min(3, len(key_terms))):
            for j in range(i+1, min(i+4, len(key_terms))):
                queries.append(clean_query(
                    f"One Piece {key_terms[i]} {key_terms[j]}"
                ))
    
    # Progressive fallbacks (more generic queries)
    fallbacks = []
    
    # Character-based fallbacks
    if characters:
        fallbacks.extend([
            clean_query(f"One Piece {char} scene") for char in characters[:2]
        ])
    
    # Location-based fallbacks
    if locations:
        fallbacks.extend([
            clean_query(f"One Piece {loc} scene") for loc in locations[:1]
        ])
    
    # Action/emotion based fallbacks
    if actions:
        fallbacks.extend([
            clean_query(f"One Piece {action} scene") for action in actions[:1]
        ])
    
    # Generic but relevant fallbacks
    fallbacks.extend([
        clean_query("One Piece " + ' '.join(summary.split()[:5])),  # First 5 words
        clean_query(f"One Piece {summary}"),
        clean_query(f"One Piece {summary} scene"),
        clean_query(f"One Piece {summary} moment")
    ])
    
    # Add context-based fallbacks
    if context.get('characters'):
        chars = list(context['characters'])[:2]
        fallbacks.append(clean_query(f"One Piece {' '.join(chars)} scene"))
    
    if context.get('locations'):
        loc = next(iter(context['locations']), '')
        if loc:
            fallbacks.append(clean_query(f"One Piece {loc} scene"))
    
    # Add some generic but useful fallbacks
    fallbacks.extend([
        "One Piece scene",
        "One Piece moment",
        "One Piece anime"
    ])
    
    # Combine all queries, ensuring uniqueness and proper order
    all_queries = []
    seen = set()
    
    # First add the most specific queries
    for q in queries + fallbacks:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            all_queries.append(q)
    
    # Ensure we always have at least one fallback
    if not all_queries:
        all_queries.append("One Piece")
    
    # Log the generated queries for debugging
    logger.debug(f"Generated {len(all_queries)} queries for slide: {all_queries}")
    
    return all_queries

def download_images_for_slides(json_path: str, out_dir: str) -> None:
    """Download images for slides based on search queries in a JSON file.
    
    Args:
        json_path: Path to the JSON file containing slide data
        out_dir: Directory to save downloaded images
    """
    # Clear the output directory before downloading new images
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    logger.info(f"Downloading images to {out_dir} from {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        slides = json.load(f)

    # Initialize context tracking
    context = {
        'characters': set(),
        'locations': set(),
        'abilities': set(),
        'current_arc': None
    }

    # Track downloaded image hashes to avoid duplicates
    downloaded_hashes = set()

    for idx, slide in enumerate(slides):
        search_query = slide.get("image_search_query")
        if not search_query:
            logger.warning(f"[Slide {idx+1}] No image_search_query")
            continue

        # Generate contextual queries for this slide
        queries = _generate_contextual_queries(slide, context)
        logger.info(f"[Slide {idx+1}] Generated {len(queries)} search queries")

        # Track if we've successfully downloaded an image
        downloaded = False

        # Try each query until we find a suitable image
        for query in queries:
            if downloaded:
                break

            logger.debug(f"[Slide {idx+1}] Trying search: {query}")
            try:
                items = google_image_search(query, num_results=5)  # Reduced number of results per query

                if not items:
                    logger.debug(f"[Slide {idx+1}] No results for: {query}")
                    continue

                logger.info(f"[Slide {idx+1}] Found {len(items)} results for: {query}")

                # Process search results
                for item in items:
                    if downloaded:
                        break

                    item_link = item.get('link', '')
                    item_title = item.get('title', '').lower()
                    item_mime = item.get('mime', '')

                    # Skip if no link or invalid MIME type
                    if not item_link or not item_mime.startswith('image/'):
                        continue

                    # Check for One Piece relevance with more specific checks
                    is_relevant = _is_relevant_image(item, context)
                    if not is_relevant:
                        logger.debug(f"[Slide {idx+1}] Skipping irrelevant image: {item_title}")
                        continue

                    # Check for duplicates using perceptual hashing
                    is_duplicate, image_hash = _is_duplicate_image(item, downloaded_hashes)
                    if is_duplicate:
                        logger.debug(f"[Slide {idx+1}] Skipping duplicate image: {item_title}")
                        continue

                    # Try to download the image with better error handling
                    filename = f"slide_{idx+1:03d}_{image_hash[:8]}.jpg"
                    save_path = os.path.join(out_dir, filename)

                    if download_image(item_link, save_path):
                        downloaded_hashes.add(image_hash)
                        slide["image_path"] = save_path
                        downloaded = True
                        logger.info(f"[Slide {idx+1}] Successfully downloaded: {os.path.basename(save_path)}")

                        # Update context with successful download info
                        _update_context_from_image(slide, item, context)

            except Exception as e:
                logger.error(f"[Slide {idx+1}] Error processing query '{query}': {str(e)}")
                continue

        if not downloaded:
            logger.warning(f"[Slide {idx+1}] Failed to download image after all queries")

    # Save the updated slides with image paths
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(slides, f, indent=2, ensure_ascii=False)

    # Log download statistics
    downloaded_count = sum(1 for slide in slides if "image_path" in slide)
    success_rate = (downloaded_count / len(slides)) * 100 if slides else 0
    logger.info(
        f"Download complete. Successfully downloaded {downloaded_count}/{len(slides)} "
        f"images ({success_rate:.1f}% success rate)"
    )

    return downloaded_count


def _is_relevant_image(item: Dict, context: Dict) -> bool:
    """Check if an image is relevant based on its metadata and context."""
    title = item.get('title', '').lower()
    snippet = item.get('snippet', '').lower()

    # Basic One Piece relevance check
    if 'one piece' not in title and 'one piece' not in snippet:
        if not any(char in title for char in context['characters']):
            return False

    # Check for common irrelevant content
    blacklist = {'fan art', 'drawing', 'sketch', 'manga panel', 'color page', 'amv', 'opening', 'ending'}
    if any(term in title for term in blacklist):
        return False

    return True


def _update_context_from_image(slide: Dict, image_data: Dict, context: Dict) -> None:
    """Update context based on successfully downloaded image."""
    # Extract potential character names from image title/description
    title = image_data.get('title', '').lower()
    snippet = image_data.get('snippet', '').lower()

    # Update context with any new characters found
    for term in _extract_key_terms(title + ' ' + snippet):
        if _is_character_name(term) and term not in context['characters']:
            context['characters'].add(term)

    # Update locations if mentioned
    for term in _extract_key_terms(snippet):
        if _is_location(term) and term not in context['locations']:
            context['locations'].add(term)
