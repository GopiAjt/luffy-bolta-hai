import re
import json
import requests
from typing import List, Dict
import os
from dotenv import load_dotenv
import google.generativeai as genai
import logging
import time
import shutil

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


def generate_gemini_image_slides(ass_path: str, out_path: str, total_duration: float, image_dir: str = None) -> str:
    # Read and group all dialogues as before
    dialogues = parse_ass_dialogues(ass_path)
    groups = group_dialogues(dialogues)
    # Prepare a single prompt with all grouped segments
    prompt_segments = []
    for group in groups:
        prompt_segments.append(
            f"[{group['start']} - {group['end']}] {group['text']}")
    all_segments = "\n".join(prompt_segments)
    prompt = (
        "You are an expert video content designer. Your task is to map subtitle segments to suggested Google image search queries, structured in JSON format.\n\n"

        f"Here are the grouped subtitle segments for a short video:\n{all_segments}\n\n"

        "First, scan all segments and determine the single most relevant main topic of the video "
        "(e.g., an anime like 'One Piece', a TV series, a movie, a historical event, or a sports match). "
        "If no single clear topic is found, choose the most recurring proper noun, entity, or theme across all segments. "
        "This topic should be the central subject that appears most frequently in names, settings, or themes. "
        "Always include this detected topic in every 'image_search_query', along with specific characters, abilities, settings, or iconic moments from that segment.\n\n"

        "Your goal is to generate a JSON array where each object includes:\n"
        "- start_time (string, e.g., '0:00:00')\n"
        "- end_time\n"
        "- summary (short subtitle text that captures the essence of the segment)\n"
        "- image_search_query (a concise, highly relevant Google image search string based on the segment, "
        "always containing the detected topic and additional details like characters, abilities, locations, or iconic moments)\n\n"

        "🕒 MERGING RULES (Strict):\n"
        "- Merge subtitle segments into one slide if BOTH of these are true:\n"
        "    1. The time gap between the end of one segment and the start of the next is ≤ 1.5 seconds.\n"
        "    2. The combined text length of the merged segments is ≤ 250 characters.\n"
        "- Do NOT merge if the time gap is greater than 1.5 seconds, unless the segments clearly form a single sentence that was split into multiple lines.\n"
        "- Treat small interjections (“Yeah”, “Okay”, “Right”) as part of the preceding or following segment if they occur within 1.5 seconds and are related in context.\n"
        "- If merging causes the segment to exceed 250 characters, split at a natural sentence boundary.\n"
        "- The start_time of a merged slide should be the start_time of the first segment, and the end_time should be from the last segment in the merge.\n\n"

        "⏳ TIMING COVERAGE RULES:\n"
        "- Ensure the generated slides cover the entire duration of the provided subtitle segments — no gaps.\n"
        "- Include even short or filler subtitle lines if they are part of an introduction, transition, or build-up to a key idea.\n"
        "- Each object must represent one complete thought or concept, even if it spans multiple subtitle lines.\n\n"

        "Output your result as valid JSON using this schema:\n"
        "[\n"
        "  {\n"
        "    \"start_time\": \"string\",\n"
        "    \"end_time\": \"string\",\n"
        "    \"summary\": \"string\",\n"
        "    \"image_search_query\": \"string\"\n"
        "  },\n"
        "  ...\n"
        "]\n\n"
        "Respond ONLY with valid JSON. Do not include any extra commentary, explanations, or code fences."
    )



    try:
        response = model.generate_content(prompt)
        if response.text:
            logger.info(f"Gemini raw response: {response.text}")
            import json as _json
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split('\n', 1)[-1]
                if text.endswith("```"):
                    text = text.rsplit('```', 1)[0]
                text = text.strip()
            try:
                slides = _json.loads(text)
                slides = filter_irrelevant_summaries(slides)
                slides = fill_gaps_in_slides(slides, total_duration)
            except Exception as parse_err:
                logger.error(
                    f"Failed to parse Gemini response as JSON: {response.text}")
                raise
        else:
            logger.error("No text generated by Gemini API")
            raise ValueError("No text generated by Gemini API")
    except Exception as e:
        logger.error(f"Error from Gemini API: {str(e)}")
        raise
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(slides, f, indent=2)
    # Automatically download images if image_dir is provided
    if image_dir:
        download_images_for_slides(out_path, image_dir)
    return out_path


def generate_image_slides(ass_path: str, out_path: str, total_duration: float, image_dir: str = None) -> str:
    """
    Backward-compatible alias for generate_gemini_image_slides.
    """
    return generate_gemini_image_slides(ass_path, out_path, total_duration, image_dir)


def google_image_search(query, api_key=None, cse_id=None, num_results=10):
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    cse_id = cse_id or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    url = "https://www.googleapis.com/customsearch/v1"
    
    # Add site: operators to prioritize specific anime and art websites
    prioritized_sites = [
        "onepiece.fandom.com",
        "zerochan.net",
        "sakugabooru.com",
        "artstation.com",
        "anidb.net",
        "crunchyroll.com",
        "myanimelist.net",
        "deviantart.com"
    ]
    if "site:" not in query.lower():
        sites_query = " OR ".join(f"site:{site}" for site in prioritized_sites)
        query = f"{query} ({sites_query})"
    
    params = {
        "q": query,
        "cx": cse_id,
        "key": api_key,
        "searchType": "image",
        "num": num_results,
        "safe": "active"
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    results = resp.json()
    if "items" in results and results["items"]:
        return results["items"]
    return []


def download_image(url, save_path):
    """Downloads an image from a URL to a given path."""
    try:
        resp = requests.get(url, stream=True, timeout=10)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image from {url}: {e}")
        raise


def download_images_for_slides(json_path, out_dir):
    # Clear the output directory before downloading new images
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading images to {out_dir} from {json_path}")
    logger.info(f"Downloading images to {out_dir} from {json_path}")
    
    # List of prioritized sites
    prioritized_sites = [
        # "pinterest.com",
        "onepiece.fandom.com",
        "zerochan.net",
        "sakugabooru.com",
        "artstation.com",
        "anidb.net",
        "crunchyroll.com",
        "myanimelist.net",
        "deviantart.com"
    ]
    
    with open(json_path, "r", encoding="utf-8") as f:
        slides = json.load(f)
    
    for idx, slide in enumerate(slides):
        search_query = slide.get("image_search_query")
        logger.info(f"[Slide {idx+1}] Image search query: {search_query}")
        if not search_query:
            logger.warning(f"No image_search_query for slide {idx+1}")
            continue

        try:
            # First, try with the original query and prioritized sites
            if "one piece" not in search_query.lower():
                search_query = f"{search_query} One Piece Luffy"
                
            # Try with prioritized sites first
            items = google_image_search(search_query, num_results=10)
            
            # If no results, try a broader search without site restrictions
            if not items:
                logger.info(f"[Slide {idx+1}] No results with prioritized sites, trying broader search...")
                # Remove any existing site: filters from the query
                base_query = ' '.join([word for word in search_query.split() if not word.startswith('site:')])
                items = google_image_search(base_query, num_results=10)
            logger.info(
                f"[Slide {idx+1}] Google image search returned {len(items)} results.")

            # Filter for actual images, prioritizing One Piece content from preferred sites
            valid_images = []
            for item_idx, item in enumerate(items):
                item_link = item.get('link', 'N/A')
                item_title = item.get('title', '').lower()
                item_mime = item.get('mime', 'N/A')
                item_width = item.get('image', {}).get('width', 0)
                item_height = item.get('image', {}).get('height', 0)
                
                # Check if the image is from a preferred site
                is_preferred = any(site in item_link.lower() for site in prioritized_sites)
                
                # Check if the content is One Piece related
                is_onepiece = ('one piece' in item_title or 
                             any(term in item_title for term in ['luffy', 'zoro', 'mihawk', 'shanks', 'kaido', 'big mom']))

                if 'image' not in item_mime:
                    logger.debug(
                        f"[Slide {idx+1} - Item {item_idx+1}] Skipping: Not an image MIME type ({item_mime}) - {item_link}")
                    continue

                if not item_width or not item_height:
                    logger.debug(
                        f"[Slide {idx+1} - Item {item_idx+1}] Skipping: Missing dimensions ({item_width}x{item_height}) - {item_link}")
                    continue

                # Calculate aspect ratio for reference (but don't filter by it)
                aspect_ratio = item_width / item_height
                
                # Calculate priority score (higher is better)
                priority = 0
                
                # Highest priority: One Piece content from preferred sites
                if is_onepiece and is_preferred:
                    priority = 3
                # High priority: One Piece content from any site
                elif is_onepiece:
                    priority = 2
                # Medium priority: Preferred site with any content
                elif is_preferred:
                    priority = 1
                
                # Slight preference for images closer to 16:9
                aspect_score = 1 - (abs(aspect_ratio - 16/9) / (16/9)) * 0.5  # Reduced weight of aspect ratio
                
                item['_priority'] = priority + aspect_score
                valid_images.append(item)
                
                logger.debug(
                    f"[Slide {idx+1} - Item {item_idx+1}] Accepted: "
                    f"{'ONEPIECE ' if is_onepiece else ''}"
                    f"{'PREFERRED ' if is_preferred else ''}"
                    f"Aspect: {aspect_ratio:.2f} - {item_link}")

            downloaded_successfully = False
            if valid_images:
                # Sort valid images by priority (One Piece content and preferred sites first)
                valid_images.sort(key=lambda x: x.get('_priority', 0), reverse=True)
                logger.debug(
                    f"[Slide {idx+1}] Found {len(valid_images)} valid images. "
                    f"Top 3 priorities: {[round(x.get('_priority', 0), 2) for x in valid_images[:3]]}")
                
                for item_idx, best_image in enumerate(valid_images):
                    img_url = best_image['link']
                    ext = os.path.splitext(img_url)[-1].split("?")[0] or ".jpg"
                    save_path = os.path.join(
                        out_dir, f"slide_{idx+1}_16x9{ext}")
                    try:
                        logger.info(
                            f"[Slide {idx+1}] Attempting to download 16:9 image {item_idx+1}: {img_url}")
                        download_image(img_url, save_path)
                        print(f"Downloaded: {save_path}")
                        logger.info(f"Downloaded: {save_path}")
                        downloaded_successfully = True
                        break  # Exit loop if download is successful
                    except Exception as download_e:
                        logger.warning(
                            f"[Slide {idx+1}] Failed to download 16:9 image {item_idx+1} from {img_url}: {download_e}")

            if not downloaded_successfully and items:
                logger.warning(
                    f"[Slide {idx+1}] No suitable 16:9 image downloaded. Attempting fallback to any image.")
                for item_idx, fallback_item in enumerate(items):
                    img_url_fallback = fallback_item['link']
                    item_mime = fallback_item.get('mime', '')
                    if 'image' not in item_mime:
                        logger.debug(
                            f"[Slide {idx+1} - Fallback Item {item_idx+1}] Skipping: Not an image MIME type ({item_mime}) - {img_url_fallback}")
                        continue

                    ext = os.path.splitext(
                        img_url_fallback)[-1].split("?")[0] or ".jpg"
                    # Use a different name for fallback
                    save_path_fallback = os.path.join(
                        out_dir, f"slide_{idx+1}_fallback{ext}")
                    try:
                        logger.info(
                            f"[Slide {idx+1}] Attempting to download fallback image {item_idx+1}: {img_url_fallback}")
                        download_image(img_url_fallback, save_path_fallback)
                        print(f"Downloaded (fallback): {save_path_fallback}")
                        logger.info(
                            f"Downloaded (fallback): {save_path_fallback}")
                        downloaded_successfully = True
                        break  # Exit loop if download is successful
                    except Exception as fallback_e:
                        logger.warning(
                            f"[Slide {idx+1}] Failed to download fallback image {item_idx+1} from {img_url_fallback}: {fallback_e}")

            if not downloaded_successfully:
                logger.error(
                    f"[Slide {idx+1}] No image could be downloaded for query: {search_query}")
                print(f"No image could be downloaded for slide {idx+1}")

        except Exception as e:
            print(f"Failed to download for slide {idx+1}: {e}")
            logger.error(f"Failed to download for slide {idx+1}: {e}")
        finally:
            time.sleep(1)  # Rate limit our requests
