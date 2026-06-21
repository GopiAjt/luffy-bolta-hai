import re
import json
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv
import logging

from app.config import normalize_video_profile
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
    "Loki", "Dory", "Brogy", "Yggdrasil", "Little Garden", "Drum Island",
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
    "loki": "Loki",
    "prince loki": "Loki",
    "dory": "Dory",
    "brogy": "Brogy",
    "yggdrasil": "Yggdrasil",
    "little garden": "Little Garden",
    "drum island": "Drum Island",
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

PRODUCTION_BEAT_TYPES = {"hook", "setup", "evidence", "reveal", "twist", "escalation", "payoff", "resolution", "cta"}
EDITOR_VISUAL_ROLES = {
    "character",
    "evidence",
    "location",
    "object",
    "symbol",
    "comparison",
    "section_card",
    "quote_card",
    "timeline",
    "cta_card",
}
LEGACY_VISUAL_ROLES = {"title_card", "scene_broll", "evidence_card", "lower_third"}
VISUAL_ROLES = EDITOR_VISUAL_ROLES | LEGACY_VISUAL_ROLES
LAYOUT_MODES = {
    "short_vertical": {"safe_subject", "title_card", "quote_card", "evidence_card", "full_bleed"},
    "long_youtube": {"horizontal_feature", "split_context", "section_card", "quote_card", "evidence_card", "full_bleed"},
}
MOTION_PRESETS = {"subject_push", "evidence_hold", "reveal_zoom", "wide_pan", "title_card_hold", "slow_push", "hold_still", "static_hold"}
STORYBOARD_FIELDS = (
    "visual_purpose",
    "viewer_focus",
    "manual_upload_brief",
    "avoid_visual_reuse",
    "asset_metadata",
    "visual_intent",
    "emotion_state",
    "visual_diversity_score",
    "diversity_notes",
    "character_relationships",
    "retention_score",
    "retention_actions",
    "composition_layers",
    "motion_plan",
)
TRANSITION_BY_BEAT = {
    "hook": "crossfade",
    "setup": "fade_eased",
    "evidence": "crossfade",
    "reveal": "zoom_dissolve",
    "twist": "glitch_cut",
    "escalation": "whip_pan_right",
    "payoff": "fade_eased",
    "resolution": "fade_eased",
    "cta": "fade_eased",
}

# These are effect_cues names — NOT valid transition_in values.
# If the LLM accidentally puts one of these in transition_in, we sanitize it out.
_EFFECT_ONLY_NAMES: frozenset = frozenset({
    "flash_frame", "impact_shake", "red_eye_flash", "glitch",
    "dark_vignette", "desaturation", "chromatic_shift", "slow_push_in",
    "morph", "bass_hit",
})

# Motion preset aliases that sometimes leak into effect_cues. These are camera
# movements, not per-frame pixel effects, so strip them from effect_cues.
_MOTION_ONLY_NAMES: frozenset = frozenset({
    "slow_push_in", "slow_push", "pull_out", "pull_out_zoom",
    "reveal_zoom", "impact_zoom", "wide_pan", "stable_pan",
    "diagonal_pan", "subject_push", "evidence_hold", "hold_still",
    "title_card_hold", "dramatic_push", "push_in", "creep_in",
})

# Common English stop words and single-letter words that produce false-positive
# word-position matches for sfx_target_word. Any target_word in this set gets
# cleared during sanitization.
_SFX_TARGET_STOP_WORDS: frozenset = frozenset({
    "a", "an", "the", "is", "was", "were", "are", "be", "been",
    "not", "no", "nor", "but", "or", "and", "if", "in", "on",
    "to", "of", "at", "by", "for", "it", "its", "so", "as",
    "he", "she", "his", "her", "him", "we", "us", "our", "they",
    "do", "did", "had", "has", "have", "will", "can", "may",
    "that", "this", "with", "from", "who", "what", "how",
})

# Effect cues that require a minimum number of frames to render visibly.
# If a slide is shorter than _MIN_EFFECT_DURATION_SECONDS, these are stripped.
_DURATION_SENSITIVE_EFFECTS: frozenset = frozenset({
    "impact_shake", "red_eye_flash", "glitch", "chromatic_shift",
})
_MIN_EFFECT_DURATION_SECONDS: float = 0.5
VISUAL_ARCHITECTURE_STAGES = [
    "Subtitles",
    "Script Analyzer",
    "Story Beat Detector",
    "Emotional Curve Builder",
    "Visual Intent Classifier",
    "Asset Selector",
    "Motion Planner",
    "Retention Optimizer",
    "Final Slides",
]


def parse_ass_dialogues(ass_path: str, min_words=3) -> List[Dict[str, str]]:
    """Parse .ass file and extract dialogues with timestamps.
    
    Args:
        ass_path: Path to the .ass subtitle file
        min_words: Minimum number of words to consider a line complete (lines with fewer words will be grouped)
    """
    raw_dialogues = []
    storyboard_dialogues = []
    
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
            
        # Parse visible subtitles and hidden Storyboard comment lines. Long-form
        # videos burn only sparse keyword callouts, so the storyboard comments
        # carry the full narration for slide planning.
        lower_line = line.lower()
        if lower_line.startswith(('dialogue:', 'comment:')):
            parts = line.split(',', 9)  # Split into max 10 parts
            if len(parts) >= 10:
                event_type = parts[0].split(':', 1)[0].strip().lower()
                style_name = parts[3].strip().lower()
                if event_type == "comment" and style_name != "storyboard":
                    continue
                if event_type == "dialogue" and style_name == "storyboard":
                    continue
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
                    target = storyboard_dialogues if event_type == "comment" else raw_dialogues
                    target.append({
                        'start': start_time,
                        'end': end_time,
                        'text': text,
                        'word_count': len(text.split())
                    })

    if storyboard_dialogues:
        raw_dialogues = storyboard_dialogues
    
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


def _classify_production_beat(text: str, index: int, total: int) -> str:
    lowered = (text or "").lower()
    if index == 0:
        return "hook"
    if index >= max(0, total - 1) or re.search(r"\b(follow|comment|subscribe|tell me|like)\b", lowered):
        return "cta"
    if re.search(r"\b(twist|shocking|but actually|plot twist|never expected|impossible|what if|turns out)\b", lowered):
        return "twist"
    if re.search(r"\b(escalat|stronger|even more|worse|intensif|on top of that|not only|also|furthermore)\b", lowered):
        return "escalation"
    if re.search(r"\breveal|truth|secret|hidden|clue|impossible|danger|real reason)\b", lowered):
        return "reveal"
    if re.search(r"\b(answer|resolution|ending|finally|in the end|closure|conclusion|that's why)\b", lowered):
        return "resolution"
    if re.search(r"\bpayoff|means|proves|changes|therefore|because|so this\b", lowered):
        return "payoff"
    if index <= 2:
        return "setup"
    return "evidence"


def _default_visual_role(beat_type: str, text: str) -> str:
    lowered = (text or "").lower()
    if beat_type == "hook":
        return "section_card"
    if beat_type == "cta":
        return "cta_card"
    if re.search(r"\b(vs\.?|versus|opposite|unlike|contrast|compared|both|difference)\b", lowered):
        return "comparison"
    if re.search(r"\b(years?|decades?|century|timeline|chapter|episode|before|after|eventually|first|then|finally)\b", lowered):
        return "timeline"
    if re.search(r"\b(fruit|sword|weapon|poneglyph|flag|logo|symbol|crown|throne|ship|map|scar|chains?)\b", lowered):
        return "object"
    if re.search(r"\b(jaya|marineford|elbaf|wano|egghead|ohara|mary geoise|impel down|sabaody|baratie|thriller bark|island|village|kingdom|room)\b", lowered):
        return "location"
    if re.search(r"\b(world government|five elders|holy knights|nika|joy boy|will of d|straw hat|pirate flag)\b", lowered):
        return "symbol"
    if re.search(r"\bchapter|episode|sbs|quote|said|says\b", lowered):
        return "evidence"
    if '"' in text or "'" in text and len(text) < 140:
        return "quote_card"
    if re.search(r"\b(proves|evidence|because|therefore|choice|decision|action|behavior|consequence|betrayal|sacrifice)\b", lowered):
        return "evidence"
    return "character"


def _default_layout_mode(video_profile: str, beat_type: str, visual_role: str) -> str:
    video_profile = normalize_video_profile(video_profile)
    if video_profile == "long_youtube":
        if visual_role in {"title_card", "section_card", "cta_card"} or beat_type in {"hook", "payoff"}:
            return "section_card"
        if visual_role in {"evidence", "evidence_card", "timeline", "object", "symbol"}:
            return "evidence_card"
        if visual_role == "quote_card":
            return "quote_card"
        if visual_role == "comparison":
            return "split_context"
        return "horizontal_feature"
    if visual_role in {"title_card", "section_card", "cta_card"}:
        return "title_card"
    if visual_role in {"evidence", "evidence_card", "timeline", "object", "symbol"}:
        return "evidence_card"
    if visual_role == "quote_card":
        return "quote_card"
    return "safe_subject"


def _default_motion_preset(video_profile: str, beat_type: str, layout_mode: str) -> str:
    video_profile = normalize_video_profile(video_profile)
    if layout_mode in {"title_card", "section_card"} or beat_type == "cta":
        return "title_card_hold"
    if beat_type == "reveal":
        return "reveal_zoom"
    if video_profile == "long_youtube" and layout_mode == "horizontal_feature":
        return "wide_pan"
    if beat_type in {"evidence", "setup"}:
        return "evidence_hold"
    return "subject_push"


def _text_overlay_for_slide(summary: str, beat_type: str, visual_role: str) -> str:
    if visual_role not in {"title_card", "section_card", "cta_card", "evidence", "evidence_card", "quote_card", "timeline", "comparison"}:
        return ""
    cleaned = _clean_prompt_text(summary, max_len=82)
    if not cleaned:
        return ""
    if beat_type == "cta":
        return "Comment your theory"
    return cleaned


def _viewer_focus_for_slide(summary: str, subtitle_text: str, beat_type: str, visual_role: str) -> str:
    text = _clean_prompt_text(summary or subtitle_text, max_len=120)
    if not text:
        return "A clear canon visual that supports this narration beat."
    if beat_type == "hook":
        return f"The opening visual question: {text}"
    if visual_role in {"evidence", "evidence_card"}:
        return f"The specific evidence or canon behavior behind: {text}"
    if visual_role == "quote_card":
        return f"The key thesis line the viewer should remember: {text}"
    if visual_role == "object":
        return f"The object or power that makes this narration concrete: {text}"
    if visual_role == "location":
        return f"The place or arc that grounds this beat in canon: {text}"
    if visual_role == "symbol":
        return f"The symbol/institution that carries the meaning of this beat: {text}"
    if visual_role == "comparison":
        return f"The contrast the viewer should compare on screen: {text}"
    if visual_role == "timeline":
        return f"The sequence of time or escalation behind: {text}"
    if beat_type == "cta":
        return "A clean outro/CTA visual that leaves the topic in mind."
    return f"The concrete scene, character, object, or location that makes this beat visible: {text}"


def _visual_purpose_for_slide(summary: str, subtitle_text: str, beat_type: str, visual_role: str) -> str:
    text = _clean_prompt_text(summary or subtitle_text, max_len=100)
    if beat_type == "hook":
        return f"Create curiosity and establish stakes: {text}" if text else "Create curiosity and establish stakes."
    if beat_type == "setup":
        return f"Set up the core idea the viewer must understand: {text}" if text else "Set up the core idea."
    if beat_type == "reveal":
        return f"Reveal the turn or hidden implication: {text}" if text else "Reveal the turn."
    if beat_type == "payoff":
        return f"Pay off the argument with a clear takeaway: {text}" if text else "Pay off the argument."
    if beat_type == "cta":
        return "Close the video and keep the topic/question memorable."
    role_purpose = {
        "character": "Identify the person driving the beat.",
        "evidence": "Prove the claim with a concrete canon beat.",
        "location": "Ground the argument in a recognizable arc or place.",
        "object": "Make an abstract idea tangible through an object or power.",
        "symbol": "Represent a faction, ideology, or larger theme.",
        "comparison": "Show the contrast that makes the argument clear.",
        "timeline": "Show escalation or change over time.",
        "quote_card": "Make the thesis line stick.",
        "section_card": "Signal a new argument section.",
    }
    return role_purpose.get(visual_role, "Support the narration with a concrete visual function.")


def _manual_upload_brief_for_slide(query: str, viewer_focus: str) -> str:
    query = _clean_prompt_text(query, max_len=80)
    focus = _clean_prompt_text(viewer_focus, max_len=120)
    if query and focus:
        return f"Choose an image showing {query}. It should communicate: {focus}"
    if query:
        return f"Choose a clear image showing {query}."
    return "Choose a clear One Piece image that directly supports this narration beat."


def _emphasis_words(text: str, context_entities: List[str], limit: int = 4) -> List[str]:
    words = []
    for entity in context_entities:
        if entity and entity not in words:
            words.append(entity)
    for match in re.finditer(r"\b(chapter\s+\w+|void century|world government|pirate king|devil fruit|elbaf|loki|zoro|luffy)\b", text or "", re.I):
        value = match.group(0).strip()
        if value and value not in words:
            words.append(value)
    return words[:limit]


def _asset_confidence(query: str, subtitle_text: str, context_entities: List[str], duplicate_count: int = 0) -> float:
    score = 0.35
    if query and len(query.split()) >= 2:
        score += 0.22
    if any(entity.lower() in (query or "").lower() for entity in context_entities):
        score += 0.20
    if re.search(r"\bchapter|arc|island|village|bark|lobby|garden|elbaf|ohara|wano|marineford\b", f"{query} {subtitle_text}", re.I):
        score += 0.15
    score -= min(0.20, duplicate_count * 0.08)
    return round(max(0.0, min(1.0, score)), 2)


def _asset_type_for_role(visual_role: str, query: str, text: str) -> str:
    lowered = f"{query} {text}".lower()
    if visual_role in {"object", "symbol", "location", "comparison", "timeline"}:
        return visual_role
    if re.search(r"\b(island|village|kingdom|jaya|baratie|elbaf|wano|egghead|ohara|marineford|sabaody)\b", lowered):
        return "location"
    if re.search(r"\b(fruit|sword|knife|ship|map|poneglyph|throne|chains?|stairs?|scar|flag|logo)\b", lowered):
        return "object"
    if re.search(r"\b(world government|nika|joy boy|jolly roger|sun god|void century)\b", lowered):
        return "symbol"
    return "character"


def _build_asset_metadata(slide: Dict, context_entities: List[str], duplicate_count: int = 0) -> Dict:
    text = " ".join(
        part
        for part in [slide.get("subtitle_text", ""), slide.get("summary", ""), slide.get("viewer_focus", "")]
        if part
    )
    query = slide.get("image_search_query", "") or "One Piece Logo"
    entities = slide.get("context_entities") or _extract_context_entities(f"{query} {text}", limit=6) or context_entities
    asset_type = _asset_type_for_role(slide.get("visual_role", ""), query, text)
    tags = _dedupe_query_words(f"{query} {' '.join(entities[:3])} {asset_type}").split()[:10]
    confidence = slide.get("asset_confidence")
    if confidence is None:
        confidence = _asset_confidence(query, text, entities, duplicate_count)
    metadata = {
        "query": query,
        "asset_type": asset_type,
        "entities": entities[:6],
        "search_tags": tags,
        "source_priority": ["upload", "vivre_card", "manual_asset_search"],
        "confidence": confidence,
    }

    # Enrich with AssetDatabase ranked results (additive, never breaks pipeline).
    try:
        from app.utils.assets import get_asset_database
        db = get_asset_database()
        ranked = db.rank_for_beat(
            {
                "text": f"{query} {text}",
                "beat_type": slide.get("beat_type", ""),
                "entities": entities,
                "tags": tags,
                "emotion": slide.get("emotion_state", {}),
            },
            top_k=3,
        )
        if ranked:
            metadata["ranked_assets"] = [
                {"id": r.asset.id, "name": r.asset.name, "score": round(r.score, 4)}
                for r in ranked
            ]
    except Exception:
        pass  # Graceful degradation — AssetDatabase is optional.

    return metadata


def _classify_visual_intent(slide: Dict) -> Dict:
    beat_type = slide.get("beat_type", "evidence")
    visual_role = slide.get("visual_role", "character")
    purpose = slide.get("visual_purpose", "")
    intent_by_beat = {
        "hook": "curiosity_gap",
        "setup": "context_setup",
        "evidence": "proof",
        "reveal": "reversal",
        "payoff": "takeaway",
        "cta": "conversion",
    }
    return {
        "primary_intent": intent_by_beat.get(beat_type, "support"),
        "visual_role": visual_role,
        "viewer_question": slide.get("viewer_focus") or purpose,
        "intent_strength": round(
            min(1.0, 0.45 + (0.18 if beat_type in {"hook", "reveal", "payoff"} else 0.08) + (0.10 if purpose else 0.0)),
            2,
        ),
    }


def _emotion_for_text(text: str, beat_type: str, index: int, total: int) -> Dict:
    lowered = (text or "").lower()
    emotion = "curious"
    valence = 0.0
    intensity = 0.35
    if re.search(r"\b(death|died|pain|fear|trauma|loss|sacrifice|betrayal|cry|alone|weak)\b", lowered):
        emotion, valence, intensity = "grief", -0.55, 0.78
    elif re.search(r"\b(danger|threat|villain|war|fight|destroy|monster|evil)\b", lowered):
        emotion, valence, intensity = "tension", -0.35, 0.72
    elif re.search(r"\b(secret|hidden|truth|mystery|impossible|twist|reveal)\b", lowered):
        emotion, valence, intensity = "intrigue", 0.1, 0.68
    elif re.search(r"\b(dream|freedom|joy|hope|promise|future|win)\b", lowered):
        emotion, valence, intensity = "hope", 0.45, 0.55
    elif beat_type == "cta":
        emotion, valence, intensity = "resolution", 0.25, 0.28
    if beat_type == "hook":
        intensity = max(intensity, 0.76)
    if beat_type == "reveal":
        intensity = max(intensity, 0.82)
    curve_position = round(index / max(1, total - 1), 2)
    return {
        "emotion": emotion,
        "intensity": round(intensity, 2),
        "valence": round(valence, 2),
        "curve_position": curve_position,
    }


RELATIONSHIP_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bvs\.?|versus|fight|defeat|humiliat|oppos", re.I), "opposes"),
    (re.compile(r"\bcrew|protect|save|friend|ally|loyal", re.I), "allies_with"),
    (re.compile(r"\btrain|teacher|student|mentor|learn", re.I), "mentor_link"),
    (re.compile(r"\bfather|son|brother|family|lineage|inherited", re.I), "family_or_lineage"),
    (re.compile(r"\bbetray|manipulat|use|exploit", re.I), "exploits"),
    (re.compile(r"\bcontrast|unlike|opposite|mirror", re.I), "contrasts"),
]


def _character_relationships_for_slide(slide: Dict, script_entities: List[str]) -> List[Dict]:
    text = " ".join(
        part
        for part in [slide.get("subtitle_text", ""), slide.get("summary", ""), slide.get("image_search_query", "")]
        if part
    )
    entities = slide.get("context_entities") or _extract_context_entities(text, limit=4) or script_entities[:4]
    characters = [entity for entity in entities if entity in CANONICAL_ENTITIES][:4]
    if len(characters) < 2:
        return []
    relation_type = "associated_with"
    for pattern, candidate in RELATIONSHIP_PATTERNS:
        if pattern.search(text):
            relation_type = candidate
            break
    return [
        {
            "source": characters[0],
            "target": other,
            "relationship": relation_type,
            "evidence": _clean_prompt_text(text, max_len=110),
        }
        for other in characters[1:3]
    ]


def _composition_layers_for_slide(slide: Dict, video_profile: str) -> List[Dict]:
    video_profile = normalize_video_profile(video_profile)
    layout = slide.get("layout_mode") or _default_layout_mode(video_profile, slide.get("beat_type", ""), slide.get("visual_role", ""))
    layers = [
        {
            "name": "background_asset",
            "type": "image",
            "source": slide.get("image_search_query", "One Piece Logo"),
            "fit": "safe_contain" if layout in {"safe_subject", "horizontal_feature", "split_context"} else "cover",
        }
    ]
    if layout in {"title_card", "section_card", "evidence_card", "quote_card", "split_context"}:
        layers.append({"name": "editorial_panel", "type": "shape", "style": layout})
    if slide.get("text_overlay"):
        layers.append({"name": "headline", "type": "text", "text": slide["text_overlay"], "priority": "primary"})
    if slide.get("emphasis_words"):
        layers.append({"name": "emphasis_terms", "type": "text_tags", "terms": slide["emphasis_words"][:4]})
    return layers


def _advanced_motion_plan(slide: Dict, video_profile: str, previous_motion: str = "") -> Dict:
    preset = slide.get("motion_preset") or _default_motion_preset(video_profile, slide.get("beat_type", ""), slide.get("layout_mode", ""))
    emotion = slide.get("emotion_state", {})
    intensity = float(emotion.get("intensity") or 0.45)
    role = slide.get("visual_role", "character")
    camera_goal = {
        "character": "face_lock_push",
        "evidence": "inspect_detail",
        "object": "object_reveal",
        "location": "wide_context_pan",
        "symbol": "centered_icon_hold",
        "comparison": "two_subject_scan",
        "timeline": "left_to_right_progression",
        "section_card": "readable_hold",
        "quote_card": "readable_hold",
        "cta_card": "readable_hold",
    }.get(role, "support_narration")
    if previous_motion and preset == previous_motion and preset not in {"title_card_hold", "static_hold"}:
        camera_goal = f"{camera_goal}_alternate_framing"
    return {
        "preset": preset,
        "camera_goal": camera_goal,
        "motion_intensity": round(min(1.0, max(0.15, intensity)), 2),
        "focus_target": (slide.get("asset_metadata") or {}).get("asset_type", role),
        "transition_in": slide.get("transition_in") or TRANSITION_BY_BEAT.get(slide.get("beat_type"), "crossfade"),
    }


def _visual_diversity_score(slide: Dict, previous_slides: List[Dict]) -> Tuple[float, str]:
    if not previous_slides:
        return 1.0, "Opening visual establishes the palette."
    query = (slide.get("image_search_query") or "").strip().lower()
    role = (slide.get("visual_role") or "").strip().lower()
    recent = previous_slides[-3:]
    score = 1.0
    reasons = []
    if any(query and query == (prev.get("image_search_query") or "").strip().lower() for prev in recent):
        score -= 0.35
        reasons.append("query repeats nearby")
    if recent and all(role and role == (prev.get("visual_role") or "").strip().lower() for prev in recent):
        score -= 0.22
        reasons.append("visual role is overused")
    entity_set = set((slide.get("context_entities") or [])[:3])
    previous_entities = set()
    for prev in recent:
        previous_entities.update((prev.get("context_entities") or [])[:3])
    if entity_set and entity_set == previous_entities:
        score -= 0.15
        reasons.append("same entity cluster")
    if not reasons:
        reasons.append("role/query/entity mix changes from nearby slides")
    return round(max(0.0, min(1.0, score)), 2), "; ".join(reasons)


def _retention_optimizer_for_slide(slide: Dict, index: int, total: int, previous_slides: List[Dict]) -> Tuple[float, List[str], Dict]:
    optimized = dict(slide)
    actions: List[str] = []
    beat = optimized.get("beat_type", "")
    diversity = float(optimized.get("visual_diversity_score") or 0.7)
    emotion = optimized.get("emotion_state", {})
    intensity = float(emotion.get("intensity") or 0.4)
    score = 0.45 + (0.18 if beat in {"hook", "reveal", "payoff"} else 0.08) + (0.14 * diversity) + (0.12 * intensity)

    if index == 0 and not optimized.get("text_overlay"):
        optimized["text_overlay"] = _clean_prompt_text(optimized.get("summary", ""), max_len=56)
        actions.append("added hook overlay")
    if beat == "hook" and optimized.get("layout_mode") == "safe_subject":
        optimized["layout_mode"] = "title_card"
        actions.append("converted hook to readable title_card")
    if diversity < 0.55:
        actions.append("flagged nearby visual repetition")
        if optimized.get("motion_preset") not in {"title_card_hold", "reveal_zoom"}:
            optimized["motion_preset"] = "reveal_zoom" if beat == "reveal" else "subject_push"
    if total > 1 and index == total - 1 and beat != "cta":
        actions.append("final slide should resolve or invite response")
    return round(max(0.0, min(1.0, score)), 2), actions, optimized


def _apply_visual_architecture_pass(
    slides: List[Dict],
    script_context: Dict,
    video_profile: str,
) -> List[Dict]:
    """Applies the new 20-stage ProductionPipeline to the raw slides/beats."""
    from app.utils.slides.legacy_production_pipeline import run_pipeline
    
    logger.info("Running modern ProductionPipeline on storyboard beats.")
    result = run_pipeline(slides, video_profile={"platform": video_profile})
    
    if not result.passed_quality_gate:
        logger.warning(f"Quality gate found {len(result.quality_issues)} issues in the storyboard.")
        
    for slide in result.slides:
        slide["architecture_stage"] = "Final Slides"
        slide["architecture_path"] = VISUAL_ARCHITECTURE_STAGES
        
    return result.slides


def _apply_production_edit_defaults(
    slide: Dict,
    index: int,
    total: int,
    video_profile: str,
    context_entities: List[str],
) -> Dict:
    slide = dict(slide)
    video_profile = normalize_video_profile(video_profile)
    text = " ".join(
        part for part in [slide.get("subtitle_text", ""), slide.get("summary", "")] if part
    )
    beat_type = (slide.get("beat_type") or "").strip().lower()
    if beat_type not in PRODUCTION_BEAT_TYPES:
        beat_type = _classify_production_beat(text, index, total)
    visual_role = (slide.get("visual_role") or "").strip().lower()
    if visual_role not in VISUAL_ROLES:
        visual_role = _default_visual_role(beat_type, text)
    layout_mode = (slide.get("layout_mode") or "").strip().lower()
    if layout_mode not in LAYOUT_MODES[video_profile]:
        layout_mode = _default_layout_mode(video_profile, beat_type, visual_role)
    motion_preset = (slide.get("motion_preset") or "").strip().lower()
    if motion_preset not in MOTION_PRESETS:
        motion_preset = _default_motion_preset(video_profile, beat_type, layout_mode)

    slide["beat_type"] = beat_type
    slide["visual_role"] = visual_role
    slide["layout_mode"] = layout_mode
    slide["motion_preset"] = motion_preset
    slide["text_overlay"] = _clean_prompt_text(
        slide.get("text_overlay") or _text_overlay_for_slide(slide.get("summary", ""), beat_type, visual_role),
        max_len=90,
    )
    if not isinstance(slide.get("emphasis_words"), list):
        slide["emphasis_words"] = _emphasis_words(text, context_entities)
    raw_transition = (slide.get("transition_in") or "").strip().lower()
    # Sanitize: if the LLM put an effect_cue name in transition_in, replace it with the beat default
    if not raw_transition or raw_transition in _EFFECT_ONLY_NAMES:
        raw_transition = TRANSITION_BY_BEAT.get(beat_type, "crossfade")
    slide["transition_in"] = raw_transition

    raw_sfx = (slide.get("sfx_cue") or "").strip().lower()
    # Normalize literal "none" string to empty
    if raw_sfx == "none":
        raw_sfx = ""
    # Bug 1 fix: if sfx_cue duplicates transition_in, clear it — the transition
    # system already plays an SFX for transition_in, so mirroring causes double fire.
    if raw_sfx and raw_sfx == raw_transition:
        raw_sfx = ""
    # Bug 5 fix: if sfx_target_word provides a mid-slide hit, clear ALL boundary
    # sfx_cues (not just bass_hit) to prevent audio clutter on emphasis slides.
    if raw_sfx and slide.get("sfx_target_word"):
        logger.debug(
            "Clearing sfx_cue '%s' because sfx_target_word '%s' already set",
            raw_sfx, slide.get("sfx_target_word"),
        )
        raw_sfx = ""
    slide["sfx_cue"] = raw_sfx or slide["transition_in"]
    # CTA slides always get mouse_click SFX for engagement feel
    # Hook slides get pop SFX for a punchy opening
    if beat_type == "cta":
        slide["sfx_cue"] = "mouse_click"
    elif beat_type == "hook":
        slide["sfx_cue"] = "pop"
    slide["visual_purpose"] = _clean_prompt_text(
        slide.get("visual_purpose")
        or _visual_purpose_for_slide(slide.get("summary", ""), slide.get("subtitle_text", ""), beat_type, visual_role),
        max_len=180,
    )
    viewer_focus = _clean_prompt_text(
        slide.get("viewer_focus")
        or _viewer_focus_for_slide(slide.get("summary", ""), slide.get("subtitle_text", ""), beat_type, visual_role),
        max_len=160,
    )
    slide["viewer_focus"] = viewer_focus
    slide["manual_upload_brief"] = _clean_prompt_text(
        slide.get("manual_upload_brief")
        or _manual_upload_brief_for_slide(slide.get("image_search_query", ""), viewer_focus),
        max_len=220,
    )
    slide["avoid_visual_reuse"] = _clean_prompt_text(
        slide.get("avoid_visual_reuse")
        or "Do not reuse the same portrait/pose if the previous slide already used this character.",
        max_len=180,
    )
    # Bug 6 fix: cap asset_confidence for queries that mention characters absent
    # from CANONICAL_ENTITIES — the search engine likely can't resolve them.
    ac = slide.get("asset_confidence")
    if ac is not None:
        query_lower = (slide.get("image_search_query") or "").lower()
        canon_lower = {e.lower() for e in CANONICAL_ENTITIES}
        query_words = set(query_lower.split())
        # If no canonical entity appears in the query, cap confidence at 0.55
        if not query_words & canon_lower:
            capped = min(float(ac), 0.55)
            if capped < float(ac):
                logger.debug(
                    "asset_confidence capped %.2f -> %.2f for non-canonical query: %s",
                    float(ac), capped, query_lower[:60],
                )
            slide["asset_confidence"] = capped
    if "asset_confidence" not in slide:
        slide["asset_confidence"] = _asset_confidence(
            slide.get("image_search_query", ""),
            text,
            context_entities,
        )
    # Cinematic pacing fields — preserve LLM values or set sensible defaults
    VALID_PACING_INTENTS = {"rapid_montage", "hold_frame", "dramatic_pause", "parallel_cut", "standard"}
    pacing = (slide.get("pacing_intent") or "").strip().lower()
    slide["pacing_intent"] = pacing if pacing in VALID_PACING_INTENTS else "standard"

    # Bug 2 fix: filter motion-alias names out of effect_cues. These are camera
    # movements (Ken Burns presets), not per-frame pixel effects.
    if not isinstance(slide.get("effect_cues"), list):
        slide["effect_cues"] = []
    slide["effect_cues"] = [
        cue for cue in slide["effect_cues"]
        if (cue or "").strip().lower() not in _MOTION_ONLY_NAMES
    ]

    # Bug 4 fix: clear sfx_target_word if it is a common stop word.
    raw_target = (slide.get("sfx_target_word") or "").strip().lower()
    if raw_target in _SFX_TARGET_STOP_WORDS:
        logger.debug(
            "Clearing sfx_target_word '%s' (stop word) on slide '%s'",
            raw_target, slide.get("summary", "")[:40],
        )
        raw_target = ""
    slide["sfx_target_word"] = raw_target
    slide["director_note"] = slide.get("director_note") or ""

    # Bug 3 fix: strip duration-sensitive effects from very short slides where
    # they can't render enough frames to be visible (e.g. impact_shake needs ~9
    # frames at 30fps = 0.3s minimum window, so we use 0.5s as a safe floor).
    try:
        s = parse_time(slide["start_time"])
        e = parse_time(slide["end_time"])
        slide_dur = e - s
    except Exception:
        slide_dur = 999.0  # unparseable, skip duration checks

    if slide_dur < _MIN_EFFECT_DURATION_SECONDS and slide["effect_cues"]:
        original_cues = list(slide["effect_cues"])
        slide["effect_cues"] = [
            cue for cue in slide["effect_cues"]
            if (cue or "").strip().lower() not in _DURATION_SENSITIVE_EFFECTS
        ]
        if len(slide["effect_cues"]) < len(original_cues):
            logger.debug(
                "Stripped duration-sensitive effects from %.2fs slide '%s': %s",
                slide_dur, slide.get("summary", "")[:40],
                [c for c in original_cues if c not in slide["effect_cues"]],
            )

    # Cap rapid_montage slide duration to 2.0 seconds max.
    if slide["pacing_intent"] == "rapid_montage":
        try:
            s = parse_time(slide["start_time"])
            e = parse_time(slide["end_time"])
            if e - s > 2.0:
                logger.debug(
                    "rapid_montage slide %s duration %.2fs capped to 2.0s",
                    slide.get("summary", "")[:40],
                    e - s,
                )
                slide["end_time"] = format_time(s + 2.0)
        except Exception:
            pass  # timing strings non-parseable; leave as-is

    return slide


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

# Queries that are usually poor manual-search terms; rewrite them to grounded assets.
_BAD_ASSET_QUERY_RE = re.compile(
    r"\b("
    r"model\s+\w+|hito\s+hito|gear\s+v\b|laboratory|battlefield|manga\s+panel|"
    r"stretching|aura|thermodynamic|kinetic|infographic|diagram|explosion|"
    r"internal\s+friction|comparison\s+chart|theories"
    r")\b",
    re.IGNORECASE,
)

_NON_SEARCHABLE_QUERY_RE = re.compile(
    r"\b("
    r"thermodynamic|kinetic|friction|pressure|elastic|detonat|explosion|"
    r"internal\s+blast|diagram|infographic|two\s+devil\s+fruit\s+rule|"
    r"vs\s+normal|energy\s+expansion|mythical\s+zoan\s+property|abstract"
    r")\b",
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


def _image_slides_rules_block(context_entities: List[str], video_profile: str = "short_vertical") -> str:
    video_profile = normalize_video_profile(video_profile)
    entities_line = (
        ", ".join(context_entities)
        if context_entities
        else "None detected; use One Piece Logo or Grand Line Map for transitions"
    )
    profile_line = (
        "SHORTS PROFILE: 9:16 vertical, punchy 2-4 second beats, strong hook/reveal cards, large safe-centered subjects, subtitles have high priority.\n"
        if video_profile == "short_vertical"
        else (
            "LONG-FORM PROFILE: 16:9 horizontal, calmer 5-9 second beats, section/chapter cards, "
            "lower-third context, sparse keyword subtitles only, and each slide must tell the editor "
            "exactly what the viewer should inspect during the narration.\n"
        )
    )
    long_form_storyboard_rules = ""
    if video_profile == "long_youtube":
        long_form_storyboard_rules = (
            "LONG-FORM STORYBOARD RULES:\n"
            "- Think like an editor making a 6 minute anime essay, not a search engine.\n"
            "- Each slide summary must answer: what should the viewer be looking at right now, and why does it support the claim?\n"
            "- Do not make 80 near-identical Blackbeard portrait beats. Rotate visual functions: character portrait, crew/group, devil fruit/object, opposing character, location/arc, evidence card, quote card, section card.\n"
            "- Use section_card for major argument turns: hook, psychology setup, evidence block, Luffy contrast, final warning, CTA.\n"
            "- Use evidence_card when the narration cites behavior, choices, psychology, power, crew, institutions, or consequences.\n"
            "- Use quote_card only for memorable thesis/payoff lines.\n"
            "- image_search_query should be the exact manual storyboard target. Avoid numbered duplicates like 'Blackbeard 12'. If the same character repeats, change the scene/object: 'Blackbeard Jaya cherry pie', 'Blackbeard Yami Yami no Mi', 'Blackbeard Pirates Jaya', 'Whitebeard Marineford Blackbeard', 'Luffy Blackbeard Mock Town'.\n"
            "- For abstract psychology, choose concrete visual evidence: fear -> child Teach moon image if mentioned, power hunger -> Yami Yami fruit, manipulation -> Blackbeard Pirates or Impel Down/Marineford, contrast -> Luffy vs Blackbeard.\n\n"
        )
    hook_retention_rules = (
        "HOOK RETENTION PASS:\n"
        "- Think like a senior faceless anime editor trying to stop the first swipe/drop-off.\n"
        "- Treat the first 3-8 seconds as a micro-storyboard, not one static intro card.\n"
        "- If the opening sentence contains a question, accusation, twist, ranking, final-villain claim, betrayal, sacrifice, secret, or power reveal, split it into 2-4 smaller visual cues when timestamps allow.\n"
        "- Hook visual cue order should usually be: curiosity image -> threat/evidence image -> thesis/title card. Example: 'Holy Knights final villains?' can become 'Holy Knights silhouettes', 'Imu Five Elders throne room', then a title_card asking the final-villain question.\n"
        "- Each hook cue must change what the viewer inspects: face/silhouette, symbol/object, location/institution, then title/question. Do not hold only a logo during the hook unless it is the final CTA.\n"
        "- Hook text_overlay should be short, punchy, and incomplete enough to create curiosity; avoid explaining the whole video in the overlay.\n"
        "- Hook viewer_focus must tell the human editor the retention purpose: curiosity gap, proof of stakes, threat reveal, contradiction, or payoff setup.\n"
        "- Hook manual_upload_brief must be practical for manual upload: name the exact character/group/symbol/scene to pick and the desired crop/framing.\n"
        "- For short_vertical hooks, prefer 0.8-2.0 second visual cues, large centered subjects, and hard curiosity changes. For long_youtube hooks, prefer 2.0-4.0 second cues with calmer evidence/title pacing.\n\n"
    )
    return (
        "CRITICAL — each slide matches the SPOKEN lines in [start_time, end_time]. Faceless theory video (voiceover + B-roll).\n"
        "Before making slides, read ALL subtitles as one complete script. Infer the main topic, arc, characters, and argument, then choose each slide's search query from that whole-script understanding plus the local spoken beat.\n\n"
        f"{profile_line}\n"
        f"{long_form_storyboard_rules}"
        f"{hook_retention_rules}"
        "EDITOR THINKING MODEL:\n"
        "- Stop thinking in isolated images. Think: narration -> visual_purpose -> visual_role -> visual.\n"
        "- For every slide, first decide what the narration is doing: introduce, prove, locate, symbolize, compare, escalate, reveal, pay off, or close.\n"
        "- Then choose visual_role from an editor's role system, not a search-engine category.\n"
        "- The image_search_query is only the final step. It should be the concrete canon asset that best performs the visual_purpose.\n"
        "- Example: narration 'Blackbeard spent his entire life chasing a dream' should NOT default to only 'Marshall D. Teach'. Purpose: introduce obsession. Strong visual options: 'Blackbeard Yami Yami no Mi', 'Blackbeard Jaya speech', 'Whitebeard Pirates ship Blackbeard'.\n\n"
        "VISUAL ROLE ROTATION:\n"
        "- Use these visual_role values: character, evidence, location, object, symbol, comparison, section_card, quote_card, timeline, cta_card.\n"
        "- If the same character remains the topic for several slides, rotate the role: character -> object -> crew/group/evidence -> location -> symbol -> comparison -> timeline.\n"
        "- Avoid more than two consecutive character-role slides about the same person.\n"
        "- character: introduce or re-anchor a person with a clear face/pose.\n"
        "- evidence: prove the claim through a canon action, panel/scene, crew, institution, or consequence.\n"
        "- location: ground the beat in an arc/place like Jaya, Marineford, Elbaf, Mary Geoise, Ohara, Wano.\n"
        "- object: make the idea tangible through a fruit, sword, ship, throne, map, chains, scar, flag, or artifact.\n"
        "- symbol: represent a faction, ideology, mystery, or theme: World Government symbol, Nika, Jolly Roger, Poneglyph, throne.\n"
        "- comparison: show contrast like Luffy vs Blackbeard, old vs new, dream vs obsession, freedom vs control.\n"
        "- timeline: show years, escalation, sequence, chapter history, or cause-and-effect.\n\n"
        "PRODUCTION EDIT STRUCTURE:\n"
        "- First identify the full-script structure: hook, setup, evidence, reveal, payoff, CTA.\n"
        "- Every slide must include production edit metadata: beat_type, visual_role, layout_mode, visual_purpose, motion_preset, text_overlay, emphasis_words, transition_in, sfx_cue, asset_confidence, viewer_focus, manual_upload_brief, avoid_visual_reuse.\n"
        "- beat_type values: hook, setup, evidence, reveal, payoff, cta.\n"
        "- visual_role values: character, evidence, location, object, symbol, comparison, section_card, quote_card, timeline, cta_card.\n"
        "- short_vertical layout_mode values: safe_subject, title_card, quote_card, evidence_card, full_bleed.\n"
        "- long_youtube layout_mode values: horizontal_feature, split_context, section_card, quote_card, evidence_card, full_bleed.\n"
        "- motion_preset values: subject_push, evidence_hold, reveal_zoom, wide_pan, title_card_hold.\n\n"
        "VISUAL ARCHITECTURE:\n"
        "- Follow this pipeline exactly: Subtitles -> Script Analyzer -> Story Beat Detector -> Emotional Curve Builder -> Visual Intent Classifier -> Asset Selector -> Motion Planner -> Retention Optimizer -> Final Slides.\n"
        "- asset_metadata is the asset database row for the slide: query, asset_type, entities, search_tags, source_priority, confidence.\n"
        "- visual_intent explains the viewer question and why this beat exists visually.\n"
        "- emotion_state tracks emotion, intensity, valence, and curve_position across the whole script.\n"
        "- visual_diversity_score should penalize repeated query/role/entity runs; diversity_notes explains the reason.\n"
        "- character_relationships should name visible relationships when two or more characters/entities are involved.\n"
        "- retention_score and retention_actions are the optimizer pass: hook clarity, curiosity gap, repetition risk, payoff strength.\n"
        "- composition_layers describes multi-layer composition: background asset, editorial panel, headline, emphasis terms.\n"
        "- motion_plan is the advanced motion planner output: preset, camera_goal, motion_intensity, focus_target, transition_in.\n\n"
        "CONTEXT-AWARE PACING:\n"
        "- Cut on idea changes, not only character names: hook, cause, consequence, example, rebuttal, payoff.\n"
        "- Prefer 2-5 seconds per slide. Never hold one visual across a new canon event, new character, or new emotional turn.\n"
        "- Long narration sections must become multiple visual beats with different scene-search queries.\n"
        "- For trauma/psychology/symbolic narration, DO NOT invent symbolic art. Map the feeling to the closest searchable canon entity, arc, place, object, or logo.\n"
        "- Avoid generic repeats like only 'Zoro training' or 'Zoro scars'; each slide needs a new focal action or object.\n\n"
        "CINEMATIC PACING (DIRECTOR'S TIMELINE):\n"
        "- Think like a professional video editor creating a cinematic timeline, not a slideshow.\n"
        "- If a sentence contains a reveal, shock, or name-drop, split it into 2-3 rapid visual cuts (0.8-2.0 seconds each) with DIFFERENT assets per cut.\n"
        "- Example: 'Out of everyone in the world... why did Imu choose Gunko?' should become 3 cuts: Imu eyes close-up (1.5s), Empty Throne zoom (1.0s), Gunko silhouette (1.5s).\n"
        "- beat_type values now include: hook, setup, evidence, reveal, twist, escalation, payoff, resolution, cta.\n"
        "- Use pacing_intent to signal the editor how this beat should be cut:\n"
        "  - 'rapid_montage': 3+ quick cuts showing alternatives/candidates (e.g., Admirals, Gorosei, Holy Knights in rapid succession).\n"
        "  - 'hold_frame': freeze on a single powerful image for emphasis (e.g., 'That's not trust.' — hold on Gunko expression).\n"
        "  - 'dramatic_pause': brief black/silence before a payoff line.\n"
        "  - 'parallel_cut': interleave two subjects (e.g., Imu + Gunko alternating).\n"
        "  - 'standard': normal pacing.\n"
        "- Use effect_cues to request cinematic compositor effects (array of strings):\n"
        "  - 'impact_shake': camera shake on impact words.\n"
        "  - 'flash_frame': 2-frame white flash between reveals.\n"
        "  - 'slow_push_in': Ken Burns slow zoom toward subject.\n"
        "  - 'red_eye_flash': brief red overlay for menace.\n"
        "  - 'glitch': digital distortion for shock/twist.\n"
        "  - 'dark_vignette': heavy edge darkening for mystery.\n"
        "  - 'desaturation': drain color for grief/trauma.\\n"
        "  - 'morph': blend/dissolve between two character states (e.g., young Shuri → older Gunko).\n"
        "  - 'chromatic_shift': RGB channel split for glitch/distortion.\n"
        "CRITICAL effect_cues rules:\n"
        "- effect_cues go in the 'effect_cues' array ONLY. NEVER put effect names in 'transition_in'.\n"
        "- 'transition_in' only accepts: crossfade, fade_eased, zoom_dissolve, iris_wipe, radial_wipe, glitch_cut, whip_pan_left, whip_pan_right, motion_slide_left, motion_slide_right, fade_eased, cube_rotation, water_ripple.\n"
        "- 'sfx_cue' must be '' (empty string) or a transition name — NEVER 'none' (use '' instead).\n"
        "- If you set sfx_target_word, set sfx_cue to '' — they both trigger audio and having both causes a double-hit.\n"
        "RAPID MONTAGE RULE:\n"
        "- When pacing_intent is 'rapid_montage', you MUST generate at least 3 separate slides from the narration sentence, each with a completely different image_search_query and visual.\n"
        "- Example: 'Out of everyone in the world... why did Imu choose Gunko?' must produce:\n"
        "  Slide A: Imu eyes (1.5s), Slide B: Empty Throne zoom (1.0s), Slide C: Gunko silhouette (1.5s)\n"
        "- Two slides for a rapid_montage is NOT acceptable.\n"
        "- Use sfx_target_word to sync a bass_hit to the exact emphasis word in the narration (e.g., 'Gunko').\n"
        "- Use director_note for human-readable editorial intent (e.g., 'This is the first reveal moment', 'This line deserves its own shot', 'Retention spike').\n\n"
        f'For EVERY slide set visual_source to "{_VISUAL_SOURCE_ASSET}". AI image generation is disabled.\n'
        'For EVERY slide set ai_image_prompt to empty string "".\n\n'
        "USE asset_search FOR:\n"
        "- Named characters, crews, places, arcs, objects, powers, chapter beats, logos, and maps.\n"
        "- When the spoken beat is abstract, choose the closest concrete One Piece anchor from the whole script.\n"
        "- If no exact scene exists, use a searchable fallback: main character + arc/place, main object + arc, One Piece Logo, or Grand Line Map.\n\n"
        "SEARCH QUERY STYLE FOR asset_search:\n"
        "- Use specific searchable manga/anime scene phrases, not abstract emotions or cinematic descriptions.\n"
        "- Prefer 3-7 words: character + arc/place + object/action.\n"
        "- Include exact names from the script when possible: Loki, Elbaf, Yggdrasil, Dory, Brogy, Ohara, Void Century, Five Elders.\n"
        "- Good: 'Zoro Kuina Shimotsuki stairs', 'Zoro Mihawk Baratie cross knife', 'Zoro nothing happened Thriller Bark', 'Kuma Sabaody Straw Hats vanish', 'Zoro Enma King Onigashima', 'Zoro bows to Mihawk Kuraigana'.\n"
        "- Good for Elbaf: 'Loki Elbaf chains', 'Dory Brogy Little Garden', 'Ohara scholars library', 'Elbaf giant village', 'Void Century Poneglyph', 'Five Elders Mary Geoise'.\n"
        "- Bad: 'Zoro trauma', 'fear of weakness', 'emotional thesis', 'buried memory', 'giant prince sitting in darkness', 'dramatic silhouette'.\n\n"
        "FIELD RULES:\n"
        "- summary: short beat from subtitles (what the viewer should understand).\n"
        "- visual_purpose: one sentence explaining the editor's reason for the visual: introduce, prove, locate, symbolize, compare, escalate, reveal, pay off, or close.\n"
        "- viewer_focus: one concrete sentence telling the editor/viewer what should be inspected on screen and why it supports this narration beat.\n"
        "- manual_upload_brief: one practical storyboard note for a human picking the image. Name the needed character, object, arc, expression, or scene.\n"
        "- avoid_visual_reuse: short warning about what visual not to repeat from nearby beats, especially repeated portraits or the same generic crew shot.\n"
        "- pacing_intent: one of rapid_montage, hold_frame, dramatic_pause, parallel_cut, standard.\n"
        "- effect_cues: array of effect strings from the cinematic list above. Empty array [] if none needed.\n"
        "- sfx_target_word: the word in the narration to sync a bass_hit sound to. Empty string if none.\n"
        "- director_note: human-readable editorial intent. Empty string if none.\n"
        "- asset_metadata, visual_intent, emotion_state, visual_diversity_score, diversity_notes, character_relationships, retention_score, retention_actions, composition_layers, and motion_plan may be included; if uncertain, leave coherent simple values and the backend will repair them.\n"
        "- image_search_query: REQUIRED. 3–9 words, searchable canon scene/entity phrase. "
        "Never repeat the same query twice. No jargon filenames.\n"
        "- ai_image_prompt: ALWAYS empty string \"\". Do not write AI prompts, image-generation descriptions, or symbolic art instructions.\n\n"
        f"SCRIPT-WIDE ENTITIES (context only): {entities_line}\n"
        "Max 2 slides with Monkey D. Luffy in image_search_query unless the entire script is about Luffy.\n"
        "TIMING: merge short fragments; max ~3.5s per slide; continuous timestamps with no gaps. "
        "Each slide's end_time must equal the next slide's start_time.\n\n"
        "OUTPUT: ONLY a JSON array. Each object MUST include:\n"
        "start_time, end_time, summary, visual_source, image_search_query, ai_image_prompt, beat_type, visual_role, visual_purpose, layout_mode, motion_preset, text_overlay, emphasis_words, transition_in, sfx_cue, asset_confidence, viewer_focus, manual_upload_brief, avoid_visual_reuse, pacing_intent, effect_cues, sfx_target_word, director_note, asset_metadata, visual_intent, emotion_state, visual_diversity_score, diversity_notes, character_relationships, retention_score, retention_actions, composition_layers, motion_plan\n"
    )


def _build_image_slides_full_prompt(
    timestamped_dialogues: List[Dict],
    context_entities: List[str],
    video_profile: str = "short_vertical",
    total_duration: float = 0.0,
) -> str:
    raw_subtitles = "\n".join(
        f"{d['start']} - {d['end']}: {d['text']}" for d in timestamped_dialogues
    )
    return (
        "You are an expert video content designer specializing in One Piece. "
        "Map timestamped subtitles to a production edit plan and image search queries as a JSON array.\n\n"
        f"{_image_slides_rules_block(context_entities, video_profile)}\n"
        "GOOD EXAMPLE (whole-script context, asset-search-only):\n"
        '[{"start_time":"0:00:00.03","end_time":"0:00:03.08",'
        '"summary":"Zoro trains from fear of weakness",'
        f'"visual_source":"{_VISUAL_SOURCE_ASSET}",'
        '"image_search_query":"Roronoa Zoro training dojo",'
        '"ai_image_prompt":"",'
        '"beat_type":"hook","visual_role":"section_card","visual_purpose":"Create curiosity by showing the origin of Zoro ambition.",'
        '"layout_mode":"title_card",'
        '"motion_preset":"subject_push","text_overlay":"Zoro trains from fear",'
        '"emphasis_words":["Zoro"],"transition_in":"crossfade","sfx_cue":"crossfade","asset_confidence":0.82,'
        '"viewer_focus":"Inspect Zoro as a driven child so the fear behind his ambition is clear.",'
        '"manual_upload_brief":"Pick a dojo/training image with young Zoro or swords visible.",'
        '"avoid_visual_reuse":"Avoid another generic adult Zoro portrait here."},'
        '{"start_time":"0:00:03.60","end_time":"0:00:07.80",'
        '"summary":"Kuina death creates Zoro fear",'
        f'"visual_source":"{_VISUAL_SOURCE_ASSET}",'
        '"image_search_query":"Zoro Kuina Shimotsuki stairs",'
        '"ai_image_prompt":"",'
        '"beat_type":"setup","visual_role":"evidence","visual_purpose":"Prove the fear by showing the canon event that created it.",'
        '"layout_mode":"evidence_card",'
        '"motion_preset":"evidence_hold","text_overlay":"Chapter five changed Zoro",'
        '"emphasis_words":["Zoro","Kuina"],"transition_in":"fade_eased","sfx_cue":"fade_eased","asset_confidence":0.92,'
        '"viewer_focus":"Show Kuina or the staircase because this is the evidence behind Zoro fear.",'
        '"manual_upload_brief":"Pick a Kuina/Shimotsuki stairs or dojo image, not a random Zoro pose.",'
        '"avoid_visual_reuse":"Do not repeat the previous training image."},'
        '{"start_time":"0:00:09.00","end_time":"0:00:11.84",'
        '"summary":"Mihawk humiliates Zoro at Baratie",'
        f'"visual_source":"{_VISUAL_SOURCE_ASSET}",'
        '"image_search_query":"Zoro Mihawk Baratie cross knife",'
        '"ai_image_prompt":"",'
        '"beat_type":"reveal","visual_role":"comparison","visual_purpose":"Show Zoro weakness by contrasting him with Mihawk power.",'
        '"layout_mode":"safe_subject",'
        '"motion_preset":"reveal_zoom","text_overlay":"",'
        '"emphasis_words":["Zoro","Mihawk"],"transition_in":"zoom_dissolve","sfx_cue":"zoom_dissolve","asset_confidence":0.95,'
        '"viewer_focus":"Focus on Mihawk overpowering Zoro to make the humiliation visual.",'
        '"manual_upload_brief":"Pick the Baratie cross-knife scene or a clear Zoro vs Mihawk frame.",'
        '"avoid_visual_reuse":"Avoid a solo Mihawk portrait without Zoro."},'
        '{"start_time":"0:00:31.49","end_time":"0:00:32.43",'
        '"summary":"Follow CTA",'
        f'"visual_source":"{_VISUAL_SOURCE_ASSET}",'
        '"image_search_query":"One Piece Logo",'
        '"ai_image_prompt":"",'
        '"beat_type":"cta","visual_role":"cta_card","visual_purpose":"Close the video and make the comment prompt easy to read.",'
        '"layout_mode":"title_card",'
        '"motion_preset":"title_card_hold","text_overlay":"Comment your theory",'
        '"emphasis_words":["One Piece"],"transition_in":"fade_eased","sfx_cue":"fade_eased","asset_confidence":0.75,'
        '"viewer_focus":"End on a clean One Piece identity card so the call to action reads clearly.",'
        '"manual_upload_brief":"Use a clean One Piece logo or Straw Hat crew image with open space for text.",'
        '"avoid_visual_reuse":"Avoid another dense battle image behind the CTA."}]\n\n'
        "BAD: abstract emotion queries like 'Zoro trauma'; any ai_image_prompt text; five generic Zoro asset slides in a row.\n\n"
        f"CRITICAL: The total video audio length is {total_duration:.2f} seconds. Your final slide MUST have an end_time of exactly {total_duration:.2f} or higher.\n"
        f"If the subtitles end early, you MUST invent additional visual-only outro slides (with empty subtitle_text) to bridge the gap all the way to {total_duration:.2f}.\n\n"
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
            child["context_entities"] = _extract_context_entities(
                f"{subtitle_text} {summary}",
                limit=3,
            ) or slide.get("context_entities") or context_entities
            inferred_query = _infer_query_from_subtitle_text(subtitle_text, child["context_entities"])
            if inferred_query:
                child["image_search_query"] = inferred_query
            child["visual_source"] = _VISUAL_SOURCE_ASSET
            child["image_search_query"] = (
                child.get("image_search_query")
                or _infer_query_from_subtitle_text(summary, child["context_entities"])
                or " ".join(child["context_entities"][:2])
                or "One Piece Logo"
            )
            child["ai_image_prompt"] = ""
            split_slides.append(child)

    return split_slides


def _repair_slide_timing_continuity(
    slides: List[Dict],
    total_duration: float,
    dialogues: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Make slide timings continuous for the sequential slideshow renderer.

    Gemini can return small holes between slides. The renderer consumes durations
    in order, so those holes are lost and later stretched across the video. Use
    slide start times as cut points, then make every slide end at the next cut.
    """
    if not slides:
        return []

    sorted_slides = sorted(slides, key=lambda slide: parse_time(slide["start_time"]))
    dialogue_starts = [parse_time(line["start"]) for line in (dialogues or [])]

    def snap_cut(value: float) -> float:
        if not dialogue_starts:
            return value
        nearest = min(dialogue_starts, key=lambda candidate: abs(candidate - value))
        return nearest if abs(nearest - value) <= 0.18 else value

    total_end = max(total_duration, 0.0)
    cuts: List[float] = [0.0]
    for slide in sorted_slides[1:]:
        cut = snap_cut(parse_time(slide["start_time"]))
        cut = max(cuts[-1] + 0.05, min(cut, total_end))
        cuts.append(cut)
    cuts.append(total_end)

    repaired: List[Dict] = []
    for index, slide in enumerate(sorted_slides):
        start = cuts[index]
        end = cuts[index + 1]
        if end <= start:
            end = min(total_end, start + 0.05)
        fixed = dict(slide)
        fixed["start_time"] = format_time(start)
        fixed["end_time"] = format_time(end)
        repaired.append(fixed)

    return repaired


def _normalize_visual_source(raw: Optional[str]) -> str:
    value = (raw or "").strip().lower().replace("-", "_")
    if value in {"asset", "search", "asset_search", "vivre", "upload", "catalog"}:
        return _VISUAL_SOURCE_ASSET
    if value in {"ai", "ai_generate", "generate", "ai_generation", "ai_image"}:
        return _VISUAL_SOURCE_ASSET
    return _VISUAL_SOURCE_ASSET


def _clean_prompt_text(value: str, max_len: int = 260) -> str:
    text = re.sub(r"\s+", " ", (value or "").strip())
    return text[:max_len].rstrip(" ,.;")


def _apply_visual_source_plan(
    slide: Dict,
    subtitle_text: str,
    context_entities: List[str],
    luffy_asset_count: int,
) -> Dict:
    """Finalize visual_source, image_search_query, and ai_image_prompt per slide.

    The user-facing workflow is asset/search only, so this function strips any
    AI prompt Gemini returned and rewrites non-searchable ideas to a concrete
    query fallback.
    """
    slide = dict(slide)
    summary = slide.get("summary") or ""
    query = (slide.get("image_search_query") or "").strip()
    inferred_query = _infer_query_from_subtitle_text(
        f"{subtitle_text} {summary}",
        context_entities,
    )

    if not query or _BAD_ASSET_QUERY_RE.search(query) or _NON_SEARCHABLE_QUERY_RE.search(f"{subtitle_text} {summary} {query}"):
        query = inferred_query or _infer_query_from_subtitle_text(summary, context_entities) or query

    if not query or _BAD_ASSET_QUERY_RE.search(query):
        query = " ".join(context_entities[:2]) if context_entities else "One Piece Logo"
    elif len(query.split()) == 1 and len(context_entities) > 1:
        for entity in context_entities[1:3]:
            if entity.lower() != query.lower():
                query = f"{query} {entity}"
                break

    slide["visual_source"] = _VISUAL_SOURCE_ASSET
    slide["image_search_query"] = _dedupe_query_words(
        _sanitize_image_query(query or "One Piece Logo", f"{subtitle_text} {summary}", context_entities)
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
        return _sanitize_image_query(" ".join(context_entities[:2]), text, context_entities)
    return None


def _query_variant_hint(text: str) -> str:
    lowered = (text or "").lower()
    variants = [
        (r"\bcrew|member|loyalty|followers|people\b", "crew"),
        (r"\bfruit|devil fruit|darkness|yami|gura|power\b", "devil fruit"),
        (r"\bwhitebeard|marineford|edward newgate\b", "Whitebeard Marineford"),
        (r"\bluffy|roger|dream|freedom|contrast\b", "Luffy contrast"),
        (r"\bworld government|impel down|institution|marine\b", "World Government"),
        (r"\bfear|weak|child|moon|vulnerable\b", "child moon"),
        (r"\bjaya|mock town|cherry pie\b", "Jaya"),
        (r"\bmanipulat|leverage|exploit|transaction\b", "manipulation"),
        (r"\bidentity|obsession|devotion|tragedy\b", "obsession"),
        (r"\bwarning|monster|evil|humanity\b", "warning"),
    ]
    for pattern, hint in variants:
        if re.search(pattern, lowered):
            return hint
    words = [
        word
        for word in re.findall(r"\b[A-Za-z][A-Za-z'-]{3,}\b", text or "")
        if word.lower() not in {"this", "that", "there", "their", "because", "between", "dreams", "blackbeard"}
    ]
    return " ".join(words[:2])


def _refine_slides_from_subtitles(
    slides: List[Dict],
    dialogues: List[Dict],
    context_entities: List[str],
    video_profile: str = "short_vertical",
) -> List[Dict]:
    """
    Attach subtitle_text per slide, align queries to spoken content, dedupe repeats.
    """
    used_queries: Dict[str, int] = {}
    luffy_asset_count = 0
    refined: List[Dict] = []

    total = len(slides)
    for index, slide in enumerate(slides):
        start = slide["start_time"]
        end = slide["end_time"]
        subtitle_text = _collect_subtitle_text_in_range(dialogues, start, end)
        slide = dict(slide)
        slide["subtitle_text"] = subtitle_text

        summary = slide.get("summary") or ""
        query = slide.get("image_search_query") or ""
        combined = f"{subtitle_text} {summary}".strip()

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

        if not query and combined:
            query = _infer_query_from_subtitle_text(combined, context_entities) or query

        query = _sanitize_image_query(
            query or "One Piece Logo",
            combined or summary,
            _slide_context_entities(slide) or context_entities,
        )

        # Force unique queries: suffix beat hint when duplicate.
        base_key = (query or "").lower()
        if base_key in used_queries:
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
                hint = _query_variant_hint(f"{subtitle_text} {summary}")
                query = f"{query} {hint}".strip() if hint and hint.lower() not in base_key else f"{query} alternate scene"

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
        duplicate_count = used_queries.get((slide.get("image_search_query") or "").lower(), 1) - 1
        slide["asset_confidence"] = _asset_confidence(
            slide.get("image_search_query", ""),
            subtitle_text,
            slide.get("context_entities") or context_entities,
            duplicate_count=max(0, duplicate_count),
        )
        slide = _apply_production_edit_defaults(
            slide,
            index,
            total,
            video_profile,
            slide.get("context_entities") or context_entities,
        )
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


def _expand_parallel_cuts(slides: List[Dict]) -> List[Dict]:
    """Fix 5: Expand parallel_cut slides into two rapid sub-slides.

    When pacing_intent == "parallel_cut", the slide represents two subjects
    interleaved (e.g., young Shuri vs older Gunko). We split it into two
    equal-duration sub-slides so the renderer produces an actual alternating cut.

    Rules:
    - Both sub-slides get pacing_intent = "rapid_montage"
    - Sub-slide A keeps the original image_search_query
    - Sub-slide B appends " alternate view" to force a different asset search
    - Both sub-slides share effect_cues from the original
    - Transition between them becomes "crossfade" (fast)
    """
    expanded: List[Dict] = []
    for slide in slides:
        if (slide.get("pacing_intent") or "").strip().lower() != "parallel_cut":
            expanded.append(slide)
            continue

        try:
            s = parse_time(slide["start_time"])
            e = parse_time(slide["end_time"])
            mid = s + (e - s) / 2.0
        except Exception:
            expanded.append(slide)
            continue

        base = dict(slide)
        base["pacing_intent"] = "rapid_montage"
        base["end_time"] = format_time(mid)
        base["transition_in"] = slide.get("transition_in", "crossfade")

        alt = dict(slide)
        alt["pacing_intent"] = "rapid_montage"
        alt["start_time"] = format_time(mid)
        alt["transition_in"] = "crossfade"
        original_query = (slide.get("image_search_query") or "").strip()
        alt["image_search_query"] = f"{original_query} alternate view".strip()
        alt["summary"] = f"{slide.get('summary', '')} (alt angle)".strip()

        logger.info(
            "parallel_cut split: [%.2f-%.2f] -> [%.2f-%.2f] + [%.2f-%.2f]",
            s, e, s, mid, mid, e,
        )
        expanded.append(base)
        expanded.append(alt)

    return expanded


def generate_gemini_image_slides(
    ass_path: str,
    out_path: str,
    total_duration: float,
    video_profile: str = "short_vertical",
) -> str:
    video_profile = normalize_video_profile(video_profile)
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
        prompt = _build_image_slides_full_prompt(timestamped_dialogues, context_entities, video_profile, total_duration)
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
            for key in (
                "beat_type",
                "visual_role",
                "layout_mode",
                "motion_preset",
                "text_overlay",
                "emphasis_words",
                "transition_in",
                "sfx_cue",
                "asset_confidence",
                *STORYBOARD_FIELDS,
            ):
                if key in gemini_slide:
                    slide[key] = gemini_slide[key]
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
            video_profile,
        )
        final_slides = _repair_slide_timing_continuity(
            final_slides,
            total_duration,
            timestamped_dialogues,
        )
        final_slides = _apply_visual_architecture_pass(
            final_slides,
            script_context,
            video_profile,
        )
        final_slides = _expand_parallel_cuts(final_slides)

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


def generate_image_slides(
    ass_path: str,
    out_path: str,
    total_duration: float,
    video_profile: str = "short_vertical",
) -> str:
    """Generate slide timing + search terms JSON (images uploaded separately)."""
    return generate_gemini_image_slides(ass_path, out_path, total_duration, video_profile)
