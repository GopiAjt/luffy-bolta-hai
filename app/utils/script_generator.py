import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv

from app.config import normalize_video_profile

logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

REQUIRED_SECTIONS = ("title", "script", "description", "hashtags")
BANNED_HOOK_RE = re.compile(
    r"^\s*(remember\b|think about\b|what if\b|did you\b|do you\b|have you\b|"
    r"ever wondered\b|imagine\b)",
    re.IGNORECASE,
)
QUESTION_START_RE = re.compile(
    r"^\s*(what|why|how|who|when|where|is|are|was|were|can|could|would|did|do)\b",
    re.IGNORECASE,
)
CANON_ANCHOR_RE = re.compile(
    r"\b(chapter\s+\d+|episode\s+\d+|sbs\b|volume\s+\d+|arc\b|"
    r"marineford|enies lobby|wano|egghead|mary geoise|mariejois|elbaf|ohara|"
    r"dressrosa|alabasta|skypiea|whole cake|fishman island|sabaody|god valley|"
    r"onigashima|water 7|thriller bark|impel down|zou)\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "about", "after", "again", "all", "and", "anime", "are", "because", "before",
    "but", "can", "chapter", "does", "dont", "every", "for", "from", "has",
    "hidden", "how", "into", "its", "just", "like", "manga", "more", "never",
    "nobody", "not", "oda", "one", "piece", "real", "revealed", "secret", "that",
    "the", "theory", "this", "truth", "was", "what", "when", "where", "why",
    "with", "you", "your",
}
CTA_PHRASES = (
    "follow", "comment", "tell me", "prove me", "drop your", "subscribe",
    "like this", "check out", "hit follow", "smash", "notification",
)
SUBJECT_SYNONYMS = {
    "mariejois": "mary geoise",
    "mary": "mary geoise",
    "geoise": "mary geoise",
    "nika": "sun god",
    "imu": "im-sama",
    "joyboy": "joy boy",
    "gorosei": "five elders",
    "revolutionaries": "revolutionary army",
    "straw": "straw hat",
    "hats": "straw hat",
}
GENERATION_SETTINGS = {
    "short_vertical": {"temperature": 0.7, "top_p": 0.9, "max_output_tokens": 1200},
    "long_youtube": {"temperature": 0.7, "top_p": 0.9, "max_output_tokens": 5000},
}
REQUIRED_OUTPUT_TEMPLATE = (
    "TITLE: <one title under 80 characters>\n\n"
    "SCRIPT: <complete narration only, no markdown, target word count>\n\n"
    "DESCRIPTION:\n"
    "- <bullet 1>\n"
    "- <bullet 2>\n"
    "- <bullet 3>\n"
    "- <bullet 4>\n"
    "- <bullet 5>\n\n"
    "HASHTAGS: #onepiece #anime #theory <more relevant lowercase hashtags>"
)


@dataclass
class ValidationResult:
    errors: List[str]
    warnings: List[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def is_generic_topic(topic: str) -> bool:
    """Return True when frontend/default text is not an actual One Piece topic."""
    if not topic or not topic.strip():
        return True

    normalized = topic.strip().lower()
    normalized = normalized.replace("–", "-")
    generic_phrases = [
        "generate a",
        "one piece narration script",
        "30-60 second",
        "script in hindi",
        "script in english",
        "enter your script",
        "your script text here",
    ]
    return any(phrase in normalized for phrase in generic_phrases)


def _fallback_topics() -> List[str]:
    return [
        "The chapter 1 Shanks scene that defines Luffy's idea of freedom",
        "Why Zoro taking Luffy's pain at Thriller Bark still matters",
        "The Enies Lobby moment where Robin finally chooses to live",
        "How Marineford proves Whitebeard understood the next era",
        "The Sabaody separation scene that changed the Straw Hats forever",
        "Why Jinbe's blood donation on Fishman Island matters",
        "The Wano promise behind Zoro carrying Enma",
        "How Sanji's Whole Cake Island choice proves his loyalty",
        "The Egghead clues around Vegapunk's message to the world",
        "Why Ohara's destruction still drives the Void Century mystery",
        "The God Valley incident as the shadow behind Rocks and Roger",
        "Why Luffy ringing the bell in Skypiea is more than a victory",
        "The Mary Geoise scene where Shanks meets the Five Elders",
        "How Ace's final words reshape Luffy's journey",
        "Why the Going Merry farewell still hits harder than most deaths",
    ]


def _resolve_topic(topic_override: Optional[str], context_text: str, ohara_context: str, chapter_number: Optional[int]) -> str:
    if not is_generic_topic(topic_override):
        return topic_override.strip()

    if context_text or ohara_context:
        topic = f"One Piece Chapter {chapter_number} theory" if chapter_number else "Latest One Piece manga chapter theory"
    else:
        topic = random.choice(_fallback_topics())
    logger.warning("Generic topic detected; using resolved topic: %s", topic)
    return topic


def _language_config(language: str) -> Dict[str, str]:
    normalized = (language or "english").strip().lower()
    is_hindi = normalized in {"hindi", "hi", "hinglish"}
    if is_hindi:
        return {
            "label": "Hindi/Hinglish",
            "rules": (
                "LANGUAGE RULES:\n"
                "- Write TITLE, SCRIPT, and DESCRIPTION in natural Hindi/Hinglish.\n"
                "- Prefer Devanagari Hindi for narration, but keep iconic names like Luffy, Zoro, Shanks, Nika readable.\n"
                "- Use casual Indian anime-fan energy, like a friend explaining a wild theory.\n"
                "- Do NOT translate One Piece names awkwardly.\n"
                "- Do NOT use non-Hindi/non-English filler unless it is a canon One Piece term.\n"
                "- Keep section labels exactly in English: TITLE:, SCRIPT:, DESCRIPTION:, HASHTAGS:.\n"
                "- Hashtags can stay lowercase English/romanized for reach.\n\n"
            ),
            "title_rule": "- Write the title in Hindi/Hinglish, not pure English.\n",
            "power_words": (
                "- Use 2-3 high-energy Hindi/Hinglish words that convey urgency or revelation naturally in context. "
                "Do not force them, and do not bold or italicise them.\n"
            ),
            "fomo": "- Include one fresh FOMO line in Hindi/Hinglish; do not repeat the sample wording.\n",
            "final": "CRITICAL: Keep it Gen-Z hype but not spammy. Use clear Hindi/Hinglish and natural fan energy.",
        }
    return {
        "label": "English",
        "rules": (
            "LANGUAGE RULES:\n"
            "- Write TITLE, SCRIPT, and DESCRIPTION in clear, natural English.\n"
            "- Use casual anime-fan energy, like a friend explaining a wild theory.\n"
            "- Do NOT translate One Piece names awkwardly.\n"
            "- Avoid non-English filler unless it is a canon One Piece term.\n"
            "- Keep section labels exactly in English: TITLE:, SCRIPT:, DESCRIPTION:, HASHTAGS:.\n\n"
        ),
        "title_rule": "- Write the title in punchy English.\n",
        "power_words": (
            "- Use 2-3 high-energy English words that convey urgency or revelation naturally in context. "
            "Do not force them, and do not bold or italicise them.\n"
        ),
        "fomo": "- Include one fresh FOMO line in English; do not repeat the sample wording.\n",
        "final": "CRITICAL: Keep it Gen-Z hype but not spammy. Use clear English and natural fan energy.",
    }


def _build_context_block(
    context_text: str,
    ohara_context: str,
    context_sources: Optional[list],
    chapter_number: Optional[int],
) -> str:
    if not context_text and not ohara_context:
        return ""

    source_lines = []
    for source in (context_sources or [])[:3]:
        title = source.get("title") or source.get("source") or "context source"
        url = source.get("url", "")
        source_lines.append(f"- {title} {url}".strip())

    chapter_lock = ""
    if chapter_number:
        chapter_lock = (
            f"HARD CHAPTER LOCK: This video is about One Piece Chapter {chapter_number}. "
            f"The opening sentence must name Chapter {chapter_number}. Do not mention another "
            f"chapter number unless comparing directly to Chapter {chapter_number}.\n\n"
        )

    block = (
        chapter_lock +
        "GROUNDING RULES FOR PROVIDED CONTEXT:\n"
        "- Use PDF extracted text for direct scene/dialogue anchors only when readable.\n"
        "- Use The Library of Ohara context for interpretation and chapter analysis when provided.\n"
        "- Do NOT invent characters, family links, flashbacks, locations, powers, or page text absent from the context.\n"
        "- Do NOT drift to unrelated arcs unless the context explicitly mentions them.\n"
        "- If OCR text is messy, prefer the clean Ohara context over guessing from corrupted words.\n\n"
    )
    if source_lines:
        block += "CONTEXT SOURCES:\n" + "\n".join(source_lines) + "\n\n"
    if context_text:
        quality = _ocr_quality(context_text)
        if quality >= 0.7:
            block += f"CLEANED MANGA PDF TEXT:\n{context_text[:12000]}\n\n"
        else:
            logger.warning("OCR quality %.2f below threshold; skipping raw PDF context", quality)
            block += "NOTE: PDF text quality was too low to use. Rely on cleaner context only.\n\n"
    if ohara_context:
        block += f"THE LIBRARY OF OHARA CHAPTER CONTEXT:\n{ohara_context[:9000]}\n\n"
    return block


def _ocr_quality(text: str) -> float:
    if not text:
        return 1.0

    words = re.findall(r"\S+", text)
    if not words:
        return 1.0

    garbled = sum(
        1
        for word in words
        if len(word) > 20
        or bool(re.search(r"[^\x00-\x7F]{3,}", word))
        or bool(re.search(r"[A-Za-z]{12,}\d|\d[A-Za-z]{12,}", word))
    )
    return 1.0 - (garbled / len(words))


def _profile_blocks(video_profile: str, language_label: str, power_words_rule: str) -> Tuple[str, str, str]:
    if video_profile == "long_youtube":
        output_rule = f"SCRIPT: [human-like {language_label} narration, 5-8 minutes, ~750-1000 spoken words]\n\n"
        blueprint = (
            "LONG-FORM QUALITY BLUEPRINT:\n"
            "- Opening hook: 2-3 sharp sentences that name the canon anchor and central mystery.\n"
            "- Setup: explain the scene/chapter context clearly for viewers who need a reminder.\n"
            "- Evidence: use 3-5 concrete clues from the anchor/context, each tied to the same claim.\n"
            "- Theory chain: connect clues step by step to 2-3 larger lore pieces, not a random lore dump.\n"
            "- Counterpoint: include one fair objection fans may raise, then answer it.\n"
            "- Payoff: restate the theory as one memorable declarative claim before any CTA.\n"
            "- Use natural transitions, but do not output headings inside SCRIPT.\n\n"
        )
        requirements = (
            "SCRIPT REQUIREMENTS:\n"
            "- PURE narration only, no SFX/music.\n"
            "- Open with a 1-sentence powerful declarative hook, max 12 words.\n"
            "- First 20s must establish the mystery, canon anchor, and why it matters.\n"
            "- Keep a YouTube essay rhythm: energetic, clear, and paced for 5-8 minutes.\n"
            "- Show personal reaction to confirmed facts, not uncertainty about whether facts are true.\n"
            f"{power_words_rule}"
            "- Include 3-5 specific canon/context details across the script.\n"
            "- Include named scene locations only when they directly support the theory.\n"
            "- Include one counterargument and one rebuttal before the conclusion.\n"
            "- End with a concrete payoff, then a debate CTA, then a follow CTA.\n"
            f"- TOTAL WORD COUNT: 750-1000 spoken {language_label} words.\n\n"
        )
        return output_rule, blueprint, requirements

    output_rule = f"SCRIPT: [human-like {language_label} narration, 35-45s, ~85-95 spoken words]\n\n"
    blueprint = (
        "SHORT-FORM QUALITY BLUEPRINT:\n"
        "- HOOK, first 10 words: state an impossible-sounding fact or concrete claim. No questions.\n"
        "- EVIDENCE, next 40 words: name one canon anchor and one specific visual, behavior, quote, or scene detail.\n"
        "- PAYOFF, next 25 words: state the theory as a declarative claim that answers the title.\n"
        "- CTA, final 10-15 words: ask for debate only after the payoff, then add a follow CTA.\n"
        "- The chain must stay simple: anchor -> clue -> connection -> claim.\n"
        "- Do not name-drop Imu, Joy Boy, Nika, Ancient Weapons, and Gorosei together unless the topic needs them.\n\n"
    )
    requirements = (
        "SCRIPT REQUIREMENTS:\n"
        "- PURE narration only, no SFX/music.\n"
        "- Open with a 1-sentence powerful declarative hook, max 9 words.\n"
        "- First 10s must spark curiosity with a hidden truth, contradiction, or twist.\n"
        "- Keep fast pace and vary sentence length.\n"
        "- Add vivid, specific detail within the first 15s.\n"
        "- Show personal reaction to confirmed facts, not uncertainty about whether facts are true.\n"
        f"{power_words_rule}"
        "- Include 1 specific detail in the first two sentences: chapter, episode, arc, place, scene, or exact quote.\n"
        "- Include exactly 1 named scene location when possible.\n"
        "- End with a concrete payoff, then a debate CTA, then a follow CTA.\n"
        f"- TOTAL WORD COUNT: 85-95 spoken {language_label} words.\n\n"
    )
    return output_rule, blueprint, requirements


def _build_prompt(
    topic: str,
    language: str,
    context_text: str,
    chapter_number: Optional[int],
    ohara_context: str,
    context_sources: Optional[list],
    video_profile: str,
) -> str:
    language_config = _language_config(language)
    language_label = language_config["label"]
    script_output_rule, quality_blueprint, script_requirements = _profile_blocks(
        video_profile,
        language_label,
        language_config["power_words"],
    )
    context_block = _build_context_block(context_text, ohara_context, context_sources, chapter_number)
    chapter_topic_rule = f"- The first sentence must name Chapter {chapter_number}.\n" if chapter_number else ""

    return (
        "You are a creative anime scriptwriter and passionate One Piece fan.\n"
        "Your first job is accuracy. Your second job is engagement.\n\n"
        "LORE ACCURACY RULES - HIGHEST PRIORITY:\n"
        "- Every theory claim must be supported by a manga chapter, SBS, arc, scene, quote, or provided context.\n"
        "- Do NOT invent connections between characters, fruits, powers, bloodlines, or events not shown in canon/context.\n"
        "- If the topic has no confirmed catastrophic role, reframe around what is confirmed instead of exaggerating.\n"
        "- Before writing, ask: is this claim sourced from manga/context? If no, remove it.\n\n"
        "HOOK RULES - MANDATORY:\n"
        "- NEVER open with Remember, Think about, What if, Imagine, Did you, Do you, or any question.\n"
        "- Open with a declarative statement of fact or an impossible-sounding claim.\n"
        "- The opening must include or immediately lead to a concrete canon anchor.\n\n"
        "TITLE PROMISE RULES:\n"
        "- The SCRIPT must answer the TITLE with one concrete claim.\n"
        "- Do not end with only a question. Give the payoff first, then ask for debate.\n"
        "- Every sentence must move toward the title's central claim.\n"
        "- FAILURE EXAMPLE: If the title promises 'chapters 1-100 clues' but the script only uses chapter 1, that is a broken promise. Cover multiple chapters or rewrite the title to match.\n\n"
        "VOICE-FIRST WRITING RULES:\n"
        "- Write narration that naturally gives TTS emotional context: urgency in the hook, curiosity in evidence, confidence in payoff.\n"
        "- Use punctuation as pacing control: commas for breath, short full-stop sentences for impact, em dashes for weight, ellipses only for suspense.\n"
        "- Shape sentence length as emotional rhythm: short punchy hook, medium evidence beats, one slightly longer payoff sentence, then crisp CTAs.\n"
        "- Use word-level phonetic shaping for spoken clarity: prefer hard consonants for impact, open vowel sounds for wonder, and avoid tongue-twisting clusters near the reveal.\n"
        "- Put emotionally loaded words near the reveal so the voice has semantic cues to emphasize.\n"
        "- Do NOT use markdown in SCRIPT: no **bold**, no *italic*, no underscores, no asterisks of any kind.\n"
        "- Do not output stage directions or bracket tags inside SCRIPT; the TTS layer adds delivery markup separately.\n\n"
        f"{language_config['rules']}"
        f"TOPIC: \"{topic}\"\n\n"
        f"{context_block}"
        "OUTPUT STRUCTURE (in this exact order):\n"
        f"TITLE: [engaging {language_label} title, under 80 chars]\n\n"
        f"{script_output_rule}"
        f"DESCRIPTION: [{language_label}, personal, under 500 chars, bullet points, 3-5 lines, include sticky FOMO]\n\n"
        "HASHTAGS: [10-15 relevant hashtags, lowercase]\n\n"
        "TOPIC LOCK RULES:\n"
        "- Pick one main subject from TOPIC and stay with it from start to end.\n"
        "- Pick one canon anchor that directly fits TOPIC: chapter, episode, arc, island, scene, or quote.\n"
        f"{chapter_topic_rule}"
        "- Do not switch arcs mid-script unless the context directly supports the connection.\n"
        "- Mention at most 2 bigger lore pieces, and both must clearly support the same theory.\n\n"
        f"{quality_blueprint}"
        "SELF-CHECK BEFORE FINAL ANSWER:\n"
        "- Did the script answer the title's promise with a concrete claim? If no, rewrite.\n"
        "- Did punctuation create breath, weight, suspense, and impact without stage directions? If no, rewrite.\n"
        "- Does sentence length move from punchy hook to evidence rhythm to confident payoff? If no, rewrite.\n"
        "- Are reveal words easy to say aloud, with clear consonants and no awkward clusters? If no, rewrite.\n"
        "- Did the script end with a question instead of a payoff? If yes, add the payoff first.\n"
        "- Is there one clear canon anchor in the first two sentences? If no, rewrite.\n"
        "- Did any phrase come from a third language like Indonesian/Malay? If yes, remove it.\n"
        "- Did you invent a canon fact? If yes, remove it.\n\n"
        "TITLE RULES:\n"
        "- Must grab instantly, under 80 chars.\n"
        "- Randomly use one tone: shocking question, urgent warning, hidden truth, impossible claim.\n"
        f"{language_config['title_rule']}\n"
        f"{script_requirements}"
        "REFERENCE STYLE EXAMPLE (do not copy, match quality only):\n"
        "SCRIPT: Chapter 907 puts Shanks inside Mary Geoise. "
        "That single room changes how his warning reads. "
        "The Five Elders do not treat him like a normal pirate, and that is the clue. "
        "My payoff is simple: Shanks knows a government secret most pirates never reach. "
        "Is that protection or manipulation? Follow for the next hidden One Piece truth.\n\n"
        "DESCRIPTION REQUIREMENTS:\n"
        "- Must be 5-6 bullet points.\n"
        "- Each bullet should be punchy and personal.\n"
        "- Include one controversial/debate-trigger line.\n"
        f"{language_config['fomo']}"
        "- End with a community challenge and a soft follow CTA.\n"
        "- Entire description must be between 350-500 characters.\n\n"
        "HASHTAGS RULES:\n"
        "- Always include core tags: #onepiece #anime #theory.\n"
        "- Include 2+ character-specific tags when relevant.\n"
        "- Include 1 arc-specific tag when relevant.\n"
        "- Include 1+ viral bait tag like #mindblown or #plottwist.\n\n"
        f"{language_config['final']}"
    )


def _parse_response(text: str) -> Dict[str, str]:
    result = {key: "" for key in REQUIRED_SECTIONS}
    section_pattern = re.compile(
        r"^\s*(?:#+\s*)?(?:[-*]\s*)?\*{0,3}(TITLE|SCRIPT|DESCRIPTION|HASHTAGS)\s*:\*{0,3}\s*"
        r"(.*?)(?=^\s*(?:#+\s*)?(?:[-*]\s*)?\*{0,3}(?:TITLE|SCRIPT|DESCRIPTION|HASHTAGS)\s*:\*{0,3}|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    for match in section_pattern.finditer(text or ""):
        key = match.group(1).lower()
        value = match.group(2).strip()
        if key == "script":
            value = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", value)
            value = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", value)
            value = re.sub(r"\[.*?\]", "", value)
            value = re.sub(r"\s+", " ", value).strip()
        result[key] = value
    return result


def _word_count(text: str) -> int:
    return len(re.findall(r"[\w'-]+", text or "", flags=re.UNICODE))


def _first_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return ""
    return re.split(r"(?<=[.!?।])\s+", text, maxsplit=1)[0].strip()


def _opening_text(script: str, words: int = 35) -> str:
    tokens = re.findall(r"\S+", script or "")
    return " ".join(tokens[:words])


def _keywords(text: str) -> set:
    words = {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z']{2,}", text or "")
        if word.lower() not in STOPWORDS
    }
    return words


def _normalize_keywords(text: str) -> set:
    normalized = set()
    for word in _keywords(text):
        normalized.add(SUBJECT_SYNONYMS.get(word, word))
    return normalized


def _is_question_hook(hook: str) -> bool:
    return bool(QUESTION_START_RE.search(hook)) and hook.strip().endswith("?")


def _has_declarative_payoff(script: str) -> bool:
    cleaned = re.sub(r"\s+", " ", (script or "").strip())
    if not cleaned:
        return False
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    meaningful = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(phrase in lowered for phrase in CTA_PHRASES):
            continue
        meaningful.append(sentence)
    if not meaningful:
        return False
    return any(
        not sentence.endswith("?") and _word_count(sentence) >= 8
        for sentence in meaningful[-4:]
    )


def validate_generated_script(
    result: Dict[str, str],
    video_profile: str = "short_vertical",
    chapter_number: Optional[int] = None,
) -> ValidationResult:
    """Validate obvious structural quality failures without judging deep canon truth."""
    video_profile = normalize_video_profile(video_profile)
    errors: List[str] = []
    warnings: List[str] = []

    for section in REQUIRED_SECTIONS:
        if not result.get(section, "").strip():
            errors.append(f"Missing required section: {section.upper()}")

    script = result.get("script", "")
    title = result.get("title", "")
    if not script:
        return ValidationResult(errors, warnings)

    count = _word_count(script)
    if video_profile == "long_youtube":
        min_words, max_words = 700, 1050
    else:
        min_words, max_words = 75, 110
    if count < min_words or count > max_words:
        direction = "too long" if count > max_words else "too short"
        target = (min_words + max_words) // 2
        action = "Cut" if count > max_words else "Add"
        errors.append(
            f"Script word count {count} is {direction}. Target is exactly {target} words "
            f"for {video_profile}. {action} approximately {abs(count - target)} words."
        )

    hook = _first_sentence(script)
    if BANNED_HOOK_RE.search(hook):
        errors.append("Opening hook uses a banned opener")
    if _is_question_hook(hook) or hook.endswith("?"):
        errors.append("Opening hook must be declarative, not a question")

    opening = _opening_text(script)
    if not CANON_ANCHOR_RE.search(opening):
        errors.append("Opening lines do not include a concrete canon anchor")

    if chapter_number and not re.search(rf"\bchapter\s+{re.escape(str(chapter_number))}\b", opening, re.IGNORECASE):
        errors.append(f"Chapter-locked script must open with Chapter {chapter_number}")

    title_terms = _normalize_keywords(title)
    script_terms = _normalize_keywords(script)
    if title_terms and not (title_terms & script_terms):
        warnings.append("Title and script may not share a clear subject; review manually")

    if not _has_declarative_payoff(script):
        errors.append("Script ends with a question/CTA without a declarative payoff")

    if re.search(r"\b(i think maybe|maybe|it feels like|i'm starting to think|what if)\b", script, re.IGNORECASE):
        warnings.append("Script contains hedging language that can weaken authority")

    return ValidationResult(errors, warnings)


def _retry_prompt(
    bad_output: Dict[str, str],
    validation: ValidationResult,
    video_profile: str = "short_vertical",
    topic: str = "",
    language: str = "english",
    chapter_number: Optional[int] = None,
) -> str:
    failures = "\n".join(f"- {error}" for error in validation.errors)
    warnings = "\n".join(f"- {warning}" for warning in validation.warnings)
    target_words = "750-1000 words" if video_profile == "long_youtube" else "85-95 words"
    canon_anchor_rule = (
        f"- The first sentence must name Chapter {chapter_number}.\n"
        if chapter_number
        else "- The first or second sentence must include a canon anchor: chapter number, arc, island, scene, or quote.\n"
    )
    return (
        "Your previous answer failed deterministic validation.\n"
        "Rewrite the full One Piece answer. Do not explain anything before or after it.\n\n"
        f"TOPIC: {topic}\n"
        f"LANGUAGE: {language}\n"
        f"VIDEO PROFILE: {video_profile}\n\n"
        f"BLOCKING ERRORS TO FIX:\n{failures}\n\n"
        f"{'WARNINGS TO AVOID:' if warnings else ''}\n{warnings}\n\n"
        "OUTPUT FORMAT - use these four labels exactly, once each:\n"
        f"{REQUIRED_OUTPUT_TEMPLATE}\n\n"
        "PREVIOUS SCRIPT FOR REFERENCE:\n"
        f"{bad_output.get('script', '')}\n\n"
        "RULES THAT STILL APPLY:\n"
        "- Open with a declarative statement, not a question or memory-check phrase.\n"
        f"{canon_anchor_rule}"
        "- Use punctuation as pacing: commas for breath, full stops for impact, dashes for weight, ellipses only for suspense.\n"
        "- Vary sentence length for emotional rhythm: short hook, medium evidence, longer payoff, crisp CTA.\n"
        "- Choose reveal words that sound clean aloud; avoid awkward tongue-twisting clusters.\n"
        "- State the payoff as a concrete claim before asking a debate question.\n"
        "- No markdown: no **bold**, no *italic*, no underscores, no asterisks.\n"
        f"- TOTAL WORD COUNT: {target_words}.\n\n"
        "OUTPUT THE FULL CORRECTED VERSION NOW:"
    )


def _strict_repair_prompt(
    topic: str,
    language: str,
    video_profile: str,
    chapter_number: Optional[int],
    bad_output: Dict[str, str],
    validation: ValidationResult,
) -> str:
    target_words = "750-1000 words" if video_profile == "long_youtube" else "85-95 words"
    failures = "\n".join(f"- {error}" for error in validation.errors)
    canon_anchor = (
        f"Chapter {chapter_number}"
        if chapter_number
        else "a concrete One Piece canon anchor such as a chapter number, arc, island, scene, or quote"
    )
    return (
        "FORMAT REPAIR REQUIRED. Your last answer was incomplete.\n"
        "Return only the four required sections. No preface. No markdown table. No commentary.\n\n"
        f"TOPIC: {topic}\n"
        f"LANGUAGE: {language}\n"
        f"SCRIPT WORD COUNT: {target_words}\n"
        f"OPENING ANCHOR: {canon_anchor}\n\n"
        f"VALIDATION FAILURES:\n{failures}\n\n"
        "EXACT OUTPUT TEMPLATE:\n"
        f"{REQUIRED_OUTPUT_TEMPLATE}\n\n"
        "Use the previous attempt only as a rough idea, but write a complete valid version:\n"
        f"TITLE: {bad_output.get('title', '')}\n"
        f"SCRIPT: {bad_output.get('script', '')}\n"
        f"DESCRIPTION: {bad_output.get('description', '')}\n"
        f"HASHTAGS: {bad_output.get('hashtags', '')}\n"
    )


def _generate_once(prompt: str, video_profile: str = "short_vertical") -> Dict[str, str]:
    video_profile = normalize_video_profile(video_profile)
    response = model.generate_content(
        prompt,
        generation_config=GenerationConfig(**GENERATION_SETTINGS[video_profile]),
    )
    if not response.text:
        raise ValueError("No content generated by Gemini API")
    return _parse_response(response.text)


def _fallback_title(topic: str, bad_output: Dict[str, str]) -> str:
    title = (bad_output.get("title") or "").strip()
    if title:
        return title[:80]

    cleaned_topic = re.sub(r"\s+", " ", (topic or "One Piece hidden clue").strip())
    return cleaned_topic[:80]


def _fallback_anchor(topic: str, chapter_number: Optional[int]) -> str:
    if chapter_number:
        return f"Chapter {chapter_number}"

    topic_text = topic or ""
    chapter_match = re.search(r"\bchapter\s+\d+\b", topic_text, re.IGNORECASE)
    if chapter_match:
        return chapter_match.group(0).title()

    for anchor in (
        "Mary Geoise", "Marineford", "Enies Lobby", "Wano", "Egghead",
        "Ohara", "Sabaody", "God Valley", "Skypiea", "Thriller Bark",
        "Impel Down", "Whole Cake", "Fishman Island", "Alabasta",
    ):
        if re.search(rf"\b{re.escape(anchor)}\b", topic_text, re.IGNORECASE):
            return anchor

    return "Chapter 1"


def _fallback_script(topic: str, bad_output: Dict[str, str], chapter_number: Optional[int]) -> str:
    title = _fallback_title(topic, bad_output)
    anchor = _fallback_anchor(topic, chapter_number)
    subject = re.sub(r"[^A-Za-z0-9' -]", "", title).strip() or "this One Piece clue"
    return (
        f"{anchor} makes {subject} feel impossible to ignore. "
        "The scene looks simple at first, but Oda hides the pressure in the quiet details. "
        "One reaction, one pause, and one choice tell us the truth is bigger than the obvious answer. "
        "That is why this clue matters: it points to a secret motive driving the story from behind the curtain. "
        "My payoff is simple: this was not random foreshadowing, it was setup. "
        "Tell me if this changes your read, and follow for more One Piece truths."
    )


def _fallback_description(topic: str, title: str) -> str:
    subject = title or topic or "this One Piece clue"
    return (
        f"- {subject} hits harder when the canon anchor is clear.\n"
        "- The small scene detail changes the whole theory.\n"
        "- Most fans miss how quiet Oda makes the clue.\n"
        "- Debate this read if you think there is a cleaner answer.\n"
        "- Follow for more hidden One Piece truths."
    )


def _fallback_hashtags(topic: str, title: str) -> str:
    text = f"{topic} {title}".lower()
    tags = ["#onepiece", "#anime", "#theory"]
    for word, tag in (
        ("shanks", "#shanks"),
        ("luffy", "#luffy"),
        ("zoro", "#zoro"),
        ("nika", "#nika"),
        ("joy", "#joyboy"),
        ("wano", "#wano"),
        ("egghead", "#egghead"),
        ("marineford", "#marineford"),
        ("geoise", "#marygeoise"),
        ("mariejois", "#marygeoise"),
    ):
        if word in text and tag not in tags:
            tags.append(tag)
    tags.extend(tag for tag in ("#mindblown", "#plottwist", "#manga") if tag not in tags)
    return " ".join(tags[:12])


def _deterministic_fallback_result(
    topic: str,
    bad_output: Dict[str, str],
    chapter_number: Optional[int],
    failure_errors: List[str],
) -> Tuple[Dict[str, str], ValidationResult]:
    title = _fallback_title(topic, bad_output)
    script = _fallback_script(topic, bad_output, chapter_number)
    result = {
        "title": title,
        "script": script,
        "description": _fallback_description(topic, title),
        "hashtags": _fallback_hashtags(topic, title),
    }
    validation = validate_generated_script(result, chapter_number=chapter_number)
    warnings = validation.warnings + ["FALLBACK_USED: Gemini returned incomplete output after repair"]
    warnings.extend(f"VALIDATION_FAILED: {error}" for error in failure_errors)
    warnings.extend(f"FALLBACK_VALIDATION: {error}" for error in validation.errors)
    return result, ValidationResult(errors=[], warnings=warnings)


def generate_script(
    topic_override: str = None,
    language: str = "english",
    context_text: str = None,
    chapter_number: int = None,
    ohara_context: str = None,
    context_sources: list = None,
    video_profile: str = "short_vertical",
) -> dict:
    """Generate a One Piece narration script via Gemini."""
    try:
        video_profile = normalize_video_profile(video_profile)
        context_text = (context_text or "").strip()
        ohara_context = (ohara_context or "").strip()
        topic = _resolve_topic(topic_override, context_text, ohara_context, chapter_number)
        prompt = _build_prompt(
            topic,
            language,
            context_text,
            chapter_number,
            ohara_context,
            context_sources,
            video_profile,
        )

        retry_attempted = False
        result = _generate_once(prompt, video_profile)
        validation = validate_generated_script(result, video_profile, chapter_number)

        if not validation.ok:
            logger.warning("Generated script failed validation; retrying once: %s", validation.errors)
            retry_attempted = True
            retry_result = _generate_once(
                _retry_prompt(result, validation, video_profile, topic, language, chapter_number),
                video_profile,
            )
            retry_validation = validate_generated_script(retry_result, video_profile, chapter_number)
            if retry_validation.ok:
                result = retry_result
                validation = retry_validation
            else:
                logger.warning(
                    "Generated script failed validation after retry; attempting strict repair: %s",
                    retry_validation.errors,
                )
                repair_result = _generate_once(
                    _strict_repair_prompt(
                        topic,
                        language,
                        video_profile,
                        chapter_number,
                        retry_result,
                        retry_validation,
                    ),
                    video_profile,
                )
                repair_validation = validate_generated_script(repair_result, video_profile, chapter_number)
                if repair_validation.ok:
                    result = repair_result
                    validation = repair_validation
                else:
                    logger.error(
                        "Generated script failed validation after strict repair; using deterministic fallback: %s",
                        repair_validation.errors,
                    )
                    result, validation = _deterministic_fallback_result(
                        topic,
                        repair_result,
                        chapter_number,
                        repair_validation.errors,
                    )

        result["resolved_topic"] = topic
        result["quality_warnings"] = validation.warnings
        result["retry_attempted"] = retry_attempted
        return result
    except Exception as e:
        logger.error("Error generating script: %s", str(e))
        raise


if __name__ == "__main__":
    try:
        generated = generate_script()
        print("\nSCRIPT GENERATED\n")
        print("SCRIPT:")
        print("-" * 40)
        print(generated["script"])
        print("\nDESCRIPTION:")
        print("-" * 40)
        print(generated["description"])
        print("\nHASHTAGS:")
        print("-" * 40)
        print(generated["hashtags"])
        if generated.get("quality_warnings"):
            print("\nQUALITY WARNINGS:")
            print(generated["quality_warnings"])
        print()
    except Exception as exc:
        print(f"Error: {str(exc)}")
