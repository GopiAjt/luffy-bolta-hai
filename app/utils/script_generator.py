import os
import random
import re
from dotenv import load_dotenv
import google.generativeai as genai
import logging

# Set up logging
logger = logging.getLogger(__name__)

# 1. Load environment variables from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file")

# 2. Configure the Gemini client
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash-lite")


def is_generic_topic(topic: str) -> bool:
    """Return True when frontend/default text is not an actual One Piece topic."""
    if not topic or not topic.strip():
        return True

    normalized = topic.strip().lower()
    generic_phrases = [
        "generate a",
        "one piece narration script",
        "30-60 second",
        "30–60 second",
        "script in hindi",
        "script in english",
    ]
    return any(phrase in normalized for phrase in generic_phrases)


def generate_script(
    topic_override: str = None,
    language: str = "english",
    context_text: str = None,
    chapter_number: int = None,
    ohara_context: str = None,
    context_sources: list = None,
) -> dict:
    """Generate a 30-60s One Piece narration script via Gemini."""
    try:
        topics = [
            # 🔥 Character Secrets

"Zoro’s Hidden Bloodline REVEALED – Is He a Shimotsuki?!",

"The FORBIDDEN Truth About Blackbeard’s Three Devil Fruits",

"Shanks’ Secret Connection to the Gorosei Exposed",

"Why Mihawk is HIDING His True Strength",

"Luffy’s Family Secret Oda Never Wanted Us to See",

# 🧩 Oda’s Hidden Details

"Oda HID This Shocking Foreshadowing in Marineford",

"The Secret Message Buried in Enies Lobby’s “I Want to Live” Scene",

"7 Hidden Details You MISSED in Egghead Arc",

"Oda’s Secret Code in Every One Piece Bounty Poster",

"The Symbol Oda Keeps Hiding in the Straw Hat",

# 🌀 Theory Revelations

"The One Piece Treasure is Actually an Ancient Weapon",

"The REAL Reason Why Shanks Met the Gorosei",

"Joy Boy is NOT Who You Think He Is",

"Imu’s TRUE Purpose Finally REVEALED",

"Why the Will of D Changes EVERYTHING in One Piece",

# 🔗 Connections & Relationships

"The Hidden Connection Between Luffy and Rocks D. Xebec",

"How Ohara’s Destruction Secretly Affects Egghead",

"Dragon’s Role in Roger’s Final Journey You Never Noticed",

"The Bloodline Connection Oda Hides Between Garp and Blackbeard",

"Why Luffy and Shanks Share This Secret Bond",

# ⚡ Power & Ability Mysteries

"Zoro’s Sword Enma Has a Secret Ability Nobody Talks About",

"The Hidden Advanced Level of Conqueror’s Haki",

"Sanji Can Do THIS But Never Shows It",

"The REAL Weakness of Kaido’s Mythical Zoan Fruit",

"Nika’s True Origin EXPOSED – The Sun God’s Lost Power",
        ]
        context_text = (context_text or "").strip()
        ohara_context = (ohara_context or "").strip()
        topic = topic_override.strip() if not is_generic_topic(topic_override) else random.choice(topics)
        if (context_text or ohara_context) and is_generic_topic(topic_override):
            topic = f"One Piece Chapter {chapter_number} theory" if chapter_number else "Latest One Piece manga chapter theory"
        language = (language or "english").strip().lower()
        is_hindi = language in {"hindi", "hi", "hinglish"}
        if is_hindi:
            language_label = "Hindi/Hinglish"
            language_rules = (
                "LANGUAGE RULES:\n"
                "- Write TITLE, SCRIPT, and DESCRIPTION in natural Hindi/Hinglish.\n"
                "- Prefer Devanagari Hindi for narration, but keep iconic names like Luffy, Zoro, Shanks, Nika readable.\n"
                "- Use casual Indian anime-fan energy, like a friend explaining a wild theory.\n"
                "- Do NOT translate One Piece names awkwardly.\n"
                "- Do NOT use any non-Hindi/non-English language phrase. Avoid Indonesian/Malay/Spanish/Japanese filler unless it is a canon One Piece term.\n"
                "- Keep section labels exactly in English: TITLE:, SCRIPT:, DESCRIPTION:, HASHTAGS:.\n"
                "- Hashtags can stay lowercase English/romanized for reach.\n\n"
            )
            title_rule = "- Write the title in Hindi/Hinglish, not pure English.\n\n"
            power_words_rule = "- Use exactly 2-3 power words in Hindi/Hinglish (e.g., shocking, khatarnak, hidden truth).\n"
            reference_example = (
                "SCRIPT: Think about Chapter 907 for a second. Shanks walks into Mary Geoise and meets the Gorosei... "
                "but was he really just giving a warning? His calm eyes in that room feel way too controlled. "
                "Imu's shadow and the Nika fruit may not be separate threads. "
                "I think Shanks is hiding one of the most dangerous secrets in One Piece. "
                "So tell me, is he a hero or the final manipulator? Follow for the next wild truth.\n\n"
            )
            fomo_rule = "- MUST include at least one FOMO phrase in Hindi/Hinglish ('Most fans ne ye miss kiya...', 'Maine bhi pehle ignore kar diya...').\n"
            final_quality_rule = "CRITICAL: Keep it Gen-Z hype but NOT spammy. Clear Hindi/Hinglish sentences, natural fan energy, maximum scroll-stopping engagement."
        else:
            language_label = "English"
            language_rules = (
                "LANGUAGE RULES:\n"
                "- Write TITLE, SCRIPT, and DESCRIPTION in clear, natural English.\n"
                "- Use casual anime-fan energy, like a friend explaining a wild theory.\n"
                "- Do NOT translate One Piece names awkwardly.\n"
                "- Avoid non-English filler unless it is a canon One Piece term.\n"
                "- Keep section labels exactly in English: TITLE:, SCRIPT:, DESCRIPTION:, HASHTAGS:.\n\n"
            )
            title_rule = "- Write the title in punchy English.\n\n"
            power_words_rule = "- Use exactly 2-3 power words in English (e.g., shocking, dangerous, hidden truth).\n"
            reference_example = (
                "SCRIPT: Think about Chapter 907 for a second. Shanks walks into Mary Geoise and meets the Gorosei... "
                "but was he really just giving a warning? His calm eyes in that room feel way too controlled. "
                "Imu's shadow and the Nika fruit may not be separate threads. "
                "I think Shanks is hiding one of the most dangerous secrets in One Piece. "
                "So tell me, is he a hero or the final manipulator? Follow for the next wild truth.\n\n"
            )
            fomo_rule = "- MUST include at least one FOMO phrase in English ('Most fans missed this...', 'I ignored this clue at first...').\n"
            final_quality_rule = "CRITICAL: Keep it Gen-Z hype but NOT spammy. Clear English sentences, natural fan energy, maximum scroll-stopping engagement."

        context_block = ""
        chapter_topic_rule = ""
        if chapter_number:
            chapter_topic_rule = (
                f"- If a HARD CHAPTER LOCK is provided, the first sentence must name Chapter {chapter_number}.\n"
            )
        if context_text or ohara_context:
            trimmed_context = context_text[:12000]
            trimmed_ohara = ohara_context[:9000]
            sources_text = ""
            if context_sources:
                source_lines = []
                for source in context_sources[:3]:
                    title = source.get("title") or source.get("source") or "context source"
                    url = source.get("url", "")
                    source_lines.append(f"- {title} {url}".strip())
                sources_text = "CONTEXT SOURCES:\n" + "\n".join(source_lines) + "\n\n"
            chapter_lock = (
                f"HARD CHAPTER LOCK: This video is about One Piece Chapter {chapter_number}. "
                f"Do not mention another chapter number unless comparing directly to Chapter {chapter_number}.\n"
                if chapter_number else ""
            )
            context_block = (
                f"{chapter_lock}"
                f"{sources_text}"
                "GROUNDING RULES FOR PDF MODE:\n"
                "- Use PDF extracted text for direct scene/dialogue anchors only when it is readable.\n"
                "- Use The Library of Ohara context for interpretation and chapter analysis when provided.\n"
                "- Do NOT invent characters, family links, flashbacks, locations, or page text absent from the context below.\n"
                "- Do NOT drift to unrelated arcs such as Egghead, Wano, or Mary Geoise unless this context explicitly mentions them.\n"
                "- If OCR text is messy, prefer the clean Ohara context over guessing from corrupted words.\n\n"
            )
            if trimmed_context:
                context_block += (
                    "CLEANED MANGA PDF TEXT:\n"
                    f"{trimmed_context}\n\n"
                )
            if trimmed_ohara:
                context_block += (
                    "THE LIBRARY OF OHARA CHAPTER CONTEXT:\n"
                    f"{trimmed_ohara}\n\n"
                )

        prompt = (
            "You are a creative anime scriptwriter and passionate One Piece fan.\n"
            "Write in a Gen-Z, hype, casual tone, but keep it readable.\n\n"
            f"{language_rules}"
            
            f"TOPIC: \"{topic}\"\n\n"
            f"{context_block}"

            "OUTPUT STRUCTURE (in this exact order):\n"
            f"TITLE: [engaging {language_label} title, under 80 chars]\n\n"
            f"SCRIPT: [human-like {language_label} narration, 35-45s, ~85-95 spoken words]\n\n"
            f"DESCRIPTION: [{language_label}, personal, under 500 chars, BULLET POINTS + emojis, 3-5 lines, include sticky FOMO]\n\n"
            "HASHTAGS: [10–15 relevant hashtags, lowercase]\n\n"

            "RULES FOR STYLE:\n"
            "- SOUND like a real anime fan talking to friends.\n"
            "- Use hype language for energy, max 2-3 slang terms.\n"
            "- Keep sentences short, fast-paced, easy to follow.\n"
            f"- Keep the narration easy to speak in {language_label}.\n"
            "- Always add curiosity, suspense, or debate bait.\n"
            "- Avoid vague lines like 'that moment' unless the exact scene was named first.\n\n"

            "TOPIC LOCK RULES:\n"
            "- Pick ONE main subject from TOPIC and stay with it from start to end.\n"
            "- Pick ONE canon anchor that directly fits TOPIC: chapter, episode, arc, island, scene, or quote.\n"
            f"{chapter_topic_rule}"
            "- Do NOT switch arcs mid-script. If the anchor is Wano/Kaido, do not suddenly move to Egghead/Vegapunk.\n"
            "- If the anchor is Egghead/Vegapunk, do not open with Wano/Kaido.\n"
            "- Mention at most 2 bigger lore pieces, and both must clearly support the same theory.\n"
            "- Every sentence must connect to the same theory. Remove anything that feels like a different video.\n\n"

            "QUALITY BLUEPRINT:\n"
            "- Sentence 1: Name a concrete canon anchor (chapter/episode/arc/place/scene).\n"
            "- Sentence 2: Ask one suspicious question about that anchor.\n"
            "- Sentence 3: Point to one visual or behavioral clue from the scene.\n"
            "- Sentence 4: Connect that clue to 2 bigger lore pieces, not more.\n"
            "- Sentence 5: State one clear theory in first person.\n"
            "- Final lines: Ask a binary debate question, then add a follow CTA.\n"
            "- The theory chain must be easy to follow: anchor -> clue -> connection -> claim.\n"
            "- Do not name-drop Imu, Joy Boy, Nika, Ancient Weapons, Gorosei all together unless the topic truly needs them.\n\n"

            "SELF-CHECK BEFORE FINAL ANSWER:\n"
            "- Does the first line and final theory discuss the same subject? If no, rewrite.\n"
            "- Did you accidentally mix Wano with Egghead, or Kaido with Vegapunk? If yes, rewrite.\n"
            "- Did any phrase come from a third language like Indonesian/Malay? If yes, remove it.\n"
            "- Is there one clear canon anchor in the first two sentences? If no, rewrite.\n\n"

            "TITLE RULES:\n"
            "- Must grab instantly (under 80 chars).\n"
            "- Randomly use ONE of these tones: shocking question, urgent warning, hidden truth, impossible claim.\n"
            f"{title_rule}"

            "SCRIPT REQUIREMENTS:\n"
            "- PURE narration only, no SFX/music.\n"
            "- Open with a 1-sentence powerful hook (max 9 words).\n"
            "- First 10s must spark curiosity (hidden truth, contradiction, twist).\n"
            "- Keep fast pace, vary sentence length.\n"
            "- Add vivid, sensory detail within first 15s.\n"
            "- Show personal emotions + uncertainty (hesitations, incomplete thoughts).\n"
            f"{power_words_rule}"
            "- Include 1 specific detail in the first two sentences (chapter/episode/arc/place OR exact quote).\n"
            "- Include exactly 1 named scene location when possible, like Mary Geoise, Wano, Marineford, Egghead.\n"
            "- Mid-escalation: show rising excitement/doubt naturally.\n"
            "- End with a curiosity-driven CTA inviting debate (max 12 words).\n"
            "- After debate CTA, ADD a second CTA to gain followers\n"
            f"- TOTAL WORD COUNT: 85-95 spoken {language_label} words (auto-enforce brevity).\n\n"

            "REFERENCE STYLE EXAMPLE (do not copy, match quality only):\n"
            f"{reference_example}"

            "DESCRIPTION REQUIREMENTS:\n"
            "- MUST be 5-6 bullet points.\n"
            "- Each bullet MUST start with an emoji (🤯🔥💀😱⚡🚨).\n"
            f"- Each bullet MUST be 8-15 {language_label} words long (not shorter).\n"
            "- MUST include at least one controversial/debate-trigger line.\n"
            f"{fomo_rule}"
            "- MUST end with a vulnerable CTA + community challenge ('Bet you can’t prove me wrong 👀👇').\n"
            "- ALSO add a soft CTA for followers: 'Follow for more hidden One Piece truths ⚡'.\n"
            "- Entire description MUST be between 400-500 characters.\n\n"


            "HASHTAGS RULES:\n"
            "- Always include core (#onepiece #anime #theory).\n"
            "- Include 2+ character-specific tags.\n"
            "- Include 1 arc-specific tag.\n"
            "- Include 1+ viral bait tags (#mindblown, #plottwist).\n\n"

            f"{final_quality_rule}"
        )

        response = model.generate_content(prompt)
        if not response.text:
            raise ValueError("No content generated by Gemini API")
            
        # Parse the response
        result = {
            'title': '',
            'script': '',
            'description': '',
            'hashtags': ''
        }
        
        section_pattern = re.compile(
            r'^\s*(TITLE|SCRIPT|DESCRIPTION|HASHTAGS):\s*(.*?)(?=^\s*(?:TITLE|SCRIPT|DESCRIPTION|HASHTAGS):|\Z)',
            re.DOTALL | re.MULTILINE
        )

        for match in section_pattern.finditer(response.text):
            key = match.group(1).lower()
            result[key] = match.group(2).strip()
        
        return result
    except Exception as e:
        logger.error(f"Error generating script: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        result = generate_script()
        print("\n✅ SCRIPT GENERATED!\n")
        print("📜 SCRIPT:")
        print("-" * 40)
        print(result['script'])
        print("\n📝 DESCRIPTION:")
        print("-" * 40)
        print(result['description'])
        print("\n🏷️  HASHTAGS:")
        print("-" * 40)
        print(result['hashtags'])
        print()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
