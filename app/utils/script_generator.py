import os
import random
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


def generate_script() -> str:
    """Generate a 30–60s One Piece narration script via Gemini."""
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
        topic = random.choice(topics)

        prompt = (
            "You are a creative anime scriptwriter and passionate One Piece fan.\n"
            "Write in a Gen-Z, hype, casual tone — but keep it readable.\n\n"
            
            f"TOPIC(place holders to be replaced): \"{topic}\"\n\n"

            "OUTPUT STRUCTURE (in this exact order):\n"
            "TITLE: [engaging, under 80 chars]\n\n"
            "SCRIPT: [human-like narration, 35–45s, ~85–95 words]\n\n"
            "DESCRIPTION: [personal, under 500 chars, BULLET POINTS + emojis, 3–5 lines, include sticky FOMO]\n\n"
            "HASHTAGS: [10–15 relevant hashtags, lowercase]\n\n"

            "RULES FOR STYLE:\n"
            "- SOUND like a real fan talking to friends.\n"
            "- Use hype language for energy, max 2–3 slang terms.\n"
            "- Keep sentences short, fast-paced, easy to follow.\n"
            "- Blend slang with clear English — balance is key.\n"
            "- Always add curiosity, suspense, or debate bait.\n\n"

            "TITLE RULES:\n"
            "- Must grab instantly (under 80 chars).\n"
            "- Randomly use ONE of these tones: shocking question, urgent warning, hidden truth, impossible claim.\n\n"

            "SCRIPT REQUIREMENTS:\n"
            "- PURE narration only, no SFX/music.\n"
            "- Open with a 1-sentence powerful hook (max 9 words).\n"
            "- First 10s must spark curiosity (hidden truth, contradiction, twist).\n"
            "- Keep fast pace, vary sentence length.\n"
            "- Add vivid, sensory detail within first 15s.\n"
            "- Show personal emotions + uncertainty (hesitations, incomplete thoughts).\n"
            "- Use exactly 2–3 power words (e.g., shocking, dangerous, hidden).\n"
            "- Include 1 specific detail (chapter/episode OR exact quote).\n"
            "- Mid-escalation: show rising excitement/doubt naturally.\n"
            "- End with a curiosity-driven CTA inviting debate (max 12 words).\n"
            "- After debate CTA, ADD a second CTA to gain followers\n"
            "- TOTAL WORD COUNT: 85–95 words (auto-enforce brevity).\n\n"

            "DESCRIPTION REQUIREMENTS:\n"
            "- MUST be 5–6 bullet points.\n"
            "- Each bullet MUST start with an emoji (🤯🔥💀😱⚡🚨).\n"
            "- Each bullet MUST be 8–15 words long (not shorter).\n"
            "- MUST include at least one controversial/debate-trigger line.\n"
            "- MUST include at least one FOMO phrase ('Most fans missed this...', 'I almost overlooked this...').\n"
            "- MUST end with a vulnerable CTA + community challenge ('Bet you can’t prove me wrong 👀👇').\n"
            "- ALSO add a soft CTA for followers: 'Follow for more hidden One Piece truths ⚡'.\n"
            "- Entire description MUST be between 400–500 characters.\n\n"


            "HASHTAGS RULES:\n"
            "- Always include core (#onepiece #anime #theory).\n"
            "- Include 2+ character-specific tags.\n"
            "- Include 1 arc-specific tag.\n"
            "- Include 1+ viral bait tags (#mindblown, #plottwist).\n\n"

            "CRITICAL: Keep it Gen-Z hype but NOT spammy. Clear sentences, natural fan energy, maximum scroll-stopping engagement."
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
        
        # Split the response into sections
        sections = response.text.split('\n\n')
        
        for section in sections:
            if section.startswith('TITLE:'):
                result['title'] = section.replace('TITLE:', '').strip()
            elif section.startswith('SCRIPT:'):
                result['script'] = section.replace('SCRIPT:', '').strip()
            elif section.startswith('DESCRIPTION:'):
                result['description'] = section.replace('DESCRIPTION:', '').strip()
            elif section.startswith('HASHTAGS:'):
                result['hashtags'] = section.replace('HASHTAGS:', '').strip()
        
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
