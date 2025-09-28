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
            # CHARACTER SECRET TEMPLATES

            "{Character}'s Hidden {Power/Past/Connection} REVEALED",
            "The FORBIDDEN Truth About {Character}'s {Ability/Origin}",
            "{Character}'s Secret {Weakness/Strength} Oda Never Shows",
            "Why {Character} is HIDING Their True {Power/Identity}",
            "{Character}'s {Bloodline/Heritage/Family} Secret Exposed",

            # ODA'S HIDDEN DETAILS TEMPLATES

            "Oda HID This {Detail/Connection} in {Arc/Chapter}",
            "The Secret {Foreshadowing/Message} Buried in {Scene/Arc}",
            "{Number} Hidden Details You MISSED in {Arc/Episode}",
            "Oda's Secret {Pattern/Code} Throughout One Piece",
            "The {Symbol/Design/Reference} Oda Keeps Hiding",

            # THEORY REVELATION TEMPLATES

            "{Major Mystery} is Actually {Shocking Theory}",
            "The REAL Reason Behind {Event/Decision/Power}",
            "{Popular Belief} is WRONG - Here's the Truth",
            "{Character/Object/Location}'s TRUE Purpose REVEALED",
            "Why {Theory/Belief} Changes EVERYTHING",

            # CONNECTION/RELATIONSHIP TEMPLATES

            "The Hidden Connection Between {Character A} and {Character B}",
            "How {Past Event} Secretly Affects {Current Situation}",
            "{Character}'s Role in {Major Event} You Never Noticed",
            "The {Bloodline/Crew/Organization} Connection Oda Hides",
            "Why {Character} and {Character} Share This Secret"

            # POWER/ABILITY MYSTERY TEMPLATES

            "{Character}'s {Devil Fruit/Haki/Technique} Secret Ability",
            "The Hidden {Type/Level/Form} of {Power System}",
            "{Character} Can Do THIS But Never Shows It",
            "The REAL Weakness of {Powerful Character/Ability}",
            "{Power/Technique}'s True Origin EXPOSED"

            # WORLD-BUILDING SECRET TEMPLATES

            "The Dark Truth About {Organization/Location/System}",
            "{Island/Kingdom/Organization}'s Hidden {Purpose/Past}",
            "What {Government/Pirates/Marines} DON'T Want You to Know",
            "The Secret Behind {World Building Element}",
            "{Location/System}'s Connection to {Ancient Mystery}"

            # CURRENT CONTENT TEMPLATES

            "Chapter {Number}: The {Detail/Revelation} Everyone Missed",
            "{Arc Name}'s BIGGEST Secret Finally Revealed",
            "This Week's Chapter Hides {Shocking Discovery}",
            "The {Current Villain/Situation} Connection to {Past Event}",
            "Episode {Number}'s Hidden {Message/Foreshadowing}"

            # CONTROVERSIAL ANGLE TEMPLATES

            "Why {Popular Character/Theory} is Actually {Opposite Claim}",
            "{Beloved Moment/Character} Has a Dark Secret",
            "The {Disappointing/Confusing} Truth About {Popular Topic}",
            "{Character} Made the WORST Possible Decision",
            "{Fan Favorite Thing} is Actually Problematic"

            # PREDICTION/FUTURE TEMPLATES

            "{Character} Will {Action} in {Timeframe/Arc} - Here's Why",
            "The Next {Crew Member/Villain/Power-Up} is Obviously {Prediction}",
            "{Current Situation} Leads to {Major Future Event}",
            "{Character}'s Final {Battle/Moment/Revelation} Predicted",
            "How {Current Arc} Sets Up {Future Major Event}"

            # COMPARATIVE ANALYSIS TEMPLATES

            "{Character A} vs {Character B}: The Secret Difference",
            "{Past Arc/Event} vs {Current Arc/Event}: Hidden Pattern",
            "{Old Generation Character} vs {New Generation Character} Truth",
            "{Power/Ability A} vs {Power/Ability B}: Which is Really Stronger",
            "{Theory A} vs {Theory B}: Which Oda Actually Hints At"
        ]
        topic = random.choice(topics)

        prompt = (
            "You are a creative anime scriptwriter and passionate One Piece fan.\n"
            "Write in a Gen-Z, hype, casual tone — but keep it readable.\n\n"
            
            f"TOPIC: \"{topic}\"\n\n"

            "OUTPUT STRUCTURE (in this exact order):\n"
            "TITLE: [engaging, under 80 chars]\n\n"
            "SCRIPT: [human-like narration, 35–45s, ~75–85 words]\n\n"
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
            "- TOTAL WORD COUNT: 75–85 words (auto-enforce brevity).\n\n"

            "DESCRIPTION REQUIREMENTS:\n"
            "- MUST be 5–6 bullet points.\n"
            "- Each bullet MUST start with an emoji (🤯🔥💀😱⚡🚨).\n"
            "- Each bullet MUST be 8–15 words long (not shorter).\n"
            "- MUST include at least one controversial/debate-trigger line.\n"
            "- MUST include at least one FOMO phrase ('Most fans missed this...', 'I almost overlooked this...').\n"
            "- MUST end with a vulnerable CTA + community challenge ('Bet you can’t prove me wrong 👀👇').\n"
            "- Entire description MUST be between 300–500 characters.\n\n"


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
