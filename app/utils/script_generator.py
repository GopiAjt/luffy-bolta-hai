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
model = genai.GenerativeModel("gemini-2.0-flash")


def generate_script() -> str:
    """Generate a 30–60s One Piece narration script via Gemini."""
    try:
        topics = [
            # CHARACTER RANKINGS & COMPARISONS
            "Top 10 Strongest Straw Hats RANKED",
            "Most Overpowered Devil Fruits EVER",
            "Weakest to Strongest Admirals",
            "Every Yonko Ranked by Power",
            "Most Tragic One Piece Backstories",
            "Smartest Characters in One Piece",
            "Fastest Characters RANKED",
            "Most Loyal Crew Members",
            "Best One Piece Villains RANKED",
            "Most Underrated Characters",
            
            # POWER SCALING & BATTLES
            "Luffy vs Every Yonko - Who Wins?",
            "Can Zoro Beat an Admiral?",
            "Strongest Haki Users RANKED",
            "Most Broken Devil Fruit Abilities",
            "Garp vs Prime Whitebeard",
            "Revolutionary Army vs Marines",
            "Old Generation vs New Generation",
            "Who Can Beat Kaido 1v1?",
            "Mihawk's True Power Level",
            "Blackbeard's Secret Third Devil Fruit?",
            
            # THEORIES & MYSTERIES
            "The One Piece is Actually...",
            "Imu's True Identity REVEALED",
            "Joy Boy's Real Name Theory",
            "Blackbeard's Body Secret",
            "Zunesha's Ancient Crime",
            "The Will of D Explained",
            "Vegapunk's Final Message",
            "Ancient Weapons Locations",
            "Void Century Truth",
            "Laugh Tale's Real Secret",
            
            # DEVIL FRUIT CONTENT
            "Most Useless Devil Fruits",
            "Devil Fruit Awakenings RANKED",
            "Mythical Zoan Powers Explained",
            "Logia vs Paramecia vs Zoan",
            "Devil Fruit Weaknesses EXPOSED",
            "Future Devil Fruit Users",
            "Lost/Extinct Devil Fruits",
            "Devil Fruit Origins Theory",
            "Most Creative Devil Fruit Uses",
            "Artificial vs Real Devil Fruits",
            
            # EMOTIONAL/DRAMATIC MOMENTS
            "Saddest One Piece Deaths",
            "Most Emotional Straw Hat Moments",
            "Going Merry's Final Goodbye",
            "Ace's Death Impact",
            "Brook's Backstory Explained",
            "Robin's 'I Want to Live!'",
            "Sanji vs Zeff Flashback",
            "Nami's Tears for Help",
            "Chopper's Origin Story",
            "Law's Tragic Past",
            
            # WORLD BUILDING & LORE
            "Every One Piece Island RANKED",
            "Grand Line Secrets Explained",
            "Celestial Dragons' Dark Truth",
            "Revolutionary Army Goals",
            "Marine Ranks Explained",
            "Pirate Crews Power Hierarchy",
            "One Piece Timeline Explained",
            "Ancient Kingdom Theory",
            "Red Line Destruction Theory",
            "All Blue Location Theory",
            
            # CREW DYNAMICS & RELATIONSHIPS
            "Straw Hat Relationships RANKED",
            "Who Joins the Crew Next?",
            "Crew Member Roles Explained",
            "Best Straw Hat Combos",
            "Zoro vs Sanji Rivalry",
            "Captain-First Mate Duos",
            "Found Family Moments",
            "Crew Recruitment Stories",
            "Inter-Crew Friendships",
            "Mentor-Student Relationships",
            
            # CHAPTER/ARC BREAKDOWNS
            "This Week's Chapter BREAKDOWN",
            "Best One Piece Arcs RANKED",
            "Wano's Hidden Details",
            "Marineford War Analysis",
            "Whole Cake Island Secrets",
            "Dressrosa Connections",
            "Water 7 Foreshadowing",
            "Enies Lobby Epic Moments",
            "Alabasta Kingdom Politics",
            "East Blue Arc Importance",
            
            # PREDICTIONS & FUTURE CONTENT
            "Final War Predictions",
            "Who Dies in the Final Arc?",
            "Straw Hats' Final Bounties",
            "End Game Power Rankings",
            "One Piece Ending Theories",
            "Next Villain Predictions",
            "Future Alliance Theories",
            "Straw Hat Dreams Coming True",
            "Government Downfall Theory",
            "New World Changes",
            
            # FUN FACTS & TRIVIA
            "One Piece Easter Eggs",
            "Oda's Hidden Messages",
            "Real World Inspirations",
            "Voice Actor Fun Facts",
            "Animation Secrets",
            "Manga vs Anime Differences",
            "Cover Story Importance",
            "SBS Question Highlights",
            "Character Design Evolution",
            "Behind the Scenes Facts",
            
            # CONTROVERSIAL TAKES
            "Most Overrated Characters",
            "Unpopular One Piece Opinions",
            "Worst One Piece Decisions",
            "Plot Holes in One Piece",
            "Characters Who Should Be Stronger",
            "Disappointing Reveals",
            "Overhyped Moments",
            "Boring Arcs Ranked",
            "Wasted Character Potential",
            "Animation Quality Issues",
            
            # NOSTALGIA & THROWBACKS
            "Early One Piece vs Now",
            "Forgotten Plot Points",
            "Characters We Miss",
            "Old Art Style vs New",
            "First Appearances vs Current",
            "Abandoned Storylines",
            "Early Foreshadowing Payoffs",
            "Classic Moments Ranked",
            "Evolution of Animation",
            "Original vs Current Voices",
            
            # QUICK EDUCATIONAL CONTENT
            "One Piece in 60 Seconds",
            "Devil Fruit Explained Simply",
            "Haki Types Breakdown",
            "Marine Ranks Quick Guide",
            "Yonko System Explained",
            "Revolutionary Army 101",
            "World Government Structure",
            "Pirate Crew Basics",
            "Grand Line Navigation",
            "One Piece Geography",
            
            # INTERACTIVE/ENGAGEMENT CONTENT
            "Guess the Character by Laugh",
            "One Piece Quiz Challenge",
            "Rate My Top 10 List",
            "Predict Next Chapter Events",
            "Guess the Devil Fruit Power",
            "Name That Attack Move",
            "Character Voice Impressions",
            "Draw One Piece Characters",
            "React to First Time Watchers",
            "Tier List Maker",
            
            # CURRENT EVENTS TOPICS
            "Latest Chapter Reactions",
            "New Episode Highlights",
            "Manga vs Anime Comparisons",
            "Live Action Adaptation News",
            "Video Game Content",
            "Merchandise Reviews",
            "Convention Coverage",
            "Voice Actor Interviews",
            "Creator Statements",
            "Fan Art Showcases",
            
            # CROSSOVER & COMPARISON CONTENT
            "One Piece vs Naruto vs Bleach",
            "Anime Power Scaling Cross-Series",
            "Best Shonen Protagonists",
            "Manga Art Style Comparisons",
            "Voice Actor Shared Roles",
            "Similar Character Archetypes",
            "Story Structure Comparisons",
            "Animation Studio Differences",
            "Cultural Impact Analysis",
            "International Reception",
            
            # MEME & HUMOR CONTENT
            "Funniest One Piece Moments",
            "Meme-able Character Expressions",
            "Ridiculous Power-Ups",
            "Absurd Plot Armor Moments",
            "Weirdest Character Designs",
            "Strangest Devil Fruits",
            "Most Random Plot Twists",
            "Goofy Animation Moments",
            "Funny Translation Errors",
            "Community Memes Explained"
        ]
        # topic = random.choice(topics)

        prompt = (
            "You are a creative anime scriptwriter and passionate One Piece fan.\n"
            f"Create engaging content for a YouTube video about: [Search a recent trending 'one piece' topic from the internet]\n\n"
            "You MUST produce all 4 sections fully, in this exact order and format:\n\n"

            "1. TITLE (Clickbait-style, under 70 chars):\n"
            "- All caps, emojis, questions or teasers.\n"
            "- Make each title unique, NOT formulaic — vary phrasing and structure.\n"
            "- ROTATE between these styles: shocking question, impossible claim, urgent warning, hidden truth reveal.\n\n"

            "2. SCRIPT (30–40 seconds, ~70 words):\n"
            "- PURE narration only, no SFX or music cues.\n"
            "- Open with ONE short, high-impact hook (5–9 words).\n"
            "- Keep a fast pace, vary sentence length.\n"
            "- Add emotional or surprising details.\n"
            "- Use vivid, sensory language IN THE FIRST 15 SECONDS.\n"
            "- End with a curiosity-driven teaser or CTA (8–12 words max).\n"

            "- HUMANIZATION LAYER (MANDATORY):\n"
            "* Frame as personal discovery\n"
            "* Include emotional vulnerability\n"
            "* Add conversational elements\n"
            "* Show uncertainty\n"
            "* Use incomplete thoughts\n"

            "- RANDOMLY SELECT opening hook (vary each time):\n"
            "* Personal contradiction\n"
            "* Discovery moment\n"
            "* Shocking realization\n"
            "* Revelation setup\n"
            "* Hidden truth\n"
            "* Time bomb\n"

            "- POWER WORDS (use exactly 2-3):\n"
            "* Place 1 in the hook, 1–2 mid-escalation.\n"

            "- SPECIFIC DETAILS (use 1-2):\n"
            "* Use either an exact quote OR a precise chapter/episode reference.\n"

            "- EMOTIONAL STAKES (use 1):\n"
            "* Show why this matters to you personally.\n"

            "- SENSORY DESCRIPTION (use exactly 1):\n"
            "* Describe what you saw, heard, or felt physically.\n"

            "- NATURAL PACING:\n"
            "* Mix short punchy lines with longer rambling revelations.\n"
            "* Include natural hesitations: “Wait—no… hold on…”\n"

            "- CREDIBILITY BOOSTER (use 1):\n"
            "* Mention research, rewatch, or reread to sound authentic.\n"

            "- VIEWER CONNECTION (use 1-2):\n"
            "* Speak directly: “Did you catch this too?” or “Am I crazy?”\n"

            "- MID-ESCALATION with human touch (use exactly 1):\n"
            "* Confusion, excitement, fear, or disbelief rising naturally.\n"

            "- HUMAN-LIKE CLOSING (randomly select):\n"
            "* Vulnerable cliffhanger, urgent personal, confident challenge, emotional prediction,\n"
            "* community rally, nervous teaser, controversial uncertainty.\n"

            "3. VIDEO DESCRIPTION (under 500 chars, bullet points):\n"
            "- PERSONAL HOOK: Start with personal discovery moment, shocking realization, or vulnerable admission.\n"
            "- CONTROVERSY: Include 1 statement that will trigger debates in comments.\n"
            "- REVEALS: Add 3-5 personal discoveries with emojis (🤯⚡🔥💀🚨😱) and cliffhanger explanations.\n"
            "- FOMO with personality: 'Most fans missed this like I did...', 'I felt so dumb when I realized...', 'Only true fans will get why this scared me...'.\n"
            "- CREDIBILITY: Drop manga spoiler warnings, episode references, or personal research mentions.\n"
            "- COMMUNITY CHALLENGES: 'Bet you can't find what I found...', 'Prove my theory wrong...', 'Help me figure out if I'm crazy...'.\n"
            "- THEORY LEVELS: Use '🔥SPICY THEORY🔥', '⚠️DANGEROUS PREDICTION⚠️', '💀DARK REALIZATION💀', '😱MIND-BLOWN MOMENT😱'.\n"
            "- PERSONAL CONNECTIONS: Reference how this relates to other characters/arcs you've analyzed.\n"
            "- VULNERABLE CTA: End with personal questions AND community challenge that shows uncertainty.\n\n"

            "4. HASHTAGS (8-12 trending tags):\n"
            "- Core: Ex. #onepiece #anime #manga #luffy\n"
            "- Trending: Ex. #onepiecetheory #animeshorts #mangaspoilers #onepiecefan\n"
            "- Topic-specific: Add relevant character/arc hashtags\n"
            "- Viral potential: Ex. #mindblown #plottwist #hiddensecrets #animereaction\n\n"
                            
            "Format EXACTLY as:\n"
            "TITLE: [your engaging title]\n\n"
            "SCRIPT: [your human-like script here]\n\n"
            "DESCRIPTION: [your personal engaging description]\n\n"
            "HASHTAGS: [your hashtags here]\n\n"
            
            "CRITICAL: Write as a REAL FAN sharing a personal discovery, not an AI presenting information. Include natural speech patterns, emotional reactions, and genuine uncertainty. Make it sound like you're excitedly (but nervously) telling friends about something you just figured out."
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
