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
        topic = random.choice(topics)

        prompt = (
            "You are a creative anime scriptwriter and One Piece expert.\n"
            f"Create engaging content for a YouTube video about: \"{topic}\"\n\n"
            "You MUST produce all 4 sections fully, in this exact order and format:\n\n"

            "1. TITLE (Clickbait-style, under 70 chars):\n"
            "- All caps, emojis, questions or teasers.\n\n"

            "2. SCRIPT (30–40 seconds, ~80 words):\n"
            "- Pure narration only, no SFX or music cues.\n"
            "- HOOK: Start with contradiction, shocking statistic, or 'What if I told you...'\n"
            "- Use POWER WORDS: 'devastating', 'impossible', 'forbidden', 'secretly'\n"
            "- Add SPECIFIC DETAILS: Chapter numbers, exact quotes, precise locations\n"
            "- Include EMOTIONAL STAKES: 'This broke my heart', 'Fans were FURIOUS'\n"
            "- Use SENSORY LANGUAGE: Visual, auditory, tactile descriptions\n"
            "- Add PACING TECHNIQUES: Short punches. Then longer revelations that build tension.\n"
            "- Insert CREDIBILITY BOOSTERS: 'Oda confirmed', 'Hidden in plain sight', 'Twitter exploded'\n"
            "- Include FAN CONNECTION: 'You probably missed', 'Most viewers don't realize'\n"
            "- Add STAKES ESCALATION: 'But that's not even the crazy part...'\n"
            "- End with one of these + URGENT CTA: 'CLIFFHANGER or URGENT CALL TO ACTION or CHALLENGE or PREDICTION BAIT or COMMUNITY RALLYING or TEASER or CONTROVERSIAL' based on the script\n\n"

            "3. VIDEO DESCRIPTION (under 500 chars):\n"
            "- START with a shocking statistic, contradiction, or mind-bending question\n"
            "- Include 1 CONTROVERSIAL statement that will trigger debates\n"
            "- Add 3-5 punchy reveals with emojis (🤯⚡🔥) and cliffhanger explanations\n"
            "- Insert FOMO triggers: 'Most fans missed...', '99% got this wrong...'\n"
            "- Drop manga spoiler warnings or episode references for credibility\n"
            "- Include community challenges: 'Bet you can't guess...', 'Prove me wrong...'\n"
            "- Add theory confidence levels: '🔥SPICY THEORY🔥' or '⚠️DANGEROUS PREDICTION⚠️'\n"
            "- Reference other characters/arcs to expand intrigue\n"
            "- End with multiple hook questions AND a dare/challenge CTA\n\n"

            "4. HASHTAGS (5–10 relevant hashtags):\n"
            "- Include #onepiece #anime plus trending topic tags.\n\n"
            
            "Format EXACTLY as:\n"
            "TITLE: [your engaging title]\n\n"
            "SCRIPT: [your script here]\n\n"
            "DESCRIPTION: [your engaging description]\n\n"
            "HASHTAGS: [your hashtags here]"
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
