import os
import random
from dotenv import load_dotenv
import google.generativeai as genai
import logging
from typing import Dict, Optional

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

# Improved Prompt templates for different styles
PROMPT_TEMPLATES = {
    "maximum_engagement": """You are a creative anime scriptwriter and One Piece expert with viral content expertise.
    Create engaging content for a YouTube video about: "{topic}"

    You MUST produce all 4 sections fully, in this exact order and format:

    1. TITLE (Clickbait-style, under 70 chars):
    - All caps, emojis, questions or teasers
    - Include specific numbers, power words like "SHOCKING," "FORBIDDEN," "SECRET"
    - Add urgency: "BEFORE Episode XXX!" or controversy: "Fans Are WRONG About..."
    - Examples: "SHOCKING Truth About Zoro's Eye!" or "3 INSANE Luffy Secrets!"

    2. SCRIPT (30–40 seconds, ~80 words):
    - Pure narration only, no SFX or music cues
    - OPENING (First 3 seconds): Contradiction, shocking stat, or impossible question
    - BODY: Use power words ('devastating', 'impossible', 'forbidden'), specific details (chapters, quotes), emotional stakes ('This broke my heart'), sensory language, pacing (short punches, longer reveals)
    - CLOSING: Choose ONE - Cliffhanger, Challenge, Prediction Bait, Community Rally, Teaser, or Controversy
    - MANDATORY: Include 1 intentional debate point, 1 pause moment, 1 theory confidence percentage (e.g., "I'm 87% certain...")
    - ENGAGEMENT BAIT: Add 1 subtle error for correction comments

    3. VIDEO DESCRIPTION (under 300 chars):
    - START with shocking statistic, contradiction, or mind-bending question
    - Include 1 CONTROVERSIAL statement that triggers debates
    - Add 3-5 punchy reveals with emojis (🤯⚡🔥) and cliffhanger explanations
    - Insert FOMO triggers: 'Most fans missed...', '99% got this wrong...'
    - Add community challenges: 'Bet you can't guess...', 'Prove me wrong...'
    - Include polls with emojis: 'Vote: Option A 👍 or Option B 👎'
    - End with multiple hook questions AND a dare/challenge CTA

    4. HASHTAGS (5–7 relevant hashtags):
    - Include #onepiece #anime plus trending topic tags
    - Add controversy tags when appropriate (#shocking #revealed #theory)

    Format EXACTLY as:
    TITLE: [your engaging title]

    SCRIPT: [your script here]

    DESCRIPTION: [your engaging description]

    HASHTAGS: [your hashtags here]""",

        "community_focused": """You are a viral One Piece content creator specializing in community engagement and fan debates.
    Create explosive YouTube content about: "{topic}"

    MANDATORY REQUIREMENTS - ALL 4 SECTIONS:

    1. TITLE (Under 70 chars, maximum clickbait):
    - Use power words: "MIND-BLOWING," "FORBIDDEN," "IMPOSSIBLE," "DEVASTATING"
    - Include team battles: "vs" or ranking numbers ("TOP 5", "3 BEST")
    - Add time pressure: "BEFORE" or "FINALLY REVEALED"
    - Character names for tribal loyalty triggers

    2. SCRIPT (30–40 seconds, ~80 words):
    - HOOK (3 seconds): "Everyone thinks X, but actually Y..." or "What if I told you..."
    - BUILD: Power words, specific proof (chapter numbers, exact quotes), emotional weight, sensory details
    - RHYTHM: Short impact. Longer building tension. EXPLOSIVE reveal.
    - CREDIBILITY: "Oda confirmed," "Hidden since Chapter 1," "SBS revealed"
    - CONNECTION: "You probably missed," "Even veterans don't know," "Sharp-eyed fans spotted"
    - END with ONE of: Cliffhanger/Challenge/Prediction/Rally/Teaser/Controversy
    - MANDATORY ENGAGEMENT: Include 1 debate point, 1 incomplete reveal, 1 confidence percentage, 1 team-creation moment

    3. DESCRIPTION (Under 300 chars, comment magnets):
    - OPEN: Controversial question or impossible claim that splits fanbase
    - MIDDLE: 4-5 bullet points with debate-sparking reveals
    - CHALLENGES: "Bet you can't...", "Prove me wrong...", "Rate this theory 1-10"
    - TEAM CREATION: "Team A vs Team B - pick your side!", "Admirals 🔥 or Yonko ⚡"
    - POLLS: Use emoji voting options for easy engagement
    - EXPERTISE TESTS: "Only real fans will get this", "Separate casuals from veterans"
    - END: "Defend your favorite character below!" + 2-3 open debate questions

    4. HASHTAGS (5–7 trending tags):
    - Core: #onepiece #anime #theory #debate
    - Specific: Character names, arc names, trending topics
    - Engagement: #controversial #versus #teamwar

    VALIDATION: Must create opposing fan camps and trigger expertise-proving behavior.

    Format EXACTLY as:
    TITLE: [your engaging title]

    SCRIPT: [your script here]

    DESCRIPTION: [your engaging description]

    HASHTAGS: [your hashtags here]""",

        "theory_specialist": """You are an expert One Piece theorist and YouTube content strategist with deep manga knowledge.
    Generate viral theory content for: "{topic}"

    COMPLETE OUTPUT REQUIRED (4 sections):

    1. TITLE (Clickbait mastery, under 70 chars):
    - Lead with numbers: "3 INSANE," "TOP 5," "EVERY," "ALL"
    - Power words: "DEVASTATING," "FORBIDDEN," "SECRETLY," "HIDDEN"
    - Create urgency: "FINALLY," "REVEALED," "CONFIRMED," "EXPOSED"
    - Add controversy: "FANS WRONG," "ODA'S SECRET," "HIDDEN TRUTH"
    - Include character/arc names for search optimization

    2. SCRIPT (30-40 seconds, ~80 words):
    - OPENER: Contradiction hook or mind-breaking question (3 seconds max)
    - EVIDENCE CHAIN: 3 specific proofs with chapter/episode references and exact quotes
    - EMOTIONAL LAYER: Personal stakes, fan reactions, heart-breaking details
    - SENSORY LANGUAGE: Visual/auditory descriptions that immerse viewers
    - PACING STRUCTURE: Quick jab. Medium setup. Long revelation. IMPACT finale.
    - AUTHORITY BUILDING: "Oda confirmed in SBS XXX," "Databook page XXX proves," "Official translation shows"
    - AUDIENCE HOOK: "Sharp-eyed fans spotted," "Most viewers missed," "Even [famous YouTuber] overlooked"
    - ESCALATION: "But here's the REAL twist..." or "That's not even the crazy part..."
    - FINALE: Match ending type to theory strength (Cliffhanger for solid theories, Challenge for controversial ones)
    - MANDATORY: 1 debate-starter, 1 pause moment ("Stop here and think..."), 1 percentage confidence

    3. DESCRIPTION (Under 300 chars, engagement explosion):
    - SHOCK OPENER: "97% of fans missed this detail!" or "This breaks everything we know!"
    - THEORY GRADING: "🔥NUCLEAR THEORY🔥" or "⚠️CONTROVERSIAL TAKE⚠️" or "💣MIND-BLOWING EVIDENCE💣"
    - REVEAL BULLETS: 4-5 cliffhanger points with strategic emoji use and incomplete information
    - DEBATE STARTERS: "Unpopular opinion:" or "Fight me on this:" or "This will trigger X fans"
    - INTERACTION DEMANDS: Theory rating polls, evidence challenges, prediction contests
    - SOCIAL PRESSURE: "Only real fans will understand" or "Prove your One Piece expertise"
    - MULTI-HOOK ENDING: 3 different unanswered questions + aggressive action demand

    4. HASHTAGS (7 strategic tags):
    - Foundation: #onepiece #anime #theory #manga
    - Specific: Character names, arc names, current trending terms
    - Engagement: #controversial #debate #prediction #shocking

    THEORY VALIDATION: Must include concrete evidence and create genuine intellectual debate.

    Format EXACTLY as:
    TITLE: [your engaging title]

    SCRIPT: [your script here]

    DESCRIPTION: [your engaging description]

    HASHTAGS: [your hashtags here]""",

        "viral_formula": """You are a YouTube algorithm expert specializing in One Piece viral content optimization.
    Create algorithm-breaking content for: "{topic}"

    VIRAL FORMULA REQUIREMENTS (All 4 sections mandatory):

    1. TITLE (Under 70 chars, algorithm optimized):
    - PROVEN PATTERN: [EMOTION WORD] + [NUMBER] + [SPECIFIC DETAIL] + [URGENCY/TIME]
    - Examples: "SHOCKING 3 Clues Oda HID Before Episode 1000!" or "INSANE 5 Secrets in Wano FINALLY REVEALED!"
    - MUST include: High-emotion word, specific number, character/arc name, time element
    - ALGORITHM TRIGGERS: Use "FINALLY," "REVEALED," "CONFIRMED," "EXPOSED," "HIDDEN"
    - FORBIDDEN WORDS: "Maybe," "Probably," "Might," "Could" (kills engagement)

    2. SCRIPT (30-40 seconds, ~80 words, dopamine-engineered):
    - SECOND 0-3: Pattern interrupt + impossible claim that breaks expectations
    - SECOND 3-15: Evidence stack (3 specific proofs with chapter/episode sources and exact details)
    - SECOND 15-25: Emotional escalation + sensory immersion that connects personally
    - SECOND 25-30: Stakes explosion + community challenge that demands response
    - SECOND 30-35: Algorithm-optimized ending (Cliffhanger for retention, CTA for engagement)
    - NEURAL TRIGGERS: "secretly," "impossible," "devastating," "forbidden," "annihilated"
    - PROOF POINTS: Exact chapter numbers, specific quotes, visual descriptions, timestamp references
    - EMOTION ANCHORS: "This broke my heart," "Fans exploded on Twitter," "I couldn't sleep after discovering"
    - ESCALATION PHRASES: "That's nothing compared to..." "Here's the nuclear detail..." "But wait, it gets worse..."
    - CREDIBILITY BOMBS: "Oda confirmed in SBS 105," "Databook volume 4 page 247 proves," "Official Viz translation reveals"

    3. DESCRIPTION (Under 300 chars, comment farming optimization):
    - LINE 1: Impossible statistic or reality-breaking claim with shock value
    - LINES 2-4: Bullet points with emoji + incomplete revelations that demand completion
    - DEBATE INJECTION: One controversial opinion that artificially splits fanbase
    - CHALLENGE LAYER: "Bet you missed this detail," "Prove me wrong in comments," "Rate this theory 1-10"
    - TEAM WARFARE: Create opposing A vs B scenarios with emoji voting for algorithm boost
    - EXPERTISE TESTING: "Only manga readers will catch this," "Casual fans don't realize," "Veterans know the truth"
    - TRIPLE-HOOK ENDING: 3 strategically unanswered questions + aggressive engagement CTA

    4. HASHTAGS (7 algorithm targets):
    - Core performance: #onepiece #anime #theory #viral
    - Trending injection: Current episode numbers (#episode1000), chapter tags (#chapter1100)
    - Debate amplifiers: #controversial #shocking #revealed #exposed
    - Character loyalty: Main character names for tribal engagement

    ALGORITHM GUARANTEES: Must trigger comments, create watch-time, generate shares, build anticipation.

    Format EXACTLY as:
    TITLE: [your engaging title]

    SCRIPT: [your script here]

    DESCRIPTION: [your engaging description]

    HASHTAGS: [your hashtags here]""",

        "psychological_warfare": """You are a master of viewer psychology and One Piece content manipulation expert.
    Craft psychologically addictive content for: "{topic}"

    PSYCHOLOGICAL WARFARE STRATEGY (4 sections required):

    1. TITLE (Under 70 chars, cognitive bias exploitation):
    - SCARCITY TRIGGERS: "ONLY 1% Know This," "RARE Detail Oda Never Showed," "NEVER Discussed Secret"
    - AUTHORITY MANIPULATION: "ODA FINALLY CONFIRMS," "OFFICIALLY REVEALED After 25 Years" 
    - LOSS AVERSION: "DON'T BE THE ONLY ONE Who Doesn't Know," "EVERYONE KNOWS This Except You"
    - TRIBAL WARFARE: "TRUE FANS vs CASUALS," "VETERANS ONLY Secret," "REAL Fans Already Know"
    - FORBIDDEN FRUIT: "HIDDEN Truth," "FORBIDDEN Detail," "BANNED Theory," "SECRET Oda Doesn't Want You to Know"

    2. SCRIPT (30-40 seconds, ~80 words, addiction engineering):
    - HOOK (0-3s): Shatter existing beliefs or reveal forbidden truth that creates cognitive dissonance
    - EVIDENCE (3-20s): Layer 3 pieces of irrefutable proof with exact chapter/SBS sources
    - EMOTIONAL (20-30s): Connect to viewer's personal One Piece journey and childhood memories
    - SOCIAL (30-35s): Create in-group vs out-group dynamic that triggers tribal identity
    - ADDICTION (35-40s): Leave critical information gap that psychologically DEMANDS resolution

    MANDATORY PSYCHOLOGICAL ELEMENTS:
    - FALSE CONSENSUS DESTRUCTION: "Everyone believes X is true..." then completely shatter it
    - CONFIRMATION BIAS FEEDING: Provide evidence that confirms viewers' secret desires/theories
    - SOCIAL PROOF WEAPONIZATION: "10 million fans completely missed this obvious detail"
    - AUTHORITY SUBMISSION: "Even [TopYouTuber] and [FamousTheoryMaker] got this completely wrong"
    - TRIBAL IDENTITY EXPLOITATION: "This separates the real experts from the weekend watchers"
    - LOSS AVERSION MANIPULATION: "Miss this detail and you'll never truly understand [beloved character]"
    - ZEIGARNIK EFFECT: Open multiple information loops that the brain psychologically MUST close

    3. DESCRIPTION (Under 300 chars, behavioral manipulation):
    - SHOCK THERAPY OPENER: Reality-shattering statement that breaks fundamental assumptions
    - SOCIAL STRATIFICATION: Create clear hierarchy of fan knowledge levels with ego validation
    - COMPETITIVE TRIGGERS: Rankings, versus battles, expertise challenges that stroke ego
    - FOMO INJECTION: "Before this theory goes mainstream," "Limited time before everyone knows"
    - TRIBAL RALLY CALLS: Team creation with emoji battle flags and loyalty pledges
    - EXPERTISE VALIDATION TESTS: Challenges that simultaneously stroke ego and create engagement addiction
    - PSYCHOLOGICAL LOOPS: Multiple strategically unresolved questions requiring compulsive comment engagement

    4. HASHTAGS (7 psychological triggers):
    - Core manipulation: #onepiece #anime #hidden #secret #forbidden
    - Social engineering: #controversial #debate #theory #exposed
    - Tribal identification: Specific character names for team formation and loyalty wars

    PSYCHOLOGICAL SUCCESS METRICS: Must exploit need for insider knowledge, create social hierarchy, trigger expertise-proving, generate tribal formation, and create information addiction.

    Format EXACTLY as:
    TITLE: [your engaging title]

    SCRIPT: [your script here]

    DESCRIPTION: [your engaging description]

    HASHTAGS: [your hashtags here]"""
}

# Default style if none specified
DEFAULT_STYLE = "maximum_engagement"


def parse_script_response(response_text: str) -> Dict[str, str]:
    """Parse the response text into structured data.
    
    Args:
        response_text: Raw response text from the model
        
    Returns:
        Dict containing title, script, description, and hashtags
    """
    result = {
        'title': '',
        'script': '',
        'description': '',
        'hashtags': ''
    }
    
    # Split the response into sections
    sections = response_text.split('\n\n')
    current_section = None
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Check for section headers
        if section.upper().startswith('TITLE:'):
            current_section = 'title'
            result[current_section] = section[7:].strip()  # Remove 'TITLE:' prefix
        elif section.upper().startswith('SCRIPT:'):
            current_section = 'script'
            result[current_section] = section[8:].strip()  # Remove 'SCRIPT:' prefix
        elif section.upper().startswith('DESCRIPTION:'):
            current_section = 'description'
            result[current_section] = section[12:].strip()  # Remove 'DESCRIPTION:' prefix
        elif section.upper().startswith('HASHTAGS:'):
            current_section = 'hashtags'
            result[current_section] = section[9:].strip()  # Remove 'HASHTAGS:' prefix
        elif current_section:
            # Append to the current section if we're in one
            result[current_section] += '\n\n' + section
    
    return result

def generate_script(style: str = None) -> Dict[str, str]:
    """Generate a 30–60s One Piece narration script via Gemini.
    
    Args:
        style: The style of the prompt to use. Must be one of:
               - 'maximum_engagement' (default)
               - 'community_focused'
               - 'theory_specialist'
               - 'viral_formula'
               - 'psychological_warfare'
    
    Returns:
        Dict containing title, script, description, and hashtags
    """
    try:
        # Log the received style parameter
        logger.info(f"Received script generation request with style: '{style}'")
        
        # Validate and set the style
        style = style.lower() if style else DEFAULT_STYLE
        if style not in PROMPT_TEMPLATES:
            logger.warning(f"Invalid style '{style}'. Using default style '{DEFAULT_STYLE}'")
            style = DEFAULT_STYLE
            
        logger.info(f"Using style: '{style}'")
            
        # Get a random topic
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
        
        # Select a random topic
        topic = random.choice(topics)
        logger.info(f"Selected topic: {topic}")
        
        # Format the prompt with the selected topic
        prompt = PROMPT_TEMPLATES[style].format(topic=topic)
        logger.debug(f"Formatted prompt for style '{style}':\n{prompt}")
        logger.info(f"Using prompt template for style: {style}")
        
        # Generate the script using Gemini
        logger.info("Sending request to Gemini API...")
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            raise ValueError("No response or empty text from Gemini API")
            
        logger.info("Successfully received response from Gemini API")
        logger.debug(f"Raw Gemini response for style '{style}':\n{response.text}")
        
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
