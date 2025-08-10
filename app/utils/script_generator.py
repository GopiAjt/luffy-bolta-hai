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


def generate_script() -> dict:
    """
    Generate a 30-40s One Piece narration script via Gemini.
    
    Returns:
        dict: A dictionary containing title, script, description, and tags
    """
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
            f"Create a 30-40 second YouTube Shorts script about: {topic}\n\n"
            "Return the result in the following JSON format:\n\n"
            "{\n"
            "  \"title\": \"string - matches style of: 🌟 ONE PIECE'S MOST UNDERRATED LEGENDS!\",\n"
            "  \"script\": \"string - the narration only, in same tone & pacing as example\",\n"
            "  \"description\": \"string - engaging, hype-driven, 2-3 sentences like example\",\n"
            "  \"tags\": [\n"
            "    \"#OnePiece\",\n"
            "    \"#tag2\",\n"
            "    \"#tag3\",\n"
            "    \"... more tags like example\"\n"
            "  ]\n"
            "}\n\n"
            "Requirements:\n\n"
            "1. Title: Must be attention-grabbing, emoji-rich, and in ALL CAPS.\n\n"
            "2. Script:\n"
            "   - Pure narration only (no sound effects or directions)\n"
            "   - Fast-paced, high-energy delivery\n"
            "   - 30-40 seconds when read aloud (~50 words)\n"
            "   - End with a call to action\n\n"
            "3. Description:\n"
            "   - engaging, hype-driven sentences\n"
            "   - Include emojis and power scaling terms\n"
            "   - End with a question to boost engagement\n\n"
            "4. Tags:\n"
            "   - Include #OnePiece and 5-7 relevant hashtags\n"
            "   - Mix popular and niche tags\n"
            "   - Focus on characters, powers, and current arcs"
        )

        response = model.generate_content(prompt)
        # Get the first candidate from the response
        if response.text:
            try:
                # Clean and parse the JSON response
                response_text = response.text.strip()
                # Handle cases where response might be wrapped in markdown code blocks
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].strip()
                
                import json
                result = json.loads(response_text)
                
                # Ensure all required fields are present
                required_fields = ['title', 'script', 'description', 'tags']
                if not all(field in result for field in required_fields):
                    raise ValueError("Missing required fields in the response")
                    
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Response was: {response_text}")
                raise ValueError("Invalid JSON response from API")
        else:
            raise ValueError("No text generated by Gemini API")
    except Exception as e:
        logger.error(f"Error generating script: {str(e)}")
        raise


if __name__ == "__main__":
    import json
    script_data = generate_script()
    print("✅ Script generated successfully!\n")
    print("Title:", script_data['title'])
    print("\nScript:")
    print(script_data['script'])
    print("\nDescription:", script_data['description'])
    print("\nTags:", " ".join(script_data['tags']))
    
    # Save to file for reference
    with open('generated_script.json', 'w', encoding='utf-8') as f:
        json.dump(script_data, f, indent=2, ensure_ascii=False)
    print("\n✅ Script saved to 'generated_script.json'")
