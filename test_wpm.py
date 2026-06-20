import logging
from pathlib import Path
from app.utils.audio.tts_generator import generate_voiceover

logging.basicConfig(level=logging.INFO)

def test_tts_wpm():
    # A standard paragraph to test TTS pacing (exactly 100 words).
    text = (
        "The Grand Line is the most dangerous ocean in the world, filled with bizarre "
        "weather, terrifying sea monsters, and the strongest pirates in history. "
        "To conquer it, you need more than just physical strength. You need an unbreakable "
        "will and a reliable crew. Many have tried to reach the final island, Laugh Tale, "
        "but almost all have failed. Only the Pirate King, Gol D. Roger, managed to "
        "uncover its secrets. Now, a new generation is sailing out, hunting for the One "
        "Piece. The ultimate treasure awaits whoever can survive the perilous journey and "
        "claim the throne of the sea."
    )
    
    word_count = len(text.split())
    
    print(f"Generating voiceover for {word_count} words...")
    
    result = generate_voiceover(
        text=text,
        output_dir=Path("/tmp/tts_test"),
        language="English",
        video_profile="youtube_long"
    )
    
    duration = result["duration"]
    wpm = (word_count / duration) * 60
    
    print("=" * 40)
    print(f"Generated Audio Duration: {duration:.2f} seconds")
    print(f"Word Count: {word_count} words")
    print(f"Actual TTS WPM: {wpm:.2f} Words Per Minute")
    print("=" * 40)

if __name__ == "__main__":
    test_tts_wpm()
