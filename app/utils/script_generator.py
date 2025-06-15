import os
import random
from datetime import timedelta
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()

# Get API key from .env
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# One Piece topics
topics = [
    "Lore Explanation",
    "Theory or Prediction",
    "Fact or Trivia",
    "Character Power Breakdown"
]

selected_topic = random.choice(topics)

# Prompt
prompt = f"""
You are a creative anime scriptwriter and One Piece expert.
Write a concise, high-energy, 30–60 second script for a YouTube Shorts or Instagram Reel about One Piece.
The topic is: {selected_topic}.

Requirements:
- Keep it fast-paced and engaging for a narrated short-form anime video
- Use an exciting tone that fits a voiceover narration
- Do not include camera/stage directions or speaker labels — only pure narration
- End with a teaser or call to action like “But that’s not all…” or “What do you think?”

Assume the audience knows One Piece casually, but still explain terms briefly.
Length: Around 90–120 words max.
"""

# Generate the script
response = model.generate_content(prompt)
script_text = response.text.strip()

# Generate .ass subtitle content


def generate_ass_subtitles(script, duration_seconds=60):
    lines = script.split('. ')
    num_lines = len(lines)
    interval = duration_seconds / max(num_lines, 1)

    ass_content = """[Script Info]
Title: One Piece Short
ScriptType: v4.00+
Collisions: Normal
PlayResX: 1280
PlayResY: 720
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,28,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,1.5,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    for i, line in enumerate(lines):
        start_time = timedelta(seconds=i * interval)
        end_time = timedelta(seconds=(i + 1) * interval)
        ass_line = f"Dialogue: 0,{str(start_time)[:-3]},{str(end_time)[:-3]},Default,,0,0,0,,{line.strip()}"
        ass_content += ass_line + "\n"

    return ass_content


# Write to file
ass_output = generate_ass_subtitles(script_text, duration_seconds=60)

with open("one_piece_script.ass", "w", encoding="utf-8") as f:
    f.write(ass_output)

print("✅ Script generated and saved to one_piece_script.ass")
