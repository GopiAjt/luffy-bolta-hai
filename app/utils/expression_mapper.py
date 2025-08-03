import re
import json
import sys
from pathlib import Path
import google.generativeai as genai


def ass_time_to_seconds(ass_time):
    # Format: H:MM:SS.cc
    h, m, s = ass_time.split(':')
    sec, cs = s.split('.')
    return int(h)*3600 + int(m)*60 + int(sec) + float('0.'+cs)


def parse_ass_file(ass_path):
    lines = Path(ass_path).read_text(encoding='utf-8').splitlines()
    dialogue_lines = [l for l in lines if l.startswith('Dialogue:')]
    result = []
    for line in dialogue_lines:
        parts = line.split(',', 9)
        if len(parts) < 10:
            continue
        start = parts[1].strip()
        end = parts[2].strip()
        text = re.sub(r'{\\b\d}', '', parts[9]).replace(
            '{\b1}', '').replace('{\b0}', '').strip()
        text = re.sub(r'\\[Nn]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        result.append({
            'start': start,
            'end': end,
            'text': text
        })
    return result


def build_gemini_prompt(sub_lines):
    """
    Build a Gemini-compatible prompt for expression mapping from subtitle lines.
    sub_lines: list of dicts with 'start', 'end', 'text'
    Returns: prompt string
    """
    prompt = '''You are an expert video scene analyzer.\n
    I will provide you with subtitle lines from an `.ass` file, which represent spoken dialogue in a video.\n
    Your task is to map each subtitle line to the most likely facial expression the speaker would have during that line.\n\n
    Assume this is for an animated or synthesized video, and your output will be used to drive facial animation.\n\n
    Your output must be a JSON array where each object contains:\n
    - "start": timestamp of when the line begins (in seconds or H:MM:SS format)\n
    - "end": timestamp of when the line ends\n
    - "text": the spoken line\n
    - "expression": the facial expression label best suited for this line\n\n
    Use only these labels: "neutral", "happy", "angry", "surprised", "sad", "smirking", "confident", "serious", "worried", "intense", "excited", "embarrassed" etc.\n\n
    Use tone, punctuation, emphasis, and keywords to infer emotion. Humor, sarcasm, or rhetorical questions should also influence the expression choice. You may also infer if the narrator is hyped or calm, authoritative or playful.\n\n
    Example Input:\n[\n  { "start": "0:00:01.00", "end": "0:00:04.00", "text": "Okay, here we go!" },\n  
    { "start": "0:00:04.00", "end": "0:00:12.00", "text": "One Piece power levels! Let’s BREAK ‘EM DOWN!" }\n]\n\n
    Expected Output Format:\n[\n  {\n    "start": "0:00:01.00",\n    "end": "0:00:04.00",\n    "text": "Okay, here we go!",\n    "expression": "excited"\n  },\n  {\n    "start": "0:00:04.00",\n    "end": "0:00:12.00",\n    "text": "One Piece power levels! Let’s BREAK ‘EM DOWN!",\n    "expression": "intense"\n  }\n]\n\nNow here are the actual subtitle lines to analyze:\n'''
    prompt += json.dumps(sub_lines, indent=2, ensure_ascii=False)
    return prompt


def setup_gemini(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")


def generate_expression_mapping_with_gemini(model, sub_lines):
    """
    Given a Gemini model and subtitle lines (list of dicts with start, end, text),
    build a Gemini prompt and return the parsed JSON result.
    """
    prompt = build_gemini_prompt(
        [{k: l[k] for k in ('start', 'end', 'text')} for l in sub_lines])
    response = model.generate_content(prompt)
    # Try to extract JSON from the response
    try:
        # Gemini may return markdown code block, so strip if needed
        text = response.text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.endswith('```'):
            text = text[:-3]
        return json.loads(text)
    except Exception as e:
        print("Failed to parse Gemini response as JSON:", e)
        print("Raw response:\n", response.text)
        return None


def gemini_expression_mapping_from_ass(ass_path, api_key):
    """
    High-level utility: Given an .ass file and Gemini API key, return expression mapping as JSON.
    """
    model = setup_gemini(api_key)
    sub_lines = parse_ass_file(ass_path)
    # Remove 'expression' field if present (let Gemini decide)
    for l in sub_lines:
        l.pop('expression', None)
    return generate_expression_mapping_with_gemini(model, sub_lines)


def main():
    if len(sys.argv) < 3:
        print("Usage: python expression_mapper.py input.ass output.json")
        sys.exit(1)
    ass_path = sys.argv[1]
    out_path = sys.argv[2]
    mapping = parse_ass_file(ass_path)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"Expression mapping written to {out_path}")


if __name__ == "__main__":
    main()

# Example usage of build_gemini_prompt:
# lines = parse_ass_file('input.ass')
# prompt = build_gemini_prompt([{k: l[k] for k in ('start','end','text')} for l in lines])
# print(prompt)
# (Send prompt to Gemini API and parse the result)
