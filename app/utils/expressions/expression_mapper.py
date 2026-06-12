import re
import json
import sys
import logging
from pathlib import Path
import google.generativeai as genai

NARRATOR_CHARACTER = "narrator"
logger = logging.getLogger(__name__)


def ass_time_to_seconds(ass_time):
    # Format: H:MM:SS.cc
    h, m, s = ass_time.split(':')
    sec, cs = s.split('.')
    return int(h)*3600 + int(m)*60 + int(sec) + float('0.'+cs)


def _clean_ass_text(text: str) -> str:
    text = re.sub(r'\{.*?\}', '', text or '')
    text = re.sub(r'\\[Nn]', ' ', text)
    text = re.sub(r'\\[^ ]*', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def parse_ass_file(ass_path):
    lines = Path(ass_path).read_text(encoding='utf-8').splitlines()
    storyboard_lines = [
        l for l in lines
        if l.startswith('Comment:') and len(l.split(',', 9)) >= 10
        and l.split(',', 9)[3].strip().lower() == "storyboard"
    ]
    dialogue_lines = storyboard_lines or [l for l in lines if l.startswith('Dialogue:')]
    result = []
    for line in dialogue_lines:
        parts = line.split(',', 9)
        if len(parts) < 10:
            continue
        start = parts[1].strip()
        end = parts[2].strip()
        text = _clean_ass_text(parts[9])
        result.append({
            'start': start,
            'end': end,
            'text': text,
            'expression': _infer_expression_label(text, len(result)) if storyboard_lines else 'neutral',
            'character': NARRATOR_CHARACTER,
        })
    return enrich_expression_mapping(result, result) if storyboard_lines else result


def _time_distance(a: str, b: str) -> float:
    return abs(ass_time_to_seconds(a) - ass_time_to_seconds(b))


def normalize_expression_mapping(expressions, sub_lines):
    """
    Snap model expression output back to the original subtitle timings.

    Gemini sometimes preserves the text/expression but mutates one timestamp,
    creating huge overlapping expression holds. The subtitle file is the timing
    authority, so each output item is aligned by index/text to the ASS line.
    """
    if not isinstance(expressions, list):
        expressions = []

    normalized = []
    used_indexes = set()
    
    for i, sub_line in enumerate(sub_lines):
        item = {
            "start": sub_line["start"],
            "end": sub_line["end"],
            "text": sub_line.get("text", ""),
            "expression": "neutral",
            "character": NARRATOR_CHARACTER,
        }
        
        # Try to find a matching expression from Gemini's output
        sub_text = re.sub(r"\s+", " ", item["text"]).lower()
        match_expr = None
        
        # First, try exact text match
        for j, expr in enumerate(expressions):
            if not isinstance(expr, dict) or j in used_indexes:
                continue
            expr_text = re.sub(r"\s+", " ", (expr.get("text") or "").strip()).lower()
            if expr_text and expr_text == sub_text:
                match_expr = expr
                used_indexes.add(j)
                break
                
        # If no exact text match, try positional fallback (index match)
        if match_expr is None and i < len(expressions):
            if i not in used_indexes and isinstance(expressions[i], dict):
                expr = expressions[i]
                if _time_distance(expr.get("start", item["start"]), item["start"]) <= 2.0:
                    match_expr = expr
                    used_indexes.add(i)

        if match_expr:
            item["expression"] = (match_expr.get("expression") or "neutral").strip().lower()
            item["character"] = (match_expr.get("character") or NARRATOR_CHARACTER).strip().lower()
            
        normalized.append(item)

    return normalized


def _infer_expression_label(text: str, index: int = 0) -> str:
    lowered = (text or "").lower()
    if re.search(r"\b(shocking|terrifying|danger|monster|consume|sacrifice|warning|evil)\b", lowered):
        return "intense"
    if re.search(r"\b(question|why|how|what if|twist|mystery|reveal)\b", lowered) or "?" in text:
        return "surprised"
    if re.search(r"\b(fear|weak|terrified|tragic|alone|sad|cry|vulnerable)\b", lowered):
        return "worried"
    if re.search(r"\b(proves|means|truth|because|therefore|clearly|exactly)\b", lowered):
        return "confident"
    if re.search(r"\b(hook|payoff|dream|freedom|ultimate|final)\b", lowered):
        return "excited" if index < 3 else "serious"
    return "serious"


def _non_repeating_label(label: str, previous_label: str) -> str:
    """Avoid same-expression runs in generated cue plans."""
    label = (label or "serious").lower()
    previous_label = (previous_label or "").lower()
    if label != previous_label:
        return label
    alternates = {
        "serious": "confident",
        "confident": "serious",
        "intense": "surprised",
        "surprised": "intense",
        "worried": "serious",
        "sad": "worried",
        "excited": "confident",
        "angry": "intense",
        "smirking": "confident",
    }
    return alternates.get(label, "serious")


def enrich_expression_mapping(expressions, sub_lines):
    """Add sparse deterministic cues when Gemini is too conservative."""
    normalized = expressions or []
    if not sub_lines:
        return normalized

    try:
        total_duration = ass_time_to_seconds(sub_lines[-1]["end"]) - ass_time_to_seconds(sub_lines[0]["start"])
    except Exception:
        total_duration = 0.0
    if total_duration <= 0:
        return normalized

    expressive = [
        item for item in normalized
        if (item.get("expression") or "neutral").lower() not in {"neutral"}
    ]
    is_short = total_duration < 120
    target_cues = (
        max(3, min(9, int(total_duration / 7) + 1))
        if is_short
        else max(8, min(28, int(total_duration / 22)))
    )
    latest_expressive_start = max(
        (ass_time_to_seconds(item.get("start", "0:00:00.00")) for item in expressive),
        default=-1.0,
    )
    coverage_is_good = latest_expressive_start >= (total_duration * 0.72)
    if len(expressive) >= target_cues and (coverage_is_good or not is_short):
        return normalized

    existing_starts = {
        round(ass_time_to_seconds(item.get("start", "0:00:00.00")), 1)
        for item in expressive
    }
    added = []
    last_added = -999.0
    previous_label = (expressive[-1].get("expression") or "").lower() if expressive else ""
    min_gap = 5.5 if is_short else 14.0
    min_words = 3 if is_short else 5
    for index, line in enumerate(sub_lines):
        try:
            start = ass_time_to_seconds(line["start"])
        except Exception:
            continue
        if start - last_added < min_gap:
            continue
        if round(start, 1) in existing_starts:
            continue
        text = line.get("text", "")
        if len(text.split()) < min_words:
            continue
        label = _non_repeating_label(_infer_expression_label(text, index), previous_label)
        added.append({
            "start": line["start"],
            "end": line["end"],
            "text": text,
            "expression": label,
            "character": NARRATOR_CHARACTER,
            "source": "heuristic_enrichment",
        })
        last_added = start
        previous_label = label
        if len(expressive) + len(added) >= target_cues:
            break

    if added:
        logger.info(
            "Added %s heuristic expression cues for %s density",
            len(added),
            "short-form" if is_short else "long-form",
        )
    combined = [*normalized, *added]
    combined.sort(key=lambda item: ass_time_to_seconds(item.get("start", "0:00:00.00")))
    return combined


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
    - "expression": the facial expression label best suited for this line\n
    - "character": the speaking or focal character when obvious (e.g. "luffy", "zoro", "sanji"); otherwise "luffy" for narrator-style videos\n\n
    Use only these labels: "neutral", "happy", "angry", "surprised", "sad", "smirking", "confident", "serious", "worried", "intense", "excited", "embarrassed".\n\n
    Use tone, punctuation, emphasis, and keywords to infer emotion. Humor, sarcasm, or rhetorical questions should also influence the expression choice. You may also infer if the narrator is hyped or calm, authoritative or playful.\n
    For long-form narration, do not mark everything neutral. Use "serious" for thesis/evidence beats, "confident" for clear claims, "intense" for danger/reveal beats, "worried" or "sad" for tragic psychology, "surprised" for questions/twists, and "excited" for hook/payoff. Aim for a meaningful expression cue roughly every 15-30 seconds while still skipping filler lines.\n\n
    Example Input:\n[\n  { "start": "0:00:01.00", "end": "0:00:04.00", "text": "Okay, here we go!" },\n  
    { "start": "0:00:04.00", "end": "0:00:12.00", "text": "One Piece power levels! Let’s BREAK ‘EM DOWN!" }\n]\n\n
    Expected Output Format:\n[\n  {\n    "start": "0:00:01.00",\n    "end": "0:00:04.00",\n    "text": "Okay, here we go!",\n    "expression": "excited"\n  },\n  {\n    "start": "0:00:04.00",\n    "end": "0:00:12.00",\n    "text": "One Piece power levels! Let’s BREAK ‘EM DOWN!",\n    "expression": "intense"\n  }\n]\n\nNow here are the actual subtitle lines to analyze:\n'''
    prompt += json.dumps(sub_lines, indent=2, ensure_ascii=False)
    return prompt


def setup_gemini(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash-lite")


def generate_expression_mapping_with_gemini(model, sub_lines):
    """
    Given a Gemini model and subtitle lines (list of dicts with start, end, text),
    build a Gemini prompt and return the parsed JSON result.
    """
    prompt = build_gemini_prompt(
        [{k: l[k] for k in ('start', 'end', 'text')} for l in sub_lines])
    
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.35,
            max_output_tokens=8192,
            response_mime_type="application/json"
        )
    )
    # Try to extract JSON from the response
    try:
        # Gemini may return markdown code block, so strip if needed
        text = response.text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.endswith('```'):
            text = text[:-3]
        return enrich_expression_mapping(
            normalize_expression_mapping(json.loads(text), sub_lines),
            sub_lines,
        )
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
