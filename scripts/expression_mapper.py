import re
import json
import sys
from pathlib import Path

# Simple keyword-based expression mapping (expand as needed)
EXPRESSION_KEYWORDS = [
    (['smile', 'neat', 'yeah', 'right?', 'grins', 'sound off'], 'happy'),
    (['despair', 'terrible', 'forced', 'worried'], 'worried'),
    (['kicker', 'clue', 'hint', 'hidden'], 'smirking'),
    (['remember', 'did you know', 'see,'], 'curious'),
    (['not all', 'but'], 'neutral'),
    (['delicious', 'starfruit', 'jackfruit', 'durian'], 'excited'),
    (['terrible', 'despair'], 'sad'),
    (['connection', 'origins'], 'serious'),
    (['think'], 'thinking'),
    (['!'], 'excited'),
    (['?'], 'surprised'),
]


def infer_expression(text):
    t = text.lower()
    for keywords, expr in EXPRESSION_KEYWORDS:
        for kw in keywords:
            if kw in t:
                return expr
    # Default fallback
    return 'neutral'


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
        expression = infer_expression(text)
        result.append({
            'start': start,
            'end': end,
            'text': text,
            'expression': expression
        })
    return result


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
