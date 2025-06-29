import os
import sys
import json
import subprocess
from pathlib import Path


def parse_time(t):
    # Accepts H:MM:SS.ss or S.ss
    if ':' in t:
        h, m, s = t.split(':')
        return float(h)*3600 + float(m)*60 + float(s)
    return float(t)


def build_overlay_filters(expressions, expr_img_dir, video_size):
    """
    Returns ffmpeg filter_complex overlay commands for all expressions.
    Each image is overlaid only during its phrase's time window.
    """
    filters = []
    inputs = []
    overlay_idx = 1  # 0 is the background
    last_output = '[bg]'  # initial background label
    for i, expr in enumerate(expressions):
        expr_label = expr['expression'].lower()
        img_path = os.path.join(expr_img_dir, f"{expr_label}.png")
        if not os.path.exists(img_path):
            print(
                f"Warning: image for expression '{expr_label}' not found: {img_path}")
            continue
        start = parse_time(expr['start'])
        end = parse_time(expr['end'])
        # Add as input
        inputs.append(f"-i {img_path}")
        # Overlay filter
        filters.append(
            f"[{last_output}][{overlay_idx}:v] overlay=x=W-w-40:y=40:enable='between(t,{start},{end})'[bg{i+1}]"
        )
        last_output = f"[bg{i+1}]"
        overlay_idx += 1
    return inputs, filters, last_output


def main():
    if len(sys.argv) < 6:
        print("Usage: python generate_video_with_expressions.py <audio> <subtitle.ass> <expressions.json> <output.mp4> <bg_color>")
        sys.exit(1)
    audio = sys.argv[1]
    subtitle = sys.argv[2]
    expressions_json = sys.argv[3]
    output = sys.argv[4]
    bg_color = sys.argv[5] if len(sys.argv) > 5 else 'green'
    expr_img_dir = os.path.join(os.path.dirname(
        __file__), '../app/static/expressions')
    video_size = '1080x1920'

    with open(expressions_json, 'r', encoding='utf-8') as f:
        expressions = json.load(f)

    # Build overlay filters
    inputs, filters, last_output = build_overlay_filters(
        expressions, expr_img_dir, video_size)

    # Compose ffmpeg command
    filter_complex = f"color=c={bg_color}:s={video_size}:d=60:r=30[bg];" + ';'.join(
        filters)
    # Add subtitles overlay
    filter_complex += f";{last_output}subtitles='{subtitle}'[vout]"

    # Build full command
    cmd = [
        'ffmpeg', '-y',
        '-f', 'lavfi', '-i', f'color=c={bg_color}:s={video_size}:d=60:r=30',
        *sum([['-i', os.path.join(expr_img_dir, f"{expr['expression'].lower()}.png")] for expr in expressions if os.path.exists(
            os.path.join(expr_img_dir, f"{expr['expression'].lower()}.png"))], []),
        '-i', audio,
        '-filter_complex', filter_complex,
        '-map', '[vout]', '-map', f'{len(inputs)+1}:a',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-pix_fmt', 'yuv420p',
        '-c:a', 'libmp3lame', '-b:a', '192k', '-shortest', output
    ]
    print('Running ffmpeg command:')
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Video with expressions generated at {output}")


if __name__ == "__main__":
    main()
