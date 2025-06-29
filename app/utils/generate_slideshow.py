import os
import json
from moviepy.editor import ImageClip, concatenate_videoclips, TextClip, CompositeVideoClip
from moviepy.video.fx.all import fadein, fadeout


def parse_time(t):
    # Converts '0:00:06.28' to seconds
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def main(json_path, image_dir, output_path, crossfade=0.5, resolution=(576, 1024)):
    with open(json_path, "r", encoding="utf-8") as f:
        slides = json.load(f)
    clips = []
    for idx, slide in enumerate(slides):
        start = parse_time(slide["start_time"])
        end = parse_time(slide["end_time"])
        duration = end - start
        img_path = os.path.join(image_dir, f"slide_{idx+1}_9x16.jpg")
        if not os.path.exists(img_path):
            # Try png fallback
            img_path = os.path.join(image_dir, f"slide_{idx+1}_9x16.png")
            if not os.path.exists(img_path):
                print(f"Image not found for slide {idx+1}, skipping.")
                continue
        img_clip = ImageClip(img_path).set_duration(
            duration).resize(newsize=resolution)
        # Ken Burns effect: slow zoom in
        img_clip = img_clip.fx(lambda c: c.resize(
            lambda t: 1 + 0.05 * t / duration))
        # Fade in/out
        img_clip = fadein(img_clip, 0.5).fx(fadeout, 0.5)
        # Optional: overlay summary as subtitle
        txt = TextClip(slide["summary"], fontsize=40, color='white', font='Arial-Bold',
                       size=resolution, method='caption', align='center', stroke_color='black', stroke_width=2)
        txt = txt.set_duration(duration).set_position(
            ('center', 'bottom')).margin(bottom=60, opacity=0)
        final_clip = CompositeVideoClip([img_clip, txt], size=resolution)
        clips.append(final_clip)
    if not clips:
        print("No clips to concatenate.")
        return
    video = concatenate_videoclips(clips, method="compose", padding=-
                                   crossfade, transition=lambda c1, c2: c1.crossfadeout(crossfade))
    video.write_videofile(output_path, fps=30, codec="libx264", audio=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True,
                        help="Path to slides JSON file")
    parser.add_argument("--images", required=True,
                        help="Directory with 9x16 images")
    parser.add_argument("--output", required=True,
                        help="Output video file path")
    args = parser.parse_args()
    main(args.json, args.images, args.output)
