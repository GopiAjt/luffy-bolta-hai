import os
import json
from moviepy import ImageClip, concatenate_videoclips, TextClip, CompositeVideoClip, vfx


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
            # Try all common image extensions
            img_path = None
            for ext in ["jpg", "png", "jpeg"]:
                candidate = os.path.join(
                    image_dir, f"slide_{idx+1}_9x16.{ext}")
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if not img_path:
                print(f"Image not found for slide {idx+1}, skipping.")
                continue
        # Choose effect based on summary keywords
        summary = slide.get("summary", "").lower()
        effects = []
        # Default: Ken Burns + Fade In/Out
        use_ken_burns = True
        use_fade = True
        use_slide = False
        slide_side = 'left'
        # Keyword-based effect selection
        if any(word in summary for word in ["speed", "attack", "boost", "energy", "fast", "whip"]):
            # Simulate whip pan/slide for high-energy
            use_ken_burns = False
            use_slide = True
            slide_side = 'left'
        elif any(word in summary for word in ["reveal", "change", "scene", "variation", "slide"]):
            use_ken_burns = False
            use_slide = True
            slide_side = 'bottom'
        elif any(word in summary for word in ["pause", "dramatic", "intro", "outro", "end", "start"]):
            use_ken_burns = False
            use_fade = True
        # Build the image clip
        img_clip = ImageClip(img_path, duration=duration)
        img_clip = img_clip.with_effects([vfx.Resize(resolution)])
        if use_ken_burns:
            img_clip = img_clip.resized(lambda t: 1 + 0.05 * t / duration)
        if use_slide:
            img_clip = img_clip.with_effects(
                [vfx.SlideIn(0.7, side=slide_side)])
        if use_fade:
            img_clip = img_clip.with_effects([
                vfx.FadeIn(0.5),
                vfx.FadeOut(0.5)
            ])
        final_clip = CompositeVideoClip([img_clip], size=resolution)
        clips.append(final_clip)
    if not clips:
        print("No clips to concatenate.")
        return
    video = concatenate_videoclips(clips, method="compose", padding=-crossfade)
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
