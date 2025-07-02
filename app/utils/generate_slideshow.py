import os
import json
from moviepy import ImageClip, concatenate_videoclips, TextClip, CompositeVideoClip, vfx
import traceback


def parse_time(t):
    # Converts '0:00:06.28' to seconds
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def main(json_path, image_dir, output_path, crossfade=0.5, resolution=(576, 1024)):
    # Ensure resolution is a tuple of ints
    resolution = sanitize_size(resolution)
    with open(json_path, "r", encoding="utf-8") as f:
        slides = json.load(f)
    clips = []
    for idx, slide in enumerate(slides):
        try:
            print(
                f"DEBUG: Slide {idx+1} fields and types: {[ (k, type(v), v) for k,v in slide.items() ]}")
            start = parse_time(slide["start_time"])
            end = parse_time(slide["end_time"])
            # Ensure start and end are floats
            try:
                start = float(start)
                end = float(end)
            except Exception as e:
                print(
                    f"ERROR: Could not cast start/end to float: start={start} ({type(start)}), end={end} ({type(end)}), error: {e}")
                raise
            duration = end - start
            print(
                f"DEBUG: start={start} ({type(start)}), end={end} ({type(end)}), duration={duration} ({type(duration)})")
            img_path = os.path.join(image_dir, f"slide_{idx+1}_9x16.jpg")
            if not os.path.exists(img_path):
                # Try all common image extensions for 16x9 images
                img_path = None
                for ext in ["jpg", "png", "jpeg"]:
                    candidate = os.path.join(
                        image_dir, f"slide_{idx+1}_16x9.{ext}")
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
            use_ken_burns = False  # No zoom
            use_fade = True
            use_slide = True  # Always slide
            slide_side = 'left'  # Default to horizontal slide
            # If you want to alternate direction, you can use idx % 2
            # --- ZOOM TO 9:16 AND SLIDE ---
            from moviepy import vfx
            # 1. Load image
            img_clip = ImageClip(img_path, duration=duration)
            import numpy as np  # Needed for grayscale to RGB conversion
            # Initialize effect chain for this slide
            effect_chain = []
            # Ensure image is RGB (avoid broadcasting errors with bg_color)
            frame0 = img_clip.get_frame(0)
            if len(frame0.shape) == 2:  # grayscale
                def to_rgb(get_frame, t):
                    frame = get_frame(t)
                    return np.stack([frame]*3, axis=-1)
                img_clip = img_clip.transform(to_rgb)
            # Get original image size before resizing
            w, h = sanitize_size(img_clip.size)
            # 2. Resize and crop to enable sliding
            if slide_side == 'left':
                # Horizontal slide: match height, allow width > frame
                scale = resolution[1] / h
                new_w = int(w * scale)
                effect_chain.append(
                    f"Resize(scale={scale}) -> Crop(width={new_w}, height={resolution[1]})")
                img_clip = img_clip.with_effects([
                    vfx.Resize(scale),
                    vfx.Crop(width=new_w, height=resolution[1],
                             x_center=new_w // 2, y_center=resolution[1] // 2)
                ])
                w, h = sanitize_size(img_clip.size)
            else:
                # Vertical slide: match width, allow height > frame
                scale = resolution[0] / w
                new_h = int(h * scale)
                effect_chain.append(
                    f"Resize(scale={scale}) -> Crop(width={resolution[0]}, height={new_h})")
                img_clip = img_clip.with_effects([
                    vfx.Resize(scale),
                    vfx.Crop(width=resolution[0], height=new_h,
                             x_center=resolution[0] // 2, y_center=new_h // 2)
                ])
                w, h = sanitize_size(img_clip.size)
            # Ensure again after any possible mutation
            resolution = sanitize_size(resolution)
            # Debug: print type and value of w, h after crop
            print(
                f"DEBUG: img_clip.size after crop: {img_clip.size} (types: {type(w)}, {type(h)})")
            # Extra debug: print before scroll calculation
            if use_slide:
                print(
                    f"DEBUG before scroll: w={w} ({type(w)}), h={h} ({type(h)}), resolution[0]={resolution[0]} ({type(resolution[0])}), resolution[1]={resolution[1]} ({type(resolution[1])}), duration={duration} ({type(duration)})")
            # 3. Calculate scroll distance and speed for horizontal sliding
            effects = []
            if use_slide:
                try:
                    # After resizing/cropping, get new width/height
                    if slide_side == 'left':
                        scroll_distance = w - resolution[0]
                        if idx % 2 == 0:
                            # Left to right
                            x_speed = scroll_distance / duration
                            direction = 'left-to-right'
                        else:
                            # Right to left: start at right edge
                            x_speed = -scroll_distance / duration
                            direction = 'right-to-left'
                        y_speed = 0
                        print(
                            f"DEBUG SLIDE: HORIZONTAL scroll_distance={scroll_distance}, x_speed={x_speed}, direction={direction}, duration={duration}")
                        # Use vfx.Scroll for all movement (no start_x)
                        effects.append(vfx.Scroll(
                            x_speed=x_speed, y_speed=y_speed, apply_to=['mask', 'video']))
                    else:
                        scroll_distance = h - resolution[1]
                        x_speed = 0
                        y_speed = scroll_distance / duration
                        print(
                            f"DEBUG SLIDE: VERTICAL scroll_distance={scroll_distance}, y_speed={y_speed}, duration={duration}")
                        effects.append(vfx.Scroll(
                            x_speed=x_speed, y_speed=y_speed, apply_to=['mask', 'video']))
                except Exception as e:
                    print(
                        f"ERROR in scroll/slide calculation: w={w} ({type(w)}), h={h} ({type(h)}), resolution={resolution} ({[type(x) for x in resolution]}), duration={duration} ({type(duration)}), error: {e}")
                    raise
            if use_fade:
                effects.append(vfx.FadeIn(0.5))
                effects.append(vfx.FadeOut(0.5))
            if effects:
                img_clip = img_clip.with_effects(effects)
            # Center the image in the frame with black background
            # Ensure final clip is exactly the target resolution for MoviePy concat
            # Calculate scroll distances
            scroll_distance_x = w - resolution[0]
            scroll_distance_y = h - resolution[1]
            horizontal = scroll_distance_x > 0
            # Set up position lambda for scrolling
            if horizontal:
                if idx % 2 == 0:
                    # Left to right: start at -scroll_distance, move to 0
                    def pos(t): return (-scroll_distance_x +
                                        scroll_distance_x * (t / duration), 0)
                else:
                    # Right to left: start at +scroll_distance, move to 0
                    def pos(t): return (scroll_distance_x -
                                        scroll_distance_x * (t / duration), 0)
            else:
                # Vertical scroll (if ever needed)
                if idx % 2 == 0:
                    def pos(t): return (0, -scroll_distance_y +
                                        scroll_distance_y * (t / duration))
                else:
                    def pos(t): return (0, scroll_distance_y -
                                        scroll_distance_y * (t / duration))

            # Build final composite with scrolling and fades
            final_clip = CompositeVideoClip(
                [img_clip.with_position(pos)],
                size=resolution, bg_color=(0, 0, 0)
            ).with_effects([
                vfx.FadeIn(0.5),
                vfx.FadeOut(0.5)
            ])
            clips.append(final_clip)
            # Print effect chain for this slide
            print(f"DEBUG: Effect chain for slide {idx+1}: {effect_chain}")
            print(f"DEBUG: img_clip.size after all effects: {img_clip.size}")
        except Exception as e:
            print(f"FATAL ERROR in slide {idx+1}: {e}")
            traceback.print_exc()
            raise
    if not clips:
        print("No clips to concatenate.")
        return
    video = concatenate_videoclips(clips, method="compose", padding=0)
    video.write_videofile(output_path, fps=30, codec="libx264", audio=False)


def sanitize_size(size):
    """Ensure size is a tuple of two ints (width, height), even if input has more values."""
    try:
        # Only take the first two elements if more are present
        w, h = size[:2]
        w = int(float(w))
        h = int(float(h))
        return w, h
    except Exception as e:
        print(f"ERROR: Could not sanitize size: {size}, error: {e}")
        raise


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
