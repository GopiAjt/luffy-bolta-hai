import os
import json
import traceback
import numpy as np
import cv2


def parse_time(t):
    """Converts '0:00:06.28' to seconds."""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def sanitize_size(size):
    """Ensure size is a tuple of two ints (width, height)."""
    try:
        w, h = size[:2]
        return int(float(w)), int(float(h))
    except Exception as e:
        print(f"ERROR: Could not sanitize size: {size}, error: {e}")
        raise


def apply_panoramic_pan(temp_frame_dir, frame_counter, img_path, duration, resolution, idx, fps=30):
    """
    Fits the image height to the video height, then pans horizontally
    so that over the course of `duration` seconds, the window moves
    from one edge of the image to the other.

    - Even idx: pan left→right
    - Odd  idx: pan right→left
    """
    res_w, res_h = resolution
    num_frames = int(duration * fps)
    

    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image at {img_path}. Skipping.")
        return frame_counter # Return current frame_counter

    # 1. Scale image so its height matches video height
    img_h, img_w = img.shape[:2]
    scale = res_h / img_h
    scaled_w = int(img_w * scale)
    scaled = cv2.resize(img, (scaled_w, res_h))

    # If the scaled width is smaller than the video width, pad with black
    if scaled_w < res_w:
        pad = res_w - scaled_w
        left = pad // 2
        right = pad - left
        scaled = cv2.copyMakeBorder(
            scaled, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        scaled_w = res_w

    # 2. For each frame, compute horizontal crop window
    frames_actually_written = 0
    for i in range(num_frames):
        # fraction from 0.0→1.0
        t = i / (num_frames - 1) if num_frames > 1 else 0
        if idx % 2 == 0:
            # left → right
            x = int(t * (scaled_w - res_w))
        else:
            # right → left
            x = int((1 - t) * (scaled_w - res_w))

        frame = scaled[:, x: x + res_w]
        cv2.imwrite(os.path.join(temp_frame_dir, f"frame_{frame_counter:05d}.jpg"), frame)
        frame_counter += 1
        frames_actually_written += 1
    
    return frame_counter # Return updated frame_counter


def main(json_path, image_dir, output_path, total_duration=None, resolution=(576, 1024), fps=30):
    """Generates the final slideshow video using OpenCV."""
    resolution = sanitize_size(resolution)
    res_w, res_h = resolution

    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # writer = cv2.VideoWriter(output_path, fourcc, fps, (res_w, res_h), isColor=True)
    frames_written = 0
    total_output_frames = int(total_duration * fps)
    print(f"Expected total output frames: {total_output_frames}")
    
    temp_frame_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
    os.makedirs(temp_frame_dir, exist_ok=True)

    print(f"Initial total_duration: {total_duration:.2f}s")

    with open(json_path, "r", encoding="utf-8") as f:
        slides = json.load(f)

    last_slide_end_time = 0
    for idx, slide in enumerate(slides):
        try:
            start = parse_time(slide["start_time"])
            end = parse_time(slide["end_time"])
            duration = end - start
            last_slide_end_time = max(last_slide_end_time, end)

            img_path = None
            for ext in ["jpg", "png", "jpeg"]:
                for style in ["9x16", "16x9"]:
                    candidate = os.path.join(
                        image_dir, f"slide_{idx+1}_{style}.{ext}")
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                if img_path:
                    break

            if not img_path:
                fallback_path = os.path.join(
                    image_dir, f"slide_{idx+1}_fallback.jpg")
                if os.path.exists(fallback_path):
                    img_path = fallback_path
                else:
                    continue

            frames_written = apply_panoramic_pan(temp_frame_dir, frames_written, img_path, duration,
                                resolution, idx, fps)

        except Exception as e:
            traceback.print_exc()

    current_video_duration = last_slide_end_time
    print(f"Initial current_video_duration (after first pass): {current_video_duration:.2f}s")
    slide_idx_counter = 0  # To keep track of overall slides processed for pan direction

    while frames_written < total_output_frames:
        print(f"Loop start: frames_written={frames_written}, total_output_frames={total_output_frames}")
        # Cycle through the original slides
        original_slide_index = slide_idx_counter % len(slides)
        slide = slides[original_slide_index]

        try:
            start = parse_time(slide["start_time"])
            end = parse_time(slide["end_time"])
            original_slide_duration = end - start

            remaining_frames_to_fill = total_output_frames - frames_written
            # Calculate duration for this segment based on remaining frames
            duration_for_this_segment = min(original_slide_duration, remaining_frames_to_fill / fps)

            if duration_for_this_segment <= 0:
                print("Duration for this segment is zero or negative, breaking loop.")
                break  # No more duration to fill

            img_path = None
            # The image path logic needs to use the original slide index for filename
            for ext in ["jpg", "png", "jpeg"]:
                for style in ["9x16", "16x9"]:
                    candidate = os.path.join(
                        image_dir, f"slide_{original_slide_index+1}_{style}.{ext}")
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                if img_path:
                    break

            if not img_path:
                fallback_path = os.path.join(
                    image_dir, f"slide_{original_slide_index+1}_fallback.jpg")
                if os.path.exists(fallback_path):
                    img_path = fallback_path
                else:
                    # If image is missing, we still need to advance frames to avoid infinite loop
                    frames_written += int(duration_for_this_segment * fps) # Approximate advance
                    slide_idx_counter += 1
                    print(f"Image not found for original slide {original_slide_index+1}, skipping this segment. Advanced frames_written to {frames_written}")
                    continue

            print(f"  Processing segment: original_slide_duration={original_slide_duration:.2f}s, remaining_frames_to_fill={remaining_frames_to_fill}, duration_for_this_segment={duration_for_this_segment:.2f}s")
            frames_generated_this_segment = int(duration_for_this_segment * fps)
            frames_written = apply_panoramic_pan(
                temp_frame_dir, frames_written, img_path, duration_for_this_segment, resolution, slide_idx_counter, fps)
            slide_idx_counter += 1
            print(f"  After segment: frames_written updated to {frames_written}")

        except Exception as e:
            traceback.print_exc()
            # To prevent infinite loops on error, advance frames by a small amount
            frames_written += fps  # Advance by 1 second worth of frames
            slide_idx_counter += 1

    print(f"Total frames written: {frames_written}")

    total_output_frames = int(total_duration * fps)
    print(f"Expected total output frames: {total_output_frames}")

    # Use ffmpeg to stitch frames into a video
    ffmpeg_command = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-f", "image2",  # Input format
        "-framerate", str(fps),  # Input frame rate
        "-i", os.path.join(temp_frame_dir, "frame_%05d.jpg"),  # Input image sequence
        "-vframes", str(total_output_frames), # Process only the required number of frames
        "-c:v", "libx264",  # Video codec
        "-r", str(fps),  # Output frame rate
        "-vf", f"format=yuv420p",  # Video filter for pixel format
        "-pix_fmt", "yuv420p",  # Pixel format for wider compatibility
        "-crf", "23",  # Constant Rate Factor (quality)
        output_path
    ]

    print(f"Running FFmpeg command: {' '.join(ffmpeg_command)}")
    try:
        import subprocess
        subprocess.run(ffmpeg_command, check=True)
        print(f"Video successfully generated at {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg command failed: {e}")
        print(f"FFmpeg stderr: {e.stderr.decode()}")
        raise
    finally:
        # Clean up temporary frames
        import shutil
        shutil.rmtree(temp_frame_dir)
        print(f"Cleaned up temporary frame directory: {temp_frame_dir}")


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Generate a slideshow video from images and a JSON file using OpenCV.")
    parser.add_argument("--json", required=True,
                        help="Path to slides JSON file")
    parser.add_argument("--images", required=True,
                        help="Directory with images")
    parser.add_argument(
        "--duration", type=float, default=None, help="Total duration for the video. If provided, it will be padded."
    )
    parser.add_argument("--resolution", type=str, default="576x1024",
                        help="Video resolution as WxH (e.g., 576x1024)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second for the output video.")

    args = parser.parse_args()

    try:
        w, h = map(int, args.resolution.split('x'))
        resolution_arg = (w, h)
    except ValueError:
        print("Invalid resolution format. Use WxH (e.g., 576x1024).")
        exit(1)

    output_dir = "app/output"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"slideshow_{timestamp}.mp4")

    main(args.json, args.images, output_path, total_duration=args.duration,
         resolution=resolution_arg, fps=args.fps)
