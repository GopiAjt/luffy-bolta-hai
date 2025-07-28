import os
import json
import traceback
import numpy as np
import cv2
import logging
import subprocess # Added this import
from app.config import VIDEO_RESOLUTION

logger = logging.getLogger(__name__)

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


def apply_panoramic_pan(temp_frame_dir, frame_counter, img_path, duration, resolution, idx, fps=30, blur_amount=5):
    """
    Fits the image height to the video height, then pans horizontally
    so that over the course of `duration` seconds, the window moves
    from one edge of the image to the other.

    - Even idx: pan left→right
    - Odd  idx: pan right→left
    
    Args:
        temp_frame_dir: Directory to save temporary frames
        frame_counter: Starting frame number
        img_path: Path to the source image
        duration: Duration to display the slide (in seconds)
        resolution: Tuple of (width, height) for output frames
        idx: Slide index (used to determine pan direction)
        fps: Frames per second
        blur_amount: Radius of the Gaussian blur (0 = no blur, higher = more blur)
    """
    res_w, res_h = resolution
    num_frames = int(duration * fps)
    
    # Log the panning direction and blur status
    direction = "left→right" if idx % 2 == 0 else "right→left"
    blur_status = f"with blur (kernel size: {blur_amount*2 + 1})" if blur_amount > 0 else "without blur"
    logger.info(f"Processing slide {img_path} with index {idx}, panning {direction} {blur_status}")

    img = cv2.imread(img_path)
    if img is None:
        logger.warning(f"Could not read image at {img_path}. Skipping.")
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

    # Apply blur to the entire scaled image if specified
    if blur_amount > 0:
        # Ensure kernel size is odd and reasonable
        kernel_size = max(1, blur_amount * 2 + 1)  # Ensure at least 1x1
        kernel_size = min(kernel_size, 99)  # Keep kernel size reasonable
        if kernel_size % 2 == 0:  # Ensure odd kernel size
            kernel_size += 1
        
        # Convert to float32 and normalize to [0,1] for better blur
        scaled_float = scaled.astype(np.float32) / 255.0
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(scaled_float, (kernel_size, kernel_size), 0)
        
        # Convert back to uint8
        scaled = (blurred * 255).astype(np.uint8)
    
    # 2. For each frame, compute horizontal crop window
    frames_actually_written = 0
    
    # Calculate the maximum possible x position (0 if image is narrower than video width)
    max_x = max(0, scaled_w - res_w)
    
    for i in range(num_frames):
        # fraction from 0.0→1.0
        t = i / (num_frames - 1) if num_frames > 1 else 0.0
        
        # Only apply panning if the image is wider than the video
        if max_x > 0:
            if idx % 2 == 0:
                # left → right
                x = int(t * max_x)
            else:
                # right → left
                x = int((1 - t) * max_x)
        else:
            # Center the image if it's not wide enough to pan
            x = 0
            
        logger.debug(f"Frame {i}: x={x}, t={t:.2f}, max_x={max_x}, direction={'left→right' if idx % 2 == 0 else 'right→left'}")

        frame = scaled[:, x: x + res_w]
            
        cv2.imwrite(os.path.join(temp_frame_dir, f"frame_{frame_counter:05d}.jpg"), frame)
        frame_counter += 1
        frames_actually_written += 1
    
    return frame_counter # Return updated frame_counter

def get_video_duration(filepath):
    """Get the duration of a video file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        filepath
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"Error getting video duration for {filepath}: {e}")
        return 0

def extract_gif_frames(gif_path, temp_frame_dir, frame_counter_start, duration, resolution, fps=30):
    """
    Extracts frames from a GIF and saves them as JPGs in the temp_frame_dir.
    Handles looping/trimming to match the desired duration.
    Returns the updated frame_counter.
    """
    res_w, res_h = resolution
    target_num_frames = int(duration * fps)

    # Get GIF duration using ffprobe
    gif_duration = get_video_duration(gif_path)
    gif_num_frames = int(gif_duration * fps)

    if gif_num_frames == 0:
        logger.warning(f"Could not determine duration or extract frames from GIF: {gif_path}. Skipping.")
        return frame_counter_start

    # Use ffmpeg to extract frames
    # We'll extract more frames than needed and then trim/loop in Python
    # This is to ensure we have enough frames if the GIF is short
    output_pattern = os.path.join(temp_frame_dir, "gif_temp_frame_%05d.jpg")
    ffmpeg_extract_cmd = [
        "ffmpeg",
        "-i", gif_path,
        "-vf", f"scale={res_w}:{res_h}:force_original_aspect_ratio=decrease,pad={res_w}:{res_h}:(ow-iw)/2:(oh-ih)/2,setsar=1",
        "-vsync", "0", # Ensure all frames are extracted
        output_pattern
    ]
    try:
        subprocess.run(ffmpeg_extract_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed to extract frames from GIF {gif_path}: {e.stderr}")
        return frame_counter_start

    extracted_frames = sorted([f for f in os.listdir(temp_frame_dir) if f.startswith("gif_temp_frame_") and f.endswith(".jpg")])
    if not extracted_frames:
        logger.warning(f"No frames extracted from GIF: {gif_path}. Skipping.")
        return frame_counter_start

    current_frame_idx = 0
    frames_written = 0
    for i in range(target_num_frames):
        if current_frame_idx >= len(extracted_frames):
            # Loop the GIF if it's shorter than the desired duration
            current_frame_idx = 0

        src_frame_path = os.path.join(temp_frame_dir, extracted_frames[current_frame_idx])
        dest_frame_path = os.path.join(temp_frame_dir, f"frame_{frame_counter_start + frames_written:05d}.jpg")
        
        # Copy the frame to the main sequence
        os.rename(src_frame_path, dest_frame_path) # Use rename for efficiency if within same filesystem

        frames_written += 1
        current_frame_idx += 1
    
    # Clean up any remaining gif_temp_frame_ files
    for f in os.listdir(temp_frame_dir):
        if f.startswith("gif_temp_frame_"):
            os.remove(os.path.join(temp_frame_dir, f))

    return frame_counter_start + frames_written


def main(json_path, image_dir, output_path, total_duration=None, resolution=VIDEO_RESOLUTION, fps=30, blur_amount=5):
    """
    Generates the final slideshow video using OpenCV.
    
    Args:
        json_path: Path to the JSON file containing slide timings
        image_dir: Directory containing slide images
        output_path: Path where the output video will be saved
        total_duration: Total duration of the video in seconds
        resolution: Video resolution as (width, height)
        fps: Frames per second for the output video
        blur_amount: Amount of blur to apply (0 = no blur, higher = more blur)
    """
    resolution = sanitize_size(resolution)
    res_w, res_h = resolution
    
    if blur_amount > 0:
        logger.info(f"Applying blur effect with amount: {blur_amount}")
        # Ensure blur amount is a positive integer
        blur_amount = max(0, int(blur_amount))

    # Calculate total frames needed
    if total_duration is None:
        # If no total duration provided, calculate from slides
        with open(json_path, "r", encoding="utf-8") as f:
            slides = json.load(f)
        total_duration = sum(slide['end'] - slide['start'] for slide in slides)
    total_output_frames = int(total_duration * fps)
    logger.info(f"Total video duration: {total_duration:.2f}s, {total_output_frames} frames")
    logger.info(f"Target total frames: {total_output_frames} for a duration of {total_duration:.2f}s at {fps} fps.")

    temp_frame_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
    os.makedirs(temp_frame_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        slides = json.load(f)

    last_slide_end_time = 0
    # Track the actual number of slides processed for panning direction
    slides_processed = 0
    
    logger.info(f"Starting to process {len(slides)} slides...")
    
    # Initialize frames_written counter
    frames_written = 0
    
    for idx, slide in enumerate(slides):
        try:
            start = parse_time(slide["start_time"])
            end = parse_time(slide["end_time"])
            duration = end - start
            last_slide_end_time = max(last_slide_end_time, end)
            
            logger.info(f"Processing slide {idx+1} (slides_processed={slides_processed}): start={start:.2f}s, end={end:.2f}s, duration={duration:.2f}s")

            img_path = None
            for ext in ["jpg", "png", "jpeg", "gif"]:
                for style in ["9x16", "16x9"]:
                    candidate = os.path.join(image_dir, f"slide_{idx+1}_{style}.{ext}")
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                if img_path:
                    break

            if not img_path:
                # Check for fallback images with both .jpg and .png extensions
                for ext in ["jpg", "png"]:
                    fallback_path = os.path.join(image_dir, f"slide_{idx+1}_fallback.{ext}")
                    if os.path.exists(fallback_path):
                        img_path = fallback_path
                        logger.info(f"Using fallback image for slide {idx+1} with extension .{ext}.")
                        break
                if not img_path:
                    logger.warning(f"Image for slide {idx+1} not found, and no fallback available. Skipping this slide.")
                    continue

            if img_path.lower().endswith(".gif"):
                logger.info(f"Processing GIF: {img_path}")
                frames_written = extract_gif_frames(img_path, temp_frame_dir, frames_written, duration, resolution, fps)
                slides_processed += 1
            else:
                # Log the slide number and panning direction before processing
                direction = "left→right" if slides_processed % 2 == 0 else "right→left"
                logger.info(f"Processing slide {idx+1} (slides_processed={slides_processed}): panning {direction}")
                
                # Use slides_processed for panning direction and apply blur
                frames_written = apply_panoramic_pan(
                    temp_frame_dir=temp_frame_dir,
                    frame_counter=frames_written,
                    img_path=img_path,
                    duration=duration,
                    resolution=resolution,
                    idx=slides_processed,
                    fps=fps,
                    blur_amount=30  # Increased blur amount for more pronounced effect
                )
                slides_processed += 1

        except Exception as e:
            logger.error(f"Error processing slide {idx+1}: {e}")
            traceback.print_exc()

    logger.info(f"Frames written after initial pass: {frames_written}")

    # Padding logic to ensure the video reaches the total_duration
    if frames_written < total_output_frames:
        padding_needed = total_output_frames - frames_written
        logger.info(f"Padding needed for {padding_needed} frames.")

        # Create a blank frame to use for padding
        blank_frame = np.zeros((res_h, res_w, 3), dtype=np.uint8)
        blank_frame_path = os.path.join(temp_frame_dir, "blank_frame.jpg")
        cv2.imwrite(blank_frame_path, blank_frame)

        for i in range(padding_needed):
            frame_path = os.path.join(temp_frame_dir, f"frame_{frames_written + i:05d}.jpg")
            # For simplicity, we just copy the blank frame. A more advanced implementation could repeat the last slide.
            cv2.imwrite(frame_path, blank_frame)
        frames_written += padding_needed

    logger.info(f"Total frames written after padding: {frames_written}")

    # Use ffmpeg to stitch frames into a video
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", os.path.join(temp_frame_dir, "frame_%05d.jpg"),
        "-vframes", str(frames_written),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        output_path
    ]

    logger.info(f"Running FFmpeg command: {' '.join(ffmpeg_command)}")
    try:
        import subprocess
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        logger.info(f"Video successfully generated at {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg command failed: {e}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        raise
    finally:
        import shutil
        if os.path.exists(temp_frame_dir):
            shutil.rmtree(temp_frame_dir)
            logger.info(f"Cleaned up temporary frame directory: {temp_frame_dir}")


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
    parser.add_argument("--resolution", type=str, default=f"{VIDEO_RESOLUTION[0]}x{VIDEO_RESOLUTION[1]}",
                        help=f"Video resolution as WxH (e.g., {VIDEO_RESOLUTION[0]}x{VIDEO_RESOLUTION[1]})")
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
