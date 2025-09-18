import os
import json
import traceback
import numpy as np
import cv2
from PIL import Image
import io
import logging
import subprocess # Added this import
import random
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


def apply_blur(image, blur_amount):
    """Apply Gaussian blur to an image.
    
    Args:
        image: Input image as a numpy array
        blur_amount: Radius of the Gaussian blur (must be odd and positive)
        
    Returns:
        Blurred image as a numpy array
    """
    if blur_amount > 0:
        # Ensure blur_amount is odd and at least 1
        blur_amount = max(1, blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1  # Make it odd
        return cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)
    return image


def fade_effect(current_frame, next_frame, progress):
    """Fade between current and next frame.
    
    Args:
        current_frame: Current frame as numpy array
        next_frame: Next frame as numpy array
        progress: Float between 0 and 1 representing transition progress
        
    Returns:
        Transitioned frame as numpy array
    """
    logger.debug(f"Applying fade effect - Progress: {progress:.2f} ({(progress*100):.0f}%)")
    return cv2.addWeighted(current_frame, 1 - progress, next_frame, progress, 0)


def slide_effect(current_frame, next_frame, progress, direction='right'):
    """Slide transition between current and next frame.
    
    Args:
        current_frame: Current frame as numpy array
        next_frame: Next frame as numpy array
        progress: Float between 0 and 1 representing transition progress
        direction: Direction of slide ('left', 'right', 'up', 'down')
        
    Returns:
        Transitioned frame as numpy array
    """
    h, w = current_frame.shape[:2]
    result = current_frame.copy()
    
    if direction == 'right':
        # Slide right: next frame comes from the right
        x = int(w * progress)
        result[:, :w-x] = next_frame[:, x:]
        result[:, w-x:] = current_frame[:, w-x:]
        logger.debug(f"Slide {direction} - Progress: {progress:.2f}, x-offset: {x}px")
    elif direction == 'left':
        # Slide left: next frame comes from the left
        x = int(w * (1 - progress))
        result[:, x:] = next_frame[:, :w-x]
        result[:, :x] = current_frame[:, w-x:]
        logger.debug(f"Slide {direction} - Progress: {progress:.2f}, x-offset: {x}px")
    elif direction == 'down':
        # Slide down: next frame comes from the bottom
        y = int(h * progress)
        result[:h-y, :] = next_frame[y:, :]
        result[h-y:, :] = current_frame[h-y:, :]
        logger.debug(f"Slide {direction} - Progress: {progress:.2f}, y-offset: {y}px")
    elif direction == 'up':
        # Slide up: next frame comes from the top
        y = int(h * (1 - progress))
        result[y:, :] = next_frame[:h-y, :]
        result[:y, :] = current_frame[h-y:, :]
        logger.debug(f"Slide {direction} - Progress: {progress:.2f}, y-offset: {y}px")
    else:
        logger.warning(f"Unknown slide direction: {direction}. Defaulting to right slide.")
        
    return result


def crossfade_effect(current_frame, next_frame, progress, blur_amount=5):
    """Crossfade with blur effect between frames.
    
    Args:
        current_frame: Current frame as numpy array
        next_frame: Next frame as numpy array
        progress: Float between 0 and 1 representing transition progress
        blur_amount: Amount of blur to apply during transition
        
    Returns:
        Transitioned frame as numpy array
    """
    # Log the crossfade progress
    logger.debug(f"Crossfade effect - Progress: {progress:.2f}, Blur amount: {blur_amount}")
    
    # Apply blur to both frames during transition
    if 0 < progress < 1:
        # Ensure Gaussian kernel is a positive odd integer
        k = max(1, int(blur_amount))
        if k % 2 == 0:
            k += 1
        current_blurred = cv2.GaussianBlur(current_frame, (k, k), 0)
        next_blurred = cv2.GaussianBlur(next_frame, (k, k), 0)
        result = cv2.addWeighted(current_blurred, 1 - progress, next_blurred, progress, 0)
        logger.debug(f"Crossfade frame - Current weight: {1 - progress:.2f}, Next weight: {progress:.2f}")
        return result
        
    if progress >= 1:
        logger.debug("Crossfade complete - Showing next frame")
        return next_frame
    else:
        logger.debug("Crossfade starting - Showing current frame")
        return current_frame


def apply_panoramic_pan(temp_frame_dir, frame_counter, img_path, duration, resolution, idx, fps=30, blur_amount=50, start_frame=0):
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
        start_frame: Frame number to start from (used to skip frames after transition)
    """
    res_w, res_h = resolution
    num_frames = int(duration * fps)
    
    # Log the panning direction and blur status with more details
    direction = "left→right" if idx % 2 == 0 else "right→left"
    blur_status = f"with blur (kernel size: {blur_amount*2 + 1})" if blur_amount > 0 else "without blur"
    logger.info(f"Processing slide {os.path.basename(img_path)} with index {idx}, panning {direction} {blur_status}")
    logger.debug(f"  - Start frame: {start_frame}, Total frames: {num_frames}, Duration: {duration:.2f}s")

    # Read the image and apply blur
    img = cv2.imread(img_path)
    if img is None:
        logger.error(f"Failed to load image: {img_path}")
        return frame_counter
    
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
        # Use the standard blur function for consistency
        scaled = apply_blur(scaled, blur_amount)
    
    # 2. For each frame, compute horizontal crop window
    frames_actually_written = 0
    
    # Calculate the maximum possible x position (0 if image is narrower than video width)
    max_x = max(0, scaled_w - res_w)
    
    if start_frame >= num_frames:
        logger.warning(f"start_frame ({start_frame}) is greater than total frames ({num_frames}). Using last frame.")
        start_frame = max(0, num_frames - 1)
    
    # Generate frames starting from start_frame
    effective_frames = max(1, num_frames - start_frame)
    for i in range(start_frame, num_frames):
        # Calculate progress using absolute index so skipping frames advances t
        t = (i / (num_frames - 1)) if num_frames > 1 else 0.0
        # Clamp t to [0.0, 1.0]
        t = max(0.0, min(1.0, t))
        
        # Only apply panning if the image is wider than the video
        if max_x > 0:
            if idx % 2 == 0:
                # left → right
                x = int(round(t * max_x))
            else:
                # right → left
                x = int(round((1 - t) * max_x))
            # Clamp x within valid bounds
            x = max(0, min(x, max_x))
        else:
            # Center the image if it's not wide enough to pan
            x = 0
            
        logger.debug(f"Frame {i}: x={x}, t={t:.2f}, max_x={max_x}, direction={'left→right' if idx % 2 == 0 else 'right→left'}")

        frame = scaled[:, x: x + res_w]
        
        # Ensure the frame is not empty before writing
        if frame.size == 0:
            logger.warning(f"Empty frame generated at x={x}, using black frame instead")
            frame = np.zeros((res_h, res_w, 3), dtype=np.uint8)
        
        frame_path = os.path.join(temp_frame_dir, f"frame_{frame_counter:05d}.jpg")
        try:
            success = cv2.imwrite(frame_path, frame)
            if not success:
                logger.error(f"Failed to write frame {frame_path}")
                # Skip this frame but continue with the rest
                continue
            frame_counter += 1
            frames_actually_written += 1
        except Exception as e:
            logger.error(f"Error writing frame {frame_path}: {str(e)}")
            continue
    
    return frame_counter # Return updated frame_counter


def render_panned_frame(img_path, resolution, idx, t, blur_amount=0):
    """Render a single frame of a slide at a given pan progress t in [0,1].
    Mirrors the preprocessing in apply_panoramic_pan to ensure visual continuity.
    """
    res_w, res_h = resolution
    img = cv2.imread(img_path)
    if img is None:
        logger.warning(f"render_panned_frame: Failed to load {img_path}, using black frame")
        return np.zeros((res_h, res_w, 3), dtype=np.uint8)

    img_h, img_w = img.shape[:2]
    scale = res_h / img_h
    scaled_w = int(img_w * scale)
    scaled = cv2.resize(img, (scaled_w, res_h))

    if scaled_w < res_w:
        pad = res_w - scaled_w
        left = pad // 2
        right = pad - left
        scaled = cv2.copyMakeBorder(scaled, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        scaled_w = res_w

    if blur_amount > 0:
        scaled = apply_blur(scaled, blur_amount)

    max_x = max(0, scaled_w - res_w)
    # Clamp t and compute x using same direction logic
    t = max(0.0, min(1.0, float(t)))
    if max_x > 0:
        if idx % 2 == 0:
            x = int(round(t * max_x))
        else:
            x = int(round((1 - t) * max_x))
        x = max(0, min(x, max_x))
    else:
        x = 0

    frame = scaled[:, x: x + res_w]
    if frame.size == 0:
        logger.warning(f"render_panned_frame: Empty crop generated (x={x}), using black frame")
        frame = np.zeros((res_h, res_w, 3), dtype=np.uint8)
    return frame

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

def convert_to_jpg(image_path, quality=95):
    """Convert any image to JPG format.
    
    Args:
        image_path: Path to the input image
        quality: JPEG quality (1-100)
        
    Returns:
        Path to the converted JPG file or None if conversion fails
    """
    try:
        # If already a JPG, return as is
        if image_path.lower().endswith(('.jpg', '.jpeg')):
            return image_path
            
        # Generate output path with .jpg extension
        base_path = os.path.splitext(image_path)[0]
        output_path = f"{base_path}.jpg"
        
        # Skip if already converted
        if os.path.exists(output_path):
            return output_path
            
        # Open and convert the image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (e.g., for PNG with transparency)
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Save as JPG
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
            
        logger.info(f"Converted {image_path} to JPG format")
        return output_path
        
    except Exception as e:
        logger.error(f"Error converting {image_path} to JPG: {e}")
        return None


def main(json_path, image_dir, output_path, total_duration=None, resolution=VIDEO_RESOLUTION, fps=30, blur_amount=50, transition_type='random', transition_duration=0.5):
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
        transition_type: Type of transition between slides ('fade', 'crossfade', 'slide_left', 'slide_right', 'slide_up', 'slide_down')
        transition_duration: Duration of transition in seconds
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
    # Track last selected transition to avoid immediate repeats when using 'random'
    last_selected_transition = None
    
    for idx, slide in enumerate(slides):
        try:
            start = parse_time(slide["start_time"])
            end = parse_time(slide["end_time"])
            duration = end - start
            last_slide_end_time = max(last_slide_end_time, end)
            
            logger.info(f"Processing slide {idx+1} (slides_processed={slides_processed}): start={start:.2f}s, end={end:.2f}s, duration={duration:.2f}s, transition={transition_type}")

            # First try to get the image path from the slide data if it exists
            img_path = slide.get('image_path')
            
            # If not in slide data or path doesn't exist, try to find the image
            if not img_path or not os.path.exists(img_path):
                # Get the image hash if available
                image_hash = slide.get('image_hash', '')[:8] if 'image_hash' in slide else ''
                
                # List of possible extensions to try
                extensions = ["jpg", "jpeg", "png"]
                
                # Try different naming patterns in order of preference
                patterns = [
                    f"slide_{idx+1:03d}_{image_hash}.{{ext}}" if image_hash else None,  # With hash
                    f"slide_{idx+1:03d}.{{ext}}",                                        # Just number
                    f"slide_{idx+1}_{image_hash}.{{ext}}" if image_hash else None,      # Old format with hash
                    f"slide_{idx+1}.{{ext}}"                                            # Old format just number
                ]
                
                # Try each pattern with each extension
                img_path = None
                for pattern in patterns:
                    if not pattern:
                        continue
                        
                    for ext in extensions:
                        candidate = os.path.join(image_dir, pattern.format(ext=ext))
                        if os.path.exists(candidate):
                            img_path = candidate
                            logger.info(f"Found image for slide {idx+1} at: {img_path}")
                            break
                    if img_path:
                        break
            
            # If still no image found, try to find any file starting with the slide number
            if not img_path or not os.path.exists(img_path):
                try:
                    # List all files in the directory
                    all_files = os.listdir(image_dir)
                    # Look for files that start with the slide number
                    for f in all_files:
                        if f.startswith(f"slide_{idx+1}"):
                            candidate = os.path.join(image_dir, f)
                            if os.path.isfile(candidate):
                                img_path = candidate
                                logger.info(f"Found image for slide {idx+1} using fallback pattern: {img_path}")
                                break
                    
                    # If still not found, try with zero-padded number
                    if not img_path and idx + 1 < 100:  # Only try if it makes sense to pad
                        padded_num = f"{idx+1:03d}"  # 1 -> 001, 12 -> 012
                        for f in all_files:
                            if f.startswith(f"slide_{padded_num}"):
                                candidate = os.path.join(image_dir, f)
                                if os.path.isfile(candidate):
                                    img_path = candidate
                                    logger.info(f"Found image for slide {idx+1} using padded fallback: {img_path}")
                                    break
                except Exception as e:
                    logger.warning(f"Error while searching for slide {idx+1}: {e}")
            
            # If still no image found, log a warning and skip this slide
            if not img_path or not os.path.exists(img_path):
                logger.warning(f"Image for slide {idx+1} not found in {image_dir}. Tried multiple naming patterns. Skipping this slide.")
                logger.debug(f"Searched for patterns: slide_{idx+1:03d}_*.{{jpg,png,jpeg}}, slide_{idx+1}*.{{jpg,png,jpeg}}")
                continue

            # Convert image to JPG if it's not already
            _, ext = os.path.splitext(img_path)
            ext = ext.lower()
            if ext != '.jpg' and ext != '.jpeg':
                jpg_path = convert_to_jpg(img_path)
                if jpg_path and jpg_path != img_path:
                    logger.info(f"Using converted JPG: {jpg_path}")
                    img_path = jpg_path

            logger.info(f"Processing image: {img_path}")
            # Determine transition type for this boundary (support 'random')
            selected_transition_type = transition_type
            if isinstance(transition_type, str) and transition_type.lower() == 'random':
                transitions_pool = ['fade', 'crossfade', 'slide_left', 'slide_right', 'slide_up', 'slide_down']
                # Avoid picking the same transition twice in a row when possible
                if last_selected_transition in transitions_pool and len(transitions_pool) > 1:
                    pool_no_repeat = [t for t in transitions_pool if t != last_selected_transition]
                else:
                    pool_no_repeat = transitions_pool
                selected_transition_type = random.choice(pool_no_repeat)
                last_selected_transition = selected_transition_type
                logger.info(f"Selected random transition: {selected_transition_type}")

            # Calculate transition frames if not the first slide
            transition_frames = 0
            if slides_processed > 0 and transition_duration > 0 and selected_transition_type != 'none':
                transition_frames = int(transition_duration * fps)
                logger.info(
                    f"Preparing {selected_transition_type} transition between slides {slides_processed} and {slides_processed+1} | "
                    f"from: {os.path.basename(slides[idx-1].get('image_path', 'unknown')) if idx>0 else 'N/A'} -> "
                    f"to: {os.path.basename(img_path)} | duration: {transition_duration:.2f}s, frames: {transition_frames}, fps: {fps}"
                )
                
                # Adjust duration for transition
                original_duration = duration
                duration -= transition_duration
                if duration < 0:
                    duration = 0.1  # Minimum duration for the slide
                    logger.warning(f"Adjusted slide duration to minimum (0.1s) to accommodate transition")
                
                logger.debug(f"Original slide duration: {original_duration:.2f}s, After transition: {duration:.2f}s")
                
                # Load the next frame first
                logger.debug(f"Preparing next frame using panned start of next slide: {img_path}")
                # Estimate number of content frames for next slide (duration already adjusted)
                next_num_frames = max(1, int(duration * fps))
                # We will skip the first content frame after transition (start_frame=1),
                # so align the transition's last frame to that first used progress
                start_t = (1 / (next_num_frames - 1)) if next_num_frames > 1 else 0.0
                next_frame = render_panned_frame(img_path, resolution, idx=slides_processed, t=start_t, blur_amount=blur_amount)
                logger.debug(
                    f"Next frame prepared via render_panned_frame | t={start_t:.4f}, frames={next_num_frames}, idx={slides_processed}"
                )
                
                # For the first slide, use a black frame as current to fade in from black
                if slides_processed == 0:
                    current_frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
                else:
                    # Calculate the frame number we should be using for the previous slide's last frame
                    # This should be the last frame before the transition starts
                    transition_start_frame = frames_written - 1
                    current_frame_path = os.path.join(temp_frame_dir, f"frame_{transition_start_frame:05d}.jpg")
                    logger.debug(f"Loading current frame from: {current_frame_path}")
                    current_frame = cv2.imread(current_frame_path)
                    
                    # If we can't load the previous frame, try to load the first frame of the current slide
                    if current_frame is None:
                        logger.warning(f"Failed to load current frame: {current_frame_path}. Using first frame of current slide.")
                        current_frame = next_frame.copy()
                
                # Log frame details
                logger.debug(
                    f"Current frame shape: {current_frame.shape if current_frame is not None else 'None'} | "
                    f"Next frame shape: {next_frame.shape if next_frame is not None else 'None'}"
                )
                
                # Apply transition effect with error handling
                logger.info(
                    f"Applying {selected_transition_type} transition over {transition_frames} frames "
                    f"between slides {slides_processed} and {slides_processed+1} | "
                    f"frame range: {frames_written:05d}-{frames_written + transition_frames - 1:05d}"
                )
                
                # Make sure both frames have the same dimensions
                if current_frame.shape != next_frame.shape:
                    logger.warning(
                        f"Frame dimension mismatch: current {current_frame.shape} vs next {next_frame.shape}. Resizing both to {resolution[1]}x{resolution[0]}"
                    )
                    current_frame = cv2.resize(current_frame, (resolution[0], resolution[1]))
                    next_frame = cv2.resize(next_frame, (resolution[0], resolution[1]))
                
                try:
                    # Write transition frames
                    transition_written = 0
                    for i in range(transition_frames):
                        try:
                            progress = (i + 1) / transition_frames
                            frame_num = frames_written + i
                            
                            # Generate transition frame
                            if selected_transition_type == 'fade':
                                frame = fade_effect(current_frame, next_frame, progress)
                            elif selected_transition_type.startswith('slide_'):
                                direction = selected_transition_type.split('_')[1]
                                frame = slide_effect(current_frame, next_frame, progress, direction)
                            else:  # crossfade
                                frame = crossfade_effect(current_frame, next_frame, progress, blur_amount)
                            
                            # Log progress at key checkpoints to avoid log spam
                            if i in {0, max(0, transition_frames//4), max(0, transition_frames//2), max(0, 3*transition_frames//4), transition_frames-1}:
                                logger.debug(
                                    f"Transition frame {i+1}/{transition_frames} | global #{frame_num:05d} | progress={progress:.2f}"
                                )
                            
                            # Ensure frame is in the correct format
                            if frame is None:
                                logger.warning(f"Generated frame is None, using black frame instead")
                                frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
                            
                            # Ensure frame has correct dimensions
                            if frame.shape[0] != resolution[1] or frame.shape[1] != resolution[0]:
                                frame = cv2.resize(frame, (resolution[0], resolution[1]))
                            
                            # Write the frame
                            frame_path = os.path.join(temp_frame_dir, f"frame_{frame_num:05d}.jpg")
                            success = cv2.imwrite(frame_path, frame)
                            if not success:
                                raise Exception(f"Failed to write frame {frame_path}")
                            
                            transition_written += 1
                            
                        except Exception as e:
                            logger.error(f"Error in transition frame {i+1}/{transition_frames} between slides {slides_processed} and {slides_processed+1}: {str(e)}")
                            logger.error(traceback.format_exc())
                            # Skip this frame but continue with the rest
                            # Continue with next frame even if one fails
                            continue
                    
                    start_idx = frames_written
                    frames_written += transition_frames
                    end_idx = frames_written - 1
                    logger.info(
                        f"Transition complete. Wrote {transition_written}/{transition_frames} frames | "
                        f"range: {start_idx:05d}-{end_idx:05d}"
                    )
                    
                except Exception as e:
                    logger.error(f"Critical error during transition: {str(e)}")
                    logger.error(traceback.format_exc())
                    # Skip transition frames to prevent getting stuck
                    frames_written += transition_frames
            
            # Calculate the number of frames for this slide
            slide_frames = int(duration * fps)
            
            # For the first slide, start from frame 0
            # For subsequent slides with transitions, we need to account for the transition frames
            if slides_processed == 0:
                start_frame = 0
            else:
                # If we had a transition, we need to start from frame 1 to avoid duplicating the last transition frame
                start_frame = 1 if transition_frames > 0 else 0
            
            # Only generate frames if we have frames left after accounting for the start frame
            try:
                if slide_frames > start_frame:
                    # Calculate the adjusted duration to account for skipped frames
                    adjusted_duration = (slide_frames - start_frame) / fps
                    
                    logger.info(f"Processing slide {os.path.basename(img_path)} - frames: {slide_frames}, start: {start_frame}, duration: {adjusted_duration:.2f}s")
                    
                    frames_before = frames_written
                    frames_written = apply_panoramic_pan(
                        temp_frame_dir=temp_frame_dir,
                        frame_counter=frames_written,
                        img_path=img_path,
                        duration=adjusted_duration,
                        resolution=resolution,
                        idx=slides_processed,
                        fps=fps,
                        blur_amount=blur_amount,
                        start_frame=start_frame
                    )
                    
                    frames_written_this_slide = frames_written - frames_before
                    logger.info(f"Successfully wrote {frames_written_this_slide} frames for slide {slides_processed + 1}")
                    
                    # Verify the frames were written correctly
                    expected_frame = os.path.join(temp_frame_dir, f"frame_{frames_written-1:05d}.jpg")
                    if not os.path.exists(expected_frame):
                        logger.warning(f"Expected frame {expected_frame} was not created!")
                        # Create a black frame to prevent further issues
                        cv2.imwrite(expected_frame, np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8))
                        frames_written += 1
                else:
                    logger.warning(f"Skipping slide {slides_processed + 1} - not enough frames (need {start_frame + 1}, have {slide_frames})")
                    frames_written += slide_frames
                
                slides_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing slide {slides_processed + 1}: {str(e)}")
                logger.error(traceback.format_exc())
                # Skip to next slide to prevent getting stuck
                frames_written += max(1, slide_frames - start_frame)
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
