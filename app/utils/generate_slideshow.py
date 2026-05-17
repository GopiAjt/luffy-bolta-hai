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
import math
from app.config import VIDEO_RESOLUTION

logger = logging.getLogger(__name__)

def parse_time(t):
    """Converts '0:00:06.28' to seconds."""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


# Default easing to apply across transition effects
DEFAULT_EASING = 'cosine'
PAN_EASING = 'smoothstep'

# Gamma used for approximate sRGB linearization
GAMMA = 2.2

def to_linear_bgr(img: np.ndarray) -> np.ndarray:
    """Convert uint8 BGR sRGB image to linear float32 [0,1]."""
    x = img.astype(np.float32) / 255.0
    # Simple gamma approximation
    return np.power(x, GAMMA)

def to_srgb_bgr(img_linear: np.ndarray) -> np.ndarray:
    """Convert linear float32 [0,1] to uint8 BGR sRGB image."""
    x = np.clip(img_linear, 0.0, 1.0)
    x = np.power(x, 1.0 / GAMMA)
    return (x * 255.0 + 0.5).astype(np.uint8)

def blend_linear_bgr(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Gamma-correct (linear space) blend between a and b with weight t (0..1)."""
    t = float(max(0.0, min(1.0, t)))
    la = to_linear_bgr(a)
    lb = to_linear_bgr(b)
    lc = la * (1.0 - t) + lb * t
    return to_srgb_bgr(lc)


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


def normalize_blur_amount(blur_amount, max_kernel=15):
    """Clamp blur to a practical odd kernel size."""
    blur_amount = max(0, int(blur_amount))
    if blur_amount <= 0:
        return 0
    blur_amount = min(blur_amount, max_kernel)
    return blur_amount if blur_amount % 2 == 1 else blur_amount + 1


def prepare_slide_canvas(img_path, resolution, blur_amount=0):
    """Load and scale a slide image once for panned frame rendering."""
    res_w, res_h = resolution
    img = cv2.imread(img_path)
    if img is None:
        logger.error(f"Failed to load image: {img_path}")
        return None, res_w

    img_h, img_w = img.shape[:2]
    scale = res_h / img_h
    scaled_w = max(1, int(round(img_w * scale)))
    scaled = cv2.resize(img, (scaled_w, res_h), interpolation=cv2.INTER_CUBIC)

    if scaled_w < res_w:
        pad = res_w - scaled_w
        left = pad // 2
        right = pad - left
        scaled = cv2.copyMakeBorder(
            scaled, 0, 0, left, right, cv2.BORDER_REPLICATE)
        scaled_w = res_w

    blur_amount = normalize_blur_amount(blur_amount, max_kernel=9)
    if blur_amount > 0:
        scaled = apply_blur(scaled, blur_amount)

    return scaled, scaled_w


def crop_panned_frame(scaled, scaled_w, resolution, idx, t):
    """Return a sub-pixel accurate pan crop for smoother motion."""
    res_w, res_h = resolution
    max_x = max(0.0, float(scaled_w - res_w))
    t = ease_progress(t, PAN_EASING)

    if max_x > 0:
        x = t * max_x if idx % 2 == 0 else (1.0 - t) * max_x
    else:
        x = 0.0

    center = (float(x + res_w / 2.0), float(res_h / 2.0))
    frame = cv2.getRectSubPix(scaled, (res_w, res_h), center)
    if frame is None or frame.size == 0:
        logger.warning(f"Empty panned crop generated at x={x:.2f}, using black frame")
        return np.zeros((res_h, res_w, 3), dtype=np.uint8)
    return frame


def fade_effect(current_frame, next_frame, progress):
    """
    Simple fade effect that blends between two frames based on progress
        current_frame: Current frame as numpy array
        next_frame: Next frame as numpy array
        progress: Float between 0 and 1 representing transition progress
        
    Returns:
        Transitioned frame as numpy array
    """
    p = ease_progress(progress, DEFAULT_EASING)
    # Gamma-correct blend for better midtones
    return blend_linear_bgr(current_frame, next_frame, p)


def page_curl_effect(current_frame, next_frame, progress, corner='top-right'):
    """
    Creates a page curl effect from the specified corner.
    
    Args:
        current_frame: Current frame as numpy array (BGR)
        next_frame: Next frame as numpy array (BGR)
        progress: Float between 0 and 1
        corner: Corner from which to curl ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        
    Returns:
        Transitioned frame as numpy array
    """
    h, w = current_frame.shape[:2]
    result = np.zeros_like(current_frame, dtype=np.float32)
    
    # Convert to float for processing
    current = current_frame.astype(np.float32) / 255.0
    next_img = next_frame.astype(np.float32) / 255.0
    
    # Calculate curl parameters based on corner
    if corner == 'top-left':
        center = (0, 0)
        x_curve = lambda p: (1 - p) * w / 2
        y_curve = lambda p: (1 - p) * h / 2
    elif corner == 'top-right':
        center = (w, 0)
        x_curve = lambda p: w - (1 - p) * w / 2
        y_curve = lambda p: (1 - p) * h / 2
    elif corner == 'bottom-left':
        center = (0, h)
        x_curve = lambda p: (1 - p) * w / 2
        y_curve = lambda p: h - (1 - p) * h / 2
    else:  # bottom-right
        center = (w, h)
        x_curve = lambda p: w - (1 - p) * w / 2
        y_curve = lambda p: h - (1 - p) * h / 2
    
    # Create coordinate grids
    y, x = np.ogrid[:h, :w]
    
    # Calculate distance from corner
    dx = x - center[0]
    dy = y - center[1]
    distance = np.sqrt(dx*dx + dy*dy)
    max_dist = np.sqrt(w*w + h*h)
    
    # Calculate angle
    angle = np.arctan2(dy, dx)
    
    # Create curl mask
    curve_x = x_curve(progress)
    curve_y = y_curve(progress)
    curl_mask = ((x - center[0])**2 / (curve_x**2 + 1e-6) + 
                 (y - center[1])**2 / (curve_y**2 + 1e-6)) <= 1.0
    
    # Create shadow
    shadow = np.ones_like(current) * 0.3  # Darker shadow
    shadow_mask = (distance / max_dist) < (1 - progress * 0.8)
    
    # Apply the effect
    result = np.where(curl_mask[..., None], next_img, current)
    result = np.where(shadow_mask[..., None], result * shadow, result)
    
    # Add highlight on the curl
    highlight = np.ones_like(current) * 1.2
    highlight_mask = curl_mask & ((distance / max_dist) > (1 - progress * 0.5))
    result = np.where(highlight_mask[..., None], result * highlight, result)
    
    # Clamp values
    result = np.clip(result, 0, 1)
    
    return (result * 255).astype(np.uint8)


def water_ripple_effect(current_frame, next_frame, progress, center=None, amplitude=10, wavelength=30):
    """
    Creates a water ripple effect between two frames.
    
    Args:
        current_frame: Current frame as numpy array (BGR)
        next_frame: Next frame as numpy array (BGR)
        progress: Float between 0 and 1
        center: (x, y) coordinates for ripple center (None for random)
        amplitude: Maximum displacement of the ripple
        wavelength: Wavelength of the ripple
        
    Returns:
        Transitioned frame as numpy array
    """
    h, w = current_frame.shape[:2]
    
    # Use provided center or random if not specified
    if center is None:
        cx = random.randint(w//4, 3*w//4)
        cy = random.randint(h//4, 3*h//4)
        center = (cx, cy)
    
    # Create coordinate grids
    y, x = np.ogrid[:h, :w]
    
    # Calculate distance from center
    dx = x - center[0]
    dy = y - center[1]
    distance = np.sqrt(dx*dx + dy*dy)
    
    # Calculate ripple displacement
    ripple = amplitude * np.sin(2 * np.pi * distance / wavelength - progress * 2 * np.pi)
    
    # Create displacement maps
    scale = progress * ripple / (distance + 1e-6)  # Avoid division by zero
    map_x = x + dx * scale
    map_y = y + dy * scale
    
    # Ensure coordinates are within bounds
    map_x = np.clip(map_x, 0, w-1).astype(np.float32)
    map_y = np.clip(map_y, 0, h-1).astype(np.float32)
    
    # Remap the images
    current_remap = cv2.remap(current_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    next_remap = cv2.remap(next_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    # Blend based on progress
    p = ease_progress(progress, 'ease_in_out_quad')
    result = (1 - p) * current_remap.astype(np.float32) + p * next_remap.astype(np.float32)
    
    return np.clip(result, 0, 255).astype(np.uint8)


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
    
    # Apply easing to progress for weight computation
    p = ease_progress(progress, DEFAULT_EASING)
    # Apply blur to both frames during transition
    if 0 < p < 1:
        # Ensure Gaussian kernel is a positive odd integer
        k = normalize_blur_amount(blur_amount, max_kernel=15)
        if k <= 0:
            k = 1
        current_blurred = cv2.GaussianBlur(current_frame, (k, k), 0)
        next_blurred = cv2.GaussianBlur(next_frame, (k, k), 0)
        # Gamma-correct blend
        result = blend_linear_bgr(current_blurred, next_blurred, p)
        logger.debug(f"Crossfade frame - Current weight: {1 - p:.2f}, Next weight: {p:.2f}")
        return result
        
    if p >= 1:
        logger.debug("Crossfade complete - Showing next frame")
        return next_frame
    else:
        logger.debug("Crossfade starting - Showing current frame")
        return current_frame


def apply_panoramic_pan(temp_frame_dir, frame_counter, img_path, duration, resolution, idx, fps=30, blur_amount=0, start_frame=0):
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
    effective_blur = normalize_blur_amount(blur_amount, max_kernel=9)
    blur_status = f"with blur (kernel size: {effective_blur})" if effective_blur > 0 else "without blur"
    logger.info(f"Processing slide {os.path.basename(img_path)} with index {idx}, panning {direction} {blur_status}")
    logger.debug(f"  - Start frame: {start_frame}, Total frames: {num_frames}, Duration: {duration:.2f}s")

    scaled, scaled_w = prepare_slide_canvas(img_path, resolution, effective_blur)
    if scaled is None:
        return frame_counter
    
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
        
        logger.debug(f"Frame {i}: t={t:.2f}, max_x={max_x}, direction={'left→right' if idx % 2 == 0 else 'right→left'}")
        frame = crop_panned_frame(scaled, scaled_w, resolution, idx, t)
        
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
    scaled, scaled_w = prepare_slide_canvas(img_path, resolution, blur_amount)
    if scaled is None:
        logger.warning(f"render_panned_frame: Failed to load {img_path}, using black frame")
        return np.zeros((res_h, res_w, 3), dtype=np.uint8)
    return crop_panned_frame(scaled, scaled_w, resolution, idx, t)


def weighted_choice(weighted_items):
    """Pick an item from a list of (item, weight) tuples."""
    total = sum(w for _, w in weighted_items)
    if total <= 0:
        # fallback to uniform choice over items
        items = [it for it, _ in weighted_items]
        return random.choice(items)
    r = random.uniform(0, total)
    upto = 0
    for item, weight in weighted_items:
        upto += weight
        if upto >= r:
            return item
    # Fallback (floating point edge)
    return weighted_items[-1][0]


LEFT_DIR_NAMES = {"slide_left", "motion_slide_left", "whip_pan_left"}
RIGHT_DIR_NAMES = {"slide_right", "motion_slide_right", "whip_pan_right"}


def choose_transition(slides_processed, next_slide_duration, last_selected_transition, avoid_same_direction_horizontal=True):
    """Heuristic to choose a transition effect.
    
    Args:
        slides_processed: Number of slides processed so far
        next_slide_duration: Duration of the next slide in seconds
        last_selected_transition: Last used transition effect
        avoid_same_direction_horizontal: If True, avoid same horizontal direction transitions
        
    Returns:
        Name of the selected transition effect
    """
    # Base weights for different transition types
    soft_weights = [
        ("fade", 0.3), 
        ("fade_eased", 0.2), 
        ("crossfade", 0.15), 
        ("zoom_dissolve", 0.1),
        ("iris_wipe", 0.15),
        ("cube_rotation_right", 0.05),
        ("cube_rotation_left", 0.05)
    ]
    
    # Adjust weights based on next slide duration
    if next_slide_duration < 2.0:
        # For very short slides, prefer softer transitions
        soft_mix = [
            ("fade", 0.25),
            ("fade_eased", 0.2),
            ("crossfade", 0.15),
            ("iris_wipe", 0.2),
            ("zoom_dissolve", 0.2)
        ]
    elif next_slide_duration < 4.0:
        # For medium duration slides, mix in more dynamic transitions
        soft_mix = [
            ("fade", 0.2),
            ("fade_eased", 0.15),
            ("crossfade", 0.1),
            ("iris_wipe", 0.15),
            ("zoom_dissolve", 0.15),
            ("cube_rotation_right", 0.1),
            ("cube_rotation_left", 0.1),
            ("radial_wipe", 0.05)
        ]
    else:
        # For longer slides, use more dynamic transitions
        soft_mix = [
            ("fade", 0.15),
            ("fade_eased", 0.1),
            ("crossfade", 0.08),
            ("iris_wipe", 0.12),
            ("zoom_dissolve", 0.1),
            ("cube_rotation_right", 0.12),
            ("cube_rotation_left", 0.12),
            ("radial_wipe", 0.08),
            ("motion_slide_right", 0.06),
            ("motion_slide_left", 0.06),
            ("whip_pan_right", 0.03),
            ("whip_pan_left", 0.03)
        ]
    
    # Add directional transitions
    if slides_processed % 2 == 0:
        hard_mix = [
            ("slide_right", 0.12),
            ("motion_slide_right", 0.08),
            ("whip_pan_right", 0.03),
            ("cube_rotation_right", 0.12)
        ]
    else:
        hard_mix = [
            ("slide_left", 0.12),
            ("motion_slide_left", 0.08),
            ("whip_pan_left", 0.03),
            ("cube_rotation_left", 0.12)
        ]
    
    # Add vertical transitions (less frequent)
    vertical_mix = [
        ("slide_up", 0.03),
        ("slide_down", 0.03),
        ("cube_rotation_up", 0.02),
        ("cube_rotation_down", 0.02)
    ]
    
    # Special transitions (used occasionally)
    special_mix = [
        ("radial_wipe", 0.05),
        ("iris_wipe", 0.1)
    ]
    
    # Combine all transitions
    candidates = soft_mix + hard_mix + vertical_mix + special_mix
    
    # Avoid repeating the same transition
    if last_selected_transition:
        candidates = [(t, w) for t, w in candidates if t != last_selected_transition]
    
    # Avoid same direction horizontal transitions if requested
    if avoid_same_direction_horizontal and last_selected_transition:
        if 'right' in last_selected_transition:
            candidates = [(t, w) for t, w in candidates if 'right' not in t]
        elif 'left' in last_selected_transition:
            candidates = [(t, w) for t, w in candidates if 'left' not in t]
    
    # Normalize weights
    total_weight = sum(w for _, w in candidates)
    if total_weight > 0:
        candidates = [(t, w/total_weight) for t, w in candidates]
    
    # Select a transition
    if not candidates:
        return "crossfade"  # Fallback
        
    return weighted_choice(candidates)


# Normalize horizontal transition names to match upcoming pan direction
# next_pan: 'lr' (left→right) or 'rl' (right→left)
def normalize_horizontal_transition(name, next_pan):
    if not isinstance(name, str):
        return name
    # Map desired direction based on pan
    desired = 'right' if next_pan == 'lr' else 'left'
    # Simple replacements for known directional transitions
    mapping = {
        'slide_left': 'slide_right' if desired == 'right' else 'slide_left',
        'slide_right': 'slide_right' if desired == 'right' else 'slide_left',
        'motion_slide_left': 'motion_slide_right' if desired == 'right' else 'motion_slide_left',
        'motion_slide_right': 'motion_slide_right' if desired == 'right' else 'motion_slide_left',
        'whip_pan_left': 'whip_pan_right' if desired == 'right' else 'whip_pan_left',
        'whip_pan_right': 'whip_pan_right' if desired == 'right' else 'whip_pan_left',
    }
    return mapping.get(name, name)


def flip_horizontal_transition(name):
    """Flip left<->right variants for horizontal directional transitions."""
    mapping = {
        'slide_left': 'slide_right',
        'slide_right': 'slide_left',
        'motion_slide_left': 'motion_slide_right',
        'motion_slide_right': 'motion_slide_left',
        'whip_pan_left': 'whip_pan_right',
        'whip_pan_right': 'whip_pan_left',
    }
    return mapping.get(name, name)


# ------------------ New Transition Effects ------------------

def cube_rotation_effect(current_frame, next_frame, progress, direction='right'):
    """3D cube rotation transition effect.
    
    Args:
        current_frame: Current frame as numpy array (BGR)
        next_frame: Next frame as numpy array (BGR)
        progress: Float between 0 and 1
        direction: Rotation direction ('left', 'right', 'up', 'down')
        
    Returns:
        Transitioned frame as numpy array
    """
    h, w = current_frame.shape[:2]
    
    # Ease the progress for smoother motion
    p = ease_progress(progress, 'smoothstep')
    
    # Create output frame
    output = np.zeros_like(current_frame, dtype=np.float32)
    
    # Convert to float for processing
    current = current_frame.astype(np.float32) / 255.0
    next_img = next_frame.astype(np.float32) / 255.0
    
    # Calculate rotation angle (0 to 90 degrees)
    angle = p * 90.0
    
    # Convert to linear color space for better blending
    current_lin = np.power(current, GAMMA)
    next_lin = np.power(next_img, GAMMA)
    
    # Create grid of coordinates
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(x, y)
    
    # Apply rotation based on direction
    if direction in ['left', 'right']:
        # Horizontal rotation
        rot = np.radians(angle)
        if direction == 'left':
            rot = -rot
        
        # Calculate perspective
        xp = xv * np.cos(rot) + (1 if direction == 'right' else -1)
        mask = (xp + 1) / 2  # Normalize to 0-1
        
        # Create visibility masks
        mask_current = 1 - mask
        mask_next = mask
        
        # Apply perspective scaling
        scale = 0.5 + 0.5 * np.cos(np.radians(angle))
        mask_current = mask_current * scale
        mask_next = mask_next * scale
        
        # Blend images
        for c in range(3):
            output[..., c] = (current_lin[..., c] * mask_current + 
                            next_lin[..., c] * mask_next)
    
    else:  # vertical rotation
        rot = np.radians(angle)
        if direction == 'up':
            rot = -rot
            
        # Calculate perspective
        yp = yv * np.cos(rot) + (1 if direction == 'down' else -1)
        mask = (yp + 1) / 2  # Normalize to 0-1
        
        # Create visibility masks
        mask_current = 1 - mask
        mask_next = mask
        
        # Apply perspective scaling
        scale = 0.5 + 0.5 * np.cos(np.radians(angle))
        mask_current = mask_current * scale
        mask_next = mask_next * scale
        
        # Blend images
        for c in range(3):
            output[..., c] = (current_lin[..., c] * mask_current[..., np.newaxis] + 
                            next_lin[..., c] * mask_next[..., np.newaxis])
    
    # Convert back to sRGB and 8-bit
    output = np.power(np.clip(output, 0, 1), 1.0/GAMMA)
    return (output * 255).astype(np.uint8)

def iris_wipe_effect(current_frame, next_frame, progress, center=None, feather=0.1):
    """Iris wipe transition effect with soft edges.
    
    Args:
        current_frame: Current frame as numpy array (BGR)
        next_frame: Next frame as numpy array (BGR)
        progress: Float between 0 and 1
        center: (x,y) center point of the iris (None for center of image)
        feather: Feather amount for soft edges (0-1)
        
    Returns:
        Transitioned frame as numpy array
    """
    h, w = current_frame.shape[:2]
    
    # Default to center of image
    if center is None:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = center
    
    # Ease the progress for smoother motion
    p = ease_progress(progress, 'smoothstep')
    
    # Create distance map from center
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_dist = np.sqrt((w/2)**2 + (h/2)**2)  # Maximum possible distance
    
    # Calculate radius based on progress
    radius = p * max_dist * 1.2  # Slightly overshoot to ensure full coverage
    
    # Create soft mask with feathering
    mask = np.clip((radius - dist) / (feather * max_dist) + 0.5, 0, 1)
    
    # Convert to 3-channel mask
    mask_3d = mask[..., np.newaxis]
    
    # Convert to linear color space for better blending
    current_lin = to_linear_bgr(current_frame)
    next_lin = to_linear_bgr(next_frame)
    
    # Blend images using the mask
    result = current_lin * (1 - mask_3d) + next_lin * mask_3d
    
    # Convert back to sRGB
    return to_srgb_bgr(result)

def ease_progress(p, kind='cosine'):
    p = max(0.0, min(1.0, float(p)))
    if kind == 'cosine':
        return 0.5 - 0.5 * math.cos(math.pi * p)
    if kind == 'smoothstep':
        return p * p * (3 - 2 * p)
    if kind == 'sine':
        return math.sin(p * math.pi * 0.5)
    if kind == 'ease_in_out_quad':
        return 2 * p * p if p < 0.5 else 1 - ((-2 * p + 2) ** 2) / 2
    return p


def center_crop_to(frame, size_wh):
    w, h = size_wh
    fh, fw = frame.shape[:2]
    # When scaled larger than target, center-crop; when smaller, resize
    if fw < w or fh < h:
        frame = cv2.resize(frame, (w, h))
        return frame
    x0 = (fw - w) // 2
    y0 = (fh - h) // 2
    return frame[y0:y0 + h, x0:x0 + w]


def zoom_dissolve_effect(current_frame, next_frame, progress, resolution, zoom=1.05, easing='cosine'):
    p = ease_progress(progress, easing)
    res_w, res_h = resolution
    # scale next frame by factor 1 -> zoom
    scale = 1.0 + (zoom - 1.0) * p
    nh, nw = next_frame.shape[:2]
    scaled = cv2.resize(next_frame, (int(nw * scale), int(nh * scale)), interpolation=cv2.INTER_CUBIC)
    scaled = center_crop_to(scaled, (res_w, res_h))
    # Mild unsharp mask for crispness after upscaling
    us_k = 3
    if us_k % 2 == 0:
        us_k += 1
    blurred = cv2.GaussianBlur(scaled, (us_k, us_k), 0)
    sharpened = cv2.addWeighted(scaled, 1.15, blurred, -0.15, 0)
    # Gamma-correct blend
    out = blend_linear_bgr(current_frame, sharpened, p)
    return out


def motion_blur_kernel(length, orientation='horizontal'):
    k = max(1, int(length))
    if k % 2 == 0:
        k += 1
    if orientation == 'horizontal':
        kernel = np.zeros((1, k), dtype=np.float32)
        kernel[0, :] = 1.0 / k
    else:
        kernel = np.zeros((k, 1), dtype=np.float32)
        kernel[:, 0] = 1.0 / k
    return kernel


def motion_blur_slide_effect(current_frame, next_frame, progress, direction='right', strength=15):
    # Base slide composition
    h, w = current_frame.shape[:2]
    result = current_frame.copy()
    p = ease_progress(progress, 'smoothstep')
    if direction == 'right':
        # Slide right: next frame comes from the right
        x = int(w * p)
        result[:, :w-x] = next_frame[:, x:]
        result[:, w-x:] = current_frame[:, w-x:]
        orientation = 'horizontal'
    elif direction == 'left':
        # Slide left: next frame comes from the left
        x = int(w * (1 - p))
        result[:, x:] = next_frame[:, :w-x]
        result[:, :x] = current_frame[:, w-x:]
        orientation = 'horizontal'
    elif direction == 'down':
        # Slide down: next frame comes from the bottom
        y = int(h * p)
        result[:h-y, :] = next_frame[y:, :]
        result[h-y:, :] = current_frame[h-y:, :]
        orientation = 'vertical'
    else:  # up
        y = int(h * (1 - p))
        result[y:, :] = next_frame[:h-y, :]
        result[:y, :] = current_frame[h-y:, :]
        orientation = 'vertical'
    # Apply motion blur proportional to progress
    k = int(strength * max(0.1, min(1.0, 2 * abs(p - 0.5))))  # strongest mid-way
    kernel = motion_blur_kernel(max(3, k), 'horizontal' if orientation == 'horizontal' else 'vertical')
    blurred = cv2.filter2D(result, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return blurred


def slide_push_effect(current_frame, next_frame, progress, direction='right'):
    """Push one frame out while the next frame enters from the chosen direction."""
    h, w = current_frame.shape[:2]
    result = np.zeros_like(current_frame)
    p = ease_progress(progress, 'smoothstep')

    if direction == 'right':
        offset = int(round(w * p))
        result[:, :offset] = next_frame[:, w - offset:] if offset > 0 else result[:, :offset]
        result[:, offset:] = current_frame[:, :w - offset]
    elif direction == 'left':
        offset = int(round(w * p))
        result[:, :w - offset] = current_frame[:, offset:]
        if offset > 0:
            result[:, w - offset:] = next_frame[:, :offset]
    elif direction == 'down':
        offset = int(round(h * p))
        result[:offset, :] = next_frame[h - offset:, :] if offset > 0 else result[:offset, :]
        result[offset:, :] = current_frame[:h - offset, :]
    else:  # up
        offset = int(round(h * p))
        result[:h - offset, :] = current_frame[offset:, :]
        if offset > 0:
            result[h - offset:, :] = next_frame[:offset, :]

    return result


def fade_effect_eased(current_frame, next_frame, progress, easing='cosine'):
    p = ease_progress(progress, easing)
    return cv2.addWeighted(current_frame, 1 - p, next_frame, p, 0)


def radial_wipe_effect(current_frame, next_frame, progress, feather=0.05):
    # Build a circular mask from center that grows with progress
    h, w = current_frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2)
    # Apply easing to radius growth
    p = ease_progress(progress, DEFAULT_EASING)
    r = max(0.0, min(1.0, float(p))) * max_r
    feather_px = feather * max(h, w)
    # Mask grows from 0 to 1 with a feathered edge
    mask = (dist - (r - feather_px)) / (feather_px + 1e-6)
    mask = np.clip(mask, 0.0, 1.0)
    mask = 1.0 - mask  # inside is 1, outside is 0
    mask = np.clip(mask, 0.0, 1.0)
    mask3 = np.dstack([mask, mask, mask]).astype(np.float32)
    out = (next_frame.astype(np.float32) * mask3 + current_frame.astype(np.float32) * (1.0 - mask3)).astype(np.uint8)
    return out


def whip_pan_effect(current_frame, next_frame, progress, direction='right', blur_strength=25):
    # Aggressive slide with strong motion blur and easing that speeds up then snaps
    h, w = current_frame.shape[:2]
    # Use sharper ease-in-out
    p = ease_progress(progress, 'sine')
    result = current_frame.copy()
    if direction == 'right':
        x = int(w * p)
        result[:, :w - x] = next_frame[:, x:]
        result[:, w - x:] = current_frame[:, w - x:]
        orientation = 'horizontal'
    elif direction == 'left':
        x = int(w * (1 - p))
        result[:, x:] = next_frame[:, :w - x]
        result[:, :x] = current_frame[:, w - x:]
        orientation = 'horizontal'
    elif direction == 'down':
        y = int(h * p)
        result[:h - y, :] = next_frame[y:, :]
        result[h - y:, :] = current_frame[h - y:, :]
        orientation = 'vertical'
    else:
        y = int(h * (1 - p))
        result[y:, :] = next_frame[:h - y, :]
        result[:y, :] = current_frame[h - y:, :]
        orientation = 'vertical'
    # Strong motion blur regardless of p
    kernel = motion_blur_kernel(max(5, int(blur_strength)), 'horizontal' if orientation == 'horizontal' else 'vertical')
    blurred = cv2.filter2D(result, -1, kernel)
    return blurred

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


def main(json_path, image_dir, output_path, total_duration=None, resolution=VIDEO_RESOLUTION, fps=30, blur_amount=5, transition_type='auto', transition_duration=0.5, flip_horizontal_once=False, avoid_same_direction_horizontal=True, slide_blur_amount=0):
    """
    Generates the final slideshow video using OpenCV.
    
    Args:
        json_path: Path to the JSON file containing slide timings
        image_dir: Directory containing slide images
        output_path: Path where the output video will be saved
        total_duration: Total duration of the video in seconds
        resolution: Video resolution as (width, height)
        fps: Frames per second for the output video
        blur_amount: Amount of blur to apply during blur-based transitions
        slide_blur_amount: Optional blur applied to the slide images themselves
        transition_type: Type of transition between slides ('fade', 'crossfade', 'slide_left', 'slide_right', 'slide_up', 'slide_down')
        transition_duration: Duration of transition in seconds
    """
    resolution = sanitize_size(resolution)
    res_w, res_h = resolution
    
    blur_amount = normalize_blur_amount(blur_amount, max_kernel=15)
    slide_blur_amount = normalize_blur_amount(slide_blur_amount, max_kernel=9)
    if blur_amount > 0:
        logger.info(f"Applying transition blur with kernel: {blur_amount}")
    if slide_blur_amount > 0:
        logger.info(f"Applying slide blur with kernel: {slide_blur_amount}")

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

    # -------- Prepass: select transitions and allocate frames to avoid tail padding --------
    last_selected_transition = None
    flip_remaining = 1 if flip_horizontal_once else 0
    boundary_transition_types = []  # per slide index (same as processing index)
    content_frames_float = []       # desired written content frames per slide (before rounding)
    start_skips = []                # 0 for first slide or no transition, else 1
    transition_frames_list = []     # frames written for transition before this slide

    # First, select transitions per boundary deterministically for this run
    for idx, slide in enumerate(slides):
        # Parse timing
        start = parse_time(slide["start_time"]) 
        end = parse_time(slide["end_time"]) 
        duration = end - start
        # Decide transition type into THIS slide (except first)
        selected_transition_type = 'none'
        if idx > 0 and transition_duration > 0:
            if isinstance(transition_type, str):
                tt = transition_type.lower()
                if tt == 'random':
                    transitions_pool = [
                        'fade', 'crossfade', 'fade_eased', 'zoom_dissolve', 'radial_wipe',
                        'slide_up', 'slide_down',
                        'iris_wipe',
                        'cube_rotation_left', 'cube_rotation_right',
                        'cube_rotation_up', 'cube_rotation_down',
                        'page_curl_tl', 'page_curl_tr', 'page_curl_bl', 'page_curl_br',
                        'water_ripple'
                    ]
                    # Filter out the last used transition if we have alternatives
                    pool = [t for t in transitions_pool if t != last_selected_transition] or transitions_pool
                    selected_transition_type = random.choice(pool)
                elif tt == 'auto':
                    expected_content_duration = max(0.1, duration - transition_duration)
                    selected_transition_type = choose_transition(idx, expected_content_duration, last_selected_transition, avoid_same_direction_horizontal)
                else:
                    selected_transition_type = transition_type
            # Normalize horizontal transitions to match upcoming pan (unless that would violate avoidance rule)
            if selected_transition_type and selected_transition_type != 'none':
                next_pan = 'lr' if (idx % 2 == 0) else 'rl'
                normalized = normalize_horizontal_transition(selected_transition_type, next_pan)
                # If avoidance is enabled, do not normalize into same-direction
                if avoid_same_direction_horizontal:
                    if (next_pan == 'lr' and normalized in RIGHT_DIR_NAMES) or (next_pan == 'rl' and normalized in LEFT_DIR_NAMES):
                        logger.info(
                            f"Skipped normalization to '{normalized}' to avoid same-direction with pan {next_pan}; keeping '{selected_transition_type}'"
                        )
                    else:
                        if normalized != selected_transition_type:
                            logger.info(f"Normalized transition '{selected_transition_type}' to '{normalized}' to match upcoming pan {next_pan}")
                        selected_transition_type = normalized
                else:
                    if normalized != selected_transition_type:
                        logger.info(f"Normalized transition '{selected_transition_type}' to '{normalized}' to match upcoming pan {next_pan}")
                    selected_transition_type = normalized
                # Optionally flip once per run for creative variety
                if flip_remaining > 0:
                    flipped = flip_horizontal_transition(selected_transition_type)
                    if flipped != selected_transition_type:
                        logger.info(f"Flipped horizontal transition once: '{selected_transition_type}' -> '{flipped}'")
                        selected_transition_type = flipped
                        flip_remaining -= 1
            if selected_transition_type != 'none':
                last_selected_transition = selected_transition_type
        boundary_transition_types.append(selected_transition_type)

        # Compute desired content frames (the frames we intend to WRITE for the slide's content)
        content_duration = duration - (transition_duration if idx > 0 and selected_transition_type != 'none' else 0.0)
        content_frames_float.append(max(0.0, content_duration * fps))
        # Determine if we'll skip first content frame due to transition overlap
        start_skips.append(1 if (idx > 0 and selected_transition_type != 'none') else 0)
        transition_frames_list.append(int(transition_duration * fps) if (idx > 0 and selected_transition_type != 'none') else 0)

    # Allocate integer content frames using largest remainder so that total == target - transitions
    total_transition_frames = sum(transition_frames_list)
    target_content_frames = total_output_frames - total_transition_frames
    content_floor = [int(np.floor(x)) for x in content_frames_float]
    remainders = [x - f for x, f in zip(content_frames_float, content_floor)]
    allocated = content_floor[:]
    deficit = target_content_frames - sum(allocated)
    if deficit > 0:
        # Distribute +1 to the largest remainders
        order = sorted(range(len(remainders)), key=lambda i: remainders[i], reverse=True)
        for i in order[:deficit]:
            allocated[i] += 1
    elif deficit < 0:
        # Remove frames from smallest remainders (or anywhere) to match target
        order = sorted(range(len(remainders)), key=lambda i: remainders[i])
        for i in order[:abs(deficit)]:
            if allocated[i] > 0:
                allocated[i] -= 1

    # Compute planned slide_frames passed to pan function (accounting for double skip in implementation)
    planned_slide_frames = [alloc + 2 * start_skips[i] for i, alloc in enumerate(allocated)]

    # Initialize frames_written counter
    frames_written = 0

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
            # Use preselected transition for this boundary to keep frame plan consistent
            selected_transition_type = boundary_transition_types[idx]
            if selected_transition_type and selected_transition_type != 'none' and idx > 0:
                logger.info(f"Using preselected transition: {selected_transition_type}")

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
                next_frame = render_panned_frame(img_path, resolution, idx=slides_processed, t=start_t, blur_amount=slide_blur_amount)
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
                
                transition_params = {}
                if selected_transition_type == 'iris_wipe' and random.random() < 0.3:
                    h, w = current_frame.shape[:2]
                    transition_params['center'] = (
                        int(w * (0.2 + 0.6 * random.random())),
                        int(h * (0.2 + 0.6 * random.random()))
                    )
                elif selected_transition_type == 'water_ripple':
                    h, w = current_frame.shape[:2]
                    transition_params['center'] = (
                        int(w * (0.25 + 0.5 * random.random())),
                        int(h * (0.25 + 0.5 * random.random()))
                    )

                try:
                    # Write transition frames
                    transition_written = 0
                    for i in range(transition_frames):
                        try:
                            progress = (i + 1) / transition_frames
                            frame_num = frames_written + i
                            
                            # Generate transition frame
                            try:
                                if selected_transition_type == 'fade':
                                    frame = fade_effect(current_frame, next_frame, progress)
                                elif selected_transition_type == 'fade_eased':
                                    frame = fade_effect_eased(current_frame, next_frame, progress, easing='cosine')
                                elif selected_transition_type == 'crossfade':
                                    frame = crossfade_effect(current_frame, next_frame, progress, blur_amount)
                                elif selected_transition_type == 'zoom_dissolve':
                                    frame = zoom_dissolve_effect(current_frame, next_frame, progress, resolution)
                                elif selected_transition_type == 'radial_wipe':
                                    frame = radial_wipe_effect(current_frame, next_frame, progress, feather=0.06)
                                elif selected_transition_type == 'iris_wipe':
                                    frame = iris_wipe_effect(
                                        current_frame,
                                        next_frame,
                                        progress,
                                        center=transition_params.get('center'),
                                        feather=0.08
                                    )
                                elif selected_transition_type.startswith('cube_rotation_'):
                                    direction = selected_transition_type.split('_')[2]
                                    frame = cube_rotation_effect(current_frame, next_frame, progress, direction=direction)
                                elif selected_transition_type.startswith('page_curl_'):
                                    corner_key = selected_transition_type.split('_')[2]  # tl, tr, bl, or br
                                    corner_map = {
                                        'tl': 'top-left',
                                        'tr': 'top-right',
                                        'bl': 'bottom-left',
                                        'br': 'bottom-right',
                                    }
                                    frame = page_curl_effect(
                                        current_frame,
                                        next_frame,
                                        progress,
                                        corner=corner_map.get(corner_key, 'top-right')
                                    )
                                elif selected_transition_type == 'water_ripple':
                                    frame = water_ripple_effect(
                                        current_frame,
                                        next_frame,
                                        progress,
                                        center=transition_params.get('center')
                                    )
                                elif selected_transition_type.startswith('motion_slide_'):
                                    direction = selected_transition_type.split('_')[2]
                                    frame = motion_blur_slide_effect(
                                        current_frame,
                                        next_frame,
                                        progress,
                                        direction=direction,
                                        strength=10
                                    )
                                elif selected_transition_type.startswith('whip_pan_'):
                                    direction = selected_transition_type.split('_')[2]
                                    frame = whip_pan_effect(
                                        current_frame,
                                        next_frame,
                                        progress,
                                        direction=direction,
                                        blur_strength=18
                                    )
                                elif selected_transition_type.startswith('slide_'):
                                    direction = selected_transition_type.split('_')[1]
                                    frame = slide_push_effect(
                                        current_frame,
                                        next_frame,
                                        progress,
                                        direction=direction
                                    )
                                else:
                                    # Fallback to crossfade
                                    frame = crossfade_effect(current_frame, next_frame, progress, blur_amount)
                            except Exception as e:
                                logger.error(f"Error applying transition '{selected_transition_type}': {str(e)}")
                                logger.error(traceback.format_exc())
                                # Fallback to crossfade on error
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
            
            # Calculate planned frames for this slide's pan (based on pre-allocation)
            slide_frames = planned_slide_frames[idx]
            
            # For the first slide, start from frame 0
            # For subsequent slides with transitions, we need to account for the transition frames
            if slides_processed == 0:
                start_frame = 0
            else:
                # If we had a transition, we need to start from frame 1 to avoid duplicating the last transition frame
                start_frame = 1 if transition_frames > 0 else 0
            
            # Last-slide correction: ensure we exactly hit total_output_frames
            if idx == len(slides) - 1:
                remaining_needed = total_output_frames - frames_written
                corrected_slide_frames = remaining_needed + 2 * (1 if (slides_processed > 0 and transition_frames > 0) else 0)
                if corrected_slide_frames != slide_frames and corrected_slide_frames > 0:
                    logger.info(
                        f"Adjusting last slide frames from {slide_frames} to {corrected_slide_frames} to meet target {total_output_frames}."
                    )
                    slide_frames = corrected_slide_frames

            # Only generate frames if we have frames left after accounting for the start frame
            try:
                if slide_frames > start_frame:
                    # Calculate the adjusted duration to ensure the function writes (slide_frames - 2*start_frame)
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
                        blur_amount=slide_blur_amount,
                        start_frame=start_frame
                    )
                    
                    frames_written_this_slide = frames_written - frames_before
                    planned_content_frames = slide_frames - 2 * start_frame
                    logger.info(
                        f"Successfully wrote {frames_written_this_slide} frames for slide {slides_processed + 1} (planned={planned_content_frames}, start_skip={start_frame})"
                    )
                    
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

    logger.info(f"Frames written after generation: {frames_written}")
    if frames_written != total_output_frames:
        logger.warning(f"Frame count mismatch: wrote {frames_written}, expected {total_output_frames}. Consider re-checking allocation.")

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
    
