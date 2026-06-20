import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple
from app.utils.slides.generate_slideshow import ken_burns_viewport, ease_progress, DEFAULT_EASING, apply_blur, normalize_blur_amount

logger = logging.getLogger(__name__)

def render_composed_frame(
    img_bgr: np.ndarray,
    slide_meta: Dict[str, Any],
    resolution: Tuple[int, int],
    slide_idx: int,
    t: float
) -> np.ndarray:
    """
    Renders a single frame evaluating the CompositionPlanner's layers
    to create true 3D depth parallax and overlay effects.
    """
    res_w, res_h = resolution
    composition = slide_meta.get("composition", {})
    layers = composition.get("layers", [])

    if not layers:
        # Fallback to standard if no composition exists
        return _fallback_render(img_bgr, slide_meta, resolution, slide_idx, t)

    # 1. Extract motion parameters from the motion layer
    motion_layer = next((l for l in layers if l.get("category") == "motion"), None)
    motion_preset = "slow_push"
    parallax_enabled = False
    if motion_layer and motion_layer.get("metadata"):
        motion_preset = motion_layer["metadata"].get("motion_preset", "slow_push")
        parallax_enabled = motion_layer["metadata"].get("parallax", False)

    # Overwrite if slide explicitly set a motion preset
    motion_preset = slide_meta.get("motion_preset", motion_preset)
    layout_mode = slide_meta.get("layout_mode", "safe_subject")

    # 2. Extract Background
    bg_layer = next((l for l in layers if l.get("category") == "background" and l.get("layer_type") == "image"), None)
    
    # 3. Extract Foreground (default to a basic pip layer if not found)
    fg_layer = next((l for l in layers if l.get("category") == "foreground"), {"name": "default_pip", "layer_type": "glassmorphism"})

    # Canvas
    canvas = np.zeros((res_h, res_w, 3), dtype=np.uint8)

    # Parallax logic: Foreground zooms slightly faster/more than background
    # Background evaluates ken burns with 't'
    # Foreground evaluates ken burns with 't' but stronger zoom
    
    # Build Background
    if bg_layer:
        canvas = _render_background(canvas, img_bgr, resolution, slide_idx, t, motion_preset, parallax_enabled)

    # Apply treatments (mood gradients/blurs)
    treatments = [l for l in layers if l.get("category") == "background" and l.get("layer_type") != "image"]
    for treatment in treatments:
        canvas = _apply_treatment(canvas, treatment, resolution)

    # Build Foreground PIP (Editorial panel etc.)
    canvas = _render_foreground(canvas, img_bgr, resolution, slide_idx, t, motion_preset, parallax_enabled, fg_layer, layout_mode)

    # Add effects (vignette, film grain, etc)
    effects = [l for l in layers if l.get("category") == "effect"]
    for effect in effects:
        canvas = _apply_effect(canvas, effect, resolution, t)

    # Render Text Layers (Rich text with emphasis colors)
    # Only render actual text layers, ignoring metadata layers like subtitle_zone or emphasis_tags
    text_layers = [l for l in layers if l.get("category") == "text" and l.get("layer_type") == "text"]
    emphasis_words = slide_meta.get("emphasis_words", [])
    for t_layer in sorted(text_layers, key=lambda x: x.get("z_index", 0)):
        canvas = _render_text(canvas, t_layer, resolution, emphasis_words)

    return canvas

def _render_background(canvas, img_bgr, resolution, idx, t, motion_preset, parallax):
    res_w, res_h = resolution
    img_h, img_w = img_bgr.shape[:2]
    
    # Background always uses cover fit and heavily blurred if we have a foreground
    cover_scale = max(res_w / max(img_w, 1), res_h / max(img_h, 1))
    # Give it 10% headroom for pan/zoom
    cover_scale *= 1.10
    
    scaled_w = max(res_w, int(round(img_w * cover_scale)))
    scaled_h = max(res_h, int(round(img_h * cover_scale)))
    scaled = cv2.resize(img_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
    
    # Blur background to separate from foreground
    blur_amount = normalize_blur_amount(28)
    scaled = apply_blur(scaled, blur_amount)
    
    # Intense Background Parallax for panning motions
    if motion_preset in ("wide_pan", "diagonal_pan"):
        # Make the background pan very fast to compensate for the strictly contained PIP
        t_bg = t * 2.0
    else:
        # Standard parallax logic
        t_bg = t * 0.5 if parallax else t
    
    # Evaluate ken burns
    from app.utils.slides.generate_slideshow import ken_burns_viewport
    x, y, view_w, view_h = ken_burns_viewport(scaled_w, scaled_h, resolution, idx, t_bg, motion_preset)
    center = (float(x + view_w / 2.0), float(y + view_h / 2.0))
    frame = cv2.getRectSubPix(scaled, (int(round(view_w)), int(round(view_h))), center)
    
    if frame is not None and frame.size > 0:
        if frame.shape[0] != res_h or frame.shape[1] != res_w:
            frame = cv2.resize(frame, (res_w, res_h), interpolation=cv2.INTER_CUBIC)
        return frame
    return canvas

def _render_foreground(canvas, img_bgr, resolution, idx, t, motion_preset, parallax, fg_layer, layout_mode="safe_subject"):
    res_w, res_h = resolution
    img_h, img_w = img_bgr.shape[:2]
    
    from app.utils.slides.generate_slideshow import ken_burns_zoom_range, PAN_EASING
    z0, z1 = ken_burns_zoom_range(motion_preset)
    
    # If parallax, foreground moves faster/larger
    t_fg = t * 1.4 if parallax else t
    
    is_pan = motion_preset in ("stable_pan", "diagonal_pan", "wide_pan")
    is_hold = motion_preset in ("evidence_hold", "title_card_hold")
    
    # PLAN: All motions use PIP containment to prevent overflow
    # Dynamic aspect ratio sizing: Closer to 9:16 target -> larger scale
    
    # Pre-calculate max zoom to enforce strict safe margins
    max_progress = 1.4 if parallax else 1.0
    max_zoom = max(z0, z0 + (z1 - z0) * max_progress)
    
    if is_hold:
        # PLAN 1: Evidence & Title Holds
        # Maximize safe size dynamically, avoiding the strict 0.72 limit
        safe_max_w = res_w * 0.92
        safe_max_h = res_h * 0.74
        scale = min(safe_max_w / max(img_w, 1), safe_max_h / max(img_h, 1))
    else:
        # PLAN 3: Zoom & Impact Motions
        # Dynamic aspect ratio sizing: Closer to 9:16 target -> larger scale
        img_aspect = img_w / max(img_h, 1)
        target_aspect = res_w / max(res_h, 1)
        aspect_diff = abs(img_aspect - target_aspect)
        
        if aspect_diff < 0.1:
            subject_scale = 0.95
        elif aspect_diff < 0.3:
            subject_scale = 0.85
        else:
            subject_scale = 0.72
            
        subject_max_w = int(res_w * subject_scale)
        subject_max_h = int(res_h * subject_scale)
        
        safe_max_w = res_w * 0.90
        safe_max_h = res_h * 0.60
        
        scale = min(subject_max_w / max(img_w, 1), subject_max_h / max(img_h, 1))
        scale *= 1.15  # Add headroom
    
    max_scale_w = safe_max_w / max(img_w * max_zoom, 1)
    max_scale_h = safe_max_h / max(img_h * max_zoom, 1)
    
    # Strictly bound scale
    scale = min(scale, max_scale_w, max_scale_h)
    
    fg_w = int(img_w * scale)
    fg_h = int(img_h * scale)
    fg_scaled = cv2.resize(img_bgr, (fg_w, fg_h), interpolation=cv2.INTER_CUBIC)
    
    progress = ease_progress(t_fg, PAN_EASING)
    current_zoom = z0 + (z1 - z0) * progress
    
    zoomed_w = int(fg_w * current_zoom)
    zoomed_h = int(fg_h * current_zoom)
    
    if zoomed_w <= 0 or zoomed_h <= 0:
        return canvas
        
    fg_zoomed = cv2.resize(fg_scaled, (zoomed_w, zoomed_h), interpolation=cv2.INTER_CUBIC)
    
    if is_hold:
        pos_y = int(res_h * 0.40) - zoomed_h // 2
        # Clamp off-center to respect strict top/bottom text margins
        pos_y = max(int(res_h * 0.10), min(pos_y, res_h - zoomed_h - int(res_h * 0.15)))
        pos_x = (res_w - zoomed_w) // 2
    else:
        # 1. Base Alignment
        if layout_mode == "horizontal_feature":
            # Align to top 15% to leave room at bottom
            pos_y = int(res_h * 0.15)
        elif layout_mode == "split_screen":
            # Align perfectly to top
            pos_y = int(res_h * 0.05)
        else:
            # Paste onto canvas perfectly centered
            pos_y = int(res_h * 0.42) - zoomed_h // 2
            
        pos_x = (res_w - zoomed_w) // 2
        
        # 2. PIP Drift Motion
        if is_pan:
            # Drift 4% of screen width over the entire duration
            drift_amount_x = int(res_w * 0.04)
            drift_amount_y = int(res_h * 0.02)
            
            if motion_preset == "wide_pan":
                pos_x += int(drift_amount_x * (t_fg - 0.5))
            elif motion_preset == "diagonal_pan":
                pos_x -= int(drift_amount_x * (t_fg - 0.5))
                pos_y += int(drift_amount_y * (t_fg - 0.5))
            elif motion_preset == "stable_pan":
                pos_y -= int(drift_amount_y * (t_fg - 0.5))
    
    # Glassmorphism Picture-in-Picture (PIP) logic
    y1, y2 = max(0, pos_y), min(res_h, pos_y + zoomed_h)
    x1, x2 = max(0, pos_x), min(res_w, pos_x + zoomed_w)
    
    fg_y1 = 0 if pos_y >= 0 else -pos_y
    fg_y2 = fg_y1 + (y2 - y1)
    fg_x1 = 0 if pos_x >= 0 else -pos_x
    fg_x2 = fg_x1 + (x2 - x1)
    
    if y2 > y1 and x2 > x1:
        # 1. Create local rounded rectangle mask
        radius = min(zoomed_w, zoomed_h) // 25
        local_mask = np.zeros((zoomed_h, zoomed_w), dtype=np.float32)
        cv2.rectangle(local_mask, (radius, 0), (zoomed_w - radius, zoomed_h), 1.0, -1)
        cv2.rectangle(local_mask, (0, radius), (zoomed_w, zoomed_h - radius), 1.0, -1)
        cv2.circle(local_mask, (radius, radius), radius, 1.0, -1)
        cv2.circle(local_mask, (zoomed_w - radius, radius), radius, 1.0, -1)
        cv2.circle(local_mask, (radius, zoomed_h - radius), radius, 1.0, -1)
        cv2.circle(local_mask, (zoomed_w - radius, zoomed_h - radius), radius, 1.0, -1)
        
        # 2. Place it on a full canvas mask to calculate the unclipped drop shadow
        full_mask = np.zeros((res_h, res_w), dtype=np.float32)
        full_mask[y1:y2, x1:x2] = local_mask[fg_y1:fg_y2, fg_x1:fg_x2]
        
        # 3. Apply drop shadow to the background canvas
        shadow_blur = max(25, res_w // 20)
        if shadow_blur % 2 == 0: shadow_blur += 1
        
        shadow = cv2.GaussianBlur(full_mask, (shadow_blur, shadow_blur), 0)
        shadow_3d = np.repeat(shadow[:, :, np.newaxis], 3, axis=2)
        
        # Darken the background behind the PIP window (opacity 0.8)
        canvas = (canvas * (1.0 - shadow_3d * 0.8)).astype(np.uint8)
        
        # 4. Blend the foreground PIP window over the shadowed canvas
        roi = canvas[y1:y2, x1:x2]
        fg_crop = fg_zoomed[fg_y1:fg_y2, fg_x1:fg_x2]
        mask_crop = np.repeat(local_mask[fg_y1:fg_y2, fg_x1:fg_x2, np.newaxis], 3, axis=2)
        
        canvas[y1:y2, x1:x2] = (roi * (1.0 - mask_crop) + fg_crop * mask_crop).astype(np.uint8)
        
    return canvas

def _apply_treatment(canvas, treatment, resolution):
    # e.g., gradient_overlay or mood_overlay
    if treatment.get("layer_type") == "gradient":
        opacity = treatment.get("opacity", 0.5)
        # Create a simple dark gradient from top to bottom
        res_w, res_h = resolution
        gradient = np.linspace(0, 1, res_h).reshape(-1, 1, 1)
        # Multiply blend
        dark_canvas = canvas * (1.0 - gradient * opacity)
        return dark_canvas.astype(np.uint8)
    return canvas

def _apply_effect(canvas, effect, resolution, t):
    # Advanced Cinematic Effects
    layer_type = effect.get("layer_type", "")
    name = effect.get("name", "")
    
    if layer_type == "vignette" or name in ("subtle_vignette", "vignette_dark", "vignette_subtle", "dark_vignette"):
        # Create a radial gradient mask
        res_w, res_h = resolution
        opacity = effect.get("metadata", {}).get("intensity", 0.5)
        
        # We can use a pre-calculated Gaussian kernel to simulate a vignette
        kernel_x = cv2.getGaussianKernel(res_w, res_w/2)
        kernel_y = cv2.getGaussianKernel(res_h, res_h/2)
        kernel = kernel_y * kernel_x.T
        # Normalize kernel
        mask = 255 * kernel / np.linalg.norm(kernel)
        mask = mask / np.max(mask)
        # Invert and scale mask to create dark edges
        vignette_mask = (1.0 - (1.0 - mask) * opacity)
        vignette_mask = np.repeat(vignette_mask[:, :, np.newaxis], 3, axis=2)
        return (canvas * vignette_mask).astype(np.uint8)
        
    elif layer_type == "film_grain" or name in ("grain", "film_grain"):
        # Create a random Gaussian noise matrix
        res_w, res_h = resolution
        intensity = effect.get("metadata", {}).get("intensity", 0.1)
        # Generate noise
        noise = np.random.normal(0, intensity * 255, (res_h, res_w, 3))
        # Add to canvas and clip
        noisy_canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy_canvas

    elif name == "impact_shake" or layer_type == "camera_shake":
        # Camera shake — random X/Y pixel offset via affine warp
        amplitude = effect.get("metadata", {}).get("amplitude", 4)
        dx = np.random.randint(-amplitude, amplitude + 1)
        dy = np.random.randint(-amplitude, amplitude + 1)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        res_w, res_h = resolution
        return cv2.warpAffine(canvas, M, (res_w, res_h), borderMode=cv2.BORDER_REFLECT)

    elif name == "flash_frame" or layer_type == "flash":
        # Brief white flash — blend toward white for 2 frames (66ms at 30fps)
        flash_duration = effect.get("metadata", {}).get("duration", 0.066)
        # `t` is the normalized progress within this slide (0.0 to 1.0)
        # Flash at the very start of the effect layer
        if t < flash_duration:
            alpha = 1.0 - (t / flash_duration)  # Fade from white to normal
            white = np.full_like(canvas, 255)
            return cv2.addWeighted(white, alpha, canvas, 1.0 - alpha, 0)
        return canvas

    elif name == "red_eye_flash" or layer_type == "red_flash":
        # Brief red overlay for menace/threat moments
        flash_duration = effect.get("metadata", {}).get("duration", 0.15)
        if t < flash_duration:
            alpha = 0.35 * (1.0 - (t / flash_duration))  # Fade from red tint to normal
            red_overlay = np.zeros_like(canvas)
            red_overlay[:, :, 2] = 200  # Red channel in BGR
            return cv2.addWeighted(red_overlay, alpha, canvas, 1.0 - alpha, 0)
        return canvas

    elif name == "desaturation" or layer_type == "color_filter" and effect.get("metadata", {}).get("saturation", 1.0) < 0.8:
        # Drain color for grief/trauma — convert to gray and blend back
        saturation = effect.get("metadata", {}).get("saturation", 0.3)
        brightness = effect.get("metadata", {}).get("brightness", 0.85)
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Blend between full color and grayscale based on saturation
        result = cv2.addWeighted(canvas, saturation, gray_bgr, 1.0 - saturation, 0)
        if brightness != 1.0:
            result = np.clip(result.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
        return result

    elif name == "chromatic_shift" or layer_type == "distortion":
        # Split BGR channels, offset R channel by a few pixels, remerge
        offset = effect.get("metadata", {}).get("offset", 2)
        b, g, r = cv2.split(canvas)
        # Shift red channel right
        M = np.float32([[1, 0, offset], [0, 1, 0]])
        res_w, res_h = resolution
        r_shifted = cv2.warpAffine(r, M, (res_w, res_h), borderMode=cv2.BORDER_REFLECT)
        return cv2.merge([b, g, r_shifted])

    return canvas

def _resolve_dimension(val, max_val):
    if isinstance(val, str) and "%" in val:
        return int(max_val * float(val.replace("%", "")) / 100.0)
    elif str(val).lower() == "auto":
        return max_val
    return int(val)

def _render_text(canvas, layer, resolution, emphasis_words):
    text = layer.get("source", "")
    if not text:
        return canvas
        
    res_w, res_h = resolution
    pos = layer.get("position", {})
    layer_type = layer.get("layer_type", "")
    
    if layer_type == "text_tags":
        x0 = 0
        y0 = int(res_h * 0.05)  # Higher up, near the very top margin
        w = res_w
    else:
        x0 = _resolve_dimension(pos.get("x", "0%"), res_w)
        y0 = _resolve_dimension(pos.get("y", "70%"), res_h)
        w = _resolve_dimension(pos.get("width", "100%"), res_w)
    
    layer_name = layer.get("name", "")
    is_title = layer_name == "headline" or layer.get("fit") == "cover"
    is_tags = layer_type == "text_tags"
    
    # Base styling
    font = cv2.FONT_HERSHEY_SIMPLEX
    if is_tags:
        font_scale = 0.65
    else:
        font_scale = 0.90 if is_title else 0.76
    thickness = 2 if not is_tags else 1
    
    # Gold emphasis color in BGR
    accent_color = (88, 198, 238)
    text_color = (245, 245, 245)
    
    from app.utils.slides.generate_slideshow import _wrap_overlay_text
    # max_chars rough estimate based on font width
    max_chars = int(w / (18 * font_scale))
    lines = _wrap_overlay_text(text.upper() if is_title else text, max_chars)
    
    if not lines:
        return canvas
        
    line_h = int(42 * font_scale) + 16
    
    # Process emphasis words
    emp_lower = [str(w).lower() for w in emphasis_words]
    
    # If rendering tags, make everything the accent color
    if is_tags:
        emp_lower = [w.strip().lower() for w in text.split(",")]
    
    # Draw transparent panel backdrop
    panel_h = line_h * len(lines) + 40
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, min(res_h, y0 + panel_h)), (10, 12, 18), -1)
    canvas = cv2.addWeighted(overlay, 0.66, canvas, 0.34, 0)
    
    y = y0 + line_h
    for line in lines:
        words = line.split(" ")
        # Center the line within the width `w`
        # Calculate total line width first
        total_w = 0
        space_w = int(10 * font_scale)
        for i, word in enumerate(words):
            (tw, th), _ = cv2.getTextSize(word, font, font_scale, thickness)
            total_w += tw
            if i < len(words) - 1:
                total_w += space_w
                
        curr_x = x0 + max(24, (w - total_w) // 2)
        
        for word in words:
            clean_word = "".join(c for c in word.lower() if c.isalnum())
            color = accent_color if clean_word in emp_lower else text_color
            
            (tw, th), _ = cv2.getTextSize(word, font, font_scale, thickness)
            cv2.putText(canvas, word, (curr_x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            curr_x += tw + space_w
            
        y += line_h
        
    return canvas

def _fallback_render(img_bgr, slide_meta, resolution, slide_idx, t):
    from app.utils.slides.generate_slideshow import prepare_ken_burns_canvas, crop_ken_burns_frame
    motion = slide_meta.get("motion_preset", "slow_push")
    # Quick re-implementation of the legacy call for safety
    # Actually, the legacy loop caches the prepared canvas. We should let generate_slideshow do it if fallback.
    return None
