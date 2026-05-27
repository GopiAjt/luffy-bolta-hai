import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for better OCR results.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Preprocessed image
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to remove noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Correct skew in the image.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Deskewed image
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find all contours
        contours, _ = cv2.findContours(
            gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # Adjust angle
        if angle < -45:
            angle = 90 + angle
            
        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
        
    except Exception as e:
        logger.error(f"Error deskewing image: {str(e)}")
        raise

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast using CLAHE.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Contrast enhanced image
    """
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels
        merged = cv2.merge((cl, a, b))
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Error enhancing contrast: {str(e)}")
        raise

def detect_speech_bubbles(image: np.ndarray) -> list:
    """
    Detect speech bubbles in the image using contour detection.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        List of speech bubble contours
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours based on shape and size
        speech_bubbles = []
        for contour in contours:
            # Get contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Filter based on area and perimeter
            if area > 1000 and perimeter > 100:
                # Get convex hull
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                
                # Calculate solidity
                solidity = float(area) / hull_area
                
                # Filter based on solidity (speech bubbles are usually solid)
                if solidity > 0.8:
                    speech_bubbles.append(contour)
        
        return speech_bubbles
        
    except Exception as e:
        logger.error(f"Error detecting speech bubbles: {str(e)}")
        raise

def extract_bubble_tail(contour: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Extract the tail point of a speech bubble.
    
    Args:
        contour: Speech bubble contour
        
    Returns:
        Tuple of (x, y) coordinates of the tail point, or None if not found
    """
    try:
        # Get convex hull
        hull = cv2.convexHull(contour, returnPoints=False)
        
        # Get convexity defects
        defects = cv2.convexityDefects(contour, hull)
        
        if defects is None:
            return None
            
        # Find the deepest defect
        max_defect = None
        max_depth = 0
        
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            depth = d / 256.0
            
            if depth > max_depth:
                max_depth = depth
                max_defect = (s, e, f)
        
        if max_defect is None:
            return None
            
        # Get the point of the deepest defect
        s, e, f = max_defect
        tail_point = tuple(contour[f][0])
        
        return tail_point
        
    except Exception as e:
        logger.error(f"Error extracting bubble tail: {str(e)}")
        raise 