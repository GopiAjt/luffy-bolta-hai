"""
Global configuration settings for the application.
"""

import os
from pathlib import Path

# Base directory of the application
BASE_DIR = Path(__file__).resolve().parent.parent

# --- PATHS ---
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
IMAGE_SLIDES_DIR = BASE_DIR / "app" / "output" / "image_slides"
EXPRESSIONS_DIR = BASE_DIR / "app" / "static" / "expressions"
COMPILED_VIDEO_DIR = BASE_DIR / "app" / "output" / "compiled_video"

# --- AUDIO ---
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {"mp3", "wav"}

# --- VIDEO ---
VIDEO_RESOLUTION = (1080, 1920)
VIDEO_BACKGROUND_COLOR = "green"

# --- SUBTITLES ---
SUBTITLE_RESOLUTION = '1080x1920'

# --- API ---
HOST = "0.0.0.0"
PORT = 5000
DEBUG = True

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_SLIDES_DIR.mkdir(parents=True, exist_ok=True)
COMPILED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
