"""
Global configuration settings for the application.
"""

import os
from pathlib import Path

# Base directory of the application
BASE_DIR = Path(__file__).resolve().parent

# --- PATHS ---
UPLOADS_DIR = BASE_DIR / "output" / "data"
IMAGE_SLIDES_DIR = BASE_DIR / "output" / "image_slides"
EXPRESSIONS_DIR = BASE_DIR / "static" / "expressions"
COMPILED_VIDEO_DIR = BASE_DIR / "output" / "compiled_video"

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

# --- VECTOR DATABASE ---
VECTOR_DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "vectordb",
    "user": "postgres",
    "password": "postgres"
}

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_SLIDES_DIR.mkdir(parents=True, exist_ok=True)
COMPILED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
