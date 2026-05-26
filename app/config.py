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
# Expression overlays use only EXPRESSIONS_DIR ({emotion}.png). Vivre Card is for image slides only.
EXPRESSION_ASSETS_SOURCE = os.getenv("EXPRESSION_ASSETS_SOURCE", "static").strip().lower()
USE_STATIC_EXPRESSIONS_ONLY = EXPRESSION_ASSETS_SOURCE not in {"vivre", "vivre_card", "both"}
# Vivre Card PNG pack (default: app/data/vivre-card); override via VIVRE_CARD_ASSETS_DIR in .env
_DEFAULT_VIVRE_DIR = BASE_DIR / "data" / "vivre-card"
VIVRE_CARD_ASSETS_DIR = Path(
    os.getenv("VIVRE_CARD_ASSETS_DIR", str(_DEFAULT_VIVRE_DIR))
).expanduser()
NARRATOR_CHARACTER = os.getenv("NARRATOR_CHARACTER", "luffy").strip().lower()
OPARCHIVE_ENABLED = os.getenv("OPARCHIVE_ENABLED", "true").lower() not in {"0", "false", "no"}
COMPILED_VIDEO_DIR = BASE_DIR / "output" / "compiled_video"
MANGA_PDF_DIR = BASE_DIR / "output" / "manga_pdf"
BACKGROUND_MUSIC_DIR = BASE_DIR / "data"
TRANSITION_SFX_DIR = BASE_DIR / "data" / "sfx"
ENABLE_TRANSITION_SFX = os.getenv("ENABLE_TRANSITION_SFX", "true").lower() not in {"0", "false", "no"}
TRANSITION_SFX_VOLUME = float(os.getenv("TRANSITION_SFX_VOLUME", "1.25"))
TRANSITION_SFX_MIX_WEIGHT = float(os.getenv("TRANSITION_SFX_MIX_WEIGHT", "2.5"))

# --- AUDIO ---
MAX_AUDIO_FILE_SIZE_MB = float(os.getenv("MAX_AUDIO_FILE_SIZE_MB", "100"))
MAX_FILE_SIZE = int(MAX_AUDIO_FILE_SIZE_MB * 1024 * 1024)
ALLOWED_EXTENSIONS = {"mp3", "wav"}
MAX_PDF_SIZE = 100 * 1024 * 1024  # 100MB
ENABLE_BACKGROUND_MUSIC = os.getenv("ENABLE_BACKGROUND_MUSIC", "true").lower() not in {"0", "false", "no"}
BACKGROUND_MUSIC_VOLUME = float(os.getenv("BACKGROUND_MUSIC_VOLUME", "0.16"))

# --- VIDEO PROFILES ---
DEFAULT_VIDEO_PROFILE = "short_vertical"
DEFAULT_VISUAL_STYLE = "clean_pro"
VISUAL_STYLES = {"clean_pro", "manga_hype", "emotional", "dark_lore", "action"}
VIDEO_PROFILES = {
    "short_vertical": {
        "label": "Short vertical",
        "video_resolution": (1080, 1920),
        "subtitle_resolution": "1080x1920",
    },
    "long_youtube": {
        "label": "Long YouTube",
        "video_resolution": (1920, 1080),
        "subtitle_resolution": "1920x1080",
    },
}


def normalize_video_profile(video_profile: str = None) -> str:
    profile = (video_profile or DEFAULT_VIDEO_PROFILE).strip().lower()
    return profile if profile in VIDEO_PROFILES else DEFAULT_VIDEO_PROFILE


def get_video_profile_config(video_profile: str = None) -> dict:
    return VIDEO_PROFILES[normalize_video_profile(video_profile)]


def normalize_visual_style(visual_style: str = None) -> str:
    style = (visual_style or DEFAULT_VISUAL_STYLE).strip().lower()
    return style if style in VISUAL_STYLES else DEFAULT_VISUAL_STYLE


# Backwards-compatible defaults for callers that have not been profile-aware yet.
VIDEO_RESOLUTION = VIDEO_PROFILES[DEFAULT_VIDEO_PROFILE]["video_resolution"]
VIDEO_BACKGROUND_COLOR = "green"

# --- SUBTITLES ---
SUBTITLE_RESOLUTION = VIDEO_PROFILES[DEFAULT_VIDEO_PROFILE]["subtitle_resolution"]

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
MANGA_PDF_DIR.mkdir(parents=True, exist_ok=True)
TRANSITION_SFX_DIR.mkdir(parents=True, exist_ok=True)
