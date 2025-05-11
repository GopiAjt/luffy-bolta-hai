from pathlib import Path
from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_DB_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# OCR Configuration
OCR_CONFIG = {
    "lang": "jpn+eng",  # Japanese + English
    "config": "--psm 6",  # Assume uniform text block
    "dpi": 300,
}

# YOLOv8 Configuration
YOLO_CONFIG = {
    "model_path": "kitsumed/yolov8m_seg-speech-bubble",
    "conf_threshold": 0.25,
    "iou_threshold": 0.45,
}

# Vector Database Configuration
VECTOR_DB_CONFIG = {
    "collection_name": "manga_dialogues",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "vector_db_dir": str(VECTOR_DB_DIR)
}

# LLM Configuration
LLM_CONFIG = {
    "model_name": "gpt-4",  # or your preferred model
    "temperature": 0.7,
    "max_tokens": 1000,
}

# API Configuration
API_CONFIG = {
    "title": "One Piece Hindi Voiceover API",
    "description": "API for generating Hindi voiceover scripts from One Piece manga",
    "version": "1.0.0",
    "debug": os.getenv("DEBUG", "False").lower() == "true",
}

# Character-specific configurations
CHARACTER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "luffy": {
        "tone": "energetic",
        "speech_pattern": "casual",
        "catchphrase": "Gomu Gomu no...",
    },
    "zoro": {
        "tone": "serious",
        "speech_pattern": "formal",
        "catchphrase": "I'll become the greatest swordsman!",
    },
    # Add more characters as needed
}

# File paths
MODEL_PATHS = {
    "yolo": str(BASE_DIR / "models" / "yolov8m_seg-speech-bubble.pt"),
    "ocr": str(BASE_DIR / "models" / "tesseract"),
}

# API Keys and Secrets
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY", ""),
    "huggingface": os.getenv("HUGGINGFACE_API_KEY", ""),
} 