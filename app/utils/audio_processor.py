import os
import uuid
from typing import Optional
from datetime import datetime
import logging
from pathlib import Path
from pydub import AudioSegment
import ffmpeg
from config.config import UPLOADS_DIR, MAX_FILE_SIZE, ALLOWED_EXTENSIONS

logger = logging.getLogger(__name__)

def create_uploads_dir():
    """Create uploads directory if it doesn't exist."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def save_audio_file(file, filename: str) -> str:
    """
    Save audio file with a unique ID and return the path.
    
    Args:
        file: File object from request.files
        filename: Original filename
    
    Returns:
        str: Path to saved file
    """
    if not allowed_file(filename):
        raise ValueError(f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")
    
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"File too large. Maximum size: {MAX_FILE_SIZE/1024/1024:.1f}MB")
    
    # Generate unique filename
    unique_id = uuid.uuid4()
    ext = filename.rsplit(".", 1)[1].lower()
    unique_filename = f"{unique_id}.{ext}"
    
    # Save file
    file_path = UPLOADS_DIR / unique_filename
    file.save(str(file_path))
    
    return str(file_path)

def convert_to_wav(input_path: str, output_path: str) -> None:
    """
    Convert audio file to 16kHz mono WAV format.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save converted WAV file
    """
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")
    except Exception as e:
        logger.error(f"Error converting audio: {str(e)}")
        raise

def get_audio_duration(input_path: str) -> float:
    """
    Get audio duration in seconds.
    
    Args:
        input_path: Path to audio file
    
    Returns:
        float: Duration in seconds
    """
    try:
        probe = ffmpeg.probe(input_path)
        return float(probe['format']['duration'])
    except Exception as e:
        logger.error(f"Error getting audio duration: {str(e)}")
        raise
