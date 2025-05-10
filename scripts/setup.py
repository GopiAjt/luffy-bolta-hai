#!/usr/bin/env python3
import os
from pathlib import Path
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_env_file():
    """Create .env file with default values."""
    env_content = """# API Keys
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Application Settings
DEBUG=False
LOG_LEVEL=INFO

# Model Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TRANSLATION_MODEL=Helsinki-NLP/opus-mt-ja-hi

# Database Settings
VECTOR_DB_DIR=data/vector_db
COLLECTION_NAME=manga_dialogues

# Processing Settings
CHUNK_SIZE=500
CHUNK_OVERLAP=50
MAX_RETRIEVAL_RESULTS=5
"""
    
    env_path = Path(".env")
    if env_path.exists():
        logger.warning(".env file already exists. Skipping creation.")
        return
    
    with open(env_path, "w") as f:
        f.write(env_content)
    logger.info("Created .env file with default values.")

def create_directories():
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/processed",
        "data/vector_db",
        "data/scripts",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import torch
        import transformers
        import langchain
        import chromadb
        import pymupdf
        import pdf2image
        import cv2  # opencv-python
        import pytesseract
        import easyocr
        import ultralytics
        logger.info("All required Python packages are installed.")
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        logger.error("Please install all required packages using: pip install -r requirements.txt")
        sys.exit(1)

def check_tesseract():
    """Check if Tesseract OCR is installed."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        logger.info("Tesseract OCR is installed.")
    except Exception as e:
        logger.error("Tesseract OCR is not installed or not found.")
        logger.error("Please install Tesseract OCR:")
        logger.error("Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-jpn")
        logger.error("macOS: brew install tesseract tesseract-lang")
        sys.exit(1)

def main():
    """Main setup function."""
    logger.info("Starting setup...")
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Check dependencies
    check_dependencies()
    
    # Check Tesseract
    check_tesseract()
    
    logger.info("Setup complete!")
    logger.info("\nNext steps:")
    logger.info("1. Edit .env file with your API keys")
    logger.info("2. Install Tesseract OCR if not already installed")
    logger.info("3. Run the API server: uvicorn app.api.main:app --reload")
    logger.info("4. Process a chapter: python scripts/process_chapter.py path/to/chapter.pdf")
    logger.info("5. Generate a script: python scripts/generate_script.py <chapter> <page>")

if __name__ == "__main__":
    main() 