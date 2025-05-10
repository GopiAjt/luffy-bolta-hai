#!/usr/bin/env python3
import argparse
from pathlib import Path
import logging
import sys
import os
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.core.pdf_processor import PDFProcessor
from app.core.text_processor import TextProcessor
from app.core.rag_processor import RAGProcessor
from config.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_DB_DIR,
    OCR_CONFIG, YOLO_CONFIG, VECTOR_DB_CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_chapter(pdf_path: Path, output_dir: Path):
    """
    Process a manga chapter PDF.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save processed data
    """
    try:
        # Initialize processors
        pdf_processor = PDFProcessor(output_dir)
        text_processor = TextProcessor(OCR_CONFIG)
        rag_processor = RAGProcessor(VECTOR_DB_CONFIG)
        
        # Process PDF
        logger.info(f"Processing PDF: {pdf_path}")
        chapter_data = pdf_processor.process_chapter(pdf_path)
        
        # Process text
        logger.info("Extracting and translating text")
        processed_data = []
        for page in chapter_data["pages"]:
            # Create page directory
            page_dir = output_dir / f"page_{page['page_number']:03d}"
            page_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each text region
            for region in page["text_regions"]:
                text = text_processor.extract_text(region[0], region[1])
                hindi_text = text_processor.translate_to_hindi(text)
                
                processed_data.append({
                    "page": page["page_number"],
                    "original_text": text,
                    "hindi_text": hindi_text,
                    "bounding_box": region[1]
                })
        
        # Save processed data
        output_file = output_dir / "processed_data.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        # Add to vector database
        logger.info("Adding to vector database")
        rag_processor.process_chapter(processed_data)
        rag_processor.save_database()
        
        logger.info(f"Processing complete. Output saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error processing chapter: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Process a manga chapter PDF")
    parser.add_argument("input", type=str, help="Path to the input PDF file")
    parser.add_argument("--output", type=str, help="Output directory (optional)")
    args = parser.parse_args()
    
    # Convert paths
    pdf_path = Path(args.input)
    if not pdf_path.exists():
        logger.error(f"Input file not found: {pdf_path}")
        sys.exit(1)
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        # Use default output directory
        chapter_name = pdf_path.stem
        output_dir = PROCESSED_DATA_DIR / chapter_name
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process chapter
    try:
        process_chapter(pdf_path, output_dir)
    except Exception as e:
        logger.error(f"Failed to process chapter: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 