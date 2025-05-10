#!/usr/bin/env python3
import argparse
from pathlib import Path
import logging
import sys
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.core.rag_processor import RAGProcessor
from app.utils.text_processing import format_dialogue, merge_similar_dialogues
from config.config import VECTOR_DB_CONFIG, LLM_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_script(chapter: int, page: int, output_dir: Path):
    """
    Generate Hindi voiceover script for a specific page.
    
    Args:
        chapter: Chapter number
        page: Page number
        output_dir: Directory to save the script
    """
    try:
        # Initialize RAG processor
        rag_processor = RAGProcessor(VECTOR_DB_CONFIG)
        
        # Load page data
        page_path = Path(f"data/processed/chapter_{chapter:03d}/page_{page:03d}/processed_data.json")
        if not page_path.exists():
            logger.error(f"Page data not found: {page_path}")
            sys.exit(1)
        
        with open(page_path, "r", encoding="utf-8") as f:
            page_data = json.load(f)
        
        # Get context from vector database
        logger.info("Retrieving context from vector database")
        context = rag_processor.get_context_for_generation(
            " ".join(item["hindi_text"] for item in page_data)
        )
        
        # Generate script
        logger.info("Generating script")
        script = []
        for item in page_data:
            # Get character-specific context
            char_context = rag_processor.get_context_for_generation(
                item["hindi_text"],
                n_results=3
            )
            
            # Format dialogue
            dialogue = format_dialogue(
                item.get("character", "Unknown"),
                item["hindi_text"]
            )
            
            script.append({
                "character": item.get("character", "Unknown"),
                "dialogue": dialogue,
                "context": char_context
            })
        
        # Merge similar dialogues
        script = merge_similar_dialogues(script)
        
        # Save script
        output_file = output_dir / f"script_chapter_{chapter:03d}_page_{page:03d}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "chapter": chapter,
                "page": page,
                "script": script,
                "context": context
            }, f, ensure_ascii=False, indent=2)
        
        # Print script
        print("\nGenerated Hindi Voiceover Script:")
        print("=" * 50)
        print(f"Chapter {chapter}, Page {page}")
        print("=" * 50)
        for item in script:
            print(f"\n{item['dialogue']}")
            if item.get("context"):
                print("\nContext:")
                print(item["context"])
        print("\n" + "=" * 50)
        
        logger.info(f"Script saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error generating script: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Generate Hindi voiceover script")
    parser.add_argument("chapter", type=int, help="Chapter number")
    parser.add_argument("page", type=int, help="Page number")
    parser.add_argument("--output", type=str, help="Output directory (optional)")
    args = parser.parse_args()
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        # Use default output directory
        output_dir = Path("data/scripts")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate script
    try:
        generate_script(args.chapter, args.page, output_dir)
    except Exception as e:
        logger.error(f"Failed to generate script: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
 