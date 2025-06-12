#!/usr/bin/env python3
import argparse
from pathlib import Path
import logging
import sys
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.core.subtitle_rag import SubtitleRAG
from app.utils.text_processing import format_dialogue, merge_similar_dialogues
from config.config import VECTOR_DB_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_script(episode: int, output_dir: Path):
    """
    Generate Hindi voiceover script for a specific episode.
    
    Args:
        episode: Episode number
        output_dir: Directory to save the script
    """
    try:
        # Initialize RAG processor
        rag_processor = SubtitleRAG(
            data_dir=Path("data/processed/subtitles"),
            collection_name="one_piece_dialogue"
        )
        
        # Query episode dialogue
        logger.info(f"Retrieving dialogue for episode {episode}")
        results = rag_processor.query_dialogue(
            query=f"Episode {episode} dialogue",
            n_results=50,
            filter_episode=str(episode)
        )
        
        if not results:
            logger.error(f"No dialogue found for episode {episode}")
            sys.exit(1)
        
        # Generate script
        logger.info("Generating script")
        script = []
        for item in results:
            # Format dialogue
            dialogue = format_dialogue(
                "Character",  # We don't have character info in subtitles
                item["text"]
            )
            
            script.append({
                "character": "Character",  # We don't have character info in subtitles
                "dialogue": dialogue,
                "timestamp": item["timestamp"],
                "context": None  # We don't have context in subtitles
            })
        
        # Merge similar dialogues
        script = merge_similar_dialogues(script)
        
        # Save script
        output_file = output_dir / f"script_episode_{episode:03d}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "episode": episode,
                "script": script
            }, f, ensure_ascii=False, indent=2)
        
        # Print script
        print("\nGenerated Hindi Voiceover Script:")
        print("=" * 50)
        print(f"Episode {episode}")
        print("=" * 50)
        for item in script:
            print(f"\n[{item['timestamp']}] {item['dialogue']}")
        print("\n" + "=" * 50)
        
        logger.info(f"Script saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error generating script: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Generate Hindi voiceover script")
    parser.add_argument("episode", type=int, help="Episode number")
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
        generate_script(args.episode, output_dir)
    except Exception as e:
        logger.error(f"Failed to generate script: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
 