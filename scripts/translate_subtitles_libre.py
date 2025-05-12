import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LibreSubtitleTranslator:
    def __init__(self, input_dirs: List[Path], output_dir: Path):
        """Initialize the translator with input and output directories."""
        self.input_dirs = input_dirs
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_url = "http://127.0.0.1:5000/translate"
        
    def translate_text(self, text: str, source_lang: str = "ja", target_lang: str = "en") -> str:
        """Translate text using LibreTranslate API."""
        try:
            response = requests.post(
                self.api_url,
                json={
                    "q": text,
                    "source": source_lang,
                    "target": target_lang,
                    "format": "text"
                }
            )
            response.raise_for_status()
            return response.json()["translatedText"]
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text

    def get_episode_number(self, filename: str) -> str:
        """Extract episode number from filename."""
        import re
        # Try to match patterns like "One_Piece_1061" or "第1061話"
        match = re.search(r'(?:One_Piece_|第)(\d+)', filename)
        if match:
            return match.group(1)
        return ""

    def process_srt_file(self, file_path: Path) -> Dict[str, Any]:
        """Process SRT file and convert to JSON format."""
        subtitles = []
        current_subtitle = None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if not line:
                    if current_subtitle:
                        subtitles.append(current_subtitle)
                        current_subtitle = None
                    continue
                    
                if not current_subtitle:
                    current_subtitle = {"index": line, "timestamp": "", "text": ""}
                elif not current_subtitle["timestamp"]:
                    current_subtitle["timestamp"] = line
                else:
                    current_subtitle["text"] = line
                    
            if current_subtitle:
                subtitles.append(current_subtitle)
                
            return {
                "file_name": file_path.name,
                "subtitles": subtitles
            }
        except Exception as e:
            logger.error(f"Error processing SRT file {file_path}: {str(e)}")
            return None

    def translate_episode(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Translate an episode's subtitles."""
        translated_subtitles = []
        
        for subtitle in tqdm(episode_data["subtitles"], desc="Translating subtitles"):
            translated_text = self.translate_text(subtitle["text"])
            translated_subtitles.append({
                "index": subtitle["index"],
                "timestamp": subtitle["timestamp"],
                "text": translated_text
            })
            
        return {
            "file_name": episode_data["file_name"],
            "subtitles": translated_subtitles
        }

    def translate_all_episodes(self):
        """Process and translate all episodes from input directories."""
        processed_episodes = set()
        
        for input_dir in self.input_dirs:
            logger.info(f"Processing files from {input_dir}")
            files = list(input_dir.glob("*.srt"))
            logger.info(f"Found {len(files)} files in {input_dir}")
            
            for file_path in tqdm(files, desc="Processing episodes"):
                episode_number = self.get_episode_number(file_path.name)
                if not episode_number or episode_number in processed_episodes:
                    continue
                    
                processed_episodes.add(episode_number)
                logger.info(f"Processing episode {episode_number}")
                
                # Process SRT file
                episode_data = self.process_srt_file(file_path)
                if not episode_data:
                    continue
                    
                # Translate episode
                translated_episode = self.translate_episode(episode_data)
                
                # Save translated episode
                output_file = self.output_dir / f"one_piece_{episode_number}_libre.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(translated_episode, f, ensure_ascii=False, indent=2)
                    
                logger.info(f"Saved translation to {output_file}")

def main():
    # Define directories
    base_dir = Path("data")
    input_dirs = [
        base_dir / "raw",
        base_dir / "processed/subtitles"
    ]
    output_dir = base_dir / "processed/subtitles_en_libre"
    
    # Initialize translator
    translator = LibreSubtitleTranslator(input_dirs, output_dir)
    
    # Process and translate all episodes
    translator.translate_all_episodes()

if __name__ == "__main__":
    main() 