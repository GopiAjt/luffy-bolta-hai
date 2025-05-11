import json
from pathlib import Path
import logging
from typing import List, Dict
from google.cloud import translate_v2 as translate
import time
import os
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SubtitleTranslator:
    def __init__(self, input_dirs: List[Path], output_dir: Path, credentials_path: Path = None):
        """
        Initialize the subtitle translator.
        
        Args:
            input_dirs: List of directories containing the subtitle files
            output_dir: Directory to save translated subtitles
            credentials_path: Path to Google Cloud credentials JSON file
        """
        self.input_dirs = input_dirs
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up Google Cloud credentials if provided
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(credentials_path)
            
        # Initialize translation client
        self.translate_client = translate.Client()
        
    def translate_text(self, text: str, target_language: str = 'en') -> str:
        """
        Translate text using Google Cloud Translation API.
        
        Args:
            text: Text to translate
            target_language: Target language code (default: 'en' for English)
            
        Returns:
            Translated text
        """
        try:
            # Skip empty or very short text
            if not text or len(text.strip()) < 2:
                return text
                
            # Skip if text is already in English (contains only ASCII characters)
            if all(ord(c) < 128 for c in text):
                return text
                
            # Add delay to respect API rate limits
            time.sleep(0.1)
            
            result = self.translate_client.translate(
                text,
                target_language=target_language
            )
            
            return result['translatedText']
            
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            return text
            
    def translate_episode(self, episode: Dict) -> Dict:
        """
        Translate all subtitles in an episode.
        
        Args:
            episode: Episode data dictionary
            
        Returns:
            Translated episode data
        """
        translated_episode = episode.copy()
        
        # Translate episode title
        if 'episode_info' in translated_episode and 'title' in translated_episode['episode_info']:
            translated_episode['episode_info']['title'] = self.translate_text(translated_episode['episode_info']['title'])
        
        # Translate each subtitle
        for subtitle in translated_episode['subtitles']:
            subtitle['text'] = self.translate_text(subtitle['text'])
            
        return translated_episode

    def get_episode_number(self, filename: str) -> str:
        """Extract episode number from filename."""
        # Try to match One_Piece_XXXX.srt format
        match = re.search(r'One_Piece_(\d+)', filename)
        if match:
            return match.group(1)
        
        # Try to match ワンピース format
        match = re.search(r'第(\d+)話', filename)
        if match:
            return match.group(1)
            
        return "0000"  # Default if no match found

    def process_srt_file(self, file_path: Path) -> Dict:
        """Process an SRT file and convert it to our JSON format."""
        subtitles = []
        current_subtitle = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                if current_subtitle:
                    subtitles.append(current_subtitle)
                    current_subtitle = None
                continue
                
            # New subtitle block starts with a number
            if line.isdigit():
                if current_subtitle:
                    subtitles.append(current_subtitle)
                current_subtitle = {
                    'number': line,
                    'timestamp': '',
                    'text': ''
                }
                continue
                
            # Timestamp line
            if '-->' in line:
                if current_subtitle:
                    current_subtitle['timestamp'] = line
                continue
                
            # Text line
            if current_subtitle:
                if current_subtitle['text']:
                    current_subtitle['text'] += ' ' + line
                else:
                    current_subtitle['text'] = line
                    
        # Add the last subtitle if exists
        if current_subtitle:
            subtitles.append(current_subtitle)
            
        # Create episode data
        episode_number = self.get_episode_number(file_path.name)
        return {
            'file_name': file_path.name,
            'episode_info': {
                'episode_number': episode_number,
                'title': f"Episode {episode_number}"  # Default title, will be translated
            },
            'subtitles': subtitles
        }
        
    def translate_all_episodes(self):
        """
        Translate all episodes and save the results.
        """
        processed_files = set()
        
        # Process each input directory
        for input_dir in self.input_dirs:
            # Get all files in the input directory
            input_files = list(input_dir.glob('*'))  # Changed from '*.*' to '*'
            logger.info(f"Found {len(input_files)} files in {input_dir}")
            
            # Process each file
            for input_file in input_files:
                try:
                    # Skip directories and non-JSON files
                    if not input_file.is_file() or input_file.suffix.lower() != '.json':
                        continue
                        
                    # Skip if we've already processed this episode number
                    episode_number = self.get_episode_number(input_file.name)
                    if episode_number in processed_files:
                        continue
                        
                    logger.info(f"Processing {input_file.name}")
                    
                    # Load episode data
                    with open(input_file, 'r', encoding='utf-8') as f:
                        episode = json.load(f)
                    
                    # Translate episode
                    translated_episode = self.translate_episode(episode)
                    
                    # Save translated episode
                    output_file = self.output_dir / f"one_piece_{episode_number}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(translated_episode, f, ensure_ascii=False, indent=2)
                        
                    processed_files.add(episode_number)
                    logger.info(f"Saved translation to {output_file}")
                    
                except Exception as e:
                    logger.error(f"Error processing {input_file.name}: {str(e)}")
                    continue

def main():
    # Set up directories
    base_dir = Path(__file__).parent.parent
    input_dirs = [
        base_dir / 'data' / 'processed' / 'subtitles',
        base_dir / 'data' / 'raw'
    ]
    output_dir = base_dir / 'data' / 'processed' / 'subtitles_en'
    credentials_path = base_dir / 'credentials' / 'google_cloud_credentials.json'
    
    # Create translator and run translation
    translator = SubtitleTranslator(input_dirs, output_dir, credentials_path)
    translator.translate_all_episodes()

if __name__ == '__main__':
    main() 