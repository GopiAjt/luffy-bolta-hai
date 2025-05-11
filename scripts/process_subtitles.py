import os
from pathlib import Path
import re
import logging
from typing import List, Dict, Tuple
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SubtitleProcessor:
    def __init__(self, input_dirs: List[Path], output_dir: Path):
        """
        Initialize the subtitle processor.
        
        Args:
            input_dirs: List of directories containing SRT files
            output_dir: Directory to save processed text
        """
        self.input_dirs = input_dirs
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_episode_info(self, filename: str) -> Dict[str, str]:
        """
        Extract episode information from filename.
        Example: ワンピース.S10E01.第1000話　圧倒的戦力！麦わらの一味集結.WEBRip.Amazon.ja-jp[sdh].srt
        or One_Piece_1061.srt
        
        Returns:
            Dictionary containing season, episode, and title information
        """
        # Try to extract from One Piece format first
        if filename.startswith('One_Piece_'):
            episode_num = filename.replace('One_Piece_', '').replace('.srt', '')
            return {
                'season': '11',  # Assuming these are from season 11
                'episode': episode_num,
                'episode_number': episode_num,
                'title': f'Chapter {episode_num}'
            }
            
        # Try to extract from original format
        season_match = re.search(r'S(\d+)E(\d+)', filename)
        if not season_match:
            return {}
            
        season = season_match.group(1)
        episode = season_match.group(2)
        
        # Extract episode number and title
        title_match = re.search(r'第(\d+)話　(.+?)\.WEBRip', filename)
        if not title_match:
            return {}
            
        episode_num = title_match.group(1)
        title = title_match.group(2)
        
        return {
            'season': season,
            'episode': episode,
            'episode_number': episode_num,
            'title': title
        }
        
    def process_srt_file(self, file_path: Path) -> List[Dict[str, str]]:
        """
        Process a single SRT file and extract text content.
        
        Args:
            file_path: Path to the SRT file
            
        Returns:
            List of dictionaries containing timestamp and text
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split into subtitle blocks
            blocks = content.strip().split('\n\n')
            
            subtitles = []
            for block in blocks:
                lines = block.split('\n')
                if len(lines) >= 3:  # Number, timestamp, and text
                    number = lines[0]
                    timestamp = lines[1]
                    text = ' '.join(lines[2:])
                    
                    subtitles.append({
                        'number': number,
                        'timestamp': timestamp,
                        'text': text
                    })
                    
            return subtitles
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []
            
    def process_all_files(self):
        """
        Process all SRT files in the input directories.
        """
        processed_data = []
        
        for input_dir in self.input_dirs:
            for file_path in input_dir.glob('*.srt'):
                logger.info(f"Processing file: {file_path.name}")
                
                # Extract episode information
                episode_info = self.extract_episode_info(file_path.name)
                if not episode_info:
                    logger.warning(f"Could not extract episode info from {file_path.name}")
                    continue
                    
                # Process subtitles
                subtitles = self.process_srt_file(file_path)
                
                # Combine episode info with subtitles
                episode_data = {
                    'file_name': file_path.name,
                    'episode_info': episode_info,
                    'subtitles': subtitles
                }
                
                processed_data.append(episode_data)
                
                # Save individual episode data
                output_file = self.output_dir / f"{file_path.stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(episode_data, f, ensure_ascii=False, indent=2)
                    
        # Save combined data
        combined_output = self.output_dir / 'all_episodes.json'
        with open(combined_output, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Processed {len(processed_data)} files")
        logger.info(f"Output saved to {self.output_dir}")

def main():
    # Set up directories
    base_dir = Path(__file__).parent.parent
    input_dirs = [
        base_dir / 'data' / 'raw' / 'extracted_scripts',
        base_dir / 'data' / 'raw'
    ]
    output_dir = base_dir / 'data' / 'processed' / 'subtitles'
    
    # Create processor and process files
    processor = SubtitleProcessor(input_dirs, output_dir)
    processor.process_all_files()

if __name__ == '__main__':
    main() 