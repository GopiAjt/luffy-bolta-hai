import json
from pathlib import Path
import logging
from collections import Counter
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SubtitleAnalyzer:
    def __init__(self, subtitles_dir: Path):
        """
        Initialize the subtitle analyzer.
        
        Args:
            subtitles_dir: Directory containing processed subtitle JSON files
        """
        self.subtitles_dir = subtitles_dir
        self.all_episodes_file = subtitles_dir / 'all_episodes.json'
        
    def load_data(self) -> List[Dict]:
        """Load all episodes data from the combined JSON file."""
        with open(self.all_episodes_file, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def analyze_episode_lengths(self, episodes: List[Dict]) -> Dict:
        """Analyze the number of subtitles per episode."""
        lengths = {
            episode['episode_info']['episode_number']: len(episode['subtitles'])
            for episode in episodes
        }
        return lengths
        
    def analyze_character_speaking(self, episodes: List[Dict]) -> Dict[str, int]:
        """Analyze which characters speak the most."""
        character_lines = Counter()
        
        for episode in episodes:
            for subtitle in episode['subtitles']:
                text = subtitle['text']
                # Extract character names from parentheses
                if '(' in text and ')' in text:
                    char_name = text[text.find('(')+1:text.find(')')]
                    character_lines[char_name] += 1
                    
        return dict(character_lines.most_common(20))
        
    def analyze_season_stats(self, episodes: List[Dict]) -> Dict[str, Dict]:
        """Analyze statistics per season."""
        season_stats = {}
        
        for episode in episodes:
            season = episode['episode_info']['season']
            if season not in season_stats:
                season_stats[season] = {
                    'episode_count': 0,
                    'total_subtitles': 0,
                    'avg_subtitles_per_episode': 0
                }
                
            season_stats[season]['episode_count'] += 1
            season_stats[season]['total_subtitles'] += len(episode['subtitles'])
            
        # Calculate averages
        for season in season_stats:
            stats = season_stats[season]
            stats['avg_subtitles_per_episode'] = (
                stats['total_subtitles'] / stats['episode_count']
            )
            
        return season_stats
        
    def plot_episode_lengths(self, lengths: Dict[str, int], output_dir: Path):
        """Plot episode lengths as a bar chart."""
        plt.figure(figsize=(15, 6))
        episodes = list(lengths.keys())
        counts = list(lengths.values())
        
        plt.bar(episodes, counts)
        plt.title('Number of Subtitles per Episode')
        plt.xlabel('Episode Number')
        plt.ylabel('Number of Subtitles')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_file = output_dir / 'episode_lengths.png'
        plt.savefig(output_file)
        plt.close()
        
    def plot_character_speaking(self, character_lines: Dict[str, int], output_dir: Path):
        """Plot top speaking characters as a bar chart."""
        plt.figure(figsize=(12, 6))
        characters = list(character_lines.keys())
        counts = list(character_lines.values())
        
        plt.bar(characters, counts)
        plt.title('Top 20 Speaking Characters')
        plt.xlabel('Character')
        plt.ylabel('Number of Lines')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_file = output_dir / 'character_speaking.png'
        plt.savefig(output_file)
        plt.close()
        
    def generate_report(self, output_dir: Path):
        """Generate a comprehensive analysis report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        episodes = self.load_data()
        
        # Analyze data
        episode_lengths = self.analyze_episode_lengths(episodes)
        character_lines = self.analyze_character_speaking(episodes)
        season_stats = self.analyze_season_stats(episodes)
        
        # Generate plots
        self.plot_episode_lengths(episode_lengths, output_dir)
        self.plot_character_speaking(character_lines, output_dir)
        
        # Save statistics to JSON
        stats = {
            'total_episodes': len(episodes),
            'episode_lengths': episode_lengths,
            'top_speaking_characters': character_lines,
            'season_statistics': season_stats
        }
        
        with open(output_dir / 'subtitle_statistics.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Analysis complete. Results saved to {output_dir}")

def main():
    # Set up directories
    base_dir = Path(__file__).parent.parent
    subtitles_dir = base_dir / 'data' / 'processed' / 'subtitles'
    output_dir = base_dir / 'data' / 'analysis' / 'subtitles'
    
    # Create analyzer and generate report
    analyzer = SubtitleAnalyzer(subtitles_dir)
    analyzer.generate_report(output_dir)

if __name__ == '__main__':
    main() 