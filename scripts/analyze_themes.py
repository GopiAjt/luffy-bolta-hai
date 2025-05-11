import json
from pathlib import Path
import logging
from typing import Dict, List, Set
from collections import Counter, defaultdict
import re
import japanize_matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import MeCab

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ThemeAnalyzer:
    def __init__(self, subtitles_dir: Path):
        """
        Initialize the theme analyzer.
        
        Args:
            subtitles_dir: Directory containing processed subtitle JSON files
        """
        self.subtitles_dir = subtitles_dir
        self.all_episodes_file = subtitles_dir / 'all_episodes.json'
        self.mecab = MeCab.Tagger("-Owakati")
        
        # Common Japanese stop words
        self.stop_words = {
            'の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'も',
            'へ', 'から', 'まで', 'や', 'など', 'な', 'だ', 'か', 'が',
            'です', 'ます', 'です', 'ました', 'です', 'ね', 'よ', 'わ',
            'さ', 'な', 'だ', 'か', 'が', 'です', 'ます', 'です', 'ました'
        }
        
    def load_data(self) -> List[Dict]:
        """Load all episodes data from the combined JSON file."""
        with open(self.all_episodes_file, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize Japanese text using MeCab."""
        # Remove character names in parentheses
        text = re.sub(r'\([^)]*\)', '', text)
        
        # Tokenize
        tokens = self.mecab.parse(text).strip().split()
        
        # Filter out stop words and short tokens
        return [token for token in tokens 
                if token not in self.stop_words and len(token) > 1]
                
    def analyze_key_phrases(self, episodes: List[Dict], min_freq: int = 5) -> Dict[str, int]:
        """Analyze frequently occurring phrases in the subtitles."""
        phrase_counts = Counter()
        
        for episode in episodes:
            for subtitle in episode['subtitles']:
                text = subtitle['text']
                tokens = self.tokenize_text(text)
                
                # Count 2-gram phrases
                for i in range(len(tokens) - 1):
                    phrase = f"{tokens[i]} {tokens[i+1]}"
                    phrase_counts[phrase] += 1
                    
        return {phrase: count for phrase, count in phrase_counts.items() 
                if count >= min_freq}
                
    def analyze_episode_themes(self, episodes: List[Dict]) -> Dict[str, List[str]]:
        """Analyze themes for each episode based on frequent phrases."""
        episode_themes = {}
        
        for episode in episodes:
            episode_num = episode['episode_info']['episode_number']
            phrase_counts = Counter()
            
            for subtitle in episode['subtitles']:
                text = subtitle['text']
                tokens = self.tokenize_text(text)
                
                # Count individual words
                phrase_counts.update(tokens)
                
            # Get top 5 most frequent words as themes
            themes = [word for word, _ in phrase_counts.most_common(5)]
            episode_themes[episode_num] = themes
            
        return episode_themes
        
    def analyze_character_themes(self, episodes: List[Dict]) -> Dict[str, List[str]]:
        """Analyze themes associated with each character."""
        character_themes = defaultdict(Counter)
        
        for episode in episodes:
            for subtitle in episode['subtitles']:
                text = subtitle['text']
                if '(' in text and ')' in text:
                    char_name = text[text.find('(')+1:text.find(')')]
                    tokens = self.tokenize_text(text)
                    character_themes[char_name].update(tokens)
                    
        # Get top 5 themes for each character
        return {
            char: [word for word, _ in themes.most_common(5)]
            for char, themes in character_themes.items()
        }
        
    def generate_wordcloud(self, text_data: Dict[str, int], output_file: Path):
        """Generate a word cloud from text data."""
        wordcloud = WordCloud(
            font_path='/usr/share/fonts/truetype/fonts-japanese-gothic.ttf',
            width=1200,
            height=800,
            background_color='white'
        )
        
        wordcloud.generate_from_frequencies(text_data)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_theme_evolution(self, episode_themes: Dict[str, List[str]], output_dir: Path):
        """Plot the evolution of themes across episodes."""
        # Get unique themes
        all_themes = set()
        for themes in episode_themes.values():
            all_themes.update(themes)
            
        # Create theme presence matrix
        episodes = sorted(episode_themes.keys())
        theme_matrix = []
        
        for theme in all_themes:
            presence = [1 if theme in episode_themes[ep] else 0 for ep in episodes]
            theme_matrix.append(presence)
            
        # Plot heatmap
        plt.figure(figsize=(20, 10))
        plt.imshow(theme_matrix, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Theme Presence')
        
        plt.yticks(range(len(all_themes)), all_themes)
        plt.xticks(range(len(episodes)), episodes, rotation=45)
        
        plt.title('Theme Evolution Across Episodes')
        plt.xlabel('Episode Number')
        plt.ylabel('Theme')
        plt.tight_layout()
        
        output_file = output_dir / 'theme_evolution.png'
        plt.savefig(output_file)
        plt.close()
        
    def generate_analysis(self, output_dir: Path):
        """Generate comprehensive theme analysis."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        episodes = self.load_data()
        
        # Analyze themes
        key_phrases = self.analyze_key_phrases(episodes)
        episode_themes = self.analyze_episode_themes(episodes)
        character_themes = self.analyze_character_themes(episodes)
        
        # Generate visualizations
        self.generate_wordcloud(key_phrases, output_dir / 'key_phrases_wordcloud.png')
        self.plot_theme_evolution(episode_themes, output_dir)
        
        # Save analysis results
        analysis_results = {
            'key_phrases': key_phrases,
            'episode_themes': episode_themes,
            'character_themes': character_themes
        }
        
        with open(output_dir / 'theme_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Theme analysis complete. Results saved to {output_dir}")

def main():
    # Set up directories
    base_dir = Path(__file__).parent.parent
    subtitles_dir = base_dir / 'data' / 'processed' / 'subtitles'
    output_dir = base_dir / 'data' / 'analysis' / 'themes'
    
    # Create analyzer and generate analysis
    analyzer = ThemeAnalyzer(subtitles_dir)
    analyzer.generate_analysis(output_dir)

if __name__ == '__main__':
    main() 