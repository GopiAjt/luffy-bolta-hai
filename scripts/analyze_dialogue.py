import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from collections import defaultdict
import re
import japanize_matplotlib
import matplotlib.pyplot as plt
import networkx as nx

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DialogueAnalyzer:
    def __init__(self, subtitles_dir: Path):
        """
        Initialize the dialogue analyzer.
        
        Args:
            subtitles_dir: Directory containing processed subtitle JSON files
        """
        self.subtitles_dir = subtitles_dir
        self.all_episodes_file = subtitles_dir / 'all_episodes.json'
        
    def load_data(self) -> List[Dict]:
        """Load all episodes data from the combined JSON file."""
        with open(self.all_episodes_file, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def extract_character_name(self, text: str) -> str:
        """Extract character name from subtitle text if present."""
        if '(' in text and ')' in text:
            return text[text.find('(')+1:text.find(')')]
        return None
        
    def analyze_conversation_flows(self, episodes: List[Dict]) -> Dict[str, Dict[str, int]]:
        """
        Analyze conversation flows between characters.
        Returns a dictionary of character pairs and their conversation counts.
        """
        conversation_counts = defaultdict(lambda: defaultdict(int))
        
        for episode in episodes:
            current_speaker = None
            for subtitle in episode['subtitles']:
                text = subtitle['text']
                speaker = self.extract_character_name(text)
                
                if speaker:
                    if current_speaker and current_speaker != speaker:
                        # Count conversation between characters
                        conversation_counts[current_speaker][speaker] += 1
                    current_speaker = speaker
                    
        return conversation_counts
        
    def analyze_emotional_content(self, episodes: List[Dict]) -> Dict[str, Dict[str, int]]:
        """
        Analyze emotional content in dialogues using Japanese emotion indicators.
        """
        emotion_indicators = {
            '喜び': ['！', '笑', '嬉', '楽'],
            '怒り': ['！', '怒', '憤'],
            '悲しみ': ['…', '泣', '悲'],
            '驚き': ['！', '！？', 'えっ', 'まさか'],
            '恐れ': ['！', '怖', '恐']
        }
        
        character_emotions = defaultdict(lambda: defaultdict(int))
        
        for episode in episodes:
            for subtitle in episode['subtitles']:
                text = subtitle['text']
                speaker = self.extract_character_name(text)
                
                if speaker:
                    for emotion, indicators in emotion_indicators.items():
                        for indicator in indicators:
                            if indicator in text:
                                character_emotions[speaker][emotion] += 1
                                
        return character_emotions
        
    def plot_conversation_network(self, conversation_counts: Dict[str, Dict[str, int]], 
                                output_dir: Path, min_conversations: int = 5):
        """Plot conversation network between characters."""
        G = nx.DiGraph()
        
        # Add edges for conversations
        for speaker, listeners in conversation_counts.items():
            for listener, count in listeners.items():
                if count >= min_conversations:
                    G.add_edge(speaker, listener, weight=count)
                    
        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(G)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=1000, alpha=0.8)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_family='IPAexGothic')
        
        plt.title('Character Conversation Network')
        plt.axis('off')
        
        output_file = output_dir / 'conversation_network.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_emotional_distribution(self, character_emotions: Dict[str, Dict[str, int]], 
                                  output_dir: Path, top_n: int = 10):
        """Plot emotional distribution for top N characters."""
        # Get top N characters by total emotional expressions
        character_totals = {
            char: sum(emotions.values())
            for char, emotions in character_emotions.items()
        }
        top_characters = sorted(
            character_totals.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        # Prepare data for plotting
        emotions = list(next(iter(character_emotions.values())).keys())
        characters = [char for char, _ in top_characters]
        
        # Create stacked bar chart
        plt.figure(figsize=(15, 8))
        bottom = [0] * len(characters)
        
        for emotion in emotions:
            values = [character_emotions[char].get(emotion, 0) for char in characters]
            plt.bar(characters, values, bottom=bottom, label=emotion)
            bottom = [b + v for b, v in zip(bottom, values)]
            
        plt.title('Emotional Distribution by Character')
        plt.xlabel('Character')
        plt.ylabel('Number of Emotional Expressions')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        output_file = output_dir / 'emotional_distribution.png'
        plt.savefig(output_file)
        plt.close()
        
    def generate_analysis(self, output_dir: Path):
        """Generate comprehensive dialogue analysis."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        episodes = self.load_data()
        
        # Analyze conversations and emotions
        conversation_flows = self.analyze_conversation_flows(episodes)
        emotional_content = self.analyze_emotional_content(episodes)
        
        # Generate visualizations
        self.plot_conversation_network(conversation_flows, output_dir)
        self.plot_emotional_distribution(emotional_content, output_dir)
        
        # Save analysis results
        analysis_results = {
            'conversation_flows': conversation_flows,
            'emotional_content': emotional_content
        }
        
        with open(output_dir / 'dialogue_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Dialogue analysis complete. Results saved to {output_dir}")

def main():
    # Set up directories
    base_dir = Path(__file__).parent.parent
    subtitles_dir = base_dir / 'data' / 'processed' / 'subtitles'
    output_dir = base_dir / 'data' / 'analysis' / 'dialogue'
    
    # Create analyzer and generate analysis
    analyzer = DialogueAnalyzer(subtitles_dir)
    analyzer.generate_analysis(output_dir)

if __name__ == '__main__':
    main() 