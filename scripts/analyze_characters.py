import json
import re
from pathlib import Path
import logging
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import japanize_matplotlib
from typing import Dict, List, Set, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CharacterAnalyzer:
    def __init__(self, input_file: Path, output_dir: Path):
        """
        Initialize the character analyzer.
        
        Args:
            input_file: Path to the combined episodes JSON file
            output_dir: Directory to save analysis results
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Common Japanese honorifics and suffixes
        self.honorifics = ['さん', 'くん', 'ちゃん', '様', '殿', '氏']
        
        # Common character name patterns
        self.name_patterns = [
            r'([一-龯ぁ-んァ-ン]{2,4})(?:さん|くん|ちゃん|様|殿|氏)?',  # Japanese names with honorifics
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)',  # Western names
            r'([一-龯ぁ-んァ-ン]{2,4})',  # Japanese names without honorifics
        ]
        
    def extract_character_names(self, text: str) -> Set[str]:
        """
        Extract character names from text using various patterns.
        
        Args:
            text: The text to analyze
            
        Returns:
            Set of potential character names
        """
        names = set()
        
        # Skip if text is too short or contains only symbols
        if len(text) < 2 or not any(c.isalnum() for c in text):
            return names
            
        # Extract names using patterns
        for pattern in self.name_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                name = match.group(1)
                # Filter out common non-name words
                if len(name) >= 2 and not any(word in name for word in ['です', 'ます', 'です', 'でした']):
                    names.add(name)
                    
        return names
        
    def analyze_character_mentions(self, episodes: List[Dict]) -> Dict[str, Dict]:
        """
        Analyze character mentions across episodes.
        
        Args:
            episodes: List of episode data
            
        Returns:
            Dictionary of character statistics
        """
        character_stats = defaultdict(lambda: {
            'mentions': 0,
            'episodes': set(),
            'co_occurrences': defaultdict(int),
            'first_appearance': None,
            'last_appearance': None
        })
        
        for episode in episodes:
            episode_num = episode['episode_info']['episode_number']
            
            for subtitle in episode['subtitles']:
                text = subtitle['text']
                names = self.extract_character_names(text)
                
                # Update character statistics
                for name in names:
                    stats = character_stats[name]
                    stats['mentions'] += 1
                    stats['episodes'].add(episode_num)
                    
                    # Update first/last appearance
                    if not stats['first_appearance'] or episode_num < stats['first_appearance']:
                        stats['first_appearance'] = episode_num
                    if not stats['last_appearance'] or episode_num > stats['last_appearance']:
                        stats['last_appearance'] = episode_num
                        
                    # Update co-occurrences
                    for other_name in names:
                        if other_name != name:
                            stats['co_occurrences'][other_name] += 1
                            
        return character_stats
        
    def generate_character_network(self, character_stats: Dict[str, Dict], min_mentions: int = 10) -> nx.Graph:
        """
        Generate a network graph of character relationships.
        
        Args:
            character_stats: Dictionary of character statistics
            min_mentions: Minimum number of mentions to include a character
            
        Returns:
            NetworkX graph of character relationships
        """
        G = nx.Graph()
        
        # Add nodes (characters)
        for name, stats in character_stats.items():
            if stats['mentions'] >= min_mentions:
                G.add_node(name, mentions=stats['mentions'])
                
        # Add edges (co-occurrences)
        for name, stats in character_stats.items():
            if stats['mentions'] >= min_mentions:
                for other_name, co_occurrences in stats['co_occurrences'].items():
                    if other_name in G and co_occurrences >= 3:  # Minimum co-occurrences threshold
                        G.add_edge(name, other_name, weight=co_occurrences)
                        
        return G
        
    def plot_character_network(self, G: nx.Graph, output_file: Path):
        """
        Plot the character network graph.
        
        Args:
            G: NetworkX graph
            output_file: Path to save the plot
        """
        plt.figure(figsize=(20, 20))
        
        # Calculate node sizes based on mentions
        node_sizes = [G.nodes[node]['mentions'] * 10 for node in G.nodes()]
        
        # Calculate edge widths based on co-occurrences
        edge_widths = [G[u][v]['weight'] for u, v in G.edges()]
        
        # Draw the network
        pos = nx.spring_layout(G, k=1, iterations=50)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.7)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title('Character Network in One Piece')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
    def analyze_characters(self):
        """
        Perform the complete character analysis.
        """
        logger.info("Loading episode data...")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            episodes = json.load(f)
            
        logger.info("Analyzing character mentions...")
        character_stats = self.analyze_character_mentions(episodes)
        
        # Convert sets to lists for JSON serialization
        for stats in character_stats.values():
            stats['episodes'] = sorted(list(stats['episodes']))
            stats['co_occurrences'] = dict(stats['co_occurrences'])
            
        # Save character statistics
        stats_file = self.output_dir / 'character_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(character_stats, f, ensure_ascii=False, indent=2)
            
        logger.info("Generating character network...")
        G = self.generate_character_network(character_stats)
        
        # Save network data
        network_file = self.output_dir / 'character_network.json'
        network_data = {
            'nodes': [{'id': node, 'mentions': G.nodes[node]['mentions']} for node in G.nodes()],
            'edges': [{'source': u, 'target': v, 'weight': G[u][v]['weight']} for u, v in G.edges()]
        }
        with open(network_file, 'w', encoding='utf-8') as f:
            json.dump(network_data, f, ensure_ascii=False, indent=2)
            
        # Plot and save network visualization
        plot_file = self.output_dir / 'character_network.png'
        self.plot_character_network(G, plot_file)
        
        logger.info(f"Analysis complete. Results saved to {self.output_dir}")

def main():
    # Set up directories
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / 'data' / 'processed' / 'subtitles' / 'all_episodes.json'
    output_dir = base_dir / 'data' / 'analysis' / 'characters'
    
    # Create analyzer and run analysis
    analyzer = CharacterAnalyzer(input_file, output_dir)
    analyzer.analyze_characters()

if __name__ == '__main__':
    main() 