import json
from pathlib import Path
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SubtitleRAG:
    def __init__(self, 
                 data_dir: Path,
                 model_name: str = 'all-MiniLM-L6-v2',
                 collection_name: str = 'one_piece_dialogue'):
        """
        Initialize the RAG system for One Piece subtitles.
        
        Args:
            data_dir: Directory containing translated subtitle files
            model_name: Name of the sentence transformer model to use
            collection_name: Name of the ChromaDB collection
        """
        self.data_dir = data_dir
        self.model = SentenceTransformer(model_name)
        self.collection_name = collection_name
        
        # Initialize ChromaDB client with persistent storage
        persist_directory = str(data_dir.parent / 'vector_db')
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
        except Exception as e:
            logger.warning(f"Error creating persistent client: {e}")
            self.client = chromadb.Client()
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
    def load_subtitles(self) -> List[Dict[str, Any]]:
        """Load all translated subtitle files."""
        subtitle_data = []
        
        for file_path in tqdm(list(self.data_dir.glob('*.json')), desc="Loading subtitle files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    episode_data = json.load(f)
                    
                # Extract episode info
                episode_info = episode_data.get('episode_info', {})
                episode_number = episode_info.get('episode_number', 'unknown')
                title = episode_info.get('title', 'Unknown Title')
                
                # Process each subtitle
                for subtitle in episode_data.get('subtitles', []):
                    subtitle_data.append({
                        'episode_number': episode_number,
                        'title': title,
                        'timestamp': subtitle.get('timestamp', ''),
                        'text': subtitle.get('text', ''),
                        'file_name': episode_data.get('file_name', '')
                    })
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
                
        return subtitle_data
        
    def create_embeddings(self, subtitle_data: List[Dict[str, Any]]) -> None:
        """Create embeddings for subtitles and store in ChromaDB."""
        # Process in batches of 1000
        batch_size = 1000
        total_batches = (len(subtitle_data) + batch_size - 1) // batch_size
        
        for i in tqdm(range(total_batches), desc="Creating embeddings"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(subtitle_data))
            batch_data = subtitle_data[start_idx:end_idx]
            
            # Prepare data for embedding
            texts = [item['text'] for item in batch_data]
            metadatas = [{
                'episode_number': item['episode_number'],
                'title': item['title'],
                'timestamp': item['timestamp'],
                'file_name': item['file_name']
            } for item in batch_data]
            
            # Create embeddings
            embeddings = self.model.encode(texts)
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=[f"sub_{start_idx + j}" for j in range(len(texts))]
            )
        
    def query_dialogue(self, 
                      query: str, 
                      n_results: int = 5,
                      filter_episode: str = None) -> List[Dict[str, Any]]:
        """
        Query the dialogue database.
        
        Args:
            query: The search query
            n_results: Number of results to return
            filter_episode: Optional episode number to filter by
            
        Returns:
            List of matching dialogue entries with metadata
        """
        # Create query embedding
        query_embedding = self.model.encode(query)
        
        # Prepare where filter if episode specified
        where = {"episode_number": filter_episode} if filter_episode else None
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'episode_number': results['metadatas'][0][i]['episode_number'],
                'title': results['metadatas'][0][i]['title'],
                'timestamp': results['metadatas'][0][i]['timestamp'],
                'file_name': results['metadatas'][0][i]['file_name']
            })
            
        return formatted_results 