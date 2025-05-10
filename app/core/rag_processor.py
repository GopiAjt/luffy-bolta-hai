from typing import List, Dict, Any
import logging
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
from ..utils.text_processing import chunk_text

logger = logging.getLogger(__name__)

class RAGProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.embedding_model = SentenceTransformer(config["embedding_model"])
        
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            persist_directory=str(Path(config["vector_db_dir"])),
            anonymized_telemetry=False
        ))
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=config["collection_name"],
            metadata={"hnsw:space": "cosine"}
        )

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Array of embeddings
        """
        try:
            embeddings = self.embedding_model.encode(texts)
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise

    def add_to_database(self, texts: List[str], metadata: List[Dict[str, Any]]):
        """
        Add texts and their embeddings to the vector database.
        
        Args:
            texts: List of texts to add
            metadata: List of metadata dictionaries for each text
        """
        try:
            # Create embeddings
            embeddings = self.create_embeddings(texts)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadata,
                ids=[f"doc_{i}" for i in range(len(texts))]
            )
            
        except Exception as e:
            logger.error(f"Error adding to database: {str(e)}")
            raise

    def query_database(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector database for similar texts.
        
        Args:
            query: Query text
            n_results: Number of results to return
            
        Returns:
            List of dictionaries containing similar texts and metadata
        """
        try:
            # Create query embedding
            query_embedding = self.create_embeddings([query])[0]
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying database: {str(e)}")
            raise

    def process_chapter(self, chapter_data: List[Dict[str, Any]]):
        """
        Process a chapter's data and add it to the vector database.
        
        Args:
            chapter_data: List of dictionaries containing chapter information
        """
        try:
            texts = []
            metadata = []
            
            for item in chapter_data:
                # Chunk the text
                chunks = chunk_text(
                    item["hindi_text"],
                    chunk_size=self.config["chunk_size"],
                    overlap=self.config["chunk_overlap"]
                )
                
                # Add chunks to lists
                texts.extend(chunks)
                metadata.extend([{
                    "chapter": item.get("chapter", "unknown"),
                    "page": item.get("page", "unknown"),
                    "character": item.get("character", "unknown"),
                    "original_text": item["original_text"]
                } for _ in chunks])
            
            # Add to database
            self.add_to_database(texts, metadata)
            
        except Exception as e:
            logger.error(f"Error processing chapter: {str(e)}")
            raise

    def get_context_for_generation(self, current_text: str, n_results: int = 5) -> str:
        """
        Get relevant context for text generation.
        
        Args:
            current_text: Current text to generate for
            n_results: Number of similar texts to retrieve
            
        Returns:
            Formatted context string
        """
        try:
            # Query database
            results = self.query_database(current_text, n_results)
            
            # Format context
            context = "Previous similar dialogues:\n"
            for result in results:
                context += f"- {result['text']} (Character: {result['metadata']['character']})\n"
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            raise

    def save_database(self):
        """Save the vector database to disk."""
        try:
            self.client.persist()
            
        except Exception as e:
            logger.error(f"Error saving database: {str(e)}")
            raise

    def load_database(self):
        """Load the vector database from disk."""
        try:
            self.client = chromadb.Client(Settings(
                persist_directory=str(Path(self.config["vector_db_dir"])),
                anonymized_telemetry=False
            ))
            self.collection = self.client.get_collection(self.config["collection_name"])
            
        except Exception as e:
            logger.error(f"Error loading database: {str(e)}")
            raise 