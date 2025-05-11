from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import logging
from typing import List, Dict, Any, Optional
import json

from ..core.pdf_processor import PDFProcessor
from ..core.text_processor import TextProcessor
from ..core.rag_processor import RAGProcessor
from ..core.subtitle_rag import SubtitleRAG
from ..utils.text_processing import clean_text, format_dialogue
from config.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_DB_DIR,
    OCR_CONFIG, YOLO_CONFIG, VECTOR_DB_CONFIG, LLM_CONFIG
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="One Piece Hindi Voiceover API",
    description="API for generating Hindi voiceover scripts from One Piece manga and anime subtitles",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
pdf_processor = PDFProcessor(PROCESSED_DATA_DIR)
text_processor = TextProcessor(OCR_CONFIG)
rag_processor = RAGProcessor(VECTOR_DB_CONFIG)
subtitle_rag = SubtitleRAG(PROCESSED_DATA_DIR / 'subtitles_en')

@app.post("/api/v1/process-chapter")
async def process_chapter(file: UploadFile = File(...)):
    """
    Process a new manga chapter PDF.
    """
    try:
        # Save uploaded file
        file_path = RAW_DATA_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process PDF
        chapter_data = pdf_processor.process_chapter(file_path)
        
        # Process text
        processed_data = []
        for page in chapter_data["pages"]:
            # Extract text from regions
            for region in page["text_regions"]:
                text = text_processor.extract_text(region[0], region[1])
                hindi_text = text_processor.translate_to_hindi(text)
                
                processed_data.append({
                    "page": page["page_number"],
                    "original_text": text,
                    "hindi_text": hindi_text,
                    "bounding_box": region[1]
                })
        
        # Add to vector database
        rag_processor.process_chapter(processed_data)
        
        return {
            "status": "success",
            "message": "Chapter processed successfully",
            "data": {
                "pages_processed": len(chapter_data["pages"]),
                "text_regions_found": sum(len(page["text_regions"]) for page in chapter_data["pages"])
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing chapter: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/generate-script")
async def generate_script(page: int, chapter: int):
    """
    Generate Hindi voiceover script for a specific page.
    """
    try:
        # Load page data
        page_path = PROCESSED_DATA_DIR / f"chapter_{chapter:03d}" / f"page_{page:03d}.json"
        if not page_path.exists():
            raise HTTPException(status_code=404, detail="Page not found")
        
        with page_path.open("r", encoding="utf-8") as f:
            page_data = json.load(f)
        
        # Get context from vector database
        context = rag_processor.get_context_for_generation(
            " ".join(item["hindi_text"] for item in page_data)
        )
        
        # Generate script
        script = []
        for item in page_data:
            # Get character-specific context
            char_context = rag_processor.get_context_for_generation(
                item["hindi_text"],
                n_results=3
            )
            
            # Format dialogue
            dialogue = format_dialogue(
                item.get("character", "Unknown"),
                item["hindi_text"]
            )
            
            script.append({
                "character": item.get("character", "Unknown"),
                "dialogue": dialogue,
                "context": char_context
            })
        
        return {
            "status": "success",
            "data": {
                "chapter": chapter,
                "page": page,
                "script": script,
                "context": context
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating script: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/characters")
async def get_characters():
    """
    Get list of characters and their dialogue history.
    """
    try:
        # Query vector database for character information
        characters = {}
        
        # Get all documents from collection
        results = rag_processor.collection.get()
        
        # Process results
        for i in range(len(results["documents"])):
            metadata = results["metadatas"][i]
            character = metadata.get("character", "unknown")
            
            if character not in characters:
                characters[character] = {
                    "dialogue_count": 0,
                    "chapters": set(),
                    "recent_dialogues": []
                }
            
            characters[character]["dialogue_count"] += 1
            characters[character]["chapters"].add(metadata.get("chapter", "unknown"))
            characters[character]["recent_dialogues"].append({
                "text": results["documents"][i],
                "chapter": metadata.get("chapter", "unknown"),
                "page": metadata.get("page", "unknown")
            })
        
        # Convert sets to lists for JSON serialization
        for char in characters:
            characters[char]["chapters"] = list(characters[char]["chapters"])
            characters[char]["recent_dialogues"] = characters[char]["recent_dialogues"][-5:]  # Last 5 dialogues
        
        return {
            "status": "success",
            "data": characters
        }
        
    except Exception as e:
        logger.error(f"Error getting characters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

@app.post("/api/v1/subtitles/query")
async def query_subtitles(
    query: str,
    n_results: Optional[int] = 5,
    filter_episode: Optional[str] = None
):
    """
    Query the One Piece subtitle database.
    
    Args:
        query: The search query
        n_results: Number of results to return (default: 5)
        filter_episode: Optional episode number to filter by
        
    Returns:
        List of matching dialogue entries with metadata
    """
    try:
        results = subtitle_rag.query_dialogue(
            query=query,
            n_results=n_results,
            filter_episode=filter_episode
        )
        return {
            "status": "success",
            "data": results
        }
    except Exception as e:
        logger.error(f"Error querying subtitles: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/subtitles/process")
async def process_subtitles():
    """
    Process all subtitle files and create embeddings.
    """
    try:
        # Load subtitles
        subtitle_data = subtitle_rag.load_subtitles()
        logger.info(f"Loaded {len(subtitle_data)} subtitle entries")
        
        # Create embeddings
        subtitle_rag.create_embeddings(subtitle_data)
        logger.info("Embeddings created and stored in ChromaDB")
        
        return {
            "status": "success",
            "message": "Subtitles processed successfully",
            "data": {
                "entries_processed": len(subtitle_data)
            }
        }
    except Exception as e:
        logger.error(f"Error processing subtitles: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/subtitles/episodes")
async def get_episodes():
    """
    Get list of all processed episodes.
    """
    try:
        # Get all documents from collection
        results = subtitle_rag.collection.get()
        
        # Extract unique episodes
        episodes = {}
        for metadata in results["metadatas"]:
            episode_number = metadata.get("episode_number", "unknown")
            if episode_number not in episodes:
                episodes[episode_number] = {
                    "title": metadata.get("title", "Unknown Title"),
                    "file_name": metadata.get("file_name", ""),
                    "dialogue_count": 0
                }
            episodes[episode_number]["dialogue_count"] += 1
        
        return {
            "status": "success",
            "data": episodes
        }
    except Exception as e:
        logger.error(f"Error getting episodes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 