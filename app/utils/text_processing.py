from typing import List
import re
import logging

logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    try:
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_text = ' '.join(current_chunk[-overlap:])
                current_chunk = [overlap_text]
                current_size = len(overlap_text)
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error chunking text: {str(e)}")
        raise

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    try:
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Normalize punctuation
        text = re.sub(r'\.{2,}', '...', text)
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        raise

def extract_character_name(text: str, character_list: List[str]) -> str:
    """
    Extract character name from text using a list of known characters.
    
    Args:
        text: Input text
        character_list: List of known character names
        
    Returns:
        Extracted character name or "unknown"
    """
    try:
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Check for character names in text
        for character in character_list:
            if character.lower() in text_lower:
                return character
        
        return "unknown"
        
    except Exception as e:
        logger.error(f"Error extracting character name: {str(e)}")
        raise

def format_dialogue(character: str, text: str) -> str:
    """
    Format dialogue with character name.
    
    Args:
        character: Character name
        text: Dialogue text
        
    Returns:
        Formatted dialogue string
    """
    try:
        return f"{character}: {text}"
        
    except Exception as e:
        logger.error(f"Error formatting dialogue: {str(e)}")
        raise

def detect_speech_type(text: str) -> str:
    """
    Detect the type of speech (dialogue, thought, narration, etc.).
    
    Args:
        text: Input text
        
    Returns:
        Speech type
    """
    try:
        # Check for thought bubbles (usually in italics or with ellipsis)
        if text.startswith('...') or text.endswith('...'):
            return "thought"
        
        # Check for narration (usually longer, more descriptive)
        if len(text.split()) > 20:
            return "narration"
        
        # Default to dialogue
        return "dialogue"
        
    except Exception as e:
        logger.error(f"Error detecting speech type: {str(e)}")
        raise

def merge_similar_dialogues(dialogues: List[dict]) -> List[dict]:
    """
    Merge similar dialogues from the same character.
    
    Args:
        dialogues: List of dialogue dictionaries
        
    Returns:
        Merged dialogues
    """
    try:
        merged = []
        current = None
        
        for dialogue in dialogues:
            if current is None:
                current = dialogue
            elif (current["character"] == dialogue["character"] and
                  current["type"] == dialogue["type"]):
                # Merge with current dialogue
                current["text"] += " " + dialogue["text"]
            else:
                # Save current and start new
                merged.append(current)
                current = dialogue
        
        # Add the last dialogue
        if current is not None:
            merged.append(current)
        
        return merged
        
    except Exception as e:
        logger.error(f"Error merging dialogues: {str(e)}")
        raise 