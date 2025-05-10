import pytesseract
import easyocr
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
from transformers import MarianMTModel, MarianTokenizer
import torch

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.reader = easyocr.Reader(['ja', 'en'])
        
        # Initialize translation model
        self.translator = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ja-hi')
        self.tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ja-hi')
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.translator.to(self.device)

    def extract_text(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> str:
        """
        Extract text from an image region using OCR.
        
        Args:
            image: Input image
            region: Bounding box (x, y, w, h)
            
        Returns:
            Extracted text
        """
        try:
            x, y, w, h = region
            roi = image[y:y+h, x:x+w]
            
            # Use EasyOCR for better accuracy with Japanese text
            results = self.reader.readtext(roi)
            
            # Combine all detected text
            text = ' '.join([result[1] for result in results])
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            raise

    def translate_to_hindi(self, text: str) -> str:
        """
        Translate text to Hindi.
        
        Args:
            text: Input text
            
        Returns:
            Translated text in Hindi
        """
        try:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation
            translated = self.translator.generate(**inputs)
            
            # Decode
            hindi_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
            
            return hindi_text
            
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            raise

    def process_speech_bubble(self, image: np.ndarray, bubble_contour: np.ndarray) -> Dict:
        """
        Process a speech bubble and extract its text.
        
        Args:
            image: Input image
            bubble_contour: Speech bubble contour
            
        Returns:
            Dictionary containing bubble information and text
        """
        try:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(bubble_contour)
            
            # Extract text
            text = self.extract_text(image, (x, y, w, h))
            
            # Get bubble tail
            tail_point = extract_bubble_tail(bubble_contour)
            
            # Translate to Hindi
            hindi_text = self.translate_to_hindi(text)
            
            return {
                "original_text": text,
                "hindi_text": hindi_text,
                "bounding_box": (x, y, w, h),
                "tail_point": tail_point
            }
            
        except Exception as e:
            logger.error(f"Error processing speech bubble: {str(e)}")
            raise

    def process_page(self, image: np.ndarray, bubbles: List[np.ndarray]) -> List[Dict]:
        """
        Process all speech bubbles on a page.
        
        Args:
            image: Page image
            bubbles: List of speech bubble contours
            
        Returns:
            List of processed bubble information
        """
        try:
            processed_bubbles = []
            
            for bubble in bubbles:
                bubble_info = self.process_speech_bubble(image, bubble)
                processed_bubbles.append(bubble_info)
            
            return processed_bubbles
            
        except Exception as e:
            logger.error(f"Error processing page: {str(e)}")
            raise

    def save_processed_data(self, data: List[Dict], output_path: Path):
        """
        Save processed text data to JSON file.
        
        Args:
            data: List of processed bubble information
            output_path: Path to save the JSON file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise

    def load_processed_data(self, input_path: Path) -> List[Dict]:
        """
        Load processed text data from JSON file.
        
        Args:
            input_path: Path to the JSON file
            
        Returns:
            List of processed bubble information
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
            
        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            raise 