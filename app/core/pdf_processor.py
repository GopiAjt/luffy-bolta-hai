import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from pdf2image import convert_from_path
from ..utils.image_processing import preprocess_image

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_pages(self, pdf_path: Path) -> List[Tuple[int, np.ndarray]]:
        """
        Extract pages from PDF and convert to images.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of tuples containing (page_number, image_array)
        """
        try:
            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=300,
                fmt="png",
                thread_count=4
            )
            
            processed_pages = []
            for idx, image in enumerate(images):
                # Convert PIL Image to numpy array
                img_array = np.array(image)
                
                # Convert RGB to BGR (OpenCV format)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Preprocess image
                processed_img = preprocess_image(img_array)
                
                processed_pages.append((idx + 1, processed_img))
                
                # Save processed image
                output_path = self.output_dir / f"page_{idx + 1:03d}.png"
                cv2.imwrite(str(output_path), processed_img)
                
            return processed_pages
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise

    def get_page_metadata(self, pdf_path: Path) -> dict:
        """
        Extract metadata from PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing PDF metadata
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "page_count": len(doc),
                "file_size": pdf_path.stat().st_size,
            }
            doc.close()
            return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {str(e)}")
            raise

    def extract_text_regions(self, page_image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extract potential text regions from a page image.
        
        Args:
            page_image: Numpy array of the page image
            
        Returns:
            List of tuples containing (region_image, bounding_box)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter and process contours
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter based on size
                if w > 50 and h > 20:  # Minimum size threshold
                    region = page_image[y:y+h, x:x+w]
                    text_regions.append((region, (x, y, w, h)))
            
            return text_regions
            
        except Exception as e:
            logger.error(f"Error extracting text regions: {str(e)}")
            raise

    def process_chapter(self, pdf_path: Path) -> dict:
        """
        Process an entire manga chapter.
        
        Args:
            pdf_path: Path to the chapter PDF
            
        Returns:
            Dictionary containing processing results
        """
        try:
            # Get metadata
            metadata = self.get_page_metadata(pdf_path)
            
            # Extract and process pages
            pages = self.extract_pages(pdf_path)
            
            # Process each page
            processed_data = {
                "metadata": metadata,
                "pages": []
            }
            
            for page_num, page_image in pages:
                # Extract text regions
                text_regions = self.extract_text_regions(page_image)
                
                page_data = {
                    "page_number": page_num,
                    "text_regions": text_regions,
                    "image_path": str(self.output_dir / f"page_{page_num:03d}.png")
                }
                
                processed_data["pages"].append(page_data)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing chapter {pdf_path}: {str(e)}")
            raise 