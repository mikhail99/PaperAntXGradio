"""
PDF Processing utilities for Paper2ImplementationDoc
Real implementation using PyMuPDF (fitz)
"""

import logging
import re
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import pymupdf

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF text extraction and basic preprocessing."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dict containing extracted text and metadata
        """
        doc = None
        try:
            # Open PDF
            doc = pymupdf.open(pdf_path)
            
            # Extract text from all pages
            full_text = ""
            page_texts = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                page_texts.append({
                    "page_number": page_num + 1,
                    "text": page_text,
                    "char_count": len(page_text)
                })
                full_text += page_text + "\n\n"
            
            # Get page count before closing
            page_count = len(doc)
            
            # Basic cleanup
            cleaned_text = self._clean_text(full_text)
            
            # Calculate stats
            word_count = len(cleaned_text.split())
            char_count = len(cleaned_text)
            
            result = {
                "raw_text": full_text,
                "cleaned_text": cleaned_text,
                "page_texts": page_texts,
                "page_count": page_count,
                "word_count": word_count,
                "char_count": char_count,
                "extraction_method": "PyMuPDF",
                "extraction_quality": self._assess_quality(cleaned_text)
            }
            
            if self.verbose:
                logger.debug(f"Extracted {word_count} words from {page_count} pages")
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            raise
        finally:
            # Always close the document
            if doc:
                doc.close()
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleaning and normalization."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page breaks and form feeds
        text = text.replace('\f', '\n').replace('\r', '\n')
        
        # Normalize line breaks
        text = re.sub(r'\n+', '\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _assess_quality(self, text: str) -> str:
        """Assess the quality of extracted text."""
        if not text:
            return "empty"
        
        # Count readable words vs. total characters
        words = text.split()
        readable_words = [w for w in words if re.match(r'^[a-zA-Z0-9\-\.]+$', w)]
        
        if not words:
            return "empty"
        
        readability_ratio = len(readable_words) / len(words)
        
        if readability_ratio > 0.8:
            return "high"
        elif readability_ratio > 0.6:
            return "medium"
        else:
            return "low"
    
    def validate_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Validate that a PDF file exists and is readable.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dict with validation results
        """
        pdf_file = Path(pdf_path)
        
        if not pdf_file.exists():
            return {
                "valid": False,
                "error": "File not found",
                "details": f"PDF file does not exist: {pdf_path}"
            }
        
        if not pdf_file.is_file():
            return {
                "valid": False,
                "error": "Not a file",
                "details": f"Path is not a file: {pdf_path}"
            }
        
        doc = None
        try:
            # Try to open with PyMuPDF
            doc = pymupdf.open(pdf_path)
            page_count = len(doc)
            file_size = pdf_file.stat().st_size
            
            return {
                "valid": True,
                "page_count": page_count,
                "file_size": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "details": f"Valid PDF with {page_count} pages"
            }
        
        except Exception as e:
            return {
                "valid": False,
                "error": "Cannot open PDF",
                "details": f"Error opening PDF: {str(e)}"
            }
        finally:
            # Always close the document
            if doc:
                doc.close() 