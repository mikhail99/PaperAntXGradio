"""
PDF Processing Nodes for Paper2ImplementationDoc
Stage 0: PDF Processing Pipeline

This module contains nodes for PDF validation, text extraction, and metadata processing.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, TypedDict, NotRequired
from pocketflow import Node
from utils.pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)

# Type Definitions for shared state

class PdfValidationResult(TypedDict):
    valid: bool
    page_count: int
    file_size: int
    file_size_mb: float
    details: str
    error: NotRequired[str]

class PdfMetadata(TypedDict):
    page_count: int
    file_size_mb: float
    validated: bool

class PageTextInfo(TypedDict):
    page_number: int
    text: str
    char_count: int

class TextStats(TypedDict):
    word_count: int
    char_count: int
    extraction_quality: str
    extraction_method: str

class InitialSharedState(TypedDict):
    pdf_path: str
    output_dir: NotRequired[str] # Optional, defaults to "output"

class SharedStateAfterValidation(InitialSharedState):
    pdf_validation: PdfValidationResult
    pdf_metadata: PdfMetadata

class SharedStateAfterExtraction(SharedStateAfterValidation):
    raw_text: str
    cleaned_text: str
    page_texts: List[PageTextInfo]
    text_stats: TextStats

class FinalSharedState(SharedStateAfterExtraction):
    extraction_output_file: str


class ValidatePDFNode(Node):
    """
    Node to validate PDF file existence and readability.
    """
    
    def prep(self, shared: InitialSharedState) -> str:
        """
        Prepare validation by getting PDF path from shared store.
        """
        pdf_path = shared.get("pdf_path")
        if not pdf_path:
            # This case should ideally be caught by InitialSharedState if it were Pydantic
            # For TypedDict, we still need runtime checks if strictness is desired beyond static analysis.
            raise ValueError("pdf_path not found in shared store")
        return pdf_path
    
    def exec(self, pdf_path: str) -> PdfValidationResult:
        """
        Validate the PDF file using PDFProcessor.
        """
        processor = PDFProcessor(verbose=True)
        validation_result = processor.validate_pdf(pdf_path)
        
        if not validation_result["valid"]:
            raise ValueError(f"PDF validation failed: {validation_result.get('details', 'Unknown error')}")
        
        # Ensure all required fields from PdfValidationResult are present
        # This is more for runtime safety as TypedDict doesn't enforce this strictly at runtime
        # without a validator library. For now, we assume PDFProcessor.validate_pdf is well-behaved.
        return validation_result
    
    def post(self, shared: InitialSharedState, prep_res: str, exec_res: PdfValidationResult) -> str:
        """
        Store validation results in shared store.
        The `shared` dict will be updated to match `SharedStateAfterValidation`.
        """
        # Cast to allow TypedDict to track additions. In a mutable context, this is okay.
        # If shared were immutable, we'd create a new dict.
        validated_shared = shared # type: ignore 

        validated_shared["pdf_validation"] = exec_res
        validated_shared["pdf_metadata"] = {
            "page_count": exec_res["page_count"],
            "file_size_mb": exec_res["file_size_mb"],
            "validated": True
        }
        
        logger.info(f"PDF validated: {exec_res['page_count']} pages, {exec_res['file_size_mb']} MB")
        return "default"


class ExtractTextNode(Node):
    """
    Node to extract text content from validated PDF.
    """
    
    def prep(self, shared: SharedStateAfterValidation) -> str:
        """
        Prepare text extraction by getting PDF path and validation status.
        """
        pdf_path = shared["pdf_path"]
        pdf_validation = shared["pdf_validation"]
        
        if not pdf_validation.get("valid"):
            raise ValueError("PDF must be validated successfully before text extraction")
        
        return pdf_path
    
    def exec(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF using PDFProcessor.
        """
        processor = PDFProcessor(verbose=True)
        extraction_result = processor.extract_text_from_pdf(pdf_path)
        # We expect PDFProcessor to return all fields for PageTextInfo and TextStats
        return extraction_result
    
    def post(self, shared: SharedStateAfterValidation, prep_res: str, exec_res: Dict[str, Any]) -> str:
        """
        Store extracted text and metadata in shared store.
        The `shared` dict will be updated to match `SharedStateAfterExtraction`.
        """
        extracted_shared = shared # type: ignore

        extracted_shared["raw_text"] = exec_res["raw_text"]
        extracted_shared["cleaned_text"] = exec_res["cleaned_text"]
        extracted_shared["page_texts"] = exec_res["page_texts"]
        extracted_shared["text_stats"] = {
            "word_count": exec_res["word_count"],
            "char_count": exec_res["char_count"],
            "extraction_quality": exec_res["extraction_quality"],
            "extraction_method": exec_res["extraction_method"]
        }
        
        logger.info(f"Text extracted: {exec_res['word_count']} words, quality: {exec_res['extraction_quality']}")
        return "default"


class SaveExtractedDataNode(Node):
    """
    Node to save extracted text and metadata to output directory.
    """
    
    class PrepData(TypedDict):
        output_dir: str
        pdf_name: str
        # For clarity, define what's being saved explicitly rather than passing whole shared
        data_to_save: Dict[str, Any] 


    def prep(self, shared: SharedStateAfterExtraction) -> PrepData:
        """
        Prepare data saving by collecting all extracted data.
        """
        # Get output directory (default to "output")
        output_dir = shared.get("output_dir", "output")
        
        pdf_path_str = shared["pdf_path"]
        pdf_name = Path(pdf_path_str).stem
        
        # Select data to save explicitly
        data_to_save = {
            "pdf_path": shared["pdf_path"],
            "raw_text": shared["raw_text"],
            "cleaned_text": shared["cleaned_text"],
            "page_texts": shared["page_texts"],
            "pdf_metadata": shared["pdf_metadata"],
            "text_stats": shared["text_stats"]
        }

        return {
            "output_dir": output_dir,
            "pdf_name": pdf_name,
            "data_to_save": data_to_save
        }
    
    def exec(self, prep_data: PrepData) -> str:
        """
        Save extracted data to JSON file in output directory.
        """
        import json
        
        output_dir = prep_data["output_dir"]
        pdf_name = prep_data["pdf_name"]
        data_to_save = prep_data["data_to_save"]
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{pdf_name}_extracted.json")
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def post(self, shared: SharedStateAfterExtraction, prep_res: PrepData, exec_res: str) -> str:
        """
        Store output file path in shared store.
        The `shared` dict will be updated to match `FinalSharedState`.
        """
        final_shared = shared # type: ignore
        final_shared["extraction_output_file"] = exec_res
        
        logger.info(f"Extracted data saved to: {exec_res}")
        return "default" 