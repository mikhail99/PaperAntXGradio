"""
PDF Processing Flow for Paper2ImplementationDoc
Stage 0: PDF Processing Pipeline

This module defines the PocketFlow pipeline for PDF processing.
"""

from pocketflow import Flow
# Import using importlib to handle module name starting with number
import importlib.util
import sys
import os

# Load the nodes module
spec = importlib.util.spec_from_file_location("pdf_process_nodes", "0_pdf_process_nodes.py")
pdf_nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdf_nodes)

ValidatePDFNode = pdf_nodes.ValidatePDFNode
ExtractTextNode = pdf_nodes.ExtractTextNode
SaveExtractedDataNode = pdf_nodes.SaveExtractedDataNode


def create_pdf_processing_flow():
    """
    Creates and returns the PDF processing flow.
    
    Flow sequence:
    1. ValidatePDFNode - Validates PDF file
    2. ExtractTextNode - Extracts text content  
    3. SaveExtractedDataNode - Saves results to output
    
    Returns:
        Flow: The configured PDF processing flow
    """
    
    # Instantiate nodes with retry configuration
    validate_pdf = ValidatePDFNode(max_retries=2, wait=1)
    extract_text = ExtractTextNode(max_retries=3, wait=2)
    save_data = SaveExtractedDataNode(max_retries=2, wait=1)
    
    # Connect nodes in sequence
    validate_pdf >> extract_text >> save_data
    
    # Create the flow starting with validation
    pdf_flow = Flow(start=validate_pdf)
    
    return pdf_flow


# For standalone testing
if __name__ == "__main__":
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) != 2:
        print("Usage: python 0_pdf_process_flow.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Create shared store with PDF path
    shared = {
        "pdf_path": pdf_path,
        "output_dir": "output"
    }
    
    # Create and run the flow
    flow = create_pdf_processing_flow()
    
    try:
        flow.run(shared)
        print(f"\nPDF processing completed successfully!")
        print(f"Output saved to: {shared.get('extraction_output_file')}")
        print(f"Text stats: {shared.get('text_stats')}")
    except Exception as e:
        print(f"Error during PDF processing: {str(e)}")
        sys.exit(1) 