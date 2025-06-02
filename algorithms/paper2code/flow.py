"""
Paper2ImplementationDoc Flow - Core pipeline orchestration
Following PocketFlow-Tutorial-Codebase-Knowledge pattern
"""

from pocketflow import Flow
# Import all node classes from nodes.py
from nodes import (
    PDFInputNode,
    TextExtractionNode, 
    StructureAnalysisNode,
    ImplementationAnalysisNode,
    DocumentationGenerationNode
)

def create_paper2doc_flow(
    analysis_depth: str = "detailed",
    output_format: str = "markdown", 
    include_diagrams: bool = False,
    max_sections: int = 10,
    verbose: bool = False
):
    """Creates and returns the Paper2ImplementationDoc generation flow."""
    
    # Instantiate nodes with configuration
    pdf_input = PDFInputNode(verbose=verbose)
    text_extraction = TextExtractionNode(verbose=verbose, max_retries=3, wait=5)
    structure_analysis = StructureAnalysisNode(
        max_sections=max_sections, 
        verbose=verbose, 
        max_retries=3, 
        wait=5
    )
    implementation_analysis = ImplementationAnalysisNode(
        analysis_depth=analysis_depth,
        verbose=verbose,
        max_retries=5,
        wait=10
    )
    documentation_generation = DocumentationGenerationNode(
        output_format=output_format,
        include_diagrams=include_diagrams,
        verbose=verbose,
        max_retries=3,
        wait=5
    )
    
    # Connect nodes in sequence based on the pipeline design
    pdf_input >> text_extraction
    text_extraction >> structure_analysis
    structure_analysis >> implementation_analysis
    implementation_analysis >> documentation_generation
    
    # Create the flow starting with PDFInputNode
    paper2doc_flow = Flow(start=pdf_input)
    
    return paper2doc_flow 