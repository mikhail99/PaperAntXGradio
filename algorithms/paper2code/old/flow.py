"""
Paper2ImplementationDoc Flow - Core pipeline orchestration
Following PocketFlow-Tutorial-Codebase-Knowledge pattern
"""

from pocketflow import Flow
from typing import Dict, Any
# Import all node classes from nodes.py
from nodes import (
    PDFInputNode,
    TextExtractionNode, 
    StructureAnalysisNode,
    ImplementationAnalysisNode,
    DocumentationGenerationNode
)

def create_paper2doc_flow(config: Dict[str, Any]) -> Flow:
    """
    Create the Paper2ImplementationDoc flow with enhanced component analysis (Iteration 3).
    
    Args:
        config: Configuration dictionary containing processing options
        
    Returns:
        Flow: Configured PocketFlow pipeline
    """
    # Extract configuration options
    verbose = config.get("verbose", False)
    analysis_depth = config.get("analysis_depth", "detailed")
    output_format = config.get("output_format", "markdown")
    include_diagrams = config.get("include_diagrams", False)
    max_sections = config.get("max_sections", 10)
    enable_component_analysis = config.get("enable_component_analysis", True)  # NEW for Iteration 3
    
    # Create nodes with enhanced configuration
    pdf_input = PDFInputNode(
        verbose=verbose,
        max_retries=2,
        wait=1
    )
    
    text_extraction = TextExtractionNode(
        verbose=verbose,
        max_retries=3,
        wait=2
    )
    
    structure_analysis = StructureAnalysisNode(
        max_sections=max_sections,
        enable_component_analysis=enable_component_analysis,  # NEW
        verbose=verbose,
        max_retries=2,
        wait=1
    )
    
    implementation_analysis = ImplementationAnalysisNode(
        analysis_depth=analysis_depth,
        verbose=verbose,
        max_retries=2,
        wait=1
    )
    
    documentation_generation = DocumentationGenerationNode(
        output_format=output_format,
        include_diagrams=include_diagrams,
        verbose=verbose,
        max_retries=2,
        wait=1
    )
    
    # Connect nodes in sequence
    pdf_input >> text_extraction >> structure_analysis >> implementation_analysis >> documentation_generation
    
    # Create and return flow
    return Flow(start=pdf_input) 