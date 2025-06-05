"""
Planning Flow for Paper2ImplementationDoc
Stage 1: Section Splitting & Planning

This module defines the PocketFlow pipeline for the planning stage.
"""

from pocketflow import Flow
import importlib.util
import sys
import os

# Load the planning nodes module
spec = importlib.util.spec_from_file_location("planning_nodes", "1_planning_nodes.py")
planning_nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planning_nodes)

SplitSectionsNode = planning_nodes.SplitSectionsNode
SelectSectionsNode = planning_nodes.SelectSectionsNode
SavePlanningResultsNode = planning_nodes.SavePlanningResultsNode


def create_planning_flow():
    """
    Creates and returns the planning flow.
    
    Flow sequence:
    1. SplitSectionsNode - Split text into sections using regex
    2. SelectSectionsNode - Select relevant sections using LLM
    3. SavePlanningResultsNode - Save planning results to output
    
    Returns:
        Flow: The configured planning flow
    """
    
    # Instantiate nodes with retry configuration
    split_sections = SplitSectionsNode(max_retries=2, wait=1)
    select_sections = SelectSectionsNode(max_retries=3, wait=5)  # Longer wait for LLM calls
    save_results = SavePlanningResultsNode(max_retries=2, wait=1)
    
    # Connect nodes in sequence
    split_sections >> select_sections >> save_results
    
    # Create the flow starting with section splitting
    planning_flow = Flow(start=split_sections)
    
    return planning_flow


def create_combined_flow():
    """
    Creates a combined flow that includes both PDF processing and planning.
    
    Returns:
        Flow: The combined PDF + planning flow
    """
    # Import PDF processing flow
    pdf_spec = importlib.util.spec_from_file_location("pdf_flow", "0_pdf_process_flow.py")
    pdf_flow_module = importlib.util.module_from_spec(pdf_spec)
    pdf_spec.loader.exec_module(pdf_flow_module)
    
    # Get the final node from PDF processing flow  
    pdf_flow = pdf_flow_module.create_pdf_processing_flow()
    
    # Create planning flow
    planning_flow = create_planning_flow()
    
    # Connect PDF processing output to planning input
    # We need to find the last node in the PDF flow to connect it
    # For now, we'll create them separately and combine in the test
    
    return pdf_flow, planning_flow


# For standalone testing
if __name__ == "__main__":
    import logging
    import json
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) != 2:
        print("Usage: python 1_planning_flow.py <path_to_extracted_data.json>")
        print("  where extracted_data.json is the output from PDF processing (Iteration 1)")
        sys.exit(1)
    
    extracted_data_path = sys.argv[1]
    
    # Load extracted data from PDF processing
    try:
        with open(extracted_data_path, 'r', encoding='utf-8') as f:
            pdf_data = json.load(f)
    except Exception as e:
        print(f"Error loading extracted data: {str(e)}")
        sys.exit(1)
    
    # Create shared store with data from PDF processing
    shared = {
        "pdf_path": pdf_data.get("pdf_path", ""),
        "raw_text": pdf_data.get("raw_text", ""),
        "cleaned_text": pdf_data.get("cleaned_text", ""),
        "pdf_metadata": pdf_data.get("pdf_metadata", {}),
        "text_stats": pdf_data.get("text_stats", {}),
        "output_dir": "output"
    }
    
    print(f"📄 Loaded PDF data: {shared['text_stats'].get('word_count', 0)} words")
    
    # Create and run the planning flow
    flow = create_planning_flow()
    
    try:
        flow.run(shared)
        print(f"\n✅ Planning completed successfully!")
        print(f"📊 Sections detected: {len(shared.get('sections', []))}")
        print(f"🎯 Sections selected: {len(shared.get('selected_sections', []))}")
        print(f"💾 Output saved to: {shared.get('planning_output_file')}")
        
        # Print selected sections summary
        if shared.get("selected_sections"):
            print(f"\n📋 Selected sections:")
            for i, section in enumerate(shared["selected_sections"]):
                print(f"  {i+1}. {section['section_type']}: {section['title']}")
                print(f"     Reason: {section['selection_reason']}")
        
    except Exception as e:
        print(f"❌ Error during planning: {str(e)}")
        sys.exit(1) 