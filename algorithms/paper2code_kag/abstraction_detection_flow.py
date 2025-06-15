"""
Abstraction Detection Flow for Paper2ImplementationDoc
Orchestrates abstraction identification, categorization, and saving.

This module provides a function to create a pre-configured pocketflow.Flow
for the abstraction detection stage.
"""

import logging
from pocketflow import Flow
from typing import Dict, Any

# Import abstraction detection nodes
from algorithms.paper2code_kag.abstraction_detection_nodes import (
    IdentifyAbstractionsNode,
    CategorizeAbstractionsNode
)

logger = logging.getLogger(__name__)


def create_abstraction_detection_flow(use_mock_llm: bool = True, output_dir: str = "output", target_types: list = None) -> Flow:
    """
    Creates and returns the abstraction detection flow.
    
    Args:
        use_mock_llm: Whether to use the mock LLM interface.
        output_dir: Directory for saving the output plan.
        target_types: Optional list of specific abstraction types to target.
        
    Returns:
        A configured pocketflow.Flow object for abstraction detection.
    """
    # Create nodes with retry configuration
    identify_node = IdentifyAbstractionsNode(
        use_mock_llm=use_mock_llm,
        target_types=target_types,
        max_retries=3,
        wait=5
    )
    
    categorize_node = CategorizeAbstractionsNode(
        max_retries=3,
        wait=5
    )
    
    # Connect nodes in sequence
    identify_node >> categorize_node 
    
    # Create and return the flow
    logger.info(f"Initialized abstraction detection flow with mock_llm={use_mock_llm}")
    return Flow(start=identify_node)

def run_abstraction_detection(shared_state: Dict[str, Any], use_mock_llm: bool = True, output_dir: str = "output", target_types: list = None) -> Dict[str, Any]:
    """
    Runs the complete abstraction detection flow on a given shared state.
    
    Args:
        shared_state: Shared state containing selected sections.
        ... other args for flow creation ...
        
    Returns:
        Updated shared state with abstraction detection results.
    """
    logger.info("🚀 Starting abstraction detection flow")
    
    # Validate input
    if "selected_sections" not in shared_state:
        raise ValueError("Missing 'selected_sections' in shared state.")
    
    sections_count = len(shared_state["selected_sections"])
    logger.info(f"Processing {sections_count} selected sections for abstraction detection")
    
    # Initialize fields if not present
    abstraction_fields = {
        "raw_abstractions": [], "categorized_abstractions": [], "abstraction_summary": {},
        "abstraction_detection_method": "", "total_abstractions_found": 0
    }
    for field, default_value in abstraction_fields.items():
        shared_state.setdefault(field, default_value)
    
    try:
        # Create and run the flow
        detection_flow = create_abstraction_detection_flow(use_mock_llm, output_dir, target_types)
        detection_flow.run(shared_state)
        
        # Log results
        total_abstractions = shared_state.get("total_abstractions_found", 0)
        categorized_count = len(shared_state.get("categorized_abstractions", []))
        
        logger.info(f"✅ Abstraction detection completed successfully")
        logger.info(f"📊 Results: {total_abstractions} abstractions found, {categorized_count} categorized")
        
        # Add flow metadata
        shared_state["abstraction_planning_completed"] = True
        shared_state["abstraction_planning_flow_status"] = "success"
        
        return shared_state
        
    except Exception as e:
        logger.error(f"❌ Abstraction detection flow failed: {str(e)}", exc_info=True)
        shared_state["abstraction_planning_completed"] = False
        shared_state["abstraction_planning_flow_status"] = "failed"
        shared_state["abstraction_planning_error"] = str(e)
        raise

def load_section_planning_results(file_path: str = "output/planning_results.json") -> Dict[str, Any]:
    """
    Load section planning results from previous iteration.
    
    Args:
        file_path: Path to section planning results JSON file
        
    Returns:
        Shared state dict with section planning results
    """
    import json
    import os
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Section planning results not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract relevant fields for abstraction planning
    shared_state = {
        "selected_sections": data.get("selected_sections", []),
        "planning_summary": data.get("planning_summary", {}),
        "total_sections_detected": data.get("total_sections_detected", 0),
        "pdf_metadata": data.get("pdf_metadata", {}),
        "text_stats": data.get("text_stats", {})
    }
    
    logger.info(f"Loaded section planning results: {len(shared_state['selected_sections'])} sections")
    return shared_state

def test_abstraction_detection_flow():
    """Test the abstraction detection flow."""
    print("🧪 Testing Abstraction Detection Flow")
    
    # Create mock shared state (simulating section planning results)
    mock_shared_state = {
        "selected_sections": [
            {"title": "Abstract", "content": "A novel neural network...", "section_type": "abstract"},
            {"title": "Methodology", "content": "Step 1: Data preprocessing...", "section_type": "methodology"}
        ],
        "planning_summary": {"total_sections_analyzed": 2},
        "total_sections_detected": 2
    }
    
    print("\n🚀 Testing flow with mock LLM...")
    try:
        result_state = run_abstraction_detection(
            shared_state=mock_shared_state, 
            use_mock_llm=True, 
            output_dir="test_output"
        )
        
        print("✅ Flow completed successfully!")
        print(f"📊 Results:")
        print(f"  • Total abstractions found: {result_state.get('total_abstractions_found', 0)}")
        print(f"  • Categorized abstractions: {len(result_state.get('categorized_abstractions', []))}")
        print(f"  • Detection method: {result_state.get('abstraction_detection_method', 'unknown')}")
        
        # Show summary statistics
        summary = result_state.get("abstraction_summary", {})
        if summary:
            print(f"  • Type distribution: {summary.get('type_distribution', {})}")
            print(f"  • Complexity distribution: {summary.get('complexity_distribution', {})}")
            print(f"  • Average importance: {summary.get('average_importance', 0):.2f}")
        
        # Check if file was saved
        if result_state.get("abstraction_plan_saved"):
            print(f"  • Saved to: {result_state.get('abstraction_plan_file', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Flow failed: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    test_abstraction_detection_flow() 