"""
Connection Analysis Flow for Paper2ImplementationDoc
Orchestrates connection detection, categorization, and saving.

This module provides a function to create a pre-configured pocketflow.Flow
for the connection analysis stage.
"""

import logging
from pocketflow import Flow
from typing import Dict, Any

from algorithms.paper2code_kag.connection_analysis_nodes import (
    DetectConnectionsNode,
    CategorizeConnectionsNode,
    SaveConnectionsNode,
    ConnectionPlanningSharedState
)

logger = logging.getLogger(__name__)


def create_connection_analysis_flow(use_mock_llm: bool = True, output_dir: str = "output") -> Flow:
    """
    Creates and returns the connection analysis flow.
    
    Args:
        use_mock_llm: Whether to use the mock LLM interface.
        output_dir: Directory for saving the output plan.
        
    Returns:
        A configured pocketflow.Flow object for connection analysis.
    """
    detect_node = DetectConnectionsNode(use_mock_llm=use_mock_llm, max_retries=3, wait=5)
    categorize_node = CategorizeConnectionsNode(max_retries=3, wait=5)
    save_node = SaveConnectionsNode(output_dir=output_dir, max_retries=3, wait=5)
    
    detect_node >> categorize_node >> save_node
    
    logger.info(f"Initialized connection analysis flow with mock_llm={use_mock_llm}")
    return Flow(start=detect_node)

def run_connection_analysis(shared_state: Dict[str, Any], use_mock_llm: bool = True, output_dir: str = "output") -> Dict[str, Any]:
    """
    Runs the complete connection analysis flow on a given shared state.
    
    Args:
        shared_state: Shared state containing categorized abstractions.
        ... other args for flow creation ...
        
    Returns:
        Updated shared state with connection analysis results.
    """
    logger.info("🚀 Starting connection analysis flow")
    
    if "categorized_abstractions" not in shared_state or not shared_state["categorized_abstractions"]:
        raise ValueError("Missing 'categorized_abstractions' in shared state. Run abstraction detection first.")
    
    abstraction_count = len(shared_state["categorized_abstractions"])
    logger.info(f"Analyzing connections for {abstraction_count} categorized abstractions")
    
    connection_fields = {
        "raw_connections": [], "categorized_connections": [], "connection_summary": {},
        "total_connections_found": 0
    }
    for field, default_value in connection_fields.items():
        shared_state.setdefault(field, default_value)
        
    try:
        analysis_flow = create_connection_analysis_flow(use_mock_llm, output_dir)
        analysis_flow.run(shared_state)
        
        total_connections = shared_state.get("total_connections_found", 0)
        logger.info(f"✅ Connection analysis completed successfully")
        logger.info(f"📊 Results: {total_connections} connections found")
        
        shared_state["connection_planning_completed"] = True
        shared_state["connection_planning_flow_status"] = "success"
        
        return shared_state
        
    except Exception as e:
        logger.error(f"❌ Connection analysis flow failed: {str(e)}", exc_info=True)
        shared_state["connection_planning_completed"] = False
        shared_state["connection_planning_flow_status"] = "failed"
        shared_state["connection_planning_error"] = str(e)
        raise

def test_connection_analysis_flow():
    """Test the connection analysis flow."""
    print("🧪 Testing Connection Analysis Flow")
    
    mock_shared_state = {
        "categorized_abstractions": [
            {"id": "abs-1", "name": "Neural Network", "type": "model"},
            {"id": "abs-2", "name": "Adam Optimizer", "type": "algorithm"}
        ],
        "pdf_metadata": {"title": "Test Paper"}
    }
    
    print("\n🚀 Testing flow with mock LLM...")
    try:
        result_state = run_connection_analysis(
            shared_state=mock_shared_state,
            use_mock_llm=True,
            output_dir="test_output"
        )
        
        print("✅ Flow completed successfully!")
        # ... additional logging ...
        
    except Exception as e:
        print(f"❌ Flow failed: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    test_connection_analysis_flow() 