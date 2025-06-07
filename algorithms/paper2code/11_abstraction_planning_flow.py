"""
Abstraction Planning Flow for Paper2ImplementationDoc
Iteration 3: Orchestrates abstraction identification, categorization, and saving.

Flow: identify_abstractions >> categorize_abstractions >> save_abstractions
"""

import logging
from pocketflow import Flow
from typing import Dict, Any

# Import abstraction planning nodes
from algorithms.paper2code.abstraction_planning_nodes import (
    IdentifyAbstractionsNode,
    CategorizeAbstractionsNode,
    SaveAbstractionsNode,
    AbstractionPlanningSharedState
)

logger = logging.getLogger(__name__)

class AbstractionPlanningFlow:
    """
    Flow for abstraction planning stage.
    Identifies, categorizes, and saves abstractions from selected sections.
    """
    
    def __init__(self, use_mock_llm: bool = True, output_dir: str = "output", target_types: list = None):
        self.use_mock_llm = use_mock_llm
        self.output_dir = output_dir
        self.target_types = target_types or []
        
        # Create nodes with retry configuration
        self.identify_node = IdentifyAbstractionsNode(
            use_mock_llm=use_mock_llm,
            target_types=target_types,
            max_retries=3,
            wait=5
        )
        
        self.categorize_node = CategorizeAbstractionsNode(
            max_retries=3,
            wait=5
        )
        
        self.save_node = SaveAbstractionsNode(
            output_dir=output_dir,
            max_retries=3,
            wait=5
        )
        
        # Connect nodes in sequence
        self.identify_node >> self.categorize_node >> self.save_node
        
        # Create flow
        self.flow = Flow(start=self.identify_node)
        
        logger.info(f"Initialized abstraction planning flow with mock_llm={use_mock_llm}")
    
    def run(self, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the abstraction planning flow.
        
        Args:
            shared_state: Shared state containing selected sections from previous iteration
            
        Returns:
            Updated shared state with abstraction planning results
        """
        logger.info("🚀 Starting abstraction planning flow")
        
        # Validate input
        if "selected_sections" not in shared_state:
            raise ValueError("Missing 'selected_sections' in shared state. Run section planning first.")
        
        sections_count = len(shared_state["selected_sections"])
        logger.info(f"Processing {sections_count} selected sections for abstraction detection")
        
        # Initialize abstraction planning fields if not present
        abstraction_fields = {
            "raw_abstractions": [],
            "categorized_abstractions": [],
            "abstraction_summary": {},
            "abstraction_detection_method": "",
            "total_abstractions_found": 0
        }
        
        for field, default_value in abstraction_fields.items():
            if field not in shared_state:
                shared_state[field] = default_value
        
        try:
            # Run the flow
            self.flow.run(shared_state)
            
            # Log results
            total_abstractions = shared_state.get("total_abstractions_found", 0)
            categorized_count = len(shared_state.get("categorized_abstractions", []))
            
            logger.info(f"✅ Abstraction planning completed successfully")
            logger.info(f"📊 Results: {total_abstractions} abstractions found, {categorized_count} categorized")
            
            # Add flow metadata
            shared_state["abstraction_planning_completed"] = True
            shared_state["abstraction_planning_flow_status"] = "success"
            
            return shared_state
            
        except Exception as e:
            logger.error(f"❌ Abstraction planning flow failed: {str(e)}")
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

def run_abstraction_planning(
    section_results_file: str = "output/planning_results.json",
    use_mock_llm: bool = True,
    output_dir: str = "output",
    target_types: list = None
) -> Dict[str, Any]:
    """
    Complete abstraction planning workflow.
    
    Args:
        section_results_file: Path to section planning results
        use_mock_llm: Whether to use mock LLM interface
        output_dir: Directory for output files
        target_types: List of abstraction types to detect (None for all)
        
    Returns:
        Final shared state with abstraction planning results
    """
    logger.info("🎯 Running complete abstraction planning workflow")
    
    # Load section planning results
    shared_state = load_section_planning_results(section_results_file)
    
    # Create and run abstraction planning flow
    planning_flow = AbstractionPlanningFlow(
        use_mock_llm=use_mock_llm,
        output_dir=output_dir,
        target_types=target_types
    )
    
    result_state = planning_flow.run(shared_state)
    
    logger.info("🎉 Abstraction planning workflow completed successfully")
    return result_state

def test_abstraction_planning_flow():
    """Test the abstraction planning flow."""
    print("🧪 Testing Abstraction Planning Flow")
    
    # Create mock shared state (simulating section planning results)
    mock_shared_state = {
        "selected_sections": [
            {
                "title": "Abstract",
                "content": "This paper presents a novel neural network approach for data processing. Our method uses deep learning techniques with transformer architecture.",
                "section_type": "abstract",
                "selection_reason": "Selected by heuristic",
                "priority": 1
            },
            {
                "title": "2. Methodology",
                "content": "Step 1: Data preprocessing using matrix operations\nStep 2: Feature extraction with CNN layers\nStep 3: Attention mechanism for sequence processing\nThe algorithm complexity is O(n log n) for the preprocessing phase.",
                "section_type": "methodology",
                "selection_reason": "Selected by heuristic",
                "priority": 2
            },
            {
                "title": "Implementation",
                "content": "Our approach uses PyTorch framework with NumPy for numerical operations. The implementation requires TensorFlow for certain optimization procedures.",
                "section_type": "section",
                "selection_reason": "Selected by heuristic",
                "priority": 4
            }
        ],
        "planning_summary": {
            "total_sections_analyzed": 6,
            "selection_method": "heuristic_fallback",
            "selection_criteria": "Type priority + technical keyword density"
        },
        "total_sections_detected": 6
    }
    
    print(f"📊 Input: {len(mock_shared_state['selected_sections'])} selected sections")
    
    # Test flow with mock LLM
    print("\n🚀 Testing flow with mock LLM...")
    flow = AbstractionPlanningFlow(use_mock_llm=True, output_dir="test_output")
    
    try:
        result_state = flow.run(mock_shared_state)
        
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
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Run test
    test_abstraction_planning_flow() 