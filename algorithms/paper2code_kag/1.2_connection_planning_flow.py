"""
Connection Planning Flow for analyzing dependencies and relationships between abstractions.
Part of Iteration 4: Extended Planning Stage 3 (Connection Planning).
"""

import os
import json
import logging
from typing import Dict, Any

# Import PocketFlow base class
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pocketflow import Flow

# Import connection planning nodes
from connection_planning_nodes import (
    AnalyzeDependenciesNode,
    MapConnectionsNode, 
    SaveConnectionsNode
)

class ConnectionPlanningFlow:
    """Orchestrates the complete connection planning workflow."""
    
    def __init__(self, use_mock_llm: bool = True, output_dir: str = "output"):
        self.use_mock_llm = use_mock_llm
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Initialize nodes with retry configuration
        self.analyze_dependencies_node = AnalyzeDependenciesNode(
            use_mock_llm=use_mock_llm,
            max_retries=3, 
            wait=1
        )
        
        self.map_connections_node = MapConnectionsNode(
            max_retries=3,
            wait=1
        )
        
        self.save_connections_node = SaveConnectionsNode(
            output_dir=output_dir,
            max_retries=3,
            wait=1
        )
        
        # Connect nodes in sequence
        self.analyze_dependencies_node >> self.map_connections_node >> self.save_connections_node
        
        # Create the flow
        self.flow = Flow(start=self.analyze_dependencies_node)
        
        self.logger.info("Connection Planning Flow initialized")
    
    def run(self, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete connection planning flow.
        
        Args:
            shared_state: Shared state containing abstraction planning results
            
        Returns:
            Updated shared state with connection planning results
        """
        try:
            self.logger.info("Starting connection planning flow...")
            
            # Validate input
            self._validate_input(shared_state)
            
            # Run the flow
            self.flow.run(shared_state)
            
            # Log success
            self.logger.info("Connection planning flow completed successfully")
            return shared_state
            
        except Exception as e:
            self.logger.error(f"Connection planning flow failed: {str(e)}")
            shared_state["connection_planning_completed"] = False
            shared_state["connection_planning_flow_status"] = f"failed: {str(e)}"
            raise
    
    def _validate_input(self, shared_state: Dict[str, Any]) -> None:
        """Validate that required input data is present."""
        required_keys = ["categorized_abstractions"]
        
        for key in required_keys:
            if key not in shared_state:
                raise ValueError(f"Missing required key in shared state: {key}")
        
        categorized_abs = shared_state["categorized_abstractions"]
        if not categorized_abs:
            raise ValueError("No categorized abstractions found")
        
        self.logger.info(f"Input validation passed: {len(categorized_abs)} abstractions to analyze")

def load_abstraction_planning_results(planning_file: str = "output/abstraction_plan.json") -> Dict[str, Any]:
    """Load abstraction planning results from JSON file."""
    
    if not os.path.exists(planning_file):
        raise FileNotFoundError(f"Abstraction planning file not found: {planning_file}")
    
    try:
        with open(planning_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        planning_results = data.get("abstraction_planning_results", {})
        
        # Extract the data we need for connection planning
        shared_state = {
            "categorized_abstractions": planning_results.get("categorized_abstractions", []),
            "raw_abstractions": planning_results.get("raw_abstractions", []),
            "abstraction_summary": planning_results.get("abstraction_summary", {}),
            "detection_metadata": planning_results.get("detection_metadata", {}),
            
            # Copy previous planning for reference
            "previous_planning": {
                "section_planning": data.get("previous_planning", {}).get("section_planning", {}),
                "abstraction_planning": planning_results
            }
        }
        
        logging.info(f"Loaded {len(shared_state['categorized_abstractions'])} categorized abstractions")
        return shared_state
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in planning file: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to load planning results: {str(e)}")

def run_connection_planning(planning_file: str = "output/abstraction_plan.json",
                          use_mock_llm: bool = True,
                          output_dir: str = "output") -> Dict[str, Any]:
    """
    Main function to run connection planning.
    
    Args:
        planning_file: Path to abstraction planning results
        use_mock_llm: Whether to use mock LLM for testing
        output_dir: Directory for output files
        
    Returns:
        Final shared state with connection planning results
    """
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Connection Planning (Iteration 4)")
    
    try:
        # Load abstraction planning results
        shared_state = load_abstraction_planning_results(planning_file)
        logger.info(f"Loaded abstraction planning from {planning_file}")
        
        # Create and run connection planning flow
        connection_flow = ConnectionPlanningFlow(
            use_mock_llm=use_mock_llm,
            output_dir=output_dir
        )
        
        result_state = connection_flow.run(shared_state)
        
        # Log final results
        logger.info("=== Connection Planning Results ===")
        logger.info(f"Total connections: {result_state.get('total_connections_found', 0)}")
        logger.info(f"Dependency chains: {len(result_state.get('dependency_chains', []))}")
        logger.info(f"Workflow sequences: {len(result_state.get('workflow_sequences', []))}")
        logger.info(f"Output file: {result_state.get('connection_plan_file', 'N/A')}")
        logger.info(f"File size: {result_state.get('connection_plan_file_size', 0)} bytes")
        
        return result_state
        
    except Exception as e:
        logger.error(f"Connection planning failed: {str(e)}")
        raise

# Convenience function for testing
def create_test_shared_state() -> Dict[str, Any]:
    """Create mock shared state for testing connection planning."""
    
    # Mock categorized abstractions based on Iteration 3 results
    mock_abstractions = [
        {
            "abstraction": {
                "id": "abs_0_neural_network_approach",
                "name": "neural network approach",
                "type": "technique",
                "description": "This paper presents a novel neural network approach for data processing",
                "confidence": 0.9,
                "detection_method": "rule_based",
                "section_source": "Abstract",
                "keywords": ["neural", "network", "approach"],
                "context": "novel neural network approach for data processing"
            },
            "category": "Neural Architecture",
            "subcategory": "Neural Network Design",
            "importance_score": 0.9,
            "relationships": ["deep learning techniques", "transformer architecture"],
            "implementation_complexity": "high"
        },
        {
            "abstraction": {
                "id": "abs_1_transformer_architecture",
                "name": "transformer architecture", 
                "type": "architecture",
                "description": "Our method uses deep learning techniques with transformer architecture",
                "confidence": 0.8,
                "detection_method": "rule_based",
                "section_source": "Abstract",
                "keywords": ["transformer", "architecture"],
                "context": "deep learning techniques with transformer architecture"
            },
            "category": "Neural Architecture",
            "subcategory": "Model Architecture", 
            "importance_score": 0.85,
            "relationships": ["neural network approach"],
            "implementation_complexity": "high"
        },
        {
            "abstraction": {
                "id": "abs_2_data_preprocessing",
                "name": "data preprocessing",
                "type": "method",
                "description": "Step 1: Data preprocessing using matrix operations",
                "confidence": 0.8,
                "detection_method": "rule_based", 
                "section_source": "Methodology",
                "keywords": ["data", "preprocessing", "matrix"],
                "context": "Data preprocessing using matrix operations"
            },
            "category": "Data Processing",
            "subcategory": "Preprocessing",
            "importance_score": 0.75,
            "relationships": ["matrix operations"],
            "implementation_complexity": "medium"
        },
        {
            "abstraction": {
                "id": "abs_3_attention_mechanism",
                "name": "attention mechanism",
                "type": "method",
                "description": "Step 3: Attention mechanism for sequence processing", 
                "confidence": 0.85,
                "detection_method": "rule_based",
                "section_source": "Methodology",
                "keywords": ["attention", "mechanism", "sequence"],
                "context": "Attention mechanism for sequence processing"
            },
            "category": "Neural Architecture",
            "subcategory": "Attention System",
            "importance_score": 0.8,
            "relationships": ["sequence processing"],
            "implementation_complexity": "medium"
        }
    ]
    
    return {
        "categorized_abstractions": mock_abstractions,
        "raw_abstractions": [abs_info["abstraction"] for abs_info in mock_abstractions],
        "abstraction_summary": {
            "total_categorized": len(mock_abstractions),
            "category_distribution": {
                "Neural Architecture": 3,
                "Data Processing": 1
            }
        },
        "detection_metadata": {
            "total_abstractions_detected": len(mock_abstractions),
            "detection_method": "hybrid"
        }
    }

if __name__ == "__main__":
    # Run with abstraction planning results if available, otherwise use mock data
    planning_file = "output/abstraction_plan.json"
    
    if os.path.exists(planning_file):
        print("Using real abstraction planning results...")
        result = run_connection_planning(planning_file, use_mock_llm=True)
    else:
        print("Using mock test data...")
        shared_state = create_test_shared_state()
        flow = ConnectionPlanningFlow(use_mock_llm=True, output_dir="output")
        result = flow.run(shared_state)
    
    print(f"\nConnection planning completed!")
    print(f"Results saved to: {result.get('connection_plan_file', 'N/A')}") 