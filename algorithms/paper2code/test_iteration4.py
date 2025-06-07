"""
Test suite for Iteration 4: Connection Planning
Tests dependency analysis, workflow mapping, and connection planning flow.
"""

import os
import json
import sys
import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pocketflow import Flow

# Import connection planning components
from utils.connection_mapper import ConnectionMapper, Connection, DependencyChain, ConnectionType
from connection_planning_nodes import (
    AnalyzeDependenciesNode,
    MapConnectionsNode,
    SaveConnectionsNode,
    ConnectionInfo,
    DependencyChainInfo
)
from connection_planning_flow import (
    ConnectionPlanningFlow,
    load_abstraction_planning_results,
    run_connection_planning,
    create_test_shared_state
)

class TestConnectionMapper(unittest.TestCase):
    """Test the ConnectionMapper utility class."""
    
    def setUp(self):
        self.mapper = ConnectionMapper(use_mock_llm=True)
        
        # Mock abstractions for testing
        self.test_abstractions = [
            {
                "id": "abs_0_neural_network",
                "name": "neural network",
                "type": "technique",
                "description": "A neural network approach for processing data",
                "keywords": ["neural", "network"],
                "context": "neural network approach for processing"
            },
            {
                "id": "abs_1_transformer_architecture",
                "name": "transformer architecture",
                "type": "architecture", 
                "description": "Transformer architecture with attention mechanism",
                "keywords": ["transformer", "architecture"],
                "context": "transformer architecture with attention"
            },
            {
                "id": "abs_2_attention_mechanism",
                "name": "attention mechanism",
                "type": "method",
                "description": "Attention mechanism for sequence processing",
                "keywords": ["attention", "mechanism"],
                "context": "attention mechanism for sequence processing"
            }
        ]
        
        self.test_context = "Our neural network uses transformer architecture with attention mechanism for sequence processing."
    
    def test_rule_based_detection(self):
        """Test rule-based connection detection."""
        connections = self.mapper.detect_connections_rule_based(
            self.test_abstractions, 
            self.test_context
        )
        
        # Should find some connections based on context patterns
        self.assertIsInstance(connections, list)
        
        # Check connection structure
        if connections:
            conn = connections[0]
            self.assertIsInstance(conn, Connection)
            self.assertIn(conn.detection_method, ["rule_based"])
            self.assertIsInstance(conn.confidence, float)
            self.assertTrue(0 <= conn.confidence <= 1)
    
    def test_llm_detection(self):
        """Test mock LLM connection detection."""
        connections = self.mapper.detect_connections_llm(
            self.test_abstractions,
            self.test_context
        )
        
        self.assertIsInstance(connections, list)
        
        # Mock LLM should find intelligent connections
        if connections:
            conn = connections[0]
            self.assertEqual(conn.detection_method, "llm")
            self.assertIsInstance(conn.connection_type, ConnectionType)
    
    def test_hybrid_detection(self):
        """Test hybrid connection detection approach."""
        connections = self.mapper.detect_connections_hybrid(
            self.test_abstractions,
            self.test_context
        )
        
        self.assertIsInstance(connections, list)
        
        # Hybrid should combine rule-based and LLM results
        detection_methods = {conn.detection_method for conn in connections}
        possible_methods = {"rule_based", "llm", "hybrid"}
        self.assertTrue(detection_methods.issubset(possible_methods))
    
    def test_dependency_chain_analysis(self):
        """Test dependency chain analysis."""
        # Create test connections with dependencies
        test_connections = [
            Connection(
                source_id="abs_0",
                target_id="abs_1", 
                connection_type=ConnectionType.DEPENDENCY,
                confidence=0.8,
                description="Test dependency",
                evidence=["test"],
                detection_method="test"
            ),
            Connection(
                source_id="abs_1",
                target_id="abs_2",
                connection_type=ConnectionType.DEPENDENCY,
                confidence=0.7,
                description="Test dependency 2",
                evidence=["test"],
                detection_method="test"
            )
        ]
        
        chains = self.mapper.analyze_dependency_chains(test_connections)
        
        self.assertIsInstance(chains, list)
        if chains:
            chain = chains[0]
            self.assertIsInstance(chain, DependencyChain)
            self.assertGreaterEqual(len(chain.abstractions), 2)

class TestConnectionPlanningNodes(unittest.TestCase):
    """Test individual connection planning nodes."""
    
    def setUp(self):
        # Create mock shared state based on Iteration 3 results
        self.shared_state = create_test_shared_state()
        
        # Ensure output directory exists
        os.makedirs("test_output", exist_ok=True)
    
    def tearDown(self):
        # Clean up test files
        test_files = [
            "test_output/connection_plan.json"
        ]
        for file_path in test_files:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def test_analyze_dependencies_node(self):
        """Test the AnalyzeDependenciesNode."""
        node = AnalyzeDependenciesNode(use_mock_llm=True)
        
        # Test prep
        prep_result = node.prep(self.shared_state)
        self.assertIn("abstractions", prep_result)
        self.assertIn("context_text", prep_result)
        self.assertGreater(len(prep_result["abstractions"]), 0)
        
        # Test exec
        exec_result = node.exec(prep_result)
        self.assertIn("connections", exec_result)
        self.assertIn("dependency_chains", exec_result)
        self.assertIn("connection_matrix", exec_result)
        
        # Test post
        action = node.post(self.shared_state, prep_result, exec_result)
        self.assertEqual(action, "default")
        self.assertIn("detected_connections", self.shared_state)
        self.assertIn("dependency_chains", self.shared_state)
    
    def test_map_connections_node(self):
        """Test the MapConnectionsNode."""
        # First run dependency analysis to populate shared state
        dep_node = AnalyzeDependenciesNode(use_mock_llm=True)
        dep_prep = dep_node.prep(self.shared_state)
        dep_exec = dep_node.exec(dep_prep)
        dep_node.post(self.shared_state, dep_prep, dep_exec)
        
        # Now test mapping node
        map_node = MapConnectionsNode()
        
        # Test prep
        prep_result = map_node.prep(self.shared_state)
        self.assertIn("connections", prep_result)
        self.assertIn("abstractions", prep_result)
        
        # Test exec
        exec_result = map_node.exec(prep_result)
        self.assertIn("workflow_sequences", exec_result)
        self.assertIn("connection_patterns", exec_result)
        self.assertIn("workflow_graph", exec_result)
        
        # Test post
        action = map_node.post(self.shared_state, prep_result, exec_result)
        self.assertEqual(action, "default")
        self.assertIn("workflow_sequences", self.shared_state)
    
    def test_save_connections_node(self):
        """Test the SaveConnectionsNode."""
        # Prepare shared state with required data
        self.shared_state["detected_connections"] = [
            {
                "source_id": "abs_0",
                "target_id": "abs_1",
                "connection_type": "dependency",
                "confidence": 0.8,
                "description": "Test connection",
                "evidence": ["test"],
                "detection_method": "test",
                "bidirectional": False
            }
        ]
        self.shared_state["dependency_chains"] = []
        self.shared_state["connection_matrix"] = {"abs_0": ["abs_1:dependency"]}
        
        save_node = SaveConnectionsNode(output_dir="test_output")
        
        # Test prep
        prep_result = save_node.prep(self.shared_state)
        self.assertIn("connections", prep_result)
        self.assertIn("dependency_chains", prep_result)
        
        # Test exec
        exec_result = save_node.exec(prep_result)
        self.assertIn("output_file", exec_result)
        self.assertIn("file_size", exec_result)
        self.assertTrue(os.path.exists(exec_result["output_file"]))
        
        # Test post
        action = save_node.post(self.shared_state, prep_result, exec_result)
        self.assertEqual(action, "default")
        self.assertTrue(self.shared_state["connection_planning_completed"])

class TestConnectionPlanningFlow(unittest.TestCase):
    """Test the complete connection planning flow."""
    
    def setUp(self):
        self.output_dir = "test_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create test shared state
        self.shared_state = create_test_shared_state()
    
    def tearDown(self):
        # Clean up test files
        test_files = [
            "test_output/connection_plan.json"
        ]
        for file_path in test_files:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def test_connection_planning_flow_initialization(self):
        """Test ConnectionPlanningFlow initialization."""
        flow = ConnectionPlanningFlow(use_mock_llm=True, output_dir=self.output_dir)
        
        self.assertIsNotNone(flow.analyze_dependencies_node)
        self.assertIsNotNone(flow.map_connections_node)
        self.assertIsNotNone(flow.save_connections_node)
        self.assertIsInstance(flow.flow, Flow)
    
    def test_connection_planning_flow_execution(self):
        """Test complete connection planning flow execution."""
        flow = ConnectionPlanningFlow(use_mock_llm=True, output_dir=self.output_dir)
        
        result_state = flow.run(self.shared_state)
        
        # Check that flow completed successfully
        self.assertTrue(result_state.get("connection_planning_completed", False))
        self.assertEqual(result_state.get("connection_planning_flow_status"), "success")
        
        # Check that all expected data is present
        self.assertIn("detected_connections", result_state)
        self.assertIn("dependency_chains", result_state)
        self.assertIn("workflow_sequences", result_state)
        self.assertIn("connection_summary", result_state)
        
        # Check that output file was created
        self.assertIn("connection_plan_file", result_state)
        self.assertTrue(os.path.exists(result_state["connection_plan_file"]))
    
    def test_flow_validation(self):
        """Test input validation in connection planning flow."""
        flow = ConnectionPlanningFlow(use_mock_llm=True, output_dir=self.output_dir)
        
        # Test with missing required data
        invalid_state = {"some_other_data": "value"}
        
        with self.assertRaises(ValueError):
            flow.run(invalid_state)
        
        # Test with empty abstractions
        empty_state = {"categorized_abstractions": []}
        
        with self.assertRaises(ValueError):
            flow.run(empty_state)

class TestConnectionPlanningIntegration(unittest.TestCase):
    """Test integration with previous iteration results."""
    
    def setUp(self):
        self.output_dir = "test_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create mock abstraction planning file
        self.abstraction_plan_file = os.path.join(self.output_dir, "test_abstraction_plan.json")
        
        mock_abstraction_data = {
            "abstraction_planning_results": {
                "categorized_abstractions": [
                    {
                        "abstraction": {
                            "id": "abs_0_test",
                            "name": "test algorithm",
                            "type": "algorithm",
                            "description": "A test algorithm implementation",
                            "confidence": 0.9,
                            "detection_method": "rule_based",
                            "section_source": "Abstract",
                            "keywords": ["test", "algorithm"],
                            "context": "test algorithm implementation"
                        },
                        "category": "Computational",
                        "subcategory": "Algorithm Design",
                        "importance_score": 0.8,
                        "relationships": [],
                        "implementation_complexity": "medium"
                    }
                ],
                "raw_abstractions": [],
                "abstraction_summary": {"total_categorized": 1},
                "detection_metadata": {"total_abstractions_detected": 1}
            },
            "previous_planning": {
                "section_planning": {}
            }
        }
        
        with open(self.abstraction_plan_file, 'w', encoding='utf-8') as f:
            json.dump(mock_abstraction_data, f, indent=2)
    
    def tearDown(self):
        # Clean up test files
        test_files = [
            self.abstraction_plan_file,
            "test_output/connection_plan.json"
        ]
        for file_path in test_files:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def test_load_abstraction_planning_results(self):
        """Test loading abstraction planning results."""
        shared_state = load_abstraction_planning_results(self.abstraction_plan_file)
        
        self.assertIn("categorized_abstractions", shared_state)
        self.assertIn("raw_abstractions", shared_state)
        self.assertIn("abstraction_summary", shared_state)
        self.assertIn("previous_planning", shared_state)
        
        # Check structure
        categorized_abs = shared_state["categorized_abstractions"]
        self.assertGreater(len(categorized_abs), 0)
        self.assertIn("abstraction", categorized_abs[0])
    
    def test_run_connection_planning_with_file(self):
        """Test running connection planning with real abstraction file."""
        result_state = run_connection_planning(
            planning_file=self.abstraction_plan_file,
            use_mock_llm=True,
            output_dir=self.output_dir
        )
        
        self.assertIn("total_connections_found", result_state)
        self.assertIn("connection_plan_file", result_state)
        self.assertTrue(os.path.exists(result_state["connection_plan_file"]))
        
        # Verify output file structure
        with open(result_state["connection_plan_file"], 'r', encoding='utf-8') as f:
            connection_data = json.load(f)
        
        self.assertIn("connection_planning_results", connection_data)
        self.assertIn("previous_planning", connection_data)
        
        planning_results = connection_data["connection_planning_results"]
        self.assertIn("detected_connections", planning_results)
        self.assertIn("dependency_chains", planning_results)
        self.assertIn("workflow_sequences", planning_results)
        self.assertIn("connection_summary", planning_results)

class TestConnectionPlanningOutput(unittest.TestCase):
    """Test connection planning output structure and content."""
    
    def setUp(self):
        self.output_dir = "test_output"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def tearDown(self):
        # Clean up test files
        if os.path.exists("test_output/connection_plan.json"):
            os.remove("test_output/connection_plan.json")
    
    def test_connection_plan_output_structure(self):
        """Test the structure of connection plan output."""
        shared_state = create_test_shared_state()
        flow = ConnectionPlanningFlow(use_mock_llm=True, output_dir=self.output_dir)
        
        result_state = flow.run(shared_state)
        
        # Load and verify output file
        with open(result_state["connection_plan_file"], 'r', encoding='utf-8') as f:
            connection_plan = json.load(f)
        
        # Check top-level structure
        self.assertIn("connection_planning_results", connection_plan)
        self.assertIn("previous_planning", connection_plan)
        
        # Check connection planning results structure
        results = connection_plan["connection_planning_results"]
        expected_keys = [
            "detected_connections",
            "dependency_chains", 
            "workflow_sequences",
            "connection_matrix",
            "workflow_graph",
            "connection_summary",
            "workflow_insights",
            "connection_patterns",
            "analysis_metadata"
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check metadata
        metadata = results["analysis_metadata"]
        self.assertIn("total_connections_analyzed", metadata)
        self.assertIn("analysis_method", metadata)
        self.assertIn("analysis_timestamp", metadata)
    
    def test_connection_summary_metrics(self):
        """Test connection summary metrics."""
        shared_state = create_test_shared_state()
        flow = ConnectionPlanningFlow(use_mock_llm=True, output_dir=self.output_dir)
        
        result_state = flow.run(shared_state)
        
        summary = result_state["connection_summary"]
        
        # Check required metrics
        required_metrics = [
            "total_connections",
            "total_dependency_chains", 
            "total_workflow_sequences",
            "connection_type_distribution",
            "detection_method_distribution",
            "average_confidence",
            "workflow_complexity"
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, summary)
        
        # Check value types and ranges
        self.assertIsInstance(summary["total_connections"], int)
        self.assertIsInstance(summary["average_confidence"], float)
        self.assertTrue(0 <= summary["average_confidence"] <= 1)
        self.assertIn(summary["workflow_complexity"], ["low", "medium", "high"])

def run_all_tests():
    """Run all test suites."""
    test_suites = [
        TestConnectionMapper,
        TestConnectionPlanningNodes,
        TestConnectionPlanningFlow,
        TestConnectionPlanningIntegration,
        TestConnectionPlanningOutput
    ]
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    print("="*60)
    print("ITERATION 4: CONNECTION PLANNING - TEST RESULTS")
    print("="*60)
    
    for test_suite in test_suites:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_suite)
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
        
        print(f"\n{test_suite.__name__}:")
        print(f"  Tests: {result.testsRun}")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
    
    print("\n" + "="*60)
    print(f"TOTAL RESULTS:")
    print(f"  Tests: {total_tests}")
    print(f"  Failures: {total_failures}")
    print(f"  Errors: {total_errors}")
    print(f"  Success Rate: {((total_tests - total_failures - total_errors) / total_tests * 100):.1f}%")
    print("="*60)
    
    return total_failures + total_errors == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 