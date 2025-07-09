#!/usr/bin/env python3
"""
Test Phase 2 Priority Nodes: GenerateQueriesNode, PauseForQueryReviewNode, QueryProcessingRouter

These tests verify the 3-phase pattern implementation and basic functionality using parrot mode.
Focus: Simple functionality tests with minimal complexity as requested.
"""

import unittest
import sys
from pathlib import Path
from queue import Queue
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.proposal_agent_pf_dspy.nodes import (
    GenerateQueriesNode, 
    PauseForQueryReviewNode, 
    QueryProcessingRouter
)
from core.proposal_agent_pf_dspy.state import ProposalWorkflowState, create_shared_state
from core.proposal_agent_pf_dspy.dspy_modules import QueryGenerator


class TestPriorityNodes(unittest.TestCase):
    """Test the three priority nodes using simple test cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.topic = "machine learning interpretability"
        self.collection_name = "test_collection"
        self.chat_queue = Queue()
        self.flow_queue = Queue()
        
        # Create initial shared state
        self.shared_state = create_shared_state(
            topic=self.topic,
            collection_name=self.collection_name,
            chat_queue=self.chat_queue,
            flow_queue=self.flow_queue
        )
    
    def test_generate_queries_node_basic(self):
        """Test GenerateQueriesNode with mock DSPy module"""
        # Mock the DSPy module
        mock_dspy_module = Mock()
        mock_dspy_module.return_value.queries = [
            "interpretable machine learning methods",
            "explainable AI techniques",
            "model transparency frameworks"
        ]
        
        # Create and test node
        node = GenerateQueriesNode(mock_dspy_module)
        
        # Test prep phase
        prep_result = node.prep(self.shared_state)
        self.assertEqual(prep_result[0], self.topic)  # topic
        self.assertEqual(prep_result[1], [])  # initially empty queries
        
        # Test exec phase
        exec_result = node.exec(prep_result)
        self.assertEqual(len(exec_result), 3)
        self.assertIn("interpretable machine learning methods", exec_result)
        
        # Test post phase
        post_result = node.post(self.shared_state, prep_result, exec_result)
        self.assertEqual(post_result, "default")
        
        # Verify state was updated
        updated_state = ProposalWorkflowState.from_shared_dict(self.shared_state)
        self.assertEqual(updated_state.search_queries, exec_result)
        
        # Verify flow queue received messages
        flow_messages = []
        while not self.flow_queue.empty():
            flow_messages.append(self.flow_queue.get())
        self.assertTrue(any("Generated 3 search queries" in msg for msg in flow_messages))
    
    def test_pause_for_query_review_node(self):
        """Test PauseForQueryReviewNode HITL functionality"""
        # Set up state with queries
        self.shared_state['search_queries'] = [
            "interpretable ML methods",
            "explainable AI techniques"
        ]
        
        node = PauseForQueryReviewNode()
        
        # Test prep phase
        prep_result = node.prep(self.shared_state)
        queries, chat_queue, flow_queue = prep_result
        self.assertEqual(len(queries), 2)
        self.assertIs(chat_queue, self.chat_queue)
        self.assertIs(flow_queue, self.flow_queue)
        
        # Test exec phase
        exec_result = node.exec(prep_result)
        self.assertEqual(exec_result["interrupt_type"], "query_review")
        self.assertIn("queries", exec_result["context"])
        
        # Test post phase
        post_result = node.post(self.shared_state, prep_result, exec_result)
        self.assertEqual(post_result, "default")
        
        # Verify state was updated
        updated_state = ProposalWorkflowState.from_shared_dict(self.shared_state)
        self.assertEqual(updated_state.last_interrupt_type, "query_review")
        
        # Verify chat queue received review message
        chat_messages = []
        while not self.chat_queue.empty():
            msg = self.chat_queue.get()
            if msg is not None:
                chat_messages.append(msg)
        
        self.assertTrue(any("Please review the generated search queries" in msg for msg in chat_messages))
    
    def test_query_processing_router_approve(self):
        """Test QueryProcessingRouter with approval decision"""
        # Set up state with user input
        state = ProposalWorkflowState.from_shared_dict(self.shared_state)
        state._last_user_input = "approve"
        state.search_queries = ["test query 1", "test query 2"]
        shared_state = state.to_shared_dict()
        
        node = QueryProcessingRouter()
        
        # Test prep phase
        prep_result = node.prep(shared_state)
        user_input, user_input_raw, queries = prep_result
        self.assertEqual(user_input, "approve")
        self.assertEqual(len(queries), 2)
        
        # Test exec phase
        exec_result = node.exec(prep_result)
        self.assertEqual(exec_result["action"], "queries_approved")
        self.assertEqual(exec_result["reason"], "User approved queries")
        
        # Test post phase
        post_result = node.post(shared_state, prep_result, exec_result)
        self.assertEqual(post_result, "queries_approved")
        
        # Verify user input was cleared
        updated_state = ProposalWorkflowState.from_shared_dict(shared_state)
        self.assertIsNone(updated_state._last_user_input)
    
    def test_query_processing_router_regenerate(self):
        """Test QueryProcessingRouter with regeneration request"""
        # Set up state with regeneration request
        state = ProposalWorkflowState.from_shared_dict(self.shared_state)
        state._last_user_input = "!regenerate"
        shared_state = state.to_shared_dict()
        
        node = QueryProcessingRouter()
        
        # Test full cycle
        prep_result = node.prep(shared_state)
        exec_result = node.exec(prep_result)
        post_result = node.post(shared_state, prep_result, exec_result)
        
        self.assertEqual(exec_result["action"], "regenerate_queries")
        self.assertEqual(post_result, "regenerate_queries")
    
    def test_query_processing_router_edit_queries(self):
        """Test QueryProcessingRouter with edited queries"""
        # Set up state with edited queries
        state = ProposalWorkflowState.from_shared_dict(self.shared_state)
        state._last_user_input = "new query 1, new query 2, new query 3"
        state._last_user_input_raw = "new query 1, new query 2, new query 3"
        state.search_queries = ["old query 1", "old query 2"]
        shared_state = state.to_shared_dict()
        
        node = QueryProcessingRouter()
        
        # Test full cycle
        prep_result = node.prep(shared_state)
        exec_result = node.exec(prep_result)
        post_result = node.post(shared_state, prep_result, exec_result)
        
        self.assertEqual(exec_result["action"], "queries_approved")
        self.assertIn("edited_queries", exec_result)
        self.assertEqual(len(exec_result["edited_queries"]), 3)
        self.assertEqual(post_result, "queries_approved")
        
        # Verify state was updated with new queries
        updated_state = ProposalWorkflowState.from_shared_dict(shared_state)
        self.assertEqual(updated_state.search_queries, ["new query 1", "new query 2", "new query 3"])
    
    def test_three_node_sequence_integration(self):
        """Test the three nodes working together in sequence"""
        # Mock DSPy module
        mock_dspy_module = Mock()
        mock_dspy_module.return_value.queries = ["query 1", "query 2"]
        
        # Create nodes
        generate_node = GenerateQueriesNode(mock_dspy_module)
        pause_node = PauseForQueryReviewNode()
        router_node = QueryProcessingRouter()
        
        # Step 1: Generate queries
        prep1 = generate_node.prep(self.shared_state)
        exec1 = generate_node.exec(prep1)
        generate_node.post(self.shared_state, prep1, exec1)
        
        # Verify queries were generated
        state = ProposalWorkflowState.from_shared_dict(self.shared_state)
        self.assertEqual(len(state.search_queries), 2)
        
        # Step 2: Pause for review
        prep2 = pause_node.prep(self.shared_state)
        exec2 = pause_node.exec(prep2)
        pause_node.post(self.shared_state, prep2, exec2)
        
        # Verify pause was set
        state = ProposalWorkflowState.from_shared_dict(self.shared_state)
        self.assertEqual(state.last_interrupt_type, "query_review")
        
        # Step 3: Simulate user approval and route
        state._last_user_input = "approve"
        self.shared_state.update(state.to_shared_dict())
        
        prep3 = router_node.prep(self.shared_state)
        exec3 = router_node.exec(prep3)
        route_result = router_node.post(self.shared_state, prep3, exec3)
        
        # Verify routing decision
        self.assertEqual(route_result, "queries_approved")
        
        # Verify final state
        final_state = ProposalWorkflowState.from_shared_dict(self.shared_state)
        self.assertEqual(len(final_state.search_queries), 2)
        self.assertIsNone(final_state._last_user_input)


if __name__ == "__main__":
    print("Testing Phase 2 Priority Nodes...")
    print("=" * 50)
    
    unittest.main(verbosity=2) 