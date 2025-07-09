#!/usr/bin/env python3
"""
Test Phase 3 Nodes: All remaining workflow nodes

These tests verify the 3-phase pattern implementation for all nodes including async LiteratureReviewNode
and the full DSPy pipeline using parrot mode for simplicity.
"""

import unittest
import sys
import asyncio
from pathlib import Path
from queue import Queue
from unittest.mock import Mock, AsyncMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.proposal_agent_pf_dspy.nodes import (
    SynthesizeKnowledgeNode,
    WriteProposalNode,
    ReviewProposalNode,
    LiteratureReviewNode,
    PauseForProposalReviewNode,
    ProposalProcessingRouter
)
from core.proposal_agent_pf_dspy.state import ProposalWorkflowState, KnowledgeGap, Critique, create_shared_state
from core.proposal_agent_pf_dspy.dspy_modules import KnowledgeSynthesizer, ProposalWriter, ProposalReviewer


class TestPhase3Nodes(unittest.TestCase):
    """Test all Phase 3 nodes using simple test cases"""
    
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
        
        # Add some literature summaries for testing
        self.shared_state['literature_summaries'] = [
            "Summary 1: Recent advances in interpretable ML methods.",
            "Summary 2: Gaps in current explainability research.",
            "Summary 3: Need for better transparency frameworks."
        ]
    
    def test_synthesize_knowledge_node(self):
        """Test SynthesizeKnowledgeNode with mock DSPy module"""
        # Mock the DSPy module
        mock_dspy_module = Mock()
        mock_knowledge_gap = KnowledgeGap(
            synthesized_summary="Comprehensive review of interpretability methods",
            knowledge_gap="Limited formal verification for interpretable models",
            is_novel=True
        )
        mock_dspy_module.return_value.knowledge_gap = mock_knowledge_gap
        
        # Create and test node
        node = SynthesizeKnowledgeNode(mock_dspy_module)
        
        # Test prep phase
        prep_result = node.prep(self.shared_state)
        self.assertEqual(prep_result[0], self.topic)
        self.assertIn("Summary 1", prep_result[1])  # Check literature summaries included
        
        # Test exec phase
        exec_result = node.exec(prep_result)
        self.assertIsInstance(exec_result, KnowledgeGap)
        self.assertEqual(exec_result.knowledge_gap, "Limited formal verification for interpretable models")
        self.assertTrue(exec_result.is_novel)
        
        # Test post phase
        post_result = node.post(self.shared_state, prep_result, exec_result)
        self.assertEqual(post_result, "default")
        
        # Verify state was updated
        updated_state = ProposalWorkflowState.from_shared_dict(self.shared_state)
        self.assertEqual(updated_state.knowledge_gap, exec_result)
        
        # Verify flow queue received messages
        flow_messages = []
        while not self.flow_queue.empty():
            flow_messages.append(self.flow_queue.get())
        self.assertTrue(any("Synthesized knowledge gap" in msg for msg in flow_messages))
    
    def test_write_proposal_node(self):
        """Test WriteProposalNode with mock DSPy module"""
        # Set up state with knowledge gap
        knowledge_gap = KnowledgeGap(
            synthesized_summary="Test summary",
            knowledge_gap="Test gap",
            is_novel=True
        )
        self.shared_state['knowledge_gap'] = knowledge_gap.model_dump()
        
        # Mock the DSPy module
        mock_dspy_module = Mock()
        mock_dspy_module.return_value.proposal = "A comprehensive research proposal addressing the identified knowledge gap..."
        
        # Create and test node
        node = WriteProposalNode(mock_dspy_module)
        
        # Test prep phase
        prep_result = node.prep(self.shared_state)
        self.assertIn("Test gap", prep_result[0])  # knowledge gap summary
        self.assertEqual(prep_result[1], "")  # no prior feedback
        
        # Test exec phase
        exec_result = node.exec(prep_result)
        self.assertIn("comprehensive research proposal", exec_result)
        
        # Test post phase
        post_result = node.post(self.shared_state, prep_result, exec_result)
        self.assertEqual(post_result, "default")
        
        # Verify state was updated
        updated_state = ProposalWorkflowState.from_shared_dict(self.shared_state)
        self.assertEqual(updated_state.proposal_draft, exec_result)
    
    def test_review_proposal_node(self):
        """Test ReviewProposalNode with mock DSPy module"""
        # Set up state with proposal draft
        self.shared_state['proposal_draft'] = "A test proposal for machine learning interpretability research."
        
        # Mock the DSPy module
        mock_dspy_module = Mock()
        mock_critique = Critique(
            score=0.85,
            justification="Strong proposal with good theoretical foundation but needs more experimental details."
        )
        mock_dspy_module.return_value.critique = mock_critique
        
        # Create and test node
        node = ReviewProposalNode(mock_dspy_module, review_aspect="novelty and contribution")
        
        # Test prep phase
        prep_result = node.prep(self.shared_state)
        self.assertIn("test proposal", prep_result[0])
        self.assertEqual(prep_result[1], "novelty and contribution")
        
        # Test exec phase
        exec_result = node.exec(prep_result)
        self.assertIsInstance(exec_result, Critique)
        self.assertEqual(exec_result.score, 0.85)
        self.assertIn("experimental details", exec_result.justification)
        
        # Test post phase
        post_result = node.post(self.shared_state, prep_result, exec_result)
        self.assertEqual(post_result, "default")
        
        # Verify state was updated
        updated_state = ProposalWorkflowState.from_shared_dict(self.shared_state)
        self.assertIn("ai_reviewer", updated_state.review_team_feedback)
        self.assertEqual(updated_state.review_team_feedback["ai_reviewer"], exec_result)
    
    def test_literature_review_node_async(self):
        """Test LiteratureReviewNode with async mock document service"""
        # Mock the async document service
        mock_doc_service = AsyncMock()
        mock_doc_service.query_documents.side_effect = [
            {"answer_text": "Literature summary for interpretable ML methods"},
            {"answer_text": "Literature summary for explainability research"},
            {"error": "Query failed for some reason"}  # Test error handling
        ]
        
        # Set up state with search queries
        self.shared_state['search_queries'] = [
            "interpretable machine learning methods",
            "explainability research gaps", 
            "transparency frameworks"
        ]
        
        # Create and test node
        node = LiteratureReviewNode(mock_doc_service)
        
        # Run async test
        async def run_async_test():
            # Test prep phase
            prep_result = await node.prep_async(self.shared_state)
            queries, collection_name = prep_result
            self.assertEqual(len(queries), 3)
            self.assertEqual(collection_name, self.collection_name)
            
            # Test exec phase
            exec_result = await node.exec_async(prep_result)
            self.assertEqual(len(exec_result), 3)
            self.assertIn("Literature summary for interpretable ML", exec_result[0])
            self.assertIn("Literature summary for explainability", exec_result[1])
            self.assertIn("Error processing query", exec_result[2])  # Error case
            
            # Test post phase
            post_result = await node.post_async(self.shared_state, prep_result, exec_result)
            self.assertEqual(post_result, "default")
            
            # Verify state was updated
            updated_state = ProposalWorkflowState.from_shared_dict(self.shared_state)
            self.assertEqual(updated_state.literature_summaries, exec_result)
        
        # Run the async test
        asyncio.run(run_async_test())
    
    def test_pause_for_proposal_review_node(self):
        """Test PauseForProposalReviewNode HITL functionality"""
        # Set up state with proposal and review
        self.shared_state['proposal_draft'] = "A comprehensive proposal for ML interpretability research..."
        ai_critique = Critique(score=0.75, justification="Good approach but needs refinement")
        self.shared_state['review_team_feedback'] = {"ai_reviewer": ai_critique.model_dump()}
        self.shared_state['revision_cycles'] = 1
        
        node = PauseForProposalReviewNode()
        
        # Test prep phase
        prep_result = node.prep(self.shared_state)
        proposal_draft, review_feedback, revision_cycles, chat_queue, flow_queue = prep_result
        self.assertIn("comprehensive proposal", proposal_draft)
        self.assertEqual(revision_cycles, 1)
        self.assertIs(chat_queue, self.chat_queue)
        
        # Test exec phase
        exec_result = node.exec(prep_result)
        self.assertEqual(exec_result["interrupt_type"], "proposal_review")
        self.assertIn("proposal_preview", exec_result["context"])
        self.assertIn("ai_review", exec_result["context"])
        
        # Test post phase
        post_result = node.post(self.shared_state, prep_result, exec_result)
        self.assertEqual(post_result, "default")
        
        # Verify chat queue received review message
        chat_messages = []
        while not self.chat_queue.empty():
            msg = self.chat_queue.get()
            if msg is not None:
                chat_messages.append(msg)
        
        self.assertTrue(any("AI review complete" in msg for msg in chat_messages))
    
    def test_proposal_processing_router_approve(self):
        """Test ProposalProcessingRouter with approval decision"""
        # Set up state with user input
        state = ProposalWorkflowState.from_shared_dict(self.shared_state)
        state._last_user_input = "approve"
        state.revision_cycles = 2
        shared_state = state.to_shared_dict()
        
        node = ProposalProcessingRouter()
        
        # Test prep phase
        prep_result = node.prep(shared_state)
        user_input, user_input_raw, revision_cycles = prep_result
        self.assertEqual(user_input, "approve")
        self.assertEqual(revision_cycles, 2)
        
        # Test exec phase
        exec_result = node.exec(prep_result)
        self.assertEqual(exec_result["action"], "approved")
        self.assertEqual(exec_result["reason"], "User approved the proposal")
        
        # Test post phase
        post_result = node.post(shared_state, prep_result, exec_result)
        self.assertEqual(post_result, "approved")
        
        # Verify state was updated
        updated_state = ProposalWorkflowState.from_shared_dict(shared_state)
        self.assertTrue(updated_state.is_approved)
        self.assertIsNone(updated_state._last_user_input)
    
    def test_proposal_processing_router_revision(self):
        """Test ProposalProcessingRouter with revision request"""
        # Set up state with revision feedback
        state = ProposalWorkflowState.from_shared_dict(self.shared_state)
        state._last_user_input = "needs more details on methodology"
        state._last_user_input_raw = "needs more details on methodology"
        state.revision_cycles = 0
        shared_state = state.to_shared_dict()
        
        node = ProposalProcessingRouter()
        
        # Test full cycle
        prep_result = node.prep(shared_state)
        exec_result = node.exec(prep_result)
        post_result = node.post(shared_state, prep_result, exec_result)
        
        self.assertEqual(exec_result["action"], "revision_requested")
        self.assertEqual(exec_result["feedback"], "needs more details on methodology")
        self.assertEqual(post_result, "revision_requested")
        
        # Verify state was updated
        updated_state = ProposalWorkflowState.from_shared_dict(shared_state)
        self.assertEqual(updated_state.revision_cycles, 1)
        self.assertIn("user_review", updated_state.review_team_feedback)
        self.assertEqual(
            updated_state.review_team_feedback["user_review"].justification,
            "needs more details on methodology"
        )
    
    def test_full_pipeline_integration(self):
        """Test multiple nodes working together in sequence"""
        # Mock DSPy modules
        mock_synthesizer = Mock()
        mock_knowledge_gap = KnowledgeGap(
            synthesized_summary="Integrated summary",
            knowledge_gap="Test knowledge gap",
            is_novel=True
        )
        mock_synthesizer.return_value.knowledge_gap = mock_knowledge_gap
        
        mock_writer = Mock()
        mock_writer.return_value.proposal = "Generated proposal text"
        
        mock_reviewer = Mock()
        mock_critique = Critique(score=0.8, justification="Good proposal overall")
        mock_reviewer.return_value.critique = mock_critique
        
        # Create nodes
        synthesize_node = SynthesizeKnowledgeNode(mock_synthesizer)
        write_node = WriteProposalNode(mock_writer)
        review_node = ReviewProposalNode(mock_reviewer)
        
        # Step 1: Synthesize knowledge
        prep1 = synthesize_node.prep(self.shared_state)
        exec1 = synthesize_node.exec(prep1)
        synthesize_node.post(self.shared_state, prep1, exec1)
        
        # Verify knowledge gap was set
        state = ProposalWorkflowState.from_shared_dict(self.shared_state)
        self.assertEqual(state.knowledge_gap, mock_knowledge_gap)
        
        # Step 2: Write proposal
        prep2 = write_node.prep(self.shared_state)
        exec2 = write_node.exec(prep2)
        write_node.post(self.shared_state, prep2, exec2)
        
        # Verify proposal was written
        state = ProposalWorkflowState.from_shared_dict(self.shared_state)
        self.assertEqual(state.proposal_draft, "Generated proposal text")
        
        # Step 3: Review proposal
        prep3 = review_node.prep(self.shared_state)
        exec3 = review_node.exec(prep3)
        review_node.post(self.shared_state, prep3, exec3)
        
        # Verify review was added
        final_state = ProposalWorkflowState.from_shared_dict(self.shared_state)
        self.assertIn("ai_reviewer", final_state.review_team_feedback)
        self.assertEqual(final_state.review_team_feedback["ai_reviewer"], mock_critique)


if __name__ == "__main__":
    print("Testing Phase 3 Nodes...")
    print("=" * 50)
    
    unittest.main(verbosity=2) 