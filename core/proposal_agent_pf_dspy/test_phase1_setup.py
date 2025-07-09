"""
Phase 1 Setup Tests

This module tests that our Phase 1 setup is working correctly:
- All imports are functional
- Typed state management works
- PocketFlow integration is ready
- Module structure is complete

Run these tests to verify Phase 1 completion before moving to Phase 2.
"""

import pytest
from queue import Queue
from typing import Dict, Any

# Test imports work correctly
def test_imports():
    """Test that all core imports work without errors."""
    try:
        # Core PocketFlow imports
        from pocketflow import Flow, Node
        
        # Our module imports
        from .state import ProposalWorkflowState, create_shared_state, KnowledgeGap, Critique
        from .dspy_modules import QueryGenerator, KnowledgeSynthesizer, ProposalWriter, ProposalReviewer
        from .signatures import GenerateQueries, SynthesizeKnowledge, WriteProposal, ReviewProposal
        from .parrot import MockLM, MockPaperQAService
        from .main import create_research_service
        from .orchestrator import PocketFlowOrchestrator
        from .validation import PocketFlowValidator
        from .mermaid import MermaidGenerator
        
        # Check that classes are importable
        assert Flow is not None
        assert Node is not None
        assert ProposalWorkflowState is not None
        assert QueryGenerator is not None
        assert MockLM is not None
        assert PocketFlowOrchestrator is not None
        
        print("✅ All imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during imports: {e}")
        return False

def test_typed_state_management():
    """Test that our typed state management works correctly."""
    try:
        from .state import ProposalWorkflowState, create_shared_state, KnowledgeGap, Critique
        
        # Test state creation
        state = ProposalWorkflowState(
            topic="AI in Healthcare",
            collection_name="medical_research"
        )
        
        # Test state has correct default values
        assert state.topic == "AI in Healthcare"
        assert state.collection_name == "medical_research"
        assert state.search_queries == []
        assert state.literature_summaries == []
        assert state.proposal_draft == ""
        assert state.is_approved == False
        assert state.revision_cycles == 0
        
        # Test conversation ID is generated
        assert state.conversation_id is not None
        assert state.thread_id == state.conversation_id
        
        # Test shared dict conversion
        shared_dict = state.to_shared_dict()
        assert isinstance(shared_dict, dict)
        assert shared_dict["topic"] == "AI in Healthcare"
        assert shared_dict["collection_name"] == "medical_research"
        
        # Test round-trip conversion
        restored_state = ProposalWorkflowState.from_shared_dict(shared_dict)
        assert restored_state.topic == state.topic
        assert restored_state.collection_name == state.collection_name
        assert restored_state.conversation_id == state.conversation_id
        
        # Test state updates
        state.update(search_queries=["query1", "query2"])
        assert state.search_queries == ["query1", "query2"]
        
        # Test invalid field update raises error
        try:
            state.update(invalid_field="value")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        # Test append_to functionality
        state.append_to("literature_summaries", "summary1")
        assert state.literature_summaries == ["summary1"]
        
        # Test create_shared_state function
        initial_shared = create_shared_state("Test Topic", "test_collection")
        assert initial_shared["topic"] == "Test Topic"
        assert initial_shared["collection_name"] == "test_collection"
        assert isinstance(initial_shared["chat_queue"], Queue)
        assert isinstance(initial_shared["flow_queue"], Queue)
        
        print("✅ Typed state management works correctly")
        return True
        
    except Exception as e:
        print(f"❌ State management test failed: {e}")
        return False

def test_pydantic_models():
    """Test that our Pydantic models work correctly."""
    try:
        from .state import KnowledgeGap, Critique
        
        # Test KnowledgeGap model
        gap = KnowledgeGap(
            synthesized_summary="A summary of the literature",
            knowledge_gap="A specific gap in knowledge",
            is_novel=True
        )
        
        # Test serialization
        gap_dict = gap.model_dump()
        assert gap_dict["synthesized_summary"] == "A summary of the literature"
        assert gap_dict["knowledge_gap"] == "A specific gap in knowledge"
        assert gap_dict["is_novel"] == True
        
        # Test Critique model
        critique = Critique(
            score=0.85,
            justification="Good quality with minor issues"
        )
        
        critique_dict = critique.model_dump()
        assert critique_dict["score"] == 0.85
        assert critique_dict["justification"] == "Good quality with minor issues"
        
        # Test that these integrate with state management
        from .state import ProposalWorkflowState
        state = ProposalWorkflowState("Topic", "Collection")
        state.knowledge_gap = gap
        state.review_team_feedback = {"novelty": critique}
        
        # Test serialization with nested models
        shared_dict = state.to_shared_dict()
        assert isinstance(shared_dict["knowledge_gap"], dict)
        assert isinstance(shared_dict["review_team_feedback"]["novelty"], dict)
        
        # Test deserialization with nested models
        restored_state = ProposalWorkflowState.from_shared_dict(shared_dict)
        assert isinstance(restored_state.knowledge_gap, KnowledgeGap)
        assert isinstance(restored_state.review_team_feedback["novelty"], Critique)
        assert restored_state.knowledge_gap.is_novel == True
        assert restored_state.review_team_feedback["novelty"].score == 0.85
        
        print("✅ Pydantic models work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Pydantic model test failed: {e}")
        return False

def test_dspy_modules():
    """Test that DSPy modules are properly configured."""
    try:
        from .dspy_modules import QueryGenerator, KnowledgeSynthesizer, ProposalWriter, ProposalReviewer
        from .signatures import GenerateQueries, SynthesizeKnowledge, WriteProposal, ReviewProposal
        
        # Test module instantiation
        query_gen = QueryGenerator()
        assert query_gen is not None
        assert hasattr(query_gen, 'generate')
        
        knowledge_syn = KnowledgeSynthesizer()
        assert knowledge_syn is not None
        assert hasattr(knowledge_syn, 'synthesize')
        
        proposal_writer = ProposalWriter()
        assert proposal_writer is not None
        assert hasattr(proposal_writer, 'write')
        
        proposal_reviewer = ProposalReviewer()
        assert proposal_reviewer is not None
        assert hasattr(proposal_reviewer, 'review')
        
        print("✅ DSPy modules configured correctly")
        return True
        
    except Exception as e:
        print(f"❌ DSPy module test failed: {e}")
        return False

def test_parrot_services():
    """Test that parrot/mock services work correctly."""
    try:
        from .parrot import MockLM, MockPaperQAService
        import asyncio
        
        # Test MockLM
        mock_lm = MockLM()
        assert mock_lm is not None
        
        # Test query generation response
        query_response = mock_lm([{"text": "generates a list of search queries based on topic"}])
        assert len(query_response) == 1
        assert "queries" in query_response[0]["text"] or "parrot query" in query_response[0]["text"]
        
        # Test MockPaperQAService
        mock_service = MockPaperQAService()
        assert mock_service is not None
        
        # Test async query (would need to be run in async context in real usage)
        async def test_query():
            result = await mock_service.query_documents("test_collection", "test query")
            assert "answer_text" in result
            assert "parrot summary" in result["answer_text"]
            return result
        
        # Run the async test
        result = asyncio.run(test_query())
        assert result is not None
        
        print("✅ Parrot services work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Parrot service test failed: {e}")
        return False

def test_pocketflow_basic():
    """Test basic PocketFlow functionality works."""
    try:
        from pocketflow import Flow, Node
        
        # Create a simple test node
        class TestNode(Node):
            def prep(self, shared):
                return shared.get("input", "default")
            
            def exec(self, prep_res):
                return f"processed_{prep_res}"
            
            def post(self, shared, prep_res, exec_res):
                shared["output"] = exec_res
                return "default"
        
        # Create a simple flow
        node = TestNode()
        flow = Flow(start=node)
        
        # Test flow execution
        shared = {"input": "test_input"}
        result = flow.run(shared)
        
        assert shared["output"] == "processed_test_input"
        
        print("✅ Basic PocketFlow functionality works")
        return True
        
    except Exception as e:
        print(f"❌ PocketFlow basic test failed: {e}")
        return False

def test_service_creation():
    service = create_research_service(use_parrot=True)
    assert service is not None
    assert service.use_parrot == True

def run_all_phase1_tests():
    """Run all Phase 1 tests and report results."""
    print("🧪 Running Phase 1 Setup Tests...")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Typed State Management", test_typed_state_management),
        ("Pydantic Models", test_pydantic_models),
        ("DSPy Modules", test_dspy_modules),
        ("Parrot Services", test_parrot_services),
        ("PocketFlow Basic", test_pocketflow_basic),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Phase 1 Test Results:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Phase 1 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Phase 1 setup is complete and ready for Phase 2!")
        return True
    else:
        print("⚠️  Some tests failed. Please fix issues before proceeding to Phase 2.")
        return False

if __name__ == "__main__":
    run_all_phase1_tests() 