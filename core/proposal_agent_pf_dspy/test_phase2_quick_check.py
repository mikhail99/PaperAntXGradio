#!/usr/bin/env python3
"""
Quick Phase 2 Integration Check

Simple test to verify Phase 2 nodes integrate correctly with Phase 1 infrastructure.
Uses absolute imports and runs basic functionality tests.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from queue import Queue
from unittest.mock import Mock

def test_phase2_integration():
    """Test that Phase 2 nodes integrate with Phase 1 infrastructure"""
    
    print("🧪 Testing Phase 2 Integration...")
    
    try:
        # Import Phase 1 infrastructure
        from core.proposal_agent_pf_dspy.state import ProposalWorkflowState, create_shared_state
        from core.proposal_agent_pf_dspy.dspy_modules import QueryGenerator
        
        # Import Phase 2 nodes
        from core.proposal_agent_pf_dspy.nodes import (
            GenerateQueriesNode, 
            PauseForQueryReviewNode, 
            QueryProcessingRouter
        )
        
        print("✅ All imports successful")
        
        # Test state creation and typed management
        shared_state = create_shared_state("AI Ethics", "ethics_research")
        state = ProposalWorkflowState.from_shared_dict(shared_state)
        
        assert state.topic == "AI Ethics"
        assert state.collection_name == "ethics_research"
        assert isinstance(state.chat_queue, Queue)
        print("✅ Typed state management working")
        
        # Test node instantiation
        mock_dspy = Mock()
        mock_dspy.return_value.queries = ["query1", "query2", "query3"]
        
        generate_node = GenerateQueriesNode(mock_dspy)
        pause_node = PauseForQueryReviewNode()
        router_node = QueryProcessingRouter()
        
        print("✅ Node instantiation successful")
        
        # Test query generation flow
        prep_result = generate_node.prep(shared_state)
        exec_result = generate_node.exec(prep_result)
        post_result = generate_node.post(shared_state, prep_result, exec_result)
        
        # Verify state update
        updated_state = ProposalWorkflowState.from_shared_dict(shared_state)
        assert len(updated_state.search_queries) == 3
        assert post_result == "default"
        
        print("✅ Query generation node working")
        
        # Test pause node
        pause_prep = pause_node.prep(shared_state)
        pause_exec = pause_node.exec(pause_prep)
        pause_post = pause_node.post(shared_state, pause_prep, pause_exec)
        
        assert pause_exec["interrupt_type"] == "query_review"
        assert "queries" in pause_exec["context"]
        
        print("✅ Pause node working")
        
        # Test router with approval
        updated_state._last_user_input = "approve"
        shared_state.update(updated_state.to_shared_dict())
        
        router_prep = router_node.prep(shared_state)
        router_exec = router_node.exec(router_prep)
        router_post = router_node.post(shared_state, router_prep, router_exec)
        
        assert router_exec["action"] == "queries_approved"
        assert router_post == "queries_approved"
        
        print("✅ Router node working")
        
        # Test round-trip state serialization
        final_state = ProposalWorkflowState.from_shared_dict(shared_state)
        round_trip_dict = final_state.to_shared_dict()
        round_trip_state = ProposalWorkflowState.from_shared_dict(round_trip_dict)
        
        assert round_trip_state.topic == final_state.topic
        assert round_trip_state.search_queries == final_state.search_queries
        
        print("✅ State serialization working")
        
        print("\n🎉 Phase 2 Integration: ALL TESTS PASSED!")
        print("✨ Ready for flow implementation in Phase 3")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pocketflow_compatibility():
    """Test PocketFlow compatibility patterns"""
    
    print("\n🔍 Testing PocketFlow Compatibility...")
    
    try:
        from pocketflow import Node, Flow
        
        # Test that we can inherit from PocketFlow Node
        class TestNode(Node):
            def prep(self, shared):
                return "test_data"
            
            def exec(self, prep_res):
                return f"processed_{prep_res}"
            
            def post(self, shared, prep_res, exec_res):
                shared["result"] = exec_res
                return "default"
        
        # Test node execution
        test_node = TestNode()
        shared = {}
        
        prep_result = test_node.prep(shared)
        exec_result = test_node.exec(prep_result)
        post_result = test_node.post(shared, prep_result, exec_result)
        
        assert prep_result == "test_data"
        assert exec_result == "processed_test_data"
        assert post_result == "default"
        assert shared["result"] == "processed_test_data"
        
        print("✅ PocketFlow 3-phase pattern working")
        
        # Test basic flow creation
        another_node = TestNode()
        test_node >> another_node  # Test connection syntax
        
        flow = Flow(start=test_node)
        assert flow is not None
        
        print("✅ PocketFlow flow syntax working")
        
        return True
        
    except Exception as e:
        print(f"❌ PocketFlow compatibility failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 PHASE 2 INTEGRATION TEST")
    print("=" * 60)
    
    success1 = test_phase2_integration()
    success2 = test_pocketflow_compatibility()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎯 RESULT: ALL TESTS PASSED ✅")
        print("🚀 Phase 2 Priority Nodes Implementation Complete!")
        print("📋 Summary:")
        print("   • GenerateQueriesNode: ✅ Working")
        print("   • PauseForQueryReviewNode: ✅ Working") 
        print("   • QueryProcessingRouter: ✅ Working")
        print("   • State Management: ✅ Enhanced with typing")
        print("   • PocketFlow Integration: ✅ Compatible")
        exit(0)
    else:
        print("🚨 RESULT: SOME TESTS FAILED ❌")
        exit(1) 