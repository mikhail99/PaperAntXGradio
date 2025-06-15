"""
Seed Plan Flow for Paper2ImplementationDoc.
Uses a Knowledge Base to seed a plan for a new document.

This module provides a function to create a pre-configured pocketflow.Flow
for the plan seeding stage.
"""

import logging
from pocketflow import Flow
from typing import Dict, Any

from algorithms.paper2code_kag.seed_plan_nodes import (
    FindSimilarPapersNode,
    RetrieveAndScoreAbstractionsNode,
    GenerateDraftPlanNode,
    SeedPlanSharedState
)
from algorithms.paper2code_kag.utils.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

def create_seed_plan_flow(kb: KnowledgeBase) -> Flow:
    """
    Creates and returns the plan seeding flow.
    
    Args:
        kb: An initialized KnowledgeBase instance.
        
    Returns:
        A configured pocketflow.Flow for plan seeding.
    """
    find_similar_node = FindSimilarPapersNode(kb=kb, max_retries=3, wait=5)
    retrieve_abstractions_node = RetrieveAndScoreAbstractionsNode(kb=kb, max_retries=3, wait=5)
    generate_plan_node = GenerateDraftPlanNode(max_retries=3, wait=5)
    
    find_similar_node >> retrieve_abstractions_node >> generate_plan_node
    
    logger.info("Initialized plan seeding flow")
    return Flow(start=find_similar_node)

def run_seed_plan(shared_state: Dict[str, Any], kb: KnowledgeBase) -> Dict[str, Any]:
    """

    Runs the complete plan seeding flow on a given shared state.
    
    Args:
        shared_state: Shared state containing document sections.
        kb: An initialized KnowledgeBase instance.
        
    Returns:
        Updated shared state with the seeded draft plan.
    """
    logger.info("🚀 Starting plan seeding flow")
    
    if "selected_sections" not in shared_state:
        raise ValueError("Missing 'selected_sections' in shared state.")
        
    try:
        seed_flow = create_seed_plan_flow(kb)
        seed_flow.run(shared_state)
        
        logger.info("✅ Plan seeding completed successfully")
        logger.info(f"📊 Results: {len(shared_state.get('draft_abstractions', []))} draft abstractions generated.")
        
        shared_state["seed_plan_completed"] = True
        shared_state["seed_plan_status"] = "success"
        
        return shared_state
        
    except Exception as e:
        logger.error(f"❌ Plan seeding flow failed: {str(e)}", exc_info=True)
        shared_state["seed_plan_completed"] = False
        shared_state["seed_plan_status"] = "failed"
        shared_state["seed_plan_error"] = str(e)
        raise

def test_seed_plan_flow():
    """Test the plan seeding flow."""
    print("🧪 Testing Plan Seeding Flow")
    
    # This test requires a populated KB. For now, we mock it.
    # In a real scenario, we'd run bootstrapping first.
    print("NOTE: This test uses a mocked KnowledgeBase.")
    
    class MockKnowledgeBase:
        def find_similar_documents(self, *args, **kwargs):
            return [{"doc_id": "doc1", "score": 0.9}]
        def get_abstractions_by_doc_ids(self, *args, **kwargs):
            return [{"id": "abs1", "name": "Mock Abstraction", "type": "model"}]

    mock_kb = MockKnowledgeBase()
    
    mock_shared_state = {
        "doc_id": "new_doc",
        "selected_sections": [
            {"title": "Intro", "content": "This is a new paper about models."}
        ]
    }
    
    print("\n🚀 Testing flow...")
    try:
        result_state = run_seed_plan(mock_shared_state, mock_kb)
        print("✅ Flow completed successfully!")
        print(f"Draft abstractions: {result_state.get('draft_abstractions', [])}")
        
    except Exception as e:
        print(f"❌ Flow failed: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    test_seed_plan_flow() 