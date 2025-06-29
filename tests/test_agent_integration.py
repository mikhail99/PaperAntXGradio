import pytest
import asyncio
from pathlib import Path
import os
import pickle
import uuid

from core.proposal_agent.graph import graph
from core.collections_manager import CollectionsManager
from paperqa import Docs
from core.data_models import Article

# Create a dummy file for the test
TEST_DOC_CONTENT = """
# The Art of Mocking

Mocking is a technique in software testing where a substitute object is used
in place of a real object. This is often done to isolate the code under test
from its dependencies. This allows for more focused and faster tests.

There are two main types of mocks: stubs and mocks. Stubs provide canned
responses to calls made during the test, while mocks are objects that can
verify that certain calls were made on them.
"""
TEST_DOCS_DIR = Path("./test_docs_temp")
TEST_CACHE_DIR = Path("./data/collections")
TEST_COLLECTION_NAME = "LLM_Reasoning_Agents"

@pytest.fixture(scope="module")
def setup_test_collection():
    """
    Creates a temporary collection, adds a document, and builds the required
    PaperQA cache file for the integration test.
    """
    print(f"--- Setting up test collection ---")
    # 1. Create dummy document directory
    TEST_DOCS_DIR.mkdir(exist_ok=True)
    doc_path = TEST_DOCS_DIR / "mocking_paper.md"
    with open(doc_path, "w") as f:
        f.write(TEST_DOC_CONTENT)
        
    # 2. Use CollectionsManager to create the collection and add the article
    manager = CollectionsManager()
    if TEST_COLLECTION_NAME in manager.collections:
        manager.delete_collection(TEST_COLLECTION_NAME)
        
    print(f"\n--- Creating collection '{TEST_COLLECTION_NAME}' for integration test ---")
    manager.create_collection(
        name=TEST_COLLECTION_NAME, 
        description="A temporary collection for integration testing."
    )
    test_article = Article(title="The Art of Mocking", abstract=TEST_DOC_CONTENT)
    manager.add_article(TEST_COLLECTION_NAME, test_article)

    # 3. Manually build the PaperQA cache, because the service expects it
    print(f"--- Building PaperQA cache for '{TEST_COLLECTION_NAME}' ---")
    collection_cache_dir = TEST_CACHE_DIR / TEST_COLLECTION_NAME
    collection_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = collection_cache_dir / "paperqa_cache.pkl"
    
    # Use PaperQA's Docs to process the file and create the cache
    paperqa_docs = Docs(paths=[str(doc_path)])
    with open(cache_file, "wb") as f:
        pickle.dump(paperqa_docs, f)

    # Yield to let the test run
    yield
    
    # 4. Teardown
    print(f"\n--- Deleting collection '{TEST_COLLECTION_NAME}' and cache ---")
    manager.delete_collection(TEST_COLLECTION_NAME)
    os.remove(cache_file)
    os.rmdir(collection_cache_dir)
    os.remove(doc_path)
    os.rmdir(TEST_DOCS_DIR)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_agent_integration_happy_path():
    """
    Tests a simple, successful workflow using the real LLM and PaperQA services
    against the existing 'LLM_Reasoning_Agents' collection.
    This test is slow and requires a running Ollama instance.
    """
    print(f"--- Running test_agent_integration_happy_path ---")
    # Use a unique thread_id for each test run to ensure isolation.
    thread_id = f"integration-test-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    # 1. Start the agent
    initial_state = {
        "topic": "LLM reasoning abilities",
        "collection_name": TEST_COLLECTION_NAME
    }
    
    print(f"\n--- Running Test on collection '{TEST_COLLECTION_NAME}': Initial agent invocation ---")
    final_state = await graph.ainvoke(initial_state, config)
    
    assert final_state.get('paused_on') == 'human_query_review_node', "Agent should pause for query review"
    assert 'search_queries' in final_state and final_state['search_queries'], "Agent should have generated search queries"
    print(f"✅ Queries generated: {final_state['search_queries']}")

    # 2. Continue past query review
    print("\n--- Running Test: Continuing past query review ---")
    graph.update_state(config, {"human_feedback": "continue"})
    final_state = await graph.ainvoke(None, config)
        
    assert final_state.get('paused_on') == 'human_insight_review_node', "Agent should pause for insight review"
    assert 'knowledge_gap' in final_state and final_state['knowledge_gap'], "Agent should have synthesized a knowledge gap"
    print(f"✅ Knowledge gap synthesized.")

    # 3. Continue past insight review to the final review stage
    print("\n--- Running Test: Continuing past insight review ---")
    graph.update_state(config, {"human_feedback": "continue"})
    final_state = await graph.ainvoke(None, config)
    
    assert final_state.get('paused_on') == 'human_review_node', "Agent should pause for the final proposal review"
    assert 'proposal_draft' in final_state and final_state['proposal_draft'], "Agent should have drafted a proposal"
    print(f"✅ Proposal draft created.")

    print("\n✅ Integration Test Passed: Agent completed happy path workflow successfully.") 