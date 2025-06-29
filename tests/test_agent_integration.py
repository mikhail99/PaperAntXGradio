import pytest
import asyncio
from pathlib import Path
import os
import pickle
import uuid

from core.proposal_agent.modern_service import create_modern_service
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
async def test_modern_agent_integration():
    """
    Tests the modern interrupt() pattern service using the new HIL workflow.
    This test is slow and requires a running Ollama instance.
    """
    print(f"--- Running test_modern_agent_integration ---")
    
    # Create modern service
    service = create_modern_service()
    
    # 1. Start the agent
    config = {
        "topic": "LLM reasoning abilities",
        "collection_name": TEST_COLLECTION_NAME,
        "local_papers_only": True
    }
    
    print(f"\n--- Running Test: Starting modern agent ---")
    
    # Collect initial steps until first interrupt
    steps = []
    async for result in service.start_agent(config):
        steps.append(result)
        if result.get("step") == "human_input_required":
            break
    
    # Verify we got the query review interrupt
    last_step = steps[-1]
    assert last_step.get("step") == "human_input_required", "Agent should pause for input"
    assert last_step.get("interrupt_type") == "query_review", "Should pause for query review"
    thread_id = last_step.get("thread_id")
    print(f"✅ Query review interrupt detected: {last_step.get('message')}")

    # 2. Continue past query review
    print("\n--- Running Test: Continuing past query review ---")
    steps = []
    async for result in service.continue_agent(thread_id, "continue"):
        steps.append(result)
        if result.get("step") == "human_input_required":
            break
    
    # Verify we got the insight review interrupt
    last_step = steps[-1]
    assert last_step.get("step") == "human_input_required", "Agent should pause for insight review"
    assert last_step.get("interrupt_type") == "insight_review", "Should pause for insight review"
    print(f"✅ Insight review interrupt detected: {last_step.get('message')}")

    # 3. Continue past insight review to final review
    print("\n--- Running Test: Continuing past insight review ---")
    steps = []
    async for result in service.continue_agent(thread_id, "continue"):
        steps.append(result)
        if result.get("step") == "human_input_required":
            break
    
    # Verify we got the final review interrupt  
    last_step = steps[-1]
    assert last_step.get("step") == "human_input_required", "Agent should pause for final review"
    assert last_step.get("interrupt_type") == "final_review", "Should pause for final review"
    print(f"✅ Final review interrupt detected: {last_step.get('message')}")

    print("\n✅ Modern Integration Test Passed: Agent completed modern HIL workflow successfully.") 