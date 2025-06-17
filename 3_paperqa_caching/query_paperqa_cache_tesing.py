import os
import sys
import asyncio
import pickle
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from paperqa import Docs, Settings
from core.utils import get_local_llm_settings

# --- Configuration ---
LLM_MODEL = "ollama/gemma3:4b"
EMBEDDING_MODEL = "ollama/nomic-embed-text:latest"

async def main(collection_name: str, question: str):
    """
    Loads a PaperQA cache from a pickle file and asks a question.
    """
    print(f"--- Querying PaperQA cache for collection: '{collection_name}' ---")

    # 1. Define path to the cache file
    cache_file_path = Path("data/collections") / collection_name / "paperqa_cache.pkl"

    if not cache_file_path.exists():
        print(f"Error: Cache file not found at {cache_file_path}")
        print("Please run '3_paperqa_caching/build_paperqa_cache.py' first.")
        return

    # 2. Load the Docs object from the pickle file
    print(f"Loading cache from: {cache_file_path}")
    try:
        with open(cache_file_path, "rb") as f:
            docs = pickle.load(f)
        print(f"Cache loaded successfully. Contains {len(docs.docs)} documents.")
    except Exception as e:
        print(f"Failed to load pickle file: {e}")
        return

    # 3. Define a custom QA prompt as requested by the user
    my_qa_prompt = (
        "Answer the question '{question}' based *only* on the provided context.\n"
        "Use the context below to form your answer. "
        "You must cite the context using the key provided, like (Author et al., Year).\n"
        "If the context is insufficient to answer the question, write 'I cannot answer this question based on the provided context.' and nothing else.\n\n"
        "Context:\n{context}"
    )

    # 4. Configure settings for the query
    print("Configuring settings for the query...")
    # Get base settings for local LLM
    query_settings = get_local_llm_settings(LLM_MODEL, EMBEDDING_MODEL)
    # Apply the custom prompt
    query_settings.prompts.qa = my_qa_prompt
    # We can also set other parameters, like number of sources to retrieve
    query_settings.answer.answer_max_sources = 5

    # 5. Ask the question using the loaded docs and custom settings
    print(f"\nAsking question: '{question}'")
    answer = await docs.aquery(question, settings=query_settings)

    # 6. Print the final, formatted answer
    print("\n\n--- Formatted Answer ---")
    print(answer.formatted_answer)
    print("--------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a pickled PaperQA cache.")
    collection_name = "LLM_Resoning_Agents_Papers"
    question = "What are the main challenges for LLM agents?"

    asyncio.run(main(collection_name, question)) 