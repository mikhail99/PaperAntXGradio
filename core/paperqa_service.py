# core/paperqa_service.py
import asyncio
import pickle
from pathlib import Path
import os
from typing import Dict, Any

from paperqa import Docs
from core.utils import get_local_llm_settings

# --- Global PaperQA Configuration (base settings) ---

llm_model = "ollama/gemma3:27b"
embedding_model = "ollama/nomic-embed-text:latest"

my_settings = get_local_llm_settings(llm_model, embedding_model)


class PaperQAService:
    async def query_documents(
        self, collection_name: str, question: str
    ) -> Dict[str, Any]:
        """
        Queries a pre-built PaperQA cache for a given collection.
        It loads a Docs object from a pickle file and uses it to answer a question.
        Returns a dictionary with 'answer_text', 'formatted_evidence', and 'error'.
        """
        if not collection_name:
            error_msg = "No collection name provided."
            return {"answer_text": "", "formatted_evidence": "", "error": error_msg}

        try:
            # 1. Define path to the cache file
            cache_file_path = Path("data/collections") / collection_name / "paperqa_cache.pkl"

            if not cache_file_path.exists():
                error_msg = f"Cache file not found at {cache_file_path}. Please build the cache for this collection first."
                print(error_msg)
                return {"answer_text": "", "formatted_evidence": "", "error": error_msg}

            # 2. Load the Docs object from the pickle file
            print(f"Loading PaperQA cache from: {cache_file_path}")
            with open(cache_file_path, "rb") as f:
                docs = pickle.load(f)
            print(f"Cache loaded successfully. Contains {len(docs.docs)} documents.")

            # 3. Ask the question using the loaded docs and settings
            print(f"Querying PaperQA with: '{question}'")
            response = await docs.aquery(question, settings=my_settings)
            print("PaperQA query finished.")

            answer_text = response.formatted_answer if response and response.formatted_answer else "No answer found by PaperQA."
            
            return {"answer_text": answer_text, "formatted_evidence": "", "error": None}

        except Exception as e:
            error_message = f"Error during PaperQA processing: {str(e)}"
            print(error_message)
            return {"answer_text": "", "formatted_evidence": "", "error": error_message} 