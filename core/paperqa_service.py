print("--- Importing: core.paperqa_service ---")
# core/paperqa_service.py
import asyncio
import pickle
from pathlib import Path
import os
import re
from typing import Dict, Any

from paperqa import Docs
from core.utils import get_local_llm_settings
from core.collections_manager import CollectionsManager

# --- Global PaperQA Configuration (base settings) ---

llm_model = "ollama/qwen3:4b"
embedding_model = "ollama/nomic-embed-text:latest"

# --- Lazy Service Initializer ---
my_settings = None
collections_manager = None

def _initialize_paperqa_services():
    """Lazily initializes and returns service instances for PaperQA."""
    global my_settings, collections_manager
    if my_settings is None:
        print("--- Initializing PaperQA LLM settings ---")
        my_settings = get_local_llm_settings(llm_model, embedding_model)
    if collections_manager is None:
        print("--- Initializing CollectionsManager ---")
        collections_manager = CollectionsManager()


class PaperQAService:
    async def query_documents(
        self, collection_name: str, question: str
    ) -> Dict[str, Any]:
        """
        Queries a pre-built PaperQA cache for a given collection.
        It loads a Docs object from a pickle file and uses it to answer a question.
        Returns a dictionary with 'answer_text', 'formatted_evidence', and 'error'.
        """
        _initialize_paperqa_services()

        if not collection_name:
            error_msg = "No collection name provided."
            return {"answer_text": "", "formatted_evidence": "", "error": error_msg}
        
        collection = collections_manager.get_collection(collection_name)
        if not collection:
            error_msg = f"Collection with ID '{collection_name}' not found."
            return {"answer_text": "", "formatted_evidence": "", "error": error_msg}
        
        collection_name = collection.name

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

            # Create a map from citation string to its link
            citation_to_link_map = {}
            for text_obj in docs.texts:
                try:
                    doc = text_obj.doc
                    citation_text = doc.citation
                    link = None

                    # 1. Try to find a direct URL in the citation string
                    url_match = re.search(r'https?://[^\s,]+', citation_text)
                    if url_match:
                        # Clean trailing characters like periods or commas
                        link = url_match.group(0).rstrip('.,')

                    # 2. If no URL, try to find an arXiv ID
                    elif 'arXiv:' in citation_text:
                        arxiv_match = re.search(r'arXiv:(\d{4}\.\d{4,5})', citation_text)
                        if arxiv_match:
                            arxiv_id = arxiv_match.group(1)
                            link = f"https://arxiv.org/abs/{arxiv_id}"

                    # 3. If a link was found, add it to the map for replacement later
                    if link:
                        citation_to_link_map[citation_text] = link
                except Exception as e:
                    print(f"Could not create link for a document: {e}")

            # 3. Ask the question using the loaded docs and settings
            print(f"Querying PaperQA with: '{question}'")
            response = await docs.aquery(question, settings=my_settings)
            print("PaperQA query finished.")

            answer_text = response.formatted_answer if response and response.formatted_answer else "No answer found by PaperQA."
            
            # HACK: The formatted_answer sometimes includes the question. We strip it here.
            # Final attempt: A more aggressive regex approach to remove the question
            stripped_question = re.escape(question.strip())
            # This regex looks for an optional "Question: " prefix and then the question text,
            # ignoring leading/trailing whitespace and case. It removes the entire line.
            pattern = re.compile(r"^\s*(Question:\s*)?" + stripped_question + r"\s*$", re.IGNORECASE | re.MULTILINE)
            answer_text = pattern.sub('', answer_text).strip()

            # --- Definitive Link Replacement: Two-Stage Find and Replace ---

            # Stage 1: Find all potential citations in the text and map them to links
            citations_to_replace = {}
            for full_citation, link in citation_to_link_map.items():
                # Extract author (everything before the first comma) and year for a more robust match
                author_match = re.search(r'^([^,]+)', full_citation)
                year_match = re.search(r'(\d{4})', full_citation)

                if author_match and year_match:
                    author = author_match.group(1).strip()
                    year = year_match.group(1).strip()
                    
                    # Flexible regex to find citations like (Author et al., Year)
                    citation_pattern = re.compile(f"\(.*?{re.escape(author)}[^)]*?{re.escape(year)}.*?\)")
                    
                    for match in citation_pattern.finditer(answer_text):
                        # Store the exact text that was matched and its corresponding link
                        citations_to_replace[match.group(0)] = link

            # Stage 2: Replace the found citations with markdown links
            for text, link in citations_to_replace.items():
                answer_text = answer_text.replace(text, f"[{text}]({link})")

            return {"answer_text": answer_text, "context": response.context, "error": None}

        except Exception as e:
            error_message = f"Error during PaperQA processing: {str(e)}"
            print(error_message)
            return {"answer_text": "", "formatted_evidence": "", "error": error_message} 