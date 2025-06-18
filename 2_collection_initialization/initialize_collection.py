import os
import sys
import requests
from typing import Literal
from tqdm import tqdm

# Ensure project root is in sys.path for core imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.collections_manager import CollectionsManager
from core.data_models import Article

# It's good practice to handle potential dspy import errors if it's an optional dependency
import dspy


# --- Configuration ---
# Source ChromaDB where the main library of papers is stored
SOURCE_CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "1_library_chroma_db_output")
SOURCE_COLLECTION_NAME = "HuggingFaceDailyPapers"

# Base path where new curated collections will be stored
DEST_BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "collections")

# DSPy and Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
# As requested, using 'gemma3:4b'. Make sure this model is available in your Ollama instance.
OLLAMA_MODEL = "gemma3:4b"
TOP_K_SEARCH = 50  # Number of initial papers to retrieve for filtering


# --- DSPy Setup ---

class PaperRelevance(dspy.Signature):
    """
    Determines if a paper's abstract is highly relevant to a given collection description.
    """
    collection_description = dspy.InputField(
        desc="The high-level description of the new collection being created."
    )
    paper_abstract = dspy.InputField(
        desc="The abstract of a single paper to evaluate for relevance."
    )
    fit: Literal['Yes', 'No', 'Maybe'] = dspy.OutputField(
        desc="Is this paper a good fit for the collection? Answer 'Yes', 'No', or 'Maybe'."
    )


def setup_dspy_lm():
    """Initializes and configures the DSPy language model."""
    print(f"Configuring DSPy to use Ollama model '{OLLAMA_MODEL}' at {OLLAMA_BASE_URL}")
    # We use dspy.OllamaLocal, the standard client for local Ollama models.
    lm = dspy.LM('ollama_chat/'+OLLAMA_MODEL, api_base=OLLAMA_BASE_URL, api_key='')
    dspy.configure(lm=lm)

    

# --- Helper Functions ---

def download_pdf(arxiv_id: str, output_folder: str):
    """Downloads a PDF from arXiv given its ID."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    output_path = os.path.join(output_folder, f"{arxiv_id}.pdf")
    
    if os.path.exists(output_path):
        return True # Skip if already downloaded

    try:
        response = requests.get(pdf_url, stream=True, timeout=30)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF for {arxiv_id}: {e}")
        return False


# --- Main Logic ---

def main(new_collection_name: str, new_collection_description: str):
    """
    Main function to create a new curated collection.
    """
    print(f"Starting creation of new collection: '{new_collection_name}'")

    # 1. Set up all necessary paths and directories for the new collection
    new_collection_dir = os.path.join(DEST_BASE_PATH, new_collection_name)
    new_pdfs_dir = os.path.join(new_collection_dir, "pdfs")

    os.makedirs(new_pdfs_dir, exist_ok=True)
    print(f"Created directories for new collection at: {new_collection_dir}")


    # 3. Connect to the source ChromaDB and find the main collection by name
    source_manager = CollectionsManager(persist_directory=SOURCE_CHROMA_DB_DIR)
    source_collection = source_manager.get_collection_by_name(SOURCE_COLLECTION_NAME)

    if not source_collection:
        print(f"Error: Source collection '{SOURCE_COLLECTION_NAME}' not found in {SOURCE_CHROMA_DB_DIR}")
        return

    print(f"Searching for top {TOP_K_SEARCH} relevant papers from '{SOURCE_COLLECTION_NAME}'...")
    
    # 4. Perform semantic search to get candidate papers
    try:
        retrieved_articles = source_manager.search_articles(
            collection_id=source_collection.id,
            query=new_collection_description,
            limit=TOP_K_SEARCH
        )
    except Exception as e:
        print(f"Error: Failed during article search: {e}")
        return
        
    print(f"Found {len(retrieved_articles)} potentially relevant articles. Filtering with LLM...")

    # 5. Filter the retrieved articles using the LLM
    relevant_articles = []
    setup_dspy_lm()
    relevance_checker = dspy.Predict(PaperRelevance)
    for article in tqdm(retrieved_articles, desc="Filtering papers with LLM"):
        # Ensure abstract is not empty
        if not article.abstract:
            continue
            
        try:
            result = relevance_checker(
                collection_description=new_collection_description,
                paper_abstract=article.abstract
            )
            print(result.fit)
            # We include papers marked as 'Yes' or 'Maybe'
            if result.fit in ['Yes', 'Maybe']:
                relevant_articles.append(article)
        except Exception as e:
            print(f"\nError processing article {article.id} with LLM: {e}")

    print(f"Found {len(relevant_articles)} relevant papers to add to the new collection.")

    if not relevant_articles:
        print("No relevant papers found. Aborting.")
        return

    # 6. Create the new collection and add the filtered papers
    new_collection = source_manager.create_collection(
        name=new_collection_name,
        description=new_collection_description
    )

    for article in tqdm(relevant_articles, desc="Adding papers to new collection"):
        try:
            source_manager.add_article(new_collection.id, article)
        except Exception as e:
            print(f"Failed to add article {article.id} to new collection: {e}")

    print(f"Successfully created new collection '{new_collection_name}' with {len(relevant_articles)} papers.")

    # 7. Download PDFs for the newly added papers
    print("Downloading PDFs for the new collection...")
    for article in tqdm(relevant_articles, desc="Downloading PDFs"):
        download_pdf(article.id, new_pdfs_dir)

    print("Script finished successfully.")
    
    print("\n--- Debugging Information ---")
    all_collections_summary = source_manager.get_all_collections()
    if not all_collections_summary:
        print("No collections found.")
    else:
        print(f"Found {len(all_collections_summary)} collections:")
        for coll_summary in all_collections_summary:
            collection = source_manager.get_collection(coll_summary.id)
            if collection:
                num_articles = len(collection.articles)
                print(f"- Collection: '{collection.name}' contains {num_articles} articles.")
            else:
                 print(f"- Collection: '{coll_summary.name}' could not be loaded.")
    print("---------------------------\n")



if __name__ == "__main__":
    # --- Define the new collection here ---
    collection_name = "LLM_Reasoning_Agents"
    collection_description = "Papers about LLM reasoning agents, coding agents, problem solving agents, etc."
    
    print(f"Initializing collection '{collection_name}'...")
    main(collection_name, collection_description)
