"""
This script defines the main workflow for processing a single paper,
ingesting it into the Knowledge Base, and generating planning documents.
"""
import logging
import os
from pathlib import Path
from typing import Dict, Any

from algorithms.paper2code_kag.utils.path_setup import setup_paths
setup_paths()

from algorithms.paper2code_kag.utils.knowledge_base import KnowledgeBase
from algorithms.paper2code_kag.abstraction_detection_flow import run_abstraction_detection
from algorithms.paper2code_kag.connection_analysis_flow import run_connection_analysis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_and_analyze_paper(
    doc_path: str,
    kb: KnowledgeBase,
    output_dir: str = "output",
    use_mock_llm: bool = True
) -> Dict[str, Any]:
    """
    Orchestrates the end-to-end processing for a single document.
    """
    logger.info(f"🚀 Starting full processing for document: {doc_path}")
    doc_id = Path(doc_path).stem
    
    # 1. Ingest document into PaperQA index
    logger.info(f"Ingesting '{doc_id}' into PaperQA index...")
    kb.add_document(doc_path, doc_id=doc_id)
    
    # 2. Extract sections
    logger.info("Extracting sections from the document...")
    sections = extract_sections(doc_path)
    
    # Initialize shared state
    shared_state: Dict[str, Any] = {
        "doc_id": doc_id,
        "doc_path": doc_path,
        "sections": sections,
        "text_stats": calculate_text_stats("\n".join(s['content'] for s in sections)),
    }
    

    # 4. Run Abstraction Detection Flow
    shared_state = run_abstraction_detection(shared_state, use_mock_llm=use_mock_llm, output_dir=output_dir)

    # 5. Run Connection Analysis Flow
    shared_state = run_connection_analysis(shared_state, use_mock_llm=use_mock_llm, output_dir=output_dir)
    
    # 6. Add results to Knowledge Base
    logger.info(f"Adding processed abstractions and connections for '{doc_id}' to the Knowledge Base...")
    kb.add_abstractions(
        doc_id=doc_id,
        abstractions=shared_state.get("categorized_abstractions", [])
    )
    kb.add_connections(
        doc_id=doc_id,
        connections=shared_state.get("categorized_connections", [])
    )
    
    logger.info(f"✅ Successfully processed and ingested document: {doc_id}")
    
    return shared_state


def main():
    """Main function to run the ingestion flow."""
    kb_path = "knowledge_base"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Knowledge Base
    logger.info(f"Initializing Knowledge Base at '{kb_path}'...")
    kb = KnowledgeBase(kb_path)

    # Path to the paper to be processed
    doc_to_process = "seed_data/seed_paper_1.txt"
    if not os.path.exists(doc_to_process):
        logger.error(f"Document not found: {doc_to_process}")
        return
        
    try:
        # Run the full pipeline
        final_state = process_and_plan_paper(
            doc_path=doc_to_process,
            kb=kb,
            output_dir=output_dir,
            use_mock_llm=True
        )
        logger.info("🎉 End-to-end paper ingestion and planning complete.")
        
    except Exception as e:
        logger.error(f"An error occurred during the process: {e}", exc_info=True)

if __name__ == "__main__":
    main() 