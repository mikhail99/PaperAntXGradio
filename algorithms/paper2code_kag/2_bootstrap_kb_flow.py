"""
Bootstrap Knowledge Base Flow

This script orchestrates the bootstrapping of the Knowledge Base (KB) by
processing a set of seed documents. For each document, it runs the full
planning pipeline (section selection, abstraction detection, connection analysis)
and ingests the results into the KB.
"""
import logging
import os
from pathlib import Path
from typing import Dict, Any

from algorithms.paper2code_kag.utils.path_setup import setup_paths
setup_paths()

from algorithms.paper2code_kag.utils.knowledge_base import KnowledgeBase
from algorithms.paper2code_kag.utils.pdf_utils import extract_sections, calculate_text_stats
from algorithms.paper2code_kag.section_selection_flow import run_section_selection
from algorithms.paper2code_kag.abstraction_detection_flow import run_abstraction_detection
from algorithms.paper2code_kag.connection_analysis_flow import run_connection_analysis

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def bootstrap_knowledge_base(
    seed_data_dir: str,
    kb: KnowledgeBase,
    output_dir: str = "output",
    use_mock_llm: bool = True
):
    """
    Processes all seed documents and populates the Knowledge Base.
    """
    logger.info("🚀 Starting Knowledge Base bootstrapping process...")
    seed_files = [f for f in os.listdir(seed_data_dir) if f.endswith('.txt')]

    for seed_file in seed_files:
        doc_path = os.path.join(seed_data_dir, seed_file)
        doc_id = Path(doc_path).stem
        logger.info(f"--- Processing seed document: {doc_id} ---")

        try:
            # 1. Ingest document into PaperQA index
            logger.info(f"Ingesting '{doc_id}' into PaperQA index...")
            kb.add_document(doc_path, doc_id=doc_id)
            
            # 2. Extract sections
            sections = extract_sections(doc_path)
            
            # 3. Initialize shared state for this document
            shared_state: Dict[str, Any] = {
                "doc_id": doc_id,
                "doc_path": doc_path,
                "sections": sections,
                "text_stats": calculate_text_stats("\n".join(s['content'] for s in sections)),
            }
            
            # 4. Run the planning pipeline
            shared_state = run_section_selection(shared_state, use_mock_llm=use_mock_llm, output_dir=output_dir)
            shared_state = run_abstraction_detection(shared_state, use_mock_llm=use_mock_llm, output_dir=output_dir)
            shared_state = run_connection_analysis(shared_state, use_mock_llm=use_mock_llm, output_dir=output_dir)
            
            # 5. Add results to the central KB
            kb.add_abstractions(doc_id, shared_state.get("categorized_abstractions", []))
            kb.add_connections(doc_id, shared_state.get("categorized_connections", []))
            
            logger.info(f"✅ Successfully processed and added '{doc_id}' to the KB.")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {doc_id}: {e}", exc_info=True)
            # Continue to the next document
            continue
            
    logger.info("🎉 Knowledge Base bootstrapping complete.")

def main():
    """Main function to run the bootstrapping flow."""
    kb_path = "knowledge_base"
    output_dir = "output"
    seed_data_dir = "seed_data"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Knowledge Base
    logger.info(f"Initializing Knowledge Base at '{kb_path}'...")
    kb = KnowledgeBase(kb_path)
    
    # Run bootstrapping
    bootstrap_knowledge_base(
        seed_data_dir=seed_data_dir,
        kb=kb,
        output_dir=output_dir,
        use_mock_llm=True
    )

if __name__ == "__main__":
    main() 