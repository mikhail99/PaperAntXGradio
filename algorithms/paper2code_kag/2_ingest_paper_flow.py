import argparse
import logging
from pathlib import Path

from utils.path_setup import setup_paths
setup_paths()

from utils.knowledge_base import KnowledgeBase
from abstraction_detection_flow import AbstractionPlanningFlow
from connection_analysis_flow import ConnectionPlanningFlow


def process_and_plan_paper(doc_path: str):
    """
    Orchestrates the full pipeline for a single document:
    1. Ingests into the Knowledge Base.
    2. Runs abstraction planning.
    3. Runs connection planning.
    """
    logging.info(f"🚀 Starting full processing for document: {doc_path}")
    doc_file = Path(doc_path)
    if not doc_file.exists():
        logging.error(f"Document file not found at {doc_path}")
        return

    doc_name = doc_file.name
    output_dir = Path("output") / doc_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === 1. Ingest into Knowledge Base ===
    kb = KnowledgeBase("knowledge_base")
    try:
        docs = kb.load_docs()
        if doc_name not in docs.docs:
            logging.info(f"Ingesting {doc_name} into the Knowledge Base...")
            docs.add(doc_path, docname=doc_name)
            kb.save_docs(docs)
            logging.info(f"Successfully ingested {doc_name}.")
        else:
            logging.info(f"{doc_name} already exists in the Knowledge Base.")
        
        # Retrieve the full text for planning
        full_text = docs.docs[doc_name].text
        
    except Exception as e:
        logging.error(f"An error occurred during ingestion: {e}")
        return

    # === 2. Prepare Shared State for Planning ===
    # We bypass the need for a separate "section planning" step by
    # treating the entire document as a single, high-priority section.
    shared_state = {
        "selected_sections": [{
            "title": "Full Document",
            "content": full_text,
            "section_type": "full_text",
            "priority": 1
        }],
        "planning_summary": {
            "source_document": doc_name,
            "selection_method": "full_text_ingestion"
        }
    }
    logging.info("Prepared shared state using full document text.")

    # === 3. Run Abstraction Planning ===
    try:
        logging.info("🎯 Running Abstraction Planning Flow...")
        abstraction_flow = AbstractionPlanningFlow(use_mock_llm=True, output_dir=str(output_dir))
        shared_state = abstraction_flow.run(shared_state)
        logging.info("✅ Abstraction Planning completed.")
    except Exception as e:
        logging.error(f"Abstraction Planning failed: {e}", exc_info=True)
        return

    # === 4. Run Connection Planning ===
    try:
        logging.info("🔗 Running Connection Planning Flow...")
        connection_flow = ConnectionPlanningFlow(use_mock_llm=True, output_dir=str(output_dir))
        shared_state = connection_flow.run(shared_state)
        logging.info("✅ Connection Planning completed.")
    except Exception as e:
        logging.error(f"Connection Planning failed: {e}", exc_info=True)
        return
        
    logging.info(f"🎉 Full processing for {doc_name} finished successfully.")
    logging.info(f"Final plans saved in: {output_dir.resolve()}")

def main():
    parser = argparse.ArgumentParser(
        description="Run the full ingestion and planning pipeline for a single document."
    )
    parser.add_argument("doc_path", type=str, help="The path to the document file to process.")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    process_and_plan_paper(args.doc_path)

if __name__ == "__main__":
    main() 