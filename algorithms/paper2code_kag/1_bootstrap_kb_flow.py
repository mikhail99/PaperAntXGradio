import os
import sys
import json
import logging
from pathlib import Path

from utils.path_setup import setup_paths
setup_paths()

from utils.knowledge_base import KnowledgeBase
from abstraction_detection_flow import AbstractionPlanningFlow
from connection_analysis_flow import ConnectionPlanningFlow


def update_kb_from_plans(kb: KnowledgeBase, docname: str, abs_plan_path: Path, conn_plan_path: Path):
    """
    Parses the generated plans and updates the global Abstractions & Connections DB.
    """
    logging.info(f"Updating Knowledge Base for {docname} from plan files...")
    abstractions_db = kb.load_abstractions()

    try:
        with open(abs_plan_path) as f:
            # Navigate through the structure to get the categorized abstractions
            abs_data = json.load(f)
            new_abstractions = abs_data.get("abstraction_planning_results", {}).get("categorized_abstractions", [])
        
        with open(conn_plan_path) as f:
            conn_data = json.load(f)
            new_connections = conn_data.get("connection_planning_results", {}).get("connections", [])
        
        if "abstractions_by_doc" not in abstractions_db:
            abstractions_db["abstractions_by_doc"] = {}
        if "connections_by_doc" not in abstractions_db:
            abstractions_db["connections_by_doc"] = {}
        
        # Store all abstractions and connections under the docname
        abstractions_db["abstractions_by_doc"][docname] = new_abstractions
        abstractions_db["connections_by_doc"][docname] = new_connections

        kb.save_abstractions(abstractions_db)
        logging.info(f"Knowledge Base update for {docname} complete.")

    except FileNotFoundError as e:
        logging.error(f"Could not find plan file to update KB: {e}")
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Error parsing plan files for {docname}: {e}")


def bootstrap_knowledge_base():
    """
    Main orchestration script for bootstrapping the KB. It processes each seed document
    by running the full planning pipeline and then uses the results to populate the
    central Knowledge Base.
    """
    logging.info("===== Starting Knowledge Base Bootstrap Process =====")
    kb = KnowledgeBase("knowledge_base")
    seed_data_dir = Path("seed_data")
    temp_output_dir = Path("temp_bootstrap_output")
    
    if not seed_data_dir.exists():
        logging.error(f"Seed data directory not found at {seed_data_dir.resolve()}")
        return
        
    seed_files = list(seed_data_dir.glob("*.*"))
    if not seed_files:
        logging.warning(f"No seed files found in {seed_data_dir.resolve()}. Nothing to do.")
        return

    logging.info(f"Found {len(seed_files)} seed document(s) to process.")
    
    for paper_path in seed_files:
        logging.info(f"\n=================================================")
        logging.info(f"Processing: {paper_path.name}")
        logging.info(f"=================================================")
        
        doc_name = paper_path.name
        doc_output_dir = temp_output_dir / paper_path.stem
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Ingest paper and prepare shared state
        try:
            docs = kb.load_docs()
            if doc_name not in docs.docs:
                logging.info(f"Ingesting {doc_name} into the Paper-QA index...")
                docs.add(str(paper_path), docname=doc_name)
                kb.save_docs(docs)
            else:
                logging.info(f"{doc_name} already in Paper-QA index.")
            
            full_text = docs.docs[doc_name].text
            shared_state = {
                "selected_sections": [{"title": "Full Document", "content": full_text, "section_type": "full_text"}],
                "planning_summary": {"source_document": doc_name}
            }
        except Exception as e:
            logging.error(f"Failed to ingest or prepare {doc_name}: {e}", exc_info=True)
            continue # Move to the next paper

        # 2. Run Abstraction Planning
        try:
            logging.info("🎯 Running Abstraction Planning Flow...")
            abstraction_flow = AbstractionPlanningFlow(use_mock_llm=True, output_dir=str(doc_output_dir))
            shared_state = abstraction_flow.run(shared_state)
        except Exception as e:
            logging.error(f"Abstraction Planning failed for {doc_name}: {e}", exc_info=True)
            continue

        # 3. Run Connection Planning
        try:
            logging.info("🔗 Running Connection Planning Flow...")
            connection_flow = ConnectionPlanningFlow(use_mock_llm=True, output_dir=str(doc_output_dir))
            shared_state = connection_flow.run(shared_state)
        except Exception as e:
            logging.error(f"Connection Planning failed for {doc_name}: {e}", exc_info=True)
            continue
            
        # 4. Update the central KB with the results
        abs_plan_path = doc_output_dir / "abstraction_plan.json"
        conn_plan_path = doc_output_dir / "connection_plan.json"
        
        update_kb_from_plans(kb, doc_name, abs_plan_path, conn_plan_path)

    logging.info("\n===== Knowledge Base Bootstrap Process Finished =====")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    bootstrap_knowledge_base() 