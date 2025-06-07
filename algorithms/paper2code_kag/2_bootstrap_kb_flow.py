import os
import sys
import json
import subprocess
from pathlib import Path
import importlib.util

# --- Path Setup ---
# Add the root of the KAG project and the original paper2code project to the Python path
# This allows importing modules from both projects.
current_dir = Path(__file__).parent
kag_root = current_dir
paper2code_root = current_dir.parent / "paper2code"
sys.path.append(str(kag_root))
sys.path.append(str(paper2code_root))
# --- End Path Setup ---

from utils.knowledge_base import KnowledgeBase
# Import '1_ingest_paper_flow.py' using importlib because its name is not a valid identifier
spec = importlib.util.spec_from_file_location("ingest_flow", kag_root / "1_ingest_paper_flow.py")
ingest_flow_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest_flow_module)
ingest_paper = ingest_flow_module.ingest_paper

def run_legacy_planning(paper_path: Path, temp_output_dir: Path) -> tuple[Path, Path]:
    """
    Runs the existing planning flows from the original `paper2code` project.
    
    NOTE: The subprocess calls are MOCKED. You will need to replace them
    with the actual commands to run your legacy scripts. This will depend on
    the specific command-line arguments they accept.
    """
    print(f"\n--- Running legacy planning for {paper_path.name} ---")
    
    # 1. Define paths to legacy scripts
    abstraction_flow_script = paper2code_root / "11_abstraction_planning_flow.py"
    connection_flow_script = paper2code_root / "connection_planning_flow.py"
    
    # 2. Mock the subprocess calls
    print("MOCK: Running abstraction planning...")
    # This assumes the script takes --input-file and --output-dir
    # subprocess.run(
    #     ["python", str(abstraction_flow_script), "--input-file", str(paper_path), "--output-dir", str(temp_output_dir)],
    #     check=True
    # )
    abstraction_plan_path = temp_output_dir / "abstraction_plan.json"
    mock_abs_data = {
        "metadata": {"source": paper_path.name},
        "abstractions": [{"id": "abs1", "name": f"MockAbstraction_{paper_path.stem}"}]
    }
    with open(abstraction_plan_path, 'w') as f:
        json.dump(mock_abs_data, f, indent=2)
    print(f"MOCK: Created mock abstraction plan at {abstraction_plan_path}")

    print("MOCK: Running connection planning...")
    # This assumes the connection script takes the abstraction plan as input
    # subprocess.run(
    #     ["python", str(connection_flow_script), "--input-file", str(abstraction_plan_path), "--output-dir", str(temp_output_dir)],
    #     check=True
    # )
    connection_plan_path = temp_output_dir / "connection_plan.json"
    mock_conn_data = {
        "metadata": {"source": paper_path.name},
        "connections": [{"from": "abs1", "to": "abs1", "type": "mock_dependency"}]
    }
    with open(connection_plan_path, 'w') as f:
        json.dump(mock_conn_data, f, indent=2)
    print(f"MOCK: Created mock connection plan at {connection_plan_path}")

    print("--- Legacy planning finished ---")
    return abstraction_plan_path, connection_plan_path

def update_kb_from_plans(kb: KnowledgeBase, docname: str, abs_plan_path: Path, conn_plan_path: Path):
    """
    Parses the generated plans and updates the global Abstractions & Connections DB.
    """
    print(f"\n--- Updating Knowledge Base for {docname} ---")
    abstractions_db = kb.load_abstractions()

    with open(abs_plan_path) as f:
        new_abstractions = json.load(f).get("abstractions", [])
    with open(conn_plan_path) as f:
        new_connections = json.load(f).get("connections", [])
    
    if "abstractions_by_doc" not in abstractions_db:
        abstractions_db["abstractions_by_doc"] = {}
    if "connections_by_doc" not in abstractions_db:
        abstractions_db["connections_by_doc"] = {}
    
    # Store all abstractions and connections under the docname
    abstractions_db["abstractions_by_doc"][docname] = new_abstractions
    abstractions_db["connections_by_doc"][docname] = new_connections

    kb.save_abstractions(abstractions_db)
    print(f"--- Knowledge Base update for {docname} complete ---")

def bootstrap_knowledge_base():
    """
    Main orchestration script for bootstrapping the KB.
    """
    print("===== Starting Knowledge Base Bootstrap Process =====")
    kb = KnowledgeBase("knowledge_base")
    seed_data_dir = Path("seed_data")
    temp_output_dir = Path("temp_bootstrap_output")
    
    if not seed_data_dir.exists():
        print(f"Error: Seed data directory not found at {seed_data_dir.resolve()}")
        return
        
    seed_files = list(seed_data_dir.glob("*"))
    if not seed_files:
        print(f"No seed files found in {seed_data_dir.resolve()}. Nothing to do.")
        return

    print(f"Found {len(seed_files)} seed document(s) to process.")
    
    for paper_path in seed_files:
        print(f"\n=================================================")
        print(f"Processing: {paper_path.name}")
        print(f"=================================================")
        
        # 1. Ingest paper into PaperQA index
        ingest_paper(str(paper_path))
        
        # 2. Run legacy planning flows
        paper_temp_dir = temp_output_dir / paper_path.stem
        paper_temp_dir.mkdir(parents=True, exist_ok=True)
        abs_plan_path, conn_plan_path = run_legacy_planning(paper_path, paper_temp_dir)
        
        # 3. Update the KB with the results
        update_kb_from_plans(kb, paper_path.name, abs_plan_path, conn_plan_path)

    print("\n===== Knowledge Base Bootstrap Process Finished =====")

if __name__ == "__main__":
    bootstrap_knowledge_base() 