import sys
from pathlib import Path

def setup_paths():
    """
    Adds project directories to the Python path to enable cross-directory imports.
    This allows the KAG flows to import and use the original paper2code flows.
    """
    # Path to the current file's directory (paper2code_kag)
    current_dir = Path(__file__).parent.resolve()
    
    # Path to the parent of paper2code_kag, which is 'algorithms'
    algorithms_root = current_dir.parent

    # Path to the original paper2code directory
    paper2code_dir = algorithms_root / "paper2code"

    if str(current_dir) not in sys.path:
        sys.path.append(str(current_dir))
    
    # Add 'algorithms' directory to path to resolve `from algorithms.paper2code...`
    if str(algorithms_root) not in sys.path:
        sys.path.append(str(algorithms_root))
        
    # Also add the original paper2code directory just in case of relative imports
    if str(paper2code_dir) not in sys.path:
        sys.path.append(str(paper2code_dir)) 