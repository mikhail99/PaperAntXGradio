from utils.knowledge_base import KnowledgeBase

def setup_knowledge_base():
    """
    Initializes the knowledge base directory and files if they don't exist.
    This script should be run from the root of the `paper2code_kag` directory.
    """
    # The KB will be created at `paper2code_kag/knowledge_base`
    kb_path = "knowledge_base"
    kb = KnowledgeBase(kb_path)
    kb.initialize()

if __name__ == "__main__":
    setup_knowledge_base() 