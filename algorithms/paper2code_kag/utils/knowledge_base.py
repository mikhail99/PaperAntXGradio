import os
import pickle
import json
from pathlib import Path
from paperqa import Docs

class KnowledgeBase:
    """
    Manages the three components of the Knowledge Base:
    1. PaperQA Docs index for research papers.
    2. A JSON database for abstractions and connections.
    3. An implementation document store for final .md files.
    """
    def __init__(self, kb_path: str = "knowledge_base"):
        self.kb_path = Path(kb_path)
        self.docs_path = self.kb_path / "paperqa_docs.pkl"
        self.abstractions_db_path = self.kb_path / "abstractions.json"
        self.impl_store_path = self.kb_path / "implementations"
        self.impl_manifest_path = self.impl_store_path / "manifest.json"

    def initialize(self):
        """Creates the directory structure and initial files for a new KB."""
        print(f"Initializing Knowledge Base at {self.kb_path.resolve()}...")
        self.kb_path.mkdir(exist_ok=True)
        self.impl_store_path.mkdir(exist_ok=True)

        # Initialize PaperQA Docs
        if not self.docs_path.exists():
            print("Creating new PaperQA Docs object...")
            # Here we can customize the LLM model for paper-qa
            docs = Docs() 
            self.save_docs(docs)
        else:
            print("PaperQA Docs object already exists.")

        # Initialize Abstractions DB
        if not self.abstractions_db_path.exists():
            print("Creating new Abstractions DB...")
            self.save_abstractions({})
        else:
            print("Abstractions DB already exists.")
            
        # Initialize Implementation Store Manifest
        if not self.impl_manifest_path.exists():
            print("Creating new Implementation Store manifest...")
            with open(self.impl_manifest_path, 'w') as f:
                json.dump({}, f)
        else:
            print("Implementation Store manifest already exists.")
        print("Knowledge Base initialization complete.")

    def save_docs(self, docs: Docs):
        """Saves the PaperQA Docs object to a pickle file."""
        with open(self.docs_path, "wb") as f:
            pickle.dump(docs, f)
        print(f"Saved PaperQA Docs to {self.docs_path}")

    def load_docs(self) -> Docs:
        """Loads the PaperQA Docs object from a pickle file."""
        if not self.docs_path.exists():
            raise FileNotFoundError(f"PaperQA Docs file not found at {self.docs_path}. Did you initialize the KB?")
        with open(self.docs_path, "rb") as f:
            docs = pickle.load(f)
        print(f"Loaded PaperQA Docs from {self.docs_path}")
        return docs

    def save_abstractions(self, abstractions: dict):
        """Saves the abstractions database to a JSON file."""
        with open(self.abstractions_db_path, "w") as f:
            json.dump(abstractions, f, indent=2)
        print(f"Saved Abstractions DB to {self.abstractions_db_path}")

    def load_abstractions(self) -> dict:
        """Loads the abstractions database from a JSON file."""
        if not self.abstractions_db_path.exists():
            raise FileNotFoundError(f"Abstractions DB file not found at {self.abstractions_db_path}. Did you initialize the KB?")
        with open(self.abstractions_db_path, "r") as f:
            abstractions = json.load(f)
        print(f"Loaded Abstractions DB from {self.abstractions_db_path}")
        return abstractions 