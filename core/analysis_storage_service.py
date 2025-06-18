import json
import os
from pathlib import Path
import re
import datetime
from typing import Dict, Any

from core.collections_manager import CollectionsManager

class AnalysisStorageService:
    """
    A service to handle the saving and loading of agent analysis results.
    """
    def __init__(self):
        self.collections_manager = CollectionsManager()
        self.base_path = Path("data/collections")

    def _slugify(self, text: str) -> str:
        """
        A simple function to create a filesystem-safe "slug" from a string.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s-]', '', text).strip()
        text = re.sub(r'[-\s]+', '-', text)
        return text

    def _make_serializable(self, obj: Any) -> Any:
        """
        Recursively converts an object to a JSON-serializable format.
        It specifically handles Pydantic models by calling .model_dump().
        """
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        if isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        # Assume other types are directly serializable
        return obj

    def save_analysis(self, collection_id: str, research_direction: str, final_state: Dict[str, Any]):
        """
        Saves the final state of an agent analysis to a JSON file.
        The file is stored in a 'research_proposals' subdirectory of the collection's folder.
        """
        collection = self.collections_manager.get_collection(collection_id)
        if not collection:
            print(f"Error saving analysis: Collection with ID '{collection_id}' not found.")
            return

        try:
            # Prepare the directory
            proposal_dir = self.base_path / collection.name / "research_proposals"
            proposal_dir.mkdir(parents=True, exist_ok=True)

            # Prepare the filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = self._slugify(research_direction)
            truncated_slug = slug[:50]
            filename = f"{timestamp}_{truncated_slug}.json"
            
            file_path = proposal_dir / filename

            # Prepare the data for serialization
            serializable_state = {}
            for key, value in final_state.items():
                if key == 'services':  # Explicitly skip non-serializable parts
                    continue
                serializable_state[key] = self._make_serializable(value)

            # Save the file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_state, f, indent=4)
            print(f"Analysis saved successfully to {file_path}")

        except Exception as e:
            print(f"Error during analysis saving process: {e}") 