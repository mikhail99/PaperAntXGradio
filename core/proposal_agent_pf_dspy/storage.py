"""
PocketFlow Proposal Storage

Handles JSON storage of proposal results and intermediate states for the PocketFlow-based
proposal generation workflow. Provides the same functionality as the original system
but adapted for our enhanced state management.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from .state import ProposalWorkflowState


class ProposalStorage:
    """Handles JSON storage of proposal results for PocketFlow workflows."""
    
    def __init__(self, base_dir: str = "data/collections"):
        self.base_dir = Path(base_dir)
    
    def save_proposal_result(self, state: ProposalWorkflowState, session_id: str = None) -> str:
        """Save the complete proposal result to JSON and return the file path."""
        # Create directory structure: data/collections/{collection_name}/research_proposals/
        proposals_dir = self.base_dir / state.collection_name / "research_proposals"
        proposals_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp and sanitized topic
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = self._sanitize_filename(state.topic)
        filename = f"{timestamp}_{safe_topic}.json"
        file_path = proposals_dir / filename
        
        # Prepare data for storage with PocketFlow-specific fields
        proposal_data = {
            "topic": state.topic,
            "collection_name": state.collection_name,
            "search_queries": state.search_queries,
            "literature_summaries": state.literature_summaries,
            "knowledge_gap": state.knowledge_gap.model_dump() if state.knowledge_gap else None,
            "proposal_draft": state.proposal_draft,
            "review_team_feedback": {
                k: v.model_dump() if hasattr(v, 'model_dump') else v 
                for k, v in (state.review_team_feedback or {}).items()
            },
            "is_approved": state.is_approved,
            "revision_cycles": state.revision_cycles,
            "session_id": session_id,  # PocketFlow uses session_id instead of thread_id
            "saved_at": datetime.now().isoformat(),
            "workflow_completed": True,
            "engine": "pocketflow",  # Mark as PocketFlow-generated
            "last_interrupt_type": getattr(state, 'last_interrupt_type', None)
        }
        
        # Write to JSON file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(proposal_data, f, indent=2, ensure_ascii=False)
        
        print(f"--- [PocketFlow ProposalStorage] Saved proposal result to: {file_path} ---")
        return str(file_path)
    
    def save_intermediate_state(self, state: ProposalWorkflowState, step_name: str, session_id: str = None) -> str:
        """Save intermediate state during PocketFlow workflow execution."""
        # Create directory for intermediate states
        intermediates_dir = self.base_dir / state.collection_name / "intermediate_states"
        intermediates_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        safe_topic = self._sanitize_filename(state.topic)
        filename = f"{timestamp}_{safe_topic}_{step_name}.json"
        file_path = intermediates_dir / filename
        
        # Save current state using our to_shared_dict method
        state_data = state.to_shared_dict()
        state_data.update({
            "current_step": step_name,
            "saved_at": datetime.now().isoformat(),
            "workflow_completed": False,
            "session_id": session_id,
            "engine": "pocketflow"
        })
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        
        print(f"--- [PocketFlow ProposalStorage] Saved intermediate state: {step_name} to {file_path} ---")
        return str(file_path)
    
    def save_session_checkpoint(self, orchestrator, session_id: str, step_name: str) -> str:
        """Save a complete session checkpoint including orchestrator state."""
        state = orchestrator.get_session_state(session_id)
        if not state:
            raise ValueError(f"Session {session_id} not found")
        
        # Create checkpoints directory
        checkpoints_dir = self.base_dir / state.collection_name / "session_checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = self._sanitize_filename(state.topic)
        filename = f"{timestamp}_{safe_topic}_{step_name}_checkpoint.json"
        file_path = checkpoints_dir / filename
        
        # Include session metadata
        checkpoint_data = {
            "session_id": session_id,
            "step_name": step_name,
            "saved_at": datetime.now().isoformat(),
            "engine": "pocketflow",
            "session_status": orchestrator.active_sessions.get(session_id, {}).get("status", "unknown"),
            "state": state.to_shared_dict(),
            "active_sessions_count": len(orchestrator.list_active_sessions())
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        print(f"--- [PocketFlow ProposalStorage] Saved session checkpoint to: {file_path} ---")
        return str(file_path)
    
    def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
        """Convert text to a safe filename."""
        import re
        # Replace problematic characters with hyphens
        safe = re.sub(r'[^\w\s-]', '', text.lower())
        safe = re.sub(r'[-\s]+', '-', safe)
        return safe[:max_length].strip('-')
    
    def list_saved_proposals(self, collection_name: str = None, engine_filter: str = None) -> List[Dict[str, Any]]:
        """
        List all saved proposals, optionally filtered by collection and/or engine.
        
        Args:
            collection_name: Filter by specific collection
            engine_filter: Filter by engine type ('pocketflow', 'original', etc.)
        """
        proposals = []
        
        if collection_name:
            # Search in specific collection
            proposals_dir = self.base_dir / collection_name / "research_proposals"
            if proposals_dir.exists():
                for json_file in proposals_dir.glob("*.json"):
                    proposal_info = self._get_proposal_info(json_file)
                    if engine_filter is None or proposal_info.get('engine') == engine_filter:
                        proposals.append(proposal_info)
        else:
            # Search in all collections
            for collection_dir in self.base_dir.iterdir():
                if collection_dir.is_dir():
                    proposals_dir = collection_dir / "research_proposals"
                    if proposals_dir.exists():
                        for json_file in proposals_dir.glob("*.json"):
                            proposal_info = self._get_proposal_info(json_file)
                            if engine_filter is None or proposal_info.get('engine') == engine_filter:
                                proposals.append(proposal_info)
        
        # Sort by saved_at timestamp (newest first)
        proposals.sort(key=lambda x: x.get('saved_at', ''), reverse=True)
        return proposals
    
    def list_session_checkpoints(self, collection_name: str = None) -> List[Dict[str, Any]]:
        """List all session checkpoints."""
        checkpoints = []
        
        if collection_name:
            checkpoints_dir = self.base_dir / collection_name / "session_checkpoints"
            if checkpoints_dir.exists():
                for json_file in checkpoints_dir.glob("*_checkpoint.json"):
                    checkpoints.append(self._get_checkpoint_info(json_file))
        else:
            for collection_dir in self.base_dir.iterdir():
                if collection_dir.is_dir():
                    checkpoints_dir = collection_dir / "session_checkpoints"
                    if checkpoints_dir.exists():
                        for json_file in checkpoints_dir.glob("*_checkpoint.json"):
                            checkpoints.append(self._get_checkpoint_info(json_file))
        
        checkpoints.sort(key=lambda x: x.get('saved_at', ''), reverse=True)
        return checkpoints
    
    def load_proposal(self, file_path: str) -> Dict[str, Any]:
        """Load a saved proposal from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_checkpoint(self, file_path: str) -> Dict[str, Any]:
        """Load a session checkpoint from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def restore_session_from_checkpoint(self, file_path: str) -> ProposalWorkflowState:
        """Restore a ProposalWorkflowState from a checkpoint file."""
        checkpoint_data = self.load_checkpoint(file_path)
        state_dict = checkpoint_data.get('state', {})
        return ProposalWorkflowState.from_shared_dict(state_dict)
    
    def _get_proposal_info(self, file_path: Path) -> Dict[str, Any]:
        """Extract basic info about a proposal file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    "file_path": str(file_path),
                    "topic": data.get("topic", "Unknown"),
                    "collection_name": data.get("collection_name", "Unknown"),
                    "saved_at": data.get("saved_at", "Unknown"),
                    "is_approved": data.get("is_approved", False),
                    "revision_cycles": data.get("revision_cycles", 0),
                    "workflow_completed": data.get("workflow_completed", False),
                    "engine": data.get("engine", "unknown"),
                    "session_id": data.get("session_id", data.get("thread_id", "unknown"))
                }
        except Exception as e:
            return {
                "file_path": str(file_path),
                "error": f"Failed to read file: {e}"
            }
    
    def _get_checkpoint_info(self, file_path: Path) -> Dict[str, Any]:
        """Extract basic info about a checkpoint file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                state_data = data.get('state', {})
                return {
                    "file_path": str(file_path),
                    "session_id": data.get("session_id", "Unknown"),
                    "step_name": data.get("step_name", "Unknown"),
                    "topic": state_data.get("topic", "Unknown"),
                    "collection_name": state_data.get("collection_name", "Unknown"),
                    "saved_at": data.get("saved_at", "Unknown"),
                    "session_status": data.get("session_status", "Unknown"),
                    "engine": data.get("engine", "pocketflow")
                }
        except Exception as e:
            return {
                "file_path": str(file_path),
                "error": f"Failed to read checkpoint file: {e}"
            }
    
    def cleanup_old_intermediates(self, collection_name: str, days_old: int = 7) -> int:
        """Clean up old intermediate state files."""
        intermediates_dir = self.base_dir / collection_name / "intermediate_states"
        if not intermediates_dir.exists():
            return 0
        
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        cleaned_count = 0
        
        for json_file in intermediates_dir.glob("*.json"):
            if json_file.stat().st_mtime < cutoff_time:
                json_file.unlink()
                cleaned_count += 1
        
        print(f"--- [PocketFlow ProposalStorage] Cleaned {cleaned_count} old intermediate files ---")
        return cleaned_count
    
    def get_storage_stats(self, collection_name: str = None) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "total_proposals": 0,
            "pocketflow_proposals": 0,
            "original_proposals": 0,
            "intermediate_states": 0,
            "session_checkpoints": 0,
            "total_size_mb": 0.0
        }
        
        collections_to_check = [collection_name] if collection_name else [
            d.name for d in self.base_dir.iterdir() if d.is_dir()
        ]
        
        for coll_name in collections_to_check:
            coll_dir = self.base_dir / coll_name
            if not coll_dir.exists():
                continue
            
            # Count proposals
            proposals_dir = coll_dir / "research_proposals"
            if proposals_dir.exists():
                for json_file in proposals_dir.glob("*.json"):
                    stats["total_proposals"] += 1
                    stats["total_size_mb"] += json_file.stat().st_size / (1024 * 1024)
                    
                    # Check engine type
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                            engine = data.get('engine', 'original')
                            if engine == 'pocketflow':
                                stats["pocketflow_proposals"] += 1
                            else:
                                stats["original_proposals"] += 1
                    except:
                        stats["original_proposals"] += 1
            
            # Count intermediate states
            intermediates_dir = coll_dir / "intermediate_states"
            if intermediates_dir.exists():
                for json_file in intermediates_dir.glob("*.json"):
                    stats["intermediate_states"] += 1
                    stats["total_size_mb"] += json_file.stat().st_size / (1024 * 1024)
            
            # Count checkpoints
            checkpoints_dir = coll_dir / "session_checkpoints"
            if checkpoints_dir.exists():
                for json_file in checkpoints_dir.glob("*_checkpoint.json"):
                    stats["session_checkpoints"] += 1
                    stats["total_size_mb"] += json_file.stat().st_size / (1024 * 1024)
        
        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        return stats


# Factory function for easy import
def create_proposal_storage(base_dir: str = "data/collections") -> ProposalStorage:
    """Factory function to create a ProposalStorage instance."""
    return ProposalStorage(base_dir) 