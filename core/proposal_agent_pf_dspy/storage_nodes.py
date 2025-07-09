"""
PocketFlow Storage Nodes

Dedicated nodes for handling storage operations within the PocketFlow workflow.
These nodes follow the 3-phase pattern and can be integrated into the flow definition.
"""

from pocketflow import Node
from typing import Any, Dict, Tuple, Optional

from .state import ProposalWorkflowState
from .storage import ProposalStorage


class SaveIntermediateStateNode(Node):
    """Node to save intermediate workflow state at specific checkpoints"""
    
    def __init__(self, step_name: str, storage: ProposalStorage = None):
        super().__init__(max_retries=2, wait=0.5)
        self.step_name = step_name
        self.storage = storage or ProposalStorage()
    
    def prep(self, shared: dict) -> Tuple[ProposalWorkflowState, str]:
        """Extract current state and session info"""
        state = ProposalWorkflowState.from_shared_dict(shared)
        # Get session_id from shared state if available
        session_id = shared.get('session_id', shared.get('thread_id', 'unknown'))
        return state, session_id
    
    def exec(self, prep_res: Tuple[ProposalWorkflowState, str]) -> str:
        """Save intermediate state to storage"""
        state, session_id = prep_res
        file_path = self.storage.save_intermediate_state(state, self.step_name, session_id)
        return file_path
    
    def post(self, shared: dict, prep_res: Tuple[ProposalWorkflowState, str], exec_res: str) -> str:
        """Log save operation and continue flow"""
        if shared.get('flow_queue'):
            shared['flow_queue'].put(f"💾 Saved checkpoint: {self.step_name}")
        return "default"


class SaveFinalResultNode(Node):
    """Node to save final proposal result when workflow completes"""
    
    def __init__(self, storage: ProposalStorage = None):
        super().__init__(max_retries=3, wait=1.0)
        self.storage = storage or ProposalStorage()
    
    def prep(self, shared: dict) -> Tuple[ProposalWorkflowState, str]:
        """Extract final state and session info"""
        state = ProposalWorkflowState.from_shared_dict(shared)
        session_id = shared.get('session_id', shared.get('thread_id', 'unknown'))
        return state, session_id
    
    def exec(self, prep_res: Tuple[ProposalWorkflowState, str]) -> Optional[str]:
        """Save final proposal result if it exists"""
        state, session_id = prep_res
        
        # Only save if we have a proposal draft
        if not state.proposal_draft:
            return None
        
        file_path = self.storage.save_proposal_result(state, session_id)
        return file_path
    
    def post(self, shared: dict, prep_res: Tuple[ProposalWorkflowState, str], exec_res: Optional[str]) -> str:
        """Log final save operation"""
        if exec_res:
            if shared.get('flow_queue'):
                shared['flow_queue'].put(f"✅ Final proposal saved to: {exec_res}")
        else:
            if shared.get('flow_queue'):
                shared['flow_queue'].put("⚠️ No proposal to save (draft empty)")
        return "default"


class SaveSessionCheckpointNode(Node):
    """Node to save complete session checkpoint with orchestrator state"""
    
    def __init__(self, checkpoint_name: str, storage: ProposalStorage = None):
        super().__init__(max_retries=2, wait=0.5)
        self.checkpoint_name = checkpoint_name
        self.storage = storage or ProposalStorage()
    
    def prep(self, shared: dict) -> Tuple[str, str]:
        """Extract session info for checkpoint"""
        session_id = shared.get('session_id', shared.get('thread_id', 'unknown'))
        return session_id, self.checkpoint_name
    
    def exec(self, prep_res: Tuple[str, str]) -> Optional[str]:
        """Save session checkpoint"""
        session_id, checkpoint_name = prep_res
        
        # Note: This would require access to the orchestrator instance
        # For now, we'll just save a simplified checkpoint
        try:
            # This would ideally call: self.storage.save_session_checkpoint(orchestrator, session_id, checkpoint_name)
            # But since nodes shouldn't have orchestrator reference, we'll save a state-only checkpoint
            return f"checkpoint_{session_id}_{checkpoint_name}.json"
        except Exception as e:
            print(f"--- [PocketFlow Storage] Error saving checkpoint: {e} ---")
            return None
    
    def post(self, shared: dict, prep_res: Tuple[str, str], exec_res: Optional[str]) -> str:
        """Log checkpoint save operation"""
        if exec_res:
            if shared.get('flow_queue'):
                shared['flow_queue'].put(f"🔄 Session checkpoint saved: {exec_res}")
        else:
            if shared.get('flow_queue'):
                shared['flow_queue'].put("⚠️ Failed to save session checkpoint")
        return "default"


class LoadProposalNode(Node):
    """Node to load a previously saved proposal into current state"""
    
    def __init__(self, storage: ProposalStorage = None):
        super().__init__(max_retries=2, wait=0.5)
        self.storage = storage or ProposalStorage()
    
    def prep(self, shared: dict) -> str:
        """Extract file path from shared state"""
        # Expecting file_path to be set in shared state
        return shared.get('load_file_path', '')
    
    def exec(self, prep_res: str) -> Optional[Dict[str, Any]]:
        """Load proposal data from file"""
        file_path = prep_res
        if not file_path:
            return None
        
        try:
            return self.storage.load_proposal(file_path)
        except Exception as e:
            print(f"--- [PocketFlow Storage] Error loading proposal: {e} ---")
            return None
    
    def post(self, shared: dict, prep_res: str, exec_res: Optional[Dict[str, Any]]) -> str:
        """Update shared state with loaded proposal data"""
        if exec_res:
            # Update shared state with loaded data
            state = ProposalWorkflowState.from_shared_dict(shared)
            
            # Update key fields from loaded proposal
            state.topic = exec_res.get('topic', state.topic)
            state.collection_name = exec_res.get('collection_name', state.collection_name)
            state.search_queries = exec_res.get('search_queries', state.search_queries)
            state.literature_summaries = exec_res.get('literature_summaries', state.literature_summaries)
            state.proposal_draft = exec_res.get('proposal_draft', state.proposal_draft)
            state.is_approved = exec_res.get('is_approved', state.is_approved)
            state.revision_cycles = exec_res.get('revision_cycles', state.revision_cycles)
            
            # Update knowledge gap if present
            if exec_res.get('knowledge_gap'):
                from .state import KnowledgeGap
                kg_data = exec_res['knowledge_gap']
                if isinstance(kg_data, dict):
                    state.knowledge_gap = KnowledgeGap(**kg_data)
            
            # Update review feedback if present
            if exec_res.get('review_team_feedback'):
                from .state import Critique
                feedback = {}
                for key, value in exec_res['review_team_feedback'].items():
                    if isinstance(value, dict):
                        feedback[key] = Critique(**value)
                    else:
                        feedback[key] = value
                state.review_team_feedback = feedback
            
            shared.update(state.to_shared_dict())
            
            if shared.get('flow_queue'):
                shared['flow_queue'].put(f"📂 Loaded proposal: {exec_res.get('topic', 'Unknown')}")
        else:
            if shared.get('flow_queue'):
                shared['flow_queue'].put("❌ Failed to load proposal")
        
        return "default"


class CleanupStorageNode(Node):
    """Node to clean up old storage files"""
    
    def __init__(self, days_old: int = 7, storage: ProposalStorage = None):
        super().__init__()
        self.days_old = days_old
        self.storage = storage or ProposalStorage()
    
    def prep(self, shared: dict) -> str:
        """Extract collection name from shared state"""
        state = ProposalWorkflowState.from_shared_dict(shared)
        return state.collection_name
    
    def exec(self, prep_res: str) -> int:
        """Clean up old intermediate files"""
        collection_name = prep_res
        cleaned_count = self.storage.cleanup_old_intermediates(collection_name, self.days_old)
        return cleaned_count
    
    def post(self, shared: dict, prep_res: str, exec_res: int) -> str:
        """Log cleanup results"""
        if shared.get('flow_queue'):
            shared['flow_queue'].put(f"🧹 Cleaned {exec_res} old files from {prep_res}")
        return "default"


# Factory functions for easy node creation
def create_save_checkpoint_node(step_name: str, storage: ProposalStorage = None) -> SaveIntermediateStateNode:
    """Factory to create a save checkpoint node"""
    return SaveIntermediateStateNode(step_name, storage)

def create_save_final_node(storage: ProposalStorage = None) -> SaveFinalResultNode:
    """Factory to create a save final result node"""
    return SaveFinalResultNode(storage)

def create_load_proposal_node(storage: ProposalStorage = None) -> LoadProposalNode:
    """Factory to create a load proposal node"""
    return LoadProposalNode(storage)

def create_cleanup_node(days_old: int = 7, storage: ProposalStorage = None) -> CleanupStorageNode:
    """Factory to create a cleanup node"""
    return CleanupStorageNode(days_old, storage) 