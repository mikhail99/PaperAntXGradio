"""
This file implements the Proposal Agent workflow using a hybrid, declarative approach
inspired by PocketFlow, combining a readable, graph-based flow definition with
simple, lightweight nodes.
"""
import dspy
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, AsyncGenerator, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .state import WorkflowState, Critique
from .dspy_modules import QueryGenerator, KnowledgeSynthesizer, ProposalWriter, ProposalReviewer
# --- Import the REAL PaperQAService ---
from core.paperqa_service import PaperQAService
from .parrot import MockLM, MockPaperQAService

# ===============================================
# JSON Storage Utility
# ===============================================

class ProposalStorage:
    """Handles JSON storage of proposal results."""
    
    def __init__(self, base_dir: str = "data/collections"):
        self.base_dir = Path(base_dir)
    
    def save_proposal_result(self, state: WorkflowState) -> str:
        """Save the complete proposal result to JSON and return the file path."""
        # Create directory structure: data/collections/{collection_name}/research_proposals/
        proposals_dir = self.base_dir / state.collection_name / "research_proposals"
        proposals_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp and sanitized topic
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = self._sanitize_filename(state.topic)
        filename = f"{timestamp}_{safe_topic}.json"
        file_path = proposals_dir / filename
        
        # Prepare data for storage
        proposal_data = {
            "topic": state.topic,
            "collection_name": state.collection_name,
            "search_queries": state.search_queries,
            "literature_summaries": state.literature_summaries,
            "knowledge_gap": state.knowledge_gap.model_dump() if state.knowledge_gap else None,
            "proposal_draft": state.proposal_draft,
            "review_team_feedback": {
                k: v.model_dump() if hasattr(v, 'model_dump') else v 
                for k, v in state.review_team_feedback.items()
            },
            "is_approved": state.is_approved,
            "revision_cycles": state.revision_cycles,
            "thread_id": state.thread_id,
            "saved_at": datetime.now().isoformat(),
            "workflow_completed": True
        }
        
        # Write to JSON file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(proposal_data, f, indent=2, ensure_ascii=False)
        
        print(f"--- [ProposalStorage] Saved proposal result to: {file_path} ---")
        return str(file_path)
    
    def save_intermediate_state(self, state: WorkflowState, step_name: str) -> str:
        """Save intermediate state during workflow execution."""
        # Create directory for intermediate states
        intermediates_dir = self.base_dir / state.collection_name / "intermediate_states"
        intermediates_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        safe_topic = self._sanitize_filename(state.topic)
        filename = f"{timestamp}_{safe_topic}_{step_name}.json"
        file_path = intermediates_dir / filename
        
        # Save current state
        state_data = state.to_dict()
        state_data.update({
            "current_step": step_name,
            "saved_at": datetime.now().isoformat(),
            "workflow_completed": False
        })
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        
        return str(file_path)
    
    def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
        """Convert text to a safe filename."""
        import re
        # Replace problematic characters with hyphens
        safe = re.sub(r'[^\w\s-]', '', text.lower())
        safe = re.sub(r'[-\s]+', '-', safe)
        return safe[:max_length].strip('-')
    
    def list_saved_proposals(self, collection_name: str = None) -> List[Dict[str, Any]]:
        """List all saved proposals, optionally filtered by collection."""
        proposals = []
        
        if collection_name:
            # Search in specific collection
            proposals_dir = self.base_dir / collection_name / "research_proposals"
            if proposals_dir.exists():
                for json_file in proposals_dir.glob("*.json"):
                    proposals.append(self._get_proposal_info(json_file))
        else:
            # Search in all collections
            for collection_dir in self.base_dir.iterdir():
                if collection_dir.is_dir():
                    proposals_dir = collection_dir / "research_proposals"
                    if proposals_dir.exists():
                        for json_file in proposals_dir.glob("*.json"):
                            proposals.append(self._get_proposal_info(json_file))
        
        # Sort by saved_at timestamp (newest first)
        proposals.sort(key=lambda x: x.get('saved_at', ''), reverse=True)
        return proposals
    
    def load_proposal(self, file_path: str) -> Dict[str, Any]:
        """Load a saved proposal from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
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
                    "workflow_completed": data.get("workflow_completed", False)
                }
        except Exception as e:
            return {
                "file_path": str(file_path),
                "error": f"Failed to read file: {e}"
            }

# ===============================================
# Hybrid Flow Framework Core
# ===============================================

@dataclass
class FlowAction:
    """Represents a transition in the flow, returned by a Node's execute method."""
    type: str
    data: Optional[Dict[str, Any]] = None

class Node(ABC):
    """Abstract base class for a node in our workflow."""
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def execute(self, state: WorkflowState) -> FlowAction:
        pass

class Flow:
    """A declarative workflow definition, mapping node names to transitions."""
    def __init__(self, name: str, start_node: str):
        self.name = name
        self.start_node = start_node
        self.nodes: Dict[str, Node] = {}
        self.transitions: Dict[str, Dict[str, str]] = {}
    
    def add_node(self, node: Node) -> 'Flow':
        self.nodes[node.name] = node
        return self
    
    def add_transition(self, from_node: str, condition: str, to_node: str) -> 'Flow':
        if from_node not in self.transitions:
            self.transitions[from_node] = {}
        self.transitions[from_node][condition] = to_node
        return self
    
    def on_branch(self, from_node: str, branch_name: str, to_node: str) -> 'Flow':
        return self.add_transition(from_node, f"branch:{branch_name}", to_node)

    def on_continue(self, from_node: str, to_node: str) -> 'Flow':
        return self.add_transition(from_node, "continue", to_node)

class FlowEngine:
    """Executes a declarative Flow, managing state and transitions."""
    _active_flows: Dict[str, WorkflowState] = {}
    _paused_at_node: Dict[str, str] = {}

    def __init__(self):
        self.storage = ProposalStorage()

    async def start(self, flow: Flow, state: WorkflowState) -> AsyncGenerator[Dict[str, Any], None]:
        """Starts a new flow execution."""
        print(f"--- [FlowEngine] Starting flow '{flow.name}' for thread {state.thread_id} ---")
        current_node_name = flow.start_node
        self._active_flows[state.thread_id] = state
        
        async for result in self._run_from(current_node_name, flow, state):
            yield result

    async def continue_flow(self, thread_id: str, flow: Flow) -> AsyncGenerator[Dict[str, Any], None]:
        """Continues a paused flow."""
        print(f"--- [FlowEngine] Continuing flow for thread {thread_id} ---")
        if thread_id not in self._active_flows:
            yield {"step": "error", "error": f"Thread ID '{thread_id}' not found.", "thread_id": thread_id}
            return
        
        state = self._active_flows[thread_id]
        paused_node_name = self._paused_at_node.get(thread_id)
        
        if not paused_node_name:
            yield {"step": "error", "error": f"Flow for thread '{thread_id}' is not paused.", "thread_id": thread_id}
            return

        # *** THE FIX: Determine the NEXT node to run after the pause. ***
        # The node that paused has completed its action. We now follow its 'continue' transition.
        next_node_name = flow.transitions[paused_node_name].get("continue")
        if not next_node_name:
            yield {"step": "error", "error": f"No 'continue' transition defined for paused node '{paused_node_name}'", "thread_id": state.thread_id}
            return
            
        print(f"--- [FlowEngine] Resuming from paused node '{paused_node_name}', transitioning to '{next_node_name}' ---")
        
        # Clear the paused state as we are now resuming.
        del self._paused_at_node[thread_id]

        async for result in self._run_from(next_node_name, flow, state):
            yield result
            
    async def _run_from(self, start_node_name: str, flow: Flow, state: WorkflowState) -> AsyncGenerator[Dict[str, Any], None]:
        current_node_name = start_node_name
        try:
            while current_node_name:
                node = flow.nodes.get(current_node_name)
                if not node:
                    yield {"step": "error", "error": f"Node '{current_node_name}' not found", "thread_id": state.thread_id}
                    return

                print(f"--- [FlowEngine] Executing node: {node.name} ---")
                yield {"step": node.name, "state": state.to_dict(), "thread_id": state.thread_id}
                
                action = await node.execute(state)
                
                print(f"--- [FlowEngine] Node '{node.name}' returned action: {action.type} ---")
                yield {"step": node.name, "state": state.to_dict(), "thread_id": state.thread_id}

                if action.type == "pause":
                    self._paused_at_node[state.thread_id] = current_node_name
                    print(f"--- [FlowEngine] Pausing at node: {current_node_name} ---")
                    yield {"step": "human_input_required", "interrupt_type": action.data.get("interrupt_type"), "message": action.data.get("message"), "context": action.data.get("context"), "thread_id": state.thread_id}
                    return # Pause execution
                
                # Determine the next node based on the action from the completed node
                next_node_name = None
                if action.type == "complete":
                    next_node_name = "complete" # Handle direct completion action from a node
                
                elif action.type.startswith("branch:"):
                    branch_condition = action.type.split(":", 1)[1]
                    transition_key = f"branch:{branch_condition}"
                    next_node_name = flow.transitions[node.name].get(transition_key)
                    print(f"--- [FlowEngine] Branching on '{branch_condition}' to node: {next_node_name} ---")
                
                elif action.type == "continue":
                    next_node_name = flow.transitions[node.name].get("continue")
                    print(f"--- [FlowEngine] Continuing to node: {next_node_name} ---")

                # *** THE FIX: Check for the 'complete' signal BEFORE the next loop iteration ***
                if next_node_name == "complete":
                    print(f"--- [FlowEngine] Reached 'complete' signal. Finishing flow. ---")
                    
                    # Save the completed proposal to JSON
                    if state.is_approved:
                        try:
                            file_path = self.storage.save_proposal_result(state)
                            yield {"step": "workflow_complete_node", "state": state.to_dict(), "thread_id": state.thread_id, "saved_to": file_path}
                        except Exception as e:
                            print(f"--- [FlowEngine] Error saving proposal: {e} ---")
                            yield {"step": "workflow_complete_node", "state": state.to_dict(), "thread_id": state.thread_id, "save_error": str(e)}
                    else:
                        yield {"step": "workflow_complete_node", "state": state.to_dict(), "thread_id": state.thread_id}
                    
                    current_node_name = None  # This will terminate the while loop
                elif next_node_name is None:
                    # An undefined transition means the flow has implicitly ended.
                    error_msg = f"Implicit end of flow. No transition defined from node '{node.name}' for action '{action.type}'."
                    print(f"--- [FlowEngine] {error_msg} ---")
                    # We can treat this as a success if it's not a failed branch
                    if not action.type.startswith("branch:"):
                         yield {"step": "workflow_complete_node", "state": state.to_dict(), "thread_id": state.thread_id}
                    else: # A branch to a non-existent node is an error
                         yield {"step": "error", "error": error_msg, "thread_id": state.thread_id}
                    current_node_name = None # Terminate the loop
                else:
                    # Proceed to the next node in the sequence
                    current_node_name = next_node_name
        finally:
            if current_node_name is None and state.thread_id in self._active_flows:
                print(f"--- [FlowEngine] Cleaning up completed flow for thread {state.thread_id} ---")
                # Make sure we don't leak memory
                if state.thread_id in self._active_flows:
                    del self._active_flows[state.thread_id]
                if state.thread_id in self._paused_at_node:
                    del self._paused_at_node[state.thread_id]

# ===============================================
# Concrete Node Implementations for Proposal Agent
# ===============================================

class GenerateQueriesNode(Node):
    def __init__(self, dspy_module: QueryGenerator):
        super().__init__("generate_queries")
        self.module = dspy_module
    
    async def execute(self, state: WorkflowState) -> FlowAction:
        prediction = self.module(topic=state.topic, existing_queries=state.search_queries)
        state.update("search_queries", prediction.queries)
        return FlowAction(type="continue")

class UserInputRouterNode(Node):
    def __init__(self):
        super().__init__("user_input_router")
    
    async def execute(self, state: WorkflowState) -> FlowAction:
        user_input = getattr(state, '_last_user_input', '').strip().lower()
        interrupt = state.last_interrupt_type
        
        if interrupt == "query_review":
            if user_input == "!regenerate":
                return FlowAction(type="branch:regenerate_queries")
            if hasattr(state, '_last_user_input_raw'):
                edited_queries = [q.strip() for q in state._last_user_input_raw.split(',') if q.strip()]
                if edited_queries:
                    state.update("search_queries", edited_queries)
            return FlowAction(type="branch:queries_approved")
        
        elif interrupt == "final_review":
            if user_input == "approve":
                state.update("is_approved", True)
                return FlowAction(type="branch:approved")
            state.revision_cycles += 1
            critique = Critique(score=0.5, justification=getattr(state, '_last_user_input', 'User requested revision'))
            state.update("review_team_feedback", {"user_review": critique})
            return FlowAction(type="branch:revision_requested")
            
        return FlowAction(type="continue") # Default case

class LiteratureReviewNode(Node):
    def __init__(self, doc_service):
        super().__init__("literature_review")
        self.doc_service = doc_service
    
    async def execute(self, state: WorkflowState) -> FlowAction:
        # Revert to sequential processing to avoid overwhelming the local Ollama server.
        # Concurrency with asyncio.gather can be re-enabled for more robust, cloud-based LLMs.
        summaries = []
        for query in state.search_queries:
            result = await self.doc_service.query_documents(state.collection_name, query)
            if result and not result.get("error"):
                summary_text = result.get("answer_text", "No summary provided.")
                summaries.append(summary_text)
            else:
                summaries.append(f"Error processing query '{query}': {result.get('error', 'Unknown error')}")

        state.update("literature_summaries", summaries)
        return FlowAction(type="continue")

class SynthesizeKnowledgeNode(Node):
    def __init__(self, dspy_module: KnowledgeSynthesizer):
        super().__init__("synthesize_knowledge")
        self.module = dspy_module
    
    async def execute(self, state: WorkflowState) -> FlowAction:
        summaries_str = "\n---\n".join(state.literature_summaries)
        prediction = self.module(topic=state.topic, literature_summaries=summaries_str)
        state.update("knowledge_gap", prediction.knowledge_gap)
        return FlowAction(type="continue")

class WriteProposalNode(Node):
    def __init__(self, dspy_module: ProposalWriter):
        super().__init__("write_proposal")
        self.module = dspy_module
    
    async def execute(self, state: WorkflowState) -> FlowAction:
        feedback = json.dumps([fb.model_dump() for fb in state.review_team_feedback.values()]) if state.review_team_feedback else ""
        prediction = self.module(knowledge_gap_summary=state.knowledge_gap.model_dump_json(), prior_feedback=feedback)
        state.update("proposal_draft", prediction.proposal)
        return FlowAction(type="continue")

class ReviewProposalNode(Node):
    def __init__(self, dspy_module: ProposalReviewer):
        super().__init__("review_proposal")
        self.module = dspy_module
    
    async def execute(self, state: WorkflowState) -> FlowAction:
        prediction = self.module(proposal_draft=state.proposal_draft, review_aspect="novelty and contribution")
        state.update("review_team_feedback", {"ai_reviewer": prediction.critique})
        state.last_interrupt_type = "final_review"
        return FlowAction(type="pause", data={"interrupt_type": "final_review", "message": "AI review complete. Approve or request revision.", "context": {"review": prediction.critique.model_dump(), "revision_cycle": state.revision_cycles}})

class PauseForQueryReviewNode(Node):
    def __init__(self):
        super().__init__("pause_for_query_review")

    async def execute(self, state: WorkflowState) -> FlowAction:
        state.last_interrupt_type = "query_review"
        return FlowAction(type="pause", data={"interrupt_type": "query_review", "message": "Please review the generated queries.", "context": {"queries": state.search_queries}})

# ===============================================
# Declarative Flow Definition
# ===============================================

def create_proposal_flow() -> Flow:
    """Defines the entire proposal generation workflow in a readable, declarative way."""
    return Flow("proposal_generation", "generate_queries") \
        .add_node(GenerateQueriesNode(QueryGenerator())) \
        .add_node(PauseForQueryReviewNode()) \
        .add_node(UserInputRouterNode()) \
        .add_node(LiteratureReviewNode(PaperQAService())) \
        .add_node(SynthesizeKnowledgeNode(KnowledgeSynthesizer())) \
        .add_node(WriteProposalNode(ProposalWriter())) \
        .add_node(ReviewProposalNode(ProposalReviewer())) \
        .on_continue("generate_queries", "pause_for_query_review") \
        .on_continue("pause_for_query_review", "user_input_router") \
        .on_branch("user_input_router", "queries_approved", "literature_review") \
        .on_branch("user_input_router", "regenerate_queries", "generate_queries") \
        .on_continue("literature_review", "synthesize_knowledge") \
        .on_continue("synthesize_knowledge", "write_proposal") \
        .on_continue("write_proposal", "review_proposal") \
        .on_continue("review_proposal", "user_input_router") \
        .on_branch("user_input_router", "revision_requested", "write_proposal") \
        .on_branch("user_input_router", "approved", "complete")

# ===============================================
# Orchestrator (Drop-in Replacement)
# ===============================================

class DSPyOrchestrator:
    """ """
    
    def __init__(self, use_parrot: bool = False):
        if use_parrot:
            dspy.configure(lm=MockLM())
        else:
            dspy.configure(lm=dspy.LM('ollama_chat/gemma3:4b', api_base='http://localhost:11434', api_key=''))
        
        self.flow = create_proposal_flow()
        # If using parrot, swap out the real doc service node with a mock one.
        if use_parrot:
            self.flow.add_node(LiteratureReviewNode(MockPaperQAService()))
        
        self.engine = FlowEngine()  # This now includes storage

    async def start_agent(self, config: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        state = WorkflowState(config["topic"], config["collection_name"])
        async for result in self.engine.start(self.flow, state):
            yield result

    async def continue_agent(self, thread_id: str, user_input: str) -> AsyncGenerator[Dict[str, Any], None]:
        if thread_id in self.engine._active_flows:
            state = self.engine._active_flows[thread_id]
            state._last_user_input_raw = user_input
            state._last_user_input = user_input.strip().lower()
        
        async for result in self.engine.continue_flow(thread_id, self.flow):
            yield result

    def list_saved_proposals(self, collection_name: str = None) -> List[Dict[str, Any]]:
        """List all saved proposals, optionally filtered by collection."""
        return self.engine.storage.list_saved_proposals(collection_name)
    
    def load_proposal(self, file_path: str) -> Dict[str, Any]:
        """Load a saved proposal from JSON file."""
        return self.engine.storage.load_proposal(file_path)

def create_dspy_service(use_parrot: bool = False) -> DSPyOrchestrator:
    """Factory function to create the new Flow-based service."""
    return DSPyOrchestrator(use_parrot) 