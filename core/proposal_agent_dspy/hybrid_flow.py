"""
Hybrid Flow Framework: Combining PocketFlow's declarative flows with simple nodes.

This approach gives us:
- Readable, declarative workflow definitions (like PocketFlow)
- Simple, lightweight nodes (like our current approach)
- Clear separation between flow logic and step implementation
"""

from typing import Dict, Any, List, Optional, Union, AsyncGenerator, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import asyncio
from enum import Enum

from .state import WorkflowState


class FlowResult(Enum):
    CONTINUE = "continue"
    PAUSE = "pause" 
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class FlowAction:
    """Represents a transition/action in the flow."""
    type: str  # "continue", "pause", "branch", "complete"
    target: Optional[str] = None  # Next node name
    data: Optional[Dict[str, Any]] = None  # Additional data for the action


class Node(ABC):
    """Base class for workflow nodes. Keeps them simple and focused."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def execute(self, state: WorkflowState) -> FlowAction:
        """Execute the node logic. Return a FlowAction to control flow."""
        pass


class Flow:
    """
    Declarative workflow definition inspired by PocketFlow.
    The flow itself is readable and maintainable, while nodes stay simple.
    """
    
    def __init__(self, name: str, start_node: str):
        self.name = name
        self.start_node = start_node
        self.nodes: Dict[str, Node] = {}
        self.transitions: Dict[str, Dict[str, str]] = {}  # node -> {action_type -> target_node}
    
    def add_node(self, node: Node) -> 'Flow':
        """Add a node to the flow. Returns self for chaining."""
        self.nodes[node.name] = node
        return self
    
    def add_transition(self, from_node: str, action_type: str, to_node: str) -> 'Flow':
        """Define a transition between nodes. Returns self for chaining."""
        if from_node not in self.transitions:
            self.transitions[from_node] = {}
        self.transitions[from_node][action_type] = to_node
        return self
    
    def on_continue(self, from_node: str, to_node: str) -> 'Flow':
        """Shorthand for adding a 'continue' transition."""
        return self.add_transition(from_node, "continue", to_node)
    
    def on_branch(self, from_node: str, condition: str, to_node: str) -> 'Flow':
        """Shorthand for adding a conditional branch."""
        return self.add_transition(from_node, f"branch:{condition}", to_node)


class FlowEngine:
    """
    Execution engine for flows. Handles the orchestration while keeping 
    individual nodes simple and focused.
    """
    
    def __init__(self):
        self._active_flows: Dict[str, WorkflowState] = {}
    
    async def execute_flow(self, flow: Flow, state: WorkflowState) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute a flow from start to completion, yielding updates for the UI.
        This replaces our current _run_workflow method but with declarative flow definition.
        """
        current_node_name = flow.start_node
        self._active_flows[state.thread_id] = state
        
        try:
            while current_node_name:
                # Get the current node
                if current_node_name not in flow.nodes:
                    yield {"step": "error", "error": f"Node '{current_node_name}' not found", "thread_id": state.thread_id}
                    return
                
                current_node = flow.nodes[current_node_name]
                
                # Announce step start
                yield {"step": current_node_name, "state": state.to_dict(), "thread_id": state.thread_id}
                
                # Execute the node
                try:
                    action = await current_node.execute(state)
                except Exception as e:
                    yield {"step": "error", "error": str(e), "thread_id": state.thread_id}
                    return
                
                # Announce step completion
                yield {"step": current_node_name, "state": state.to_dict(), "thread_id": state.thread_id}
                
                # Handle the action
                if action.type == "pause":
                    # Pause for human input
                    yield {
                        "step": "human_input_required",
                        "interrupt_type": action.data.get("interrupt_type", "unknown"),
                        "message": action.data.get("message", "Input required"),
                        "context": action.data.get("context", {}),
                        "thread_id": state.thread_id,
                    }
                    return  # Stop execution until continue_flow is called
                
                elif action.type == "complete":
                    # Workflow complete
                    yield {"step": "workflow_complete_node", "state": state.to_dict(), "thread_id": state.thread_id}
                    if state.thread_id in self._active_flows:
                        del self._active_flows[state.thread_id]
                    return
                
                elif action.type == "continue":
                    # Find next node based on transitions
                    if current_node_name in flow.transitions and "continue" in flow.transitions[current_node_name]:
                        current_node_name = flow.transitions[current_node_name]["continue"]
                    else:
                        # No explicit transition, workflow complete
                        yield {"step": "workflow_complete_node", "state": state.to_dict(), "thread_id": state.thread_id}
                        if state.thread_id in self._active_flows:
                            del self._active_flows[state.thread_id]
                        return
                
                elif action.type.startswith("branch:"):
                    # Handle conditional branching
                    branch_condition = action.type.split(":", 1)[1]
                    transition_key = f"branch:{branch_condition}"
                    
                    if current_node_name in flow.transitions and transition_key in flow.transitions[current_node_name]:
                        current_node_name = flow.transitions[current_node_name][transition_key]
                    else:
                        yield {"step": "error", "error": f"No transition defined for branch '{branch_condition}' from node '{current_node_name}'", "thread_id": state.thread_id}
                        return
                
                else:
                    yield {"step": "error", "error": f"Unknown action type: {action.type}", "thread_id": state.thread_id}
                    return
        
        finally:
            # Cleanup
            if state.thread_id in self._active_flows:
                del self._active_flows[state.thread_id]
    
    async def continue_flow(self, thread_id: str, user_input: str, flow: Flow) -> AsyncGenerator[Dict[str, Any], None]:
        """Continue a paused flow with user input."""
        if thread_id not in self._active_flows:
            yield {"step": "error", "error": f"No active flow for thread {thread_id}", "thread_id": thread_id}
            return
        
        state = self._active_flows[thread_id]
        
        # Process user input and determine next action
        # This is where we'd handle routing logic like "!regenerate", "approve", etc.
        # For now, we'll use the existing orchestrator logic
        
        # Continue execution from current point
        async for update in self.execute_flow(flow, state):
            yield update


# =====================================
# Example: Proposal Agent as a Flow
# =====================================

class GenerateQueriesNode(Node):
    """Simple node that just calls our existing DSPy module."""
    
    def __init__(self, query_generator):
        super().__init__("generate_queries")
        self.query_generator = query_generator
    
    async def execute(self, state: WorkflowState) -> FlowAction:
        prediction = self.query_generator(topic=state.topic, existing_queries=state.search_queries)
        state.update("search_queries", prediction.queries)
        state.last_interrupt_type = "query_review"
        
        return FlowAction(
            type="pause",
            data={
                "interrupt_type": "query_review",
                "message": "Review and approve the generated queries.",
                "context": {
                    "queries": state.search_queries,
                    "query_count": len(state.search_queries),
                    "topic": state.topic
                }
            }
        )


class LiteratureReviewNode(Node):
    """Simple node for literature review."""
    
    def __init__(self, doc_service):
        super().__init__("literature_review")
        self.doc_service = doc_service
    
    async def execute(self, state: WorkflowState) -> FlowAction:
        summaries = []
        for query in state.search_queries:
            response = await self.doc_service.query_documents(state.collection_name, query)
            summaries.append(response.get("answer_text"))
        state.update("literature_summaries", summaries)
        
        return FlowAction(type="continue")


class ReviewProposalNode(Node):
    """Simple node for proposal review."""
    
    def __init__(self, reviewer):
        super().__init__("review_proposal")
        self.reviewer = reviewer
    
    async def execute(self, state: WorkflowState) -> FlowAction:
        review_aspect = "novelty and contribution"
        prediction = self.reviewer(proposal_draft=state.proposal_draft, review_aspect=review_aspect)
        state.update("review_team_feedback", {"ai_reviewer": prediction.critique})
        
        state.last_interrupt_type = "final_review"
        return FlowAction(
            type="pause",
            data={
                "interrupt_type": "final_review",
                "message": "The AI has reviewed the proposal. Type 'approve' to finish, or provide feedback for revision.",
                "context": {"review": prediction.critique.model_dump(), "revision_cycle": state.revision_cycles}
            }
        )


def create_proposal_flow(query_generator, synthesizer, writer, reviewer, doc_service) -> Flow:
    """
    Create a declarative proposal generation flow.
    This is much more readable than our current approach!
    """
    flow = Flow("proposal_generation", "generate_queries")
    
    # Add nodes
    flow.add_node(GenerateQueriesNode(query_generator))
    flow.add_node(LiteratureReviewNode(doc_service))
    # ... add other nodes
    
    # Define the happy path
    flow.on_continue("generate_queries", "literature_review")
    flow.on_continue("literature_review", "synthesize_knowledge") 
    flow.on_continue("synthesize_knowledge", "write_proposal")
    flow.on_continue("write_proposal", "review_proposal")
    
    # Define revision loops
    flow.on_branch("review_proposal", "revision_requested", "write_proposal")
    
    return flow 