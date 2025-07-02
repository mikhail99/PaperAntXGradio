"""
Complete example of the Proposal Agent using the Hybrid Flow Framework.

This demonstrates how much more readable and maintainable our workflow becomes
when we separate the flow definition from the node implementation.
"""

from .hybrid_flow import Flow, Node, FlowAction, FlowEngine
from .state import WorkflowState
from .dspy_modules import QueryGenerator, KnowledgeSynthesizer, ProposalWriter, ProposalReviewer
from typing import AsyncGenerator, Dict, Any


# ===============================================
# Simple, Focused Node Implementations
# ===============================================

class GenerateQueriesNode(Node):
    def __init__(self, query_generator: QueryGenerator):
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


class SynthesizeKnowledgeNode(Node):
    def __init__(self, synthesizer: KnowledgeSynthesizer):
        super().__init__("synthesize_knowledge")
        self.synthesizer = synthesizer
    
    async def execute(self, state: WorkflowState) -> FlowAction:
        summaries_str = "\n---\n".join(state.literature_summaries)
        prediction = self.synthesizer(topic=state.topic, literature_summaries=summaries_str)
        state.update("knowledge_gap", prediction.knowledge_gap)
        return FlowAction(type="continue")


class WriteProposalNode(Node):
    def __init__(self, writer: ProposalWriter):
        super().__init__("write_proposal")
        self.writer = writer
    
    async def execute(self, state: WorkflowState) -> FlowAction:
        # Include any prior feedback in the writing process
        prior_feedback_str = ""
        if state.review_team_feedback:
            import json
            prior_feedback_str = json.dumps([fb.model_dump() for fb in state.review_team_feedback.values()])
        
        prediction = self.writer(
            knowledge_gap_summary=state.knowledge_gap.model_dump_json(),
            prior_feedback=prior_feedback_str
        )
        state.update("proposal_draft", prediction.proposal)
        return FlowAction(type="continue")


class ReviewProposalNode(Node):
    def __init__(self, reviewer: ProposalReviewer):
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
                "context": {
                    "review": prediction.critique.model_dump(),
                    "revision_cycle": state.revision_cycles
                }
            }
        )


# ===============================================
# Routing Node for Human Input Decisions
# ===============================================

class UserInputRouterNode(Node):
    """
    This special node handles user input routing decisions.
    It examines the last interrupt type and user input to decide the next action.
    """
    
    def __init__(self):
        super().__init__("user_input_router")
    
    async def execute(self, state: WorkflowState) -> FlowAction:
        # This node should only be called after user input is received
        # The user input would be stored in state by the engine
        user_input = getattr(state, '_last_user_input', '').strip().lower()
        
        if state.last_interrupt_type == "query_review":
            if user_input == "!regenerate":
                return FlowAction(type="branch:regenerate_queries")
            else:
                # User approved/edited queries
                if hasattr(state, '_last_user_input_raw'):
                    # Parse edited queries if provided
                    edited_queries = [q.strip() for q in state._last_user_input_raw.split(',') if q.strip()]
                    if edited_queries:
                        state.update("search_queries", edited_queries)
                return FlowAction(type="branch:queries_approved")
        
        elif state.last_interrupt_type == "final_review":
            if user_input == "approve":
                state.update("is_approved", True)
                return FlowAction(type="branch:approved")
            else:
                # User wants revision
                state.revision_cycles += 1
                from .state import Critique
                critique = Critique(score=0.5, justification=getattr(state, '_last_user_input', 'User requested revision'))
                state.update("review_team_feedback", {"user_review": critique})
                return FlowAction(type="branch:revision_requested")
        
        # Default: continue
        return FlowAction(type="continue")


# ===============================================
# Declarative Flow Definition
# ===============================================

def create_proposal_flow(query_generator, synthesizer, writer, reviewer, doc_service) -> Flow:
    """
    This is the magic! Look how readable and maintainable this flow definition is.
    Compare this to our current orchestrator's complex routing logic.
    """
    flow = Flow("proposal_generation", "generate_queries")
    
    # === Add all nodes ===
    flow.add_node(GenerateQueriesNode(query_generator))
    flow.add_node(LiteratureReviewNode(doc_service))
    flow.add_node(SynthesizeKnowledgeNode(synthesizer))
    flow.add_node(WriteProposalNode(writer))
    flow.add_node(ReviewProposalNode(reviewer))
    flow.add_node(UserInputRouterNode())
    
    # === Define the main workflow path ===
    flow.on_continue("generate_queries", "user_input_router")
    flow.on_branch("user_input_router", "queries_approved", "literature_review")
    flow.on_branch("user_input_router", "regenerate_queries", "generate_queries")
    
    flow.on_continue("literature_review", "synthesize_knowledge")
    flow.on_continue("synthesize_knowledge", "write_proposal")
    flow.on_continue("write_proposal", "review_proposal")
    
    flow.on_continue("review_proposal", "user_input_router")
    flow.on_branch("user_input_router", "approved", "complete")  # Special "complete" target
    flow.on_branch("user_input_router", "revision_requested", "write_proposal")
    
    return flow


# ===============================================
# Hybrid Orchestrator (Drop-in Replacement)
# ===============================================

class HybridOrchestrator:
    """
    Drop-in replacement for DSPyOrchestrator that uses the new hybrid flow approach.
    Same interface, much more maintainable internals.
    """
    
    def __init__(self, use_parrot: bool = False):
        # Same initialization as before
        if use_parrot:
            import dspy
            from .parrot import MockLM, MockPaperQAService
            dspy.configure(lm=MockLM())
            self.doc_service = MockPaperQAService()
        else:
            import dspy
            lm = dspy.LM('ollama_chat/gemma3:4b', api_base='http://localhost:11434', api_key='')
            dspy.configure(lm=lm)
            from .orchestrator import PaperQAService  # Your real service
            self.doc_service = PaperQAService()
        
        # Initialize DSPy modules
        from .dspy_modules import QueryGenerator, KnowledgeSynthesizer, ProposalWriter, ProposalReviewer
        self.query_generator = QueryGenerator()
        self.synthesizer = KnowledgeSynthesizer()
        self.writer = ProposalWriter()
        self.reviewer = ProposalReviewer()
        
        # Create the declarative flow
        self.flow = create_proposal_flow(
            self.query_generator, self.synthesizer, self.writer, self.reviewer, self.doc_service
        )
        
        # Initialize the engine
        self.engine = FlowEngine()
    
    async def start_agent(self, config: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Same interface as before, but using the hybrid engine."""
        state = WorkflowState(config["topic"], config["collection_name"])
        async for result in self.engine.execute_flow(self.flow, state):
            yield result
    
    async def continue_agent(self, thread_id: str, user_input: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Same interface as before, but using the hybrid engine."""
        # Store user input in the state for the router node
        if thread_id in self.engine._active_flows:
            state = self.engine._active_flows[thread_id]
            state._last_user_input = user_input.strip().lower()
            state._last_user_input_raw = user_input
        
        async for result in self.engine.continue_flow(thread_id, user_input, self.flow):
            yield result


# ===============================================
# Factory Function (Drop-in Replacement)
# ===============================================

def create_hybrid_service(use_parrot: bool = False) -> HybridOrchestrator:
    """Factory function that creates the hybrid service as a drop-in replacement."""
    return HybridOrchestrator(use_parrot)


# ===============================================
# COMPARISON: Look how much cleaner this is!
# ===============================================

"""
OLD APPROACH (orchestrator.py):
- 150+ lines of complex routing logic in continue_agent()
- Flow logic mixed with execution logic
- Hard to understand the overall process
- Difficult to modify or extend

NEW HYBRID APPROACH:
- Flow definition is 20 lines and crystal clear
- Each node is 10-15 lines and does one thing well
- Easy to add new steps or modify routing
- Self-documenting workflow structure

The flow definition literally reads like documentation:
1. Generate queries → User reviews → Literature review
2. Synthesize → Write → Review → User approves or requests revision
3. If revision requested → go back to write
4. If approved → complete

This is the best of both worlds!
""" 