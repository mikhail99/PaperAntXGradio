import dspy
import asyncio
import json
from typing import Dict, Any, AsyncGenerator

from .state import WorkflowState, Critique
from .dspy_modules import QueryGenerator, KnowledgeSynthesizer, ProposalWriter, ProposalReviewer
# from .paperqa_service import PaperQAService # Your real service
from .parrot import MockLM, MockPaperQAService

# --- FAKE PaperQAService for demonstration ---
class PaperQAService:
    async def query_documents(self, collection_name: str, query: str) -> dict:
        await asyncio.sleep(1) # Simulate network latency
        return {"answer_text": f"Real summary for query: '{query}' in '{collection_name}'."}
# ---

class DSPyOrchestrator:
    """
    A DSPy-based orchestrator that implements the same async generator interface
    as ModernProposalAgentService to be a drop-in replacement for the Gradio UI.
    It uses a state-machine approach where the workflow is a list of steps.
    """
    _thread_states: Dict[str, WorkflowState] = {}

    def __init__(self, use_parrot: bool = False):
        if use_parrot:
            dspy.configure(lm=MockLM())
            self.doc_service = MockPaperQAService()
        else:
            ollama_lm = dspy.Ollama(model='llama3', base_url='http://localhost:11434/')
            dspy.configure(lm=ollama_lm)
            self.doc_service = PaperQAService() # Replace with your real service
        
        self.query_generator = QueryGenerator()
        self.synthesizer = KnowledgeSynthesizer()
        self.writer = ProposalWriter()
        self.reviewer = ProposalReviewer()
        
        # The workflow is defined as a sequence of methods.
        # The orchestrator will call the method at `state.next_step_index`.
        self.workflow_steps = [
            self._step_generate_queries,
            self._step_literature_review,
            self._step_synthesize_knowledge,
            self._step_write_proposal,
            self._step_review_proposal,
            self._step_complete_workflow,
        ]

    async def start_agent(self, config: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Starts a new agent run, compatible with the UI."""
        state = WorkflowState(config["topic"], config["collection_name"])
        self._thread_states[state.thread_id] = state
        
        async for result in self._run_workflow(state):
            yield result

    async def continue_agent(self, thread_id: str, user_input: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Continues an agent run after a HIL pause.
        This method contains the core routing logic: it decides whether to
        repeat the last step (e.g., for regeneration) or advance to the next one.
        """
        state = self._thread_states.get(thread_id)
        if not state:
            yield {"step": "error", "error": f"Thread ID '{thread_id}' not found.", "thread_id": thread_id}
            return
            
        # --- Routing Logic ---
        should_advance_workflow = True
        
        if state.last_interrupt_type == "query_review":
            # A special command `!regenerate` tells the orchestrator to loop.
            if user_input.strip().lower() == "!regenerate":
                should_advance_workflow = False # Re-run the same step
            # Any other non-empty input is treated as the user's approved/edited queries.
            elif user_input.strip():
                state.update("search_queries", [q.strip() for q in user_input.split(',')])
            # An empty input implies approval of the last generated queries.
        
        elif state.last_interrupt_type == "final_review":
            if user_input.strip().lower() == "approve":
                state.update("is_approved", True)
            else:
                # Any other input is treated as revision feedback
                should_advance_workflow = False
                state.revision_cycles += 1
                # A real implementation would collect feedback from multiple reviewers
                critique = Critique(score=0.5, justification=user_input) # Simplified for plan
                state.update("review_team_feedback", {"user_review": critique}) 
                # Loop back to the proposal writing step
                state.next_step_index = self.workflow_steps.index(self._step_write_proposal)

        if should_advance_workflow:
            state.next_step_index += 1

        # Now that the state is updated, run the new current step.
        async for result in self._run_workflow(state):
            yield result

    async def _run_workflow(self, state: WorkflowState) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Runs the single, current step defined by state.next_step_index.
        This method is a simple executor; all logic is in `continue_agent`.
        """
        if state.next_step_index < len(self.workflow_steps):
            current_step_func = self.workflow_steps[state.next_step_index]
            # Yield all messages from the current step's async generator.
            async for result in current_step_func(state):
                yield result
        else:
            # This should not be reached if the last step is _step_complete_workflow
            yield {"step": "error", "error": "Workflow attempted to run past its final step."}
    
    # --- Individual Workflow Steps ---

    async def _step_generate_queries(self, state: WorkflowState) -> AsyncGenerator[Dict[str, Any], None]:
        """Generates search queries and then pauses for human review."""
        yield {"step": "query_generation", "state": {}, "thread_id": state.thread_id}
        prediction = self.query_generator(topic=state.topic, existing_queries=state.search_queries)
        state.update("search_queries", prediction.queries)
        state.last_interrupt_type = "query_review" # Set context for continue_agent
        
        # PAUSE for HIL. The workflow will not advance until continue_agent is called.
        yield {
            "step": "human_input_required",
            "interrupt_type": "query_review",
            "message": "The AI generated queries. To regenerate, type `!regenerate`. Otherwise, edit/approve the queries below and send.",
            "context": {"queries": state.search_queries},
            "thread_id": state.thread_id,
        }

    async def _step_literature_review(self, state: WorkflowState) -> AsyncGenerator[Dict[str, Any], None]:
        """Runs the literature review using the approved queries."""
        yield {"step": "literature_review", "state": {"search_queries": state.search_queries}, "thread_id": state.thread_id}
        for query in state.search_queries:
            response = await self.doc_service.query_documents(state.collection_name, query)
            state.append_to("literature_summaries", response.get("answer_text"))
            yield {"step": "literature_review", "state": {"literature_summaries": state.literature_summaries}, "thread_id": state.thread_id}

    async def _step_synthesize_knowledge(self, state: WorkflowState) -> AsyncGenerator[Dict[str, Any], None]:
        """Synthesizes the literature into a knowledge gap."""
        yield {"step": "synthesize_knowledge", "state": {}, "thread_id": state.thread_id}
        summaries_str = "\n---\n".join(state.literature_summaries)
        prediction = self.synthesizer(topic=state.topic, literature_summaries=summaries_str)
        state.update("knowledge_gap", prediction.knowledge_gap)
        yield {"step": "synthesize_knowledge", "state": {"knowledge_gap": state.knowledge_gap.dict()}, "thread_id": state.thread_id}

    async def _step_write_proposal(self, state: WorkflowState) -> AsyncGenerator[Dict[str, Any], None]:
        """Writes the research proposal draft, incorporating feedback if it exists."""
        yield {"step": "write_proposal", "state": {}, "thread_id": state.thread_id}
        
        # Format any prior feedback for the LLM
        prior_feedback_str = json.dumps([fb.dict() for fb in state.review_team_feedback.values()])
        
        prediction = self.writer(
            knowledge_gap_summary=state.knowledge_gap.model_dump_json(),
            prior_feedback=prior_feedback_str
        )
        state.update("proposal_draft", prediction.proposal)
        yield {"step": "write_proposal", "state": {"proposal_draft": state.proposal_draft}, "thread_id": state.thread_id}

    async def _step_review_proposal(self, state: WorkflowState) -> AsyncGenerator[Dict[str, Any], None]:
        """Reviews the proposal and pauses for the user's final approval or revision request."""
        yield {"step": "review_proposal", "state": {}, "thread_id": state.thread_id}
        
        # In a real system, you might have multiple reviewers with different aspects.
        review_aspect = "novelty and contribution"
        prediction = self.reviewer(proposal_draft=state.proposal_draft, review_aspect=review_aspect)
        state.update("review_team_feedback", {"ai_reviewer": prediction.critique})
        
        yield {"step": "review_proposal", "state": {"review_team_feedback": {k: v.dict() for k,v in state.review_team_feedback.items()}}, "thread_id": state.thread_id}

        state.last_interrupt_type = "final_review"
        # PAUSE for HIL
        yield {
            "step": "human_input_required",
            "interrupt_type": "final_review",
            "message": "The AI has reviewed the proposal. Type 'approve' to finish, or provide feedback for revision.",
            "context": {"review": prediction.critique.dict(), "revision_cycles": state.revision_cycles},
            "thread_id": state.thread_id,
        }

    async def _step_complete_workflow(self, state: WorkflowState) -> AsyncGenerator[Dict[str, Any], None]:
        """Final step to clean up and signal completion."""
        yield {"step": "workflow_complete_node", "state": state.to_dict(), "thread_id": state.thread_id}
        if state.thread_id in self._thread_states:
            del self._thread_states[state.thread_id]

def create_dspy_service(use_parrot: bool = False) -> DSPyOrchestrator:
    """Factory function to create the new DSPy-based service."""
    return DSPyOrchestrator(use_parrot) 