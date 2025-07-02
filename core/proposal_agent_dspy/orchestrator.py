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
            # Using the exact configuration provided by the user to match their dspy environment.
            lm = dspy.LM(
                'ollama_chat/gemma3:4b', 
                api_base='http://localhost:11434', 
                api_key=''
            )
            dspy.configure(lm=lm)
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
                critique = Critique(score=0.5, justification=user_input) # Simplified for plan
                state.update("review_team_feedback", {"user_review": critique}) 
                # Loop back to the proposal writing step
                state.next_step_index = self.workflow_steps.index(self._step_write_proposal)

        if should_advance_workflow:
            state.next_step_index += 1

        # Now that the state is updated, resume the main workflow engine.
        async for result in self._run_workflow(state):
            yield result

    async def _run_workflow(self, state: WorkflowState) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Runs the workflow from the current step index until the next HIL pause or completion.
        This is the main engine that centralizes UI communication logic.
        """
        while state.next_step_index < len(self.workflow_steps):
            current_step_func = self.workflow_steps[state.next_step_index]
            step_name = current_step_func.__name__.replace("_step_", "")

            # Announce the step is starting
            yield {"step": step_name, "state": state.to_dict(), "thread_id": state.thread_id}
            
            # Execute the step. It will either return None or a dictionary with HIL instructions.
            pause_data = await current_step_func(state)

            # Announce the step is finished and update the UI with any new state
            yield {"step": step_name, "state": state.to_dict(), "thread_id": state.thread_id}

            # If the step returned HIL data, yield the interrupt and pause the workflow engine.
            if pause_data:
                yield {
                    "step": "human_input_required",
                    "interrupt_type": pause_data["interrupt_type"],
                    "message": pause_data["message"],
                    "context": pause_data["context"],
                    "thread_id": state.thread_id,
                }
                return # Stop the engine until continue_agent is called

            state.next_step_index += 1
        
        # If the loop completes, the workflow is finished.
        yield {"step": "workflow_complete_node", "state": state.to_dict(), "thread_id": state.thread_id}
        if state.thread_id in self._thread_states:
            del self._thread_states[state.thread_id]
    
    # --- Individual Workflow Steps ---

    async def _step_generate_queries(self, state: WorkflowState) -> Dict[str, Any]:
        """Generates search queries and then requests a pause for human review."""
        prediction = self.query_generator(topic=state.topic, existing_queries=state.search_queries)
        state.update("search_queries", prediction.queries)
        state.last_interrupt_type = "query_review"
        
        return {
            "interrupt_type": "query_review",
            "message": "The AI generated queries. To regenerate, type `!regenerate`. Otherwise, edit/approve the queries below and send.",
            "context": {
                "queries": state.search_queries,
                "query_count": len(state.search_queries),
                "topic": state.topic
            },
        }

    async def _step_literature_review(self, state: WorkflowState) -> None:
        """Runs the literature review using the approved queries."""
        summaries = []
        for query in state.search_queries:
            response = await self.doc_service.query_documents(state.collection_name, query)
            summaries.append(response.get("answer_text"))
        state.update("literature_summaries", summaries)

    async def _step_synthesize_knowledge(self, state: WorkflowState) -> None:
        """Synthesizes the literature into a knowledge gap."""
        summaries_str = "\n---\n".join(state.literature_summaries)
        prediction = self.synthesizer(topic=state.topic, literature_summaries=summaries_str)
        state.update("knowledge_gap", prediction.knowledge_gap)

    async def _step_write_proposal(self, state: WorkflowState) -> None:
        """Writes the research proposal draft, incorporating feedback if it exists."""
        prior_feedback_str = json.dumps([fb.model_dump() for fb in state.review_team_feedback.values()])
        
        prediction = self.writer(
            knowledge_gap_summary=state.knowledge_gap.model_dump_json(),
            prior_feedback=prior_feedback_str
        )
        state.update("proposal_draft", prediction.proposal)

    async def _step_review_proposal(self, state: WorkflowState) -> Dict[str, Any]:
        """Reviews the proposal and pauses for the user's final approval or revision request."""
        review_aspect = "novelty and contribution"
        prediction = self.reviewer(proposal_draft=state.proposal_draft, review_aspect=review_aspect)
        state.update("review_team_feedback", {"ai_reviewer": prediction.critique})
        
        state.last_interrupt_type = "final_review"
        return {
            "interrupt_type": "final_review",
            "message": "The AI has reviewed the proposal. Type 'approve' to finish, or provide feedback for revision.",
            "context": {"review": prediction.critique.model_dump(), "revision_cycle": state.revision_cycles},
        }

def create_dspy_service(use_parrot: bool = False) -> DSPyOrchestrator:
    """Factory function to create the new DSPy-based service."""
    return DSPyOrchestrator(use_parrot) 