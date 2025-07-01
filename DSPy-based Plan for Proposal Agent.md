# DSPy-based Refactoring Plan for Proposal Agent

## 1. Executive Summary

This document outlines a plan to refactor the `core/proposal_agent` system to a new, simpler architecture centered around `dspy`. The goal is to replace the complex `langgraph` setup with a more direct and readable `Orchestrator` class. This new orchestrator will:

1.  **Be a drop-in replacement** for the existing `ModernProposalAgentService`, ensuring full compatibility with the main Gradio UI (`ui/ui_research_plan.py`).
2.  Use **`dspy`** to define agent logic (what each agent does).
3.  Use **Python** to manage the overall workflow, state, and Human-in-the-Loop (HIL) interactions via an `AsyncGenerator` interface.
4.  Use **a custom, typed state object** for robust and easy-to-use data management.

This approach will significantly simplify the codebase while preserving the functionality of the existing user interface.

## 2. Core Architectural Components

### a. `WorkflowState` (The State Manager)

We will create a new, typed state object with built-in serialization to manage data. It will be stored in memory between HIL steps.

### b. `dspy` Signatures & Modules (The Agents)

All agent logic will be defined using `dspy`.
*   **Signatures (`signatures.py`):** Define the inputs and outputs for each agent task.
*   **Modules (`dspy_modules.py`):** Wrap the signatures in `dspy.Module` classes.

### c. The Python Orchestrator

A single Python class (`orchestrator.py`) will control the workflow, implementing the same service interface as the old system. It will:
*   Instantiate and manage `WorkflowState` across HIL pauses using a thread ID.
*   Provide `start_agent` and `continue_agent` async generator methods.
*   Handle HIL by `yield`ing interrupt messages to the UI and pausing execution.
*   Call `dspy` modules in sequence.

### d. Simple Parrot Testing

We will create a new, simple `parrot.py` for testing, as originally planned. This allows for fast, offline unit and integration tests.

---

## 3. Implementation Plan

### Phase 1: State Management & DSPy Foundations

**1. Create the `WorkflowState` object with serialization.**

**New File: `core/proposal_agent_dspy/state.py`**
```python
import uuid
from typing import List, Dict, Any, TypeVar
from pydantic import BaseModel, Field

# --- Pydantic models for structured data ---
class KnowledgeGap(BaseModel):
    synthesized_summary: str = Field(description="A coherent summary of all reviewed literature.")
    knowledge_gap: str = Field(description="A specific, actionable gap in the current body of knowledge.")
    is_novel: bool = Field(description="An assessment of whether the identified gap is truly novel.")

class Critique(BaseModel):
    score: float = Field(description="A score from 0.0 to 1.0 for the review aspect.")
    justification: str = Field(description="A clear justification for the given score.")

# --- The main state manager ---
T = TypeVar('T')

class WorkflowState:
    """A typed, class-based state manager for the proposal workflow."""
    
    def __init__(self, topic: str, collection_name: str):
        self.thread_id: str = str(uuid.uuid4())
        self.topic: str = topic
        self.collection_name: str = collection_name
        
        # --- Core state variables ---
        self.search_queries: List[str] = []
        self.literature_summaries: List[str] = []
        self.knowledge_gap: KnowledgeGap | None = None
        self.proposal_draft: str = ""
        self.review_team_feedback: Dict[str, Critique] = {}
        self.is_approved: bool = False
        self.revision_cycles: int = 0
        
        # --- Workflow control ---
        self.next_step_index: int = 0
        self.last_interrupt_type: str | None = None
        
    def update(self, key: str, value: Any) -> None:
        """Safely updates a state variable."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"'{key}' is not a valid state variable.")

    def append_to(self, key: str, value: T) -> None:
        """Appends a value to a list-based state variable."""
        current_value = getattr(self, key, None)
        if isinstance(current_value, list):
            current_value.append(value)
        else:
            raise TypeError(f"'{key}' is not a list and cannot be appended to.")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the state to a dictionary for persistence."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowState':
        """Deserializes the state from a dictionary."""
        instance = cls(data['topic'], data['collection_name'])
        for key, value in data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance

    def __repr__(self) -> str:
        return f"<WorkflowState queries={len(self.search_queries)} summaries={len(self.literature_summaries)}>"
```

**2. Define `dspy` Signatures and Modules.**

**New File: `core/proposal_agent_dspy/signatures.py`**
```python
import dspy
from typing import List
from .state import KnowledgeGap, Critique

class GenerateQueries(dspy.Signature):
    """Generates a list of search queries based on a research topic."""
    topic: str = dspy.InputField(desc="The main research topic.")
    existing_queries: str = dspy.InputField(desc="A stringified list of queries already tried.")
    queries: List[str] = dspy.OutputField(desc="A list of 3-5 new, focused search queries.")

class SynthesizeKnowledge(dspy.Signature):
    """Synthesizes literature summaries into a coherent overview and identifies a knowledge gap."""
    topic: str = dspy.InputField()
    literature_summaries: str = dspy.InputField(desc="A stringified list of summaries from literature reviews.")
    knowledge_gap: KnowledgeGap = dspy.OutputField()

class WriteProposal(dspy.Signature):
    """Writes a research proposal based on an identified knowledge gap and any prior feedback."""
    knowledge_gap_summary: str = dspy.InputField()
    prior_feedback: str = dspy.InputField(desc="A summary of feedback from previous review cycles, if any.")
    proposal: str = dspy.OutputField(desc="A well-structured research proposal.")

class ReviewProposal(dspy.Signature):
    """Reviews a proposal for a specific quality, like novelty or feasibility."""
    proposal_draft: str = dspy.InputField()
    review_aspect: str = dspy.InputField(desc="The specific aspect to review (e.g., 'technical feasibility', 'novelty').")
    critique: Critique = dspy.OutputField()
```

**New File: `core/proposal_agent_dspy/dspy_modules.py`**
```python
import dspy
from .signatures import GenerateQueries, SynthesizeKnowledge, WriteProposal, ReviewProposal

class QueryGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateQueries)

    def forward(self, topic, existing_queries):
        return self.generate(topic=topic, existing_queries=str(existing_queries))

class KnowledgeSynthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.TypedPredictor(SynthesizeKnowledge)

    def forward(self, topic, literature_summaries):
        return self.synthesize(topic=topic, literature_summaries=str(literature_summaries))

class ProposalWriter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.write = dspy.Predict(WriteProposal)

    def forward(self, knowledge_gap_summary, prior_feedback):
        return self.write(knowledge_gap_summary=knowledge_gap_summary, prior_feedback=str(prior_feedback))

class ProposalReviewer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.review = dspy.TypedPredictor(ReviewProposal)

    def forward(self, proposal_draft, review_aspect):
        return self.review(proposal_draft=proposal_draft, review_aspect=review_aspect)
```

### Phase 2: Simple Mocking for Testing (Unchanged)

**New File: `core/proposal_agent_dspy/parrot.py`**
```python
import dspy
from .signatures import GenerateQueries, SynthesizeKnowledge, WriteProposal, ReviewProposal
from .state import KnowledgeGap, Critique

class MockLM(dspy.LM):
    """A mock dspy.LM that returns hardcoded responses for fast testing."""
    def __init__(self):
        super().__init__("mock-model")

    def __call__(self, prompt, **kwargs):
        # The 'dspy.Predict' object gives us the signature via kwargs
        signature = kwargs.get("signature")
        
        if signature == GenerateQueries:
            return [{'queries': ['parrot query 1', 'parrot query 2']}]
        if signature == SynthesizeKnowledge:
            # dspy.TypedPredictor expects a pydantic object
            gap = KnowledgeGap(synthesized_summary="A parrot summary.", knowledge_gap="A parrot gap.", is_novel=True)
            return [gap]
        if signature == WriteProposal:
            return [{'proposal': 'A research proposal written by a parrot.'}]
        if signature == ReviewProposal:
            crit = Critique(score=0.9, justification="This is a fine parrot proposal.")
            return [crit]
            
        return ["Default parrot response."]
        
    def basic_request(self, prompt, **kwargs):
        # Not used by dspy.Predict, but required by the abstract class
        pass

class MockPaperQAService:
    """A mock PaperQAService that returns a fixed response."""
    async def query_documents(self, collection_name: str, query: str) -> dict:
        print(f"🦜 (PaperQA Parrot) Received query: '{query}'")
        return { "answer_text": f"This is a parrot summary for the query: '{query}'." }
```

### Phase 3: The UI-Compatible Orchestrator

**New File: `core/proposal_agent_dspy/orchestrator.py`**
```python
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
```

### Phase 4: Integration & Cleanup

1.  **Update UI Service Import & Placeholder Text**:
    *   In `ui/ui_research_plan.py`, change the import from the old service to the new factory function.
    *   **From:** `from core.proposal_agent.modern_service import create_modern_service`
    *   **To:** `from core.proposal_agent_dspy.orchestrator import create_dspy_service as create_modern_service` (using an alias for minimal changes)
    *   **Optional:** Update the `chat_input_box` placeholder text to guide the user on how to regenerate, e.g., "Provide feedback or type `!regenerate` to try again."

2.  **Delete** the old `core/proposal_agent` directory once the new system is fully validated.

This revised plan provides a clear path to a simpler system while ensuring the existing Gradio UI remains fully functional. 