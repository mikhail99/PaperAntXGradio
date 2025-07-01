# DSPy-based Refactoring Plan for Proposal Agent

## 1. Executive Summary

This document outlines a plan to refactor the `core/proposal_agent` system to a new, simpler architecture centered around `dspy`. The goal is to replace the complex `langgraph` setup with a more direct and readable approach where:

1.  **`dspy`** defines the agent logic (what each agent does).
2.  **Python** manages the overall workflow, state, and Human-in-the-Loop (HIL) interactions.
3.  **A custom, typed state object** provides a robust and easy-to-use container for workflow data.

This approach will significantly simplify the codebase, improve readability, and leverage `dspy`'s powerful features for building reliable language model pipelines.

## 2. Core Architectural Components

### a. `WorkflowState` (The State Manager)

We will create a new, typed state object to manage data throughout the workflow. It will replace the raw `TypedDict` and provide methods for safe and easy state manipulation, inspired by the ergonomics of `langgraph`'s state but implemented in a simple, self-contained class.

### b. `dspy` Signatures & Modules (The Agents)

All agent logic will be defined using `dspy`.
*   **Signatures (`signatures.py`):** Define the inputs and outputs for each agent task (e.g., generating queries, critiquing a proposal). The prompts are embedded directly as descriptions in the signature fields.
*   **Modules (`dspy_modules.py`):** Wrap the signatures in `dspy.Module` classes. These modules are the reusable "agents" that our orchestrator will call.

### c. The Python Orchestrator

A single, clear Python class (`orchestrator.py`) will control the entire workflow. It will:
*   Instantiate the `WorkflowState`.
*   Call the `dspy` modules in the correct sequence.
*   Handle all HIL interactions by pausing the script and asking for user input.
*   Make decisions based on the current state.

### d. Simple Parrot Testing

We will create a new, very simple `parrot.py` for testing.
*   A `MockLM` will mimic `dspy`'s language model, returning hardcoded data based on the signature it's asked to fulfill.
*   A `MockPaperQAService` will return a fixed string for any query.
This avoids the complexity of the previous scenario-based testing system while still allowing for fast, offline unit and integration tests.

---

## 3. Implementation Plan

### Phase 1: State Management & DSPy Foundations

**1. Create the `WorkflowState` object.**

**New File: `core/proposal_agent/state.py` (Overwrite)**
```python
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
        self.topic: str = topic
        self.collection_name: str = collection_name
        
        # Core state variables
        self.search_queries: List[str] = []
        self.literature_summaries: List[str] = []
        self.knowledge_gap: KnowledgeGap | None = None
        self.proposal_draft: str = ""
        self.review_team_feedback: Dict[str, Critique] = {}
        self.is_approved: bool = False
        self.revision_cycles: int = 0

    def update(self, key: str, value: Any) -> None:
        """Safely updates a state variable."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"'{key}' is not a valid state variable.")

    def append_to(self, key: str, value: T) -> None:
        """Appends a value to a list-based state variable."""
        if hasattr(self, key):
            current_value = getattr(self, key)
            if isinstance(current_value, list):
                current_value.append(value)
            else:
                raise TypeError(f"'{key}' is not a list and cannot be appended to.")
        else:
            raise KeyError(f"'{key}' is not a valid state variable.")
            
    def get(self, key: str, default: Any = None) -> Any:
        """Gets a value from the state, with an optional default."""
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return f"<WorkflowState queries={len(self.search_queries)} summaries={len(self.literature_summaries)} draft_exists={bool(self.proposal_draft)}>"
```

**2. Define `dspy` Signatures and Modules.**

**New File: `core/proposal_agent/signatures.py`**
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

**New File: `core/proposal_agent/dspy_modules.py`**
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

# ... and so on for a ProposalWriter and ProposalReviewer module.
```

### Phase 2: Simple Mocking for Testing

**1. Create the simple `parrot.py` file.**

**New File: `core/proposal_agent/parrot.py`**
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

### Phase 3: The Python Orchestrator

**1. Create the orchestrator.**

**New File: `core/proposal_agent/orchestrator.py`**
```python
import dspy
from .state import WorkflowState
from .dspy_modules import QueryGenerator, KnowledgeSynthesizer # etc.
from .paperqa_service import PaperQAService # Your real service
from .parrot import MockLM, MockPaperQAService

class Orchestrator:
    def __init__(self, use_parrot: bool = False):
        if use_parrot:
            dspy.configure(lm=MockLM())
            self.doc_service = MockPaperQAService()
        else:
            # Configure dspy with your real local Ollama model
            ollama_lm = dspy.Ollama(model='llama3', base_url='http://localhost:11434/')
            dspy.configure(lm=ollama_lm)
            self.doc_service = PaperQAService()
        
        # Instantiate DSPy modules
        self.query_generator = QueryGenerator()
        self.synthesizer = KnowledgeSynthesizer()
        # ... and so on for other modules

    async def run(self, topic: str, collection_name: str):
        state = WorkflowState(topic, collection_name)
        
        # --- Stage 1: Query Generation ---
        print("--- Generating Queries ---")
        prediction = self.query_generator(topic=state.topic, existing_queries=state.search_queries)
        state.update("search_queries", prediction.queries)
        print(f" -> Generated: {state.search_queries}")
        
        # --- Stage 2: Human-in-the-Loop (HIL) ---
        print("\n--- Review Queries ---")
        user_input = input(f"Press ENTER to approve or enter new queries: ")
        if user_input:
            state.update("search_queries", [q.strip() for q in user_input.split(',')])
            print(f" -> User updated queries: {state.search_queries}")

        # --- Stage 3: Literature Review ---
        print("\n--- Reviewing Literature ---")
        for query in state.search_queries:
            response = await self.doc_service.query_documents(state.collection_name, query)
            state.append_to("literature_summaries", response.get("answer_text"))
        
        # ... continue for all other stages in plain Python ...
        
        print("\n--- Workflow Complete ---")
        return state
```

### Phase 4: Cleanup

Once the new orchestrator is working and integrated with the UI:
1.  **Delete** the following old files:
    *   `graph.py`
    *   `modern_graph_builder.py`
    *   `modern_service.py`
    *   `hil_nodes.py`
    *   `parrot_services.py`
    *   `modern_parrot_services.py`
    *   `agent_config.json`
    *   `team_workflows.json`
    *   `prompts.json`
    *   `hil_configurations.json`
2.  Update the UI service (`ui/ui_proposal_agent_debugger.py`) to call the new `Orchestrator` instead of the `ModernProposalAgentService`.

This plan provides a clear path to a much simpler, more powerful, and more maintainable system. 