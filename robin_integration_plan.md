# Robin Integration Plan (Two-Flow Architecture) - v2

## Executive Summary

This document outlines a revised plan to integrate Robin's scientific discovery capabilities with the `@proposal_agent_dspy` system. We will create **two separate but interconnected flows**, mirroring the real-world research process of **Discovery (Scouting)** followed by **Proposal Writing (Architecture)**. This modular approach enhances flexibility, maintainability, and user experience.

## Architectural Overview: Reusing the Core `FlowEngine`

A critical point of this design is that we will **reuse the existing `FlowEngine` and declarative node system** from `orchestrator.py`. This powerful, generic framework will run both of our new flows.

-   The `FlowEngine` is the heart of the system. We are not rebuilding it.
-   We will define two separate `Flow` objects, each with its own set of `Node`s and transitions.
-   The `DiscoveryOrchestrator` and `DSPyOrchestrator` classes are high-level controllers that simply tell the **same `FlowEngine` instance** *which* flow to run.

This approach avoids code duplication and builds upon the robust solution we already have.

---

## Deep Dive: The `robin_service.py` Wrapper

This new file is the most critical component for integrating Robin. Its primary goal is to act as a robust **Adapter** or **Anti-Corruption Layer** between our application and the external `robin` library. This keeps our main application clean and independent from Robin's specific implementation details.

### Core Responsibilities

1.  **Configuration Management**:
    - It will be responsible for taking simple inputs from our UI (like `topic` and `num_assays`) and translating them into the complex `RobinConfiguration` object that the `robin` library requires.
    - It will manage API keys and other sensitive configurations, fetching them from environment variables or a secure store, rather than having them scattered in the main application logic.

2.  **Execution and Error Handling**:
    - It will contain the `async` calls to Robin's main functions (`experimental_assay`, `therapeutic_candidates`).
    - It will be wrapped in extensive `try...except` blocks to catch errors specific to the Robin library (e.g., API failures, dependency issues, empty results). This prevents Robin's errors from crashing our entire application.

3.  **Output Parsing and Normalization**:
    - The `robin` library outputs its results into a series of CSV and text files in a specific directory structure. This is not a clean or stable format to work with directly.
    - The `robin_service.py` wrapper will be responsible for **reading these output files**, parsing them (using `pandas` for CSVs), and transforming the raw data into our clean, standardized `ResearchOpportunity` Pydantic models. This is a crucial data transformation step.

4.  **Mocking and Fallback**:
    - The service will have a built-in "mock mode." If it detects that Robin's dependencies are not installed or API keys are missing, it will not crash. Instead, it will return a pre-defined set of mock `ResearchOpportunity` objects.
    - This is essential for frontend development and testing, allowing the UI to be built and tested without needing a full, time-consuming Robin run.

By centralizing all interaction with the `robin` library into this single, well-defined service, we make our system more robust, easier to test, and simpler to maintain.

---

## Flow 1: The Discovery Agent (Scout)

**Goal**: To explore a broad topic and produce a ranked list of saveable research opportunities.

### 1.1 New `DiscoveryOrchestrator`
We will create a new high-level controller class, `DiscoveryOrchestrator`. This class will be similar in structure to the existing `DSPyOrchestrator`. Its primary job is to:
1.  Define the "discovery" `Flow` object with all its specific nodes (`RunRobinAssayNode`, `RankCandidatesNode`, etc.).
2.  Instantiate the shared `FlowEngine`.
3.  Provide public methods (`start_discovery`, etc.) that use the `FlowEngine` to run the discovery flow.

This makes the `DiscoveryOrchestrator` a specific *application* of our generic `FlowEngine`.

### 1.2 New `DiscoveryState`
**File**: `core/discovery_agent_dspy/state.py` (New File)
```python
import uuid
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class ResearchOpportunity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    opportunity_type: str # 'assay' or 'therapeutic_candidate'
    title: str
    summary: str
    ranking_score: float
    raw_data: Dict[str, Any]

class DiscoveryState:
    """State manager for the Discovery Agent workflow."""
    def __init__(self, topic: str):
        self.thread_id: str = str(uuid.uuid4())
        self.topic: str = topic
        self.search_queries: List[str] = []
        self.opportunities: List[ResearchOpportunity] = []
```

### 1.3 New `robin_service.py` & `DiscoveryStorage`
As planned, the `DiscoveryOrchestrator` will use a `RobinService` wrapper and a `DiscoveryStorage` class for its operations, keeping it independent from the `ProposalStorage`.

### 1.4 UI Tab 1: "🔬 Scientific Discovery"
A new Gradio tab dedicated to this flow.
- **Input**: A text box for the research topic/disease.
- **Action**: A "Begin Discovery" button that calls `DiscoveryOrchestrator.start_discovery`.
- **Output**: An interactive table displaying the ranked `ResearchOpportunity` results, with a "Save Opportunity" button.

---

## Flow 2: The Proposal Agent (Architect)

**Goal**: To take a single, validated research opportunity and generate a full technical proposal.

### 2.1 Modifications to `DSPyOrchestrator` and `WorkflowState`
The existing `DSPyOrchestrator` and `WorkflowState` in `core/proposal_agent_dspy/` will be modified to accept a `ResearchOpportunity` as an optional input.

**File**: `core/proposal_agent_dspy/state.py`
```python
# Modify WorkflowState to accept a discovery opportunity
class WorkflowState:
    def __init__(self, topic: str, collection_name: str, opportunity: ResearchOpportunity = None):
        # ... existing init ...
        self.linked_opportunity: ResearchOpportunity | None = opportunity
        
        # If an opportunity is provided, seed the state
        if opportunity:
            self.topic = opportunity.title
            # Pre-fill knowledge gap or summaries based on opportunity data
            self.knowledge_gap = self._create_gap_from_opportunity(opportunity)
```
This change allows the proposal flow to "bootstrap" itself with the high-quality, validated starting point from the Discovery Agent.

### 2.2 Modifications to the Proposal Flow
**File**: `core/proposal_agent_dspy/orchestrator.py`
- The `create_proposal_flow` function will be updated with a new starting node: `check_for_opportunity`.
- This new node will check if `state.linked_opportunity` exists.
    - If **yes**, it branches directly to `synthesize_knowledge` or `write_proposal`, bypassing the initial query generation.
    - If **no**, it proceeds with the original flow starting from `generate_queries`.

### 2.3 UI Tab 2: "📝 Proposal Generation"
The existing UI will be slightly modified.
- A new "Load Saved Discovery" dropdown will be added.
- When a discovery opportunity is loaded, the `topic` and `proposal_input` text boxes are pre-filled, readying the Architect to begin its work on a validated idea.

---

## Revised Implementation Plan

This modular plan is easier to implement and test.

### Week 1-2: Build the Scout
- [ ] Create the `discovery_agent_dspy` directory.
- [ ] Implement `DiscoveryState` and `DiscoveryStorage`.
- [ ] Implement `robin_service.py` to wrap Robin's core functions.
- [ ] Create the `DiscoveryOrchestrator` and define its simple flow using the existing `FlowEngine`.
- [ ] Build the "Scientific Discovery" UI tab.

### Week 3-4: Enhance the Architect
- [ ] Modify `WorkflowState` in the proposal agent to accept an opportunity.
- [ ] Update the `DSPyOrchestrator` and its `create_proposal_flow` to handle the new input path.
- [ ] Update the "Proposal Generation" UI to load saved discoveries.
- [ ] End-to-end testing of the handoff from Scout to Architect.

### Week 5-6: Refinement & Testing
- [ ] Full user journey testing.
- [ ] Refine the format of the saved `ResearchOpportunity` object.
- [ ] Improve error handling and user feedback for both flows.
- [ ] Write documentation for both new user-facing features.

## Conclusion: A More Robust and User-Centric Design

This two-flow architecture is a superior design because it:
- **Reuses Proven Components**: Leverages the existing `FlowEngine` for both processes.
- **Aligns with the Mental Model of Research**: Supports the natural process of broad exploration followed by deep, focused work.
- **Increases Modularity**: Makes the system easier to build, test, and maintain.
- **Enhances Flexibility**: Each agent can be used as a powerful standalone tool.

This approach delivers more value by creating a system that is not just automated, but is a true partner in the research process. 