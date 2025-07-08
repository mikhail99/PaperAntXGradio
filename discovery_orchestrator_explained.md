# Explaining the `DiscoveryOrchestrator`

You are absolutely right to focus on this. The plan is to **reuse the core `FlowEngine` and `Node` system** you've already built. The `DiscoveryOrchestrator` is not a replacement; it's a new *user* of that existing engine.

Think of it this way:
-   **`FlowEngine`**: The generic, reusable car engine. It knows how to run any process.
-   **`DSPyOrchestrator`**: A driver who uses the engine to follow a "Proposal Writing" roadmap.
-   **`DiscoveryOrchestrator`**: A *different* driver who uses the *same* engine to follow a "Scientific Discovery" roadmap.

Here is what the code for the `DiscoveryOrchestrator` should look like in practice.

---

### File: `core/discovery_agent_dspy/orchestrator.py` (New File)

```python
# --- Imports ---
# REUSE the core components from the existing proposal agent
from core.proposal_agent_dspy.orchestrator import Flow, FlowEngine, Node, FlowAction

# Import the NEW components specific to this discovery flow
from .state import DiscoveryState
from .robin_service import RobinService
from .storage import DiscoveryStorage
from typing import Dict, Any, AsyncGenerator


# --- 1. Define Nodes for the Discovery Flow ---
# These are the specific steps for the "Scout's" roadmap.

class RunRobinDiscoveryNode(Node):
    """
    A specific node that encapsulates the call to the Robin service.
    Its only job is to run the discovery and put the results into the state.
    """
    def __init__(self, robin_service: RobinService):
        super().__init__("run_robin_discovery")
        self.robin_service = robin_service

    async def execute(self, state: DiscoveryState) -> FlowAction:
        opportunities = await self.robin_service.run_discovery(state)
        state.opportunities = opportunities
        # This action tells the FlowEngine to proceed to the next step.
        return FlowAction(type="continue")


class SaveDiscoveryResultsNode(Node):
    """
    A specific node that saves the results from the state.
    """
    def __init__(self, storage: DiscoveryStorage):
        super().__init__("save_discovery_results")
        self.storage = storage

    async def execute(self, state: DiscoveryState) -> FlowAction:
        session_path = self.storage.save_discovery_session(state)
        # Optionally save individual opportunities for the proposal agent
        for opp in state.opportunities:
            self.storage.save_opportunity(opp)
        
        # This action tells the FlowEngine that the flow is complete.
        return FlowAction(type="complete")


# --- 2. Define the Discovery Flow ---
# This is the "roadmap" for the discovery process.

def create_discovery_flow() -> Flow:
    """
    Defines the discovery workflow by chaining the nodes together.
    It's a simple, linear flow for now.
    """
    # We instantiate the generic Flow class with our specific nodes
    return Flow(name="scientific_discovery", start_node="run_robin_discovery") \
        .add_node(RunRobinDiscoveryNode(RobinService(config={}))) \
        .add_node(SaveDiscoveryResultsNode(DiscoveryStorage())) \
        .on_continue("run_robin_discovery", "save_discovery_results")


# --- 3. Create the High-Level Orchestrator Class ---
# This is the clean, public-facing interface for the UI to use.

class DiscoveryOrchestrator:
    """
    The main controller for the discovery workflow. It initializes and uses
    the generic FlowEngine to run its specific discovery flow.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # Instantiate the REUSABLE engine
        self.engine = FlowEngine()
        # Define the SPECIFIC flow for discovery
        self.flow = create_discovery_flow()

    async def start_discovery(self, topic: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Public method to start a new discovery session. The UI will call this.
        It creates the initial state and tells the engine to run our flow.
        """
        initial_state = DiscoveryState(topic=topic, config=self.config)
        
        # The engine does all the hard work of running the nodes in sequence
        async for result in self.engine.start(self.flow, initial_state):
            yield result
            
    # You would add other methods here for listing/loading saved discoveries.
```

### Key Takeaways

1.  **Clear Separation**: All the new discovery-specific logic (`RunRobinDiscoveryNode`, `DiscoveryState`, `create_discovery_flow`) is in its own directory (`core/discovery_agent_dspy`).
2.  **Maximum Reuse**: The core, generic classes (`Flow`, `FlowEngine`, `Node`, `FlowAction`) are imported directly from your existing, working solution. We are not rewriting the engine.
3.  **Encapsulation**: The `DiscoveryOrchestrator` class provides a clean and simple API (`start_discovery`) for the user interface to call, hiding the complexity of the underlying `FlowEngine`.

This structure is robust, scalable, and avoids duplicating the excellent workflow logic you've already established. 