# Simplified Proposal Agent Refactoring Plan

## 1. Executive Summary

This document outlines a simplified refactoring plan for the `core/proposal_agent` system. The goal is to achieve our primary objectives—decoupling agent logic and enabling unit tests—with minimal complexity, preserving the existing fast and effective testing workflow provided by `modern_parrot_services.py`. This plan prioritizes simplicity and speed of implementation, which is ideal for a research prototype.

## 2. Refactoring Goals

1.  **Decompose Monolithic Function**: Break down the large `create_llm_node` function in `graph.py` into smaller, single-responsibility classes.
2.  **Enable Unit Testing**: Introduce clean interfaces to allow for mocking of external services (LLM, PaperQA) in tests.
3.  **Preserve Testing Workflow**: Ensure that the `USE_PARROT_SERVICES` flag and the scenario-based testing in `modern_parrot_services.py` continue to work seamlessly.
4.  **Minimize Architectural Changes**: Avoid adding unnecessary layers of abstraction like complex factories or new service management systems.

## 3. The Simplified Architecture

### Phase 1: Establish Clean Service Interfaces

This is the most critical step and the key to testability. We define abstract base classes that both our real services and our existing parrot services can conform to.

**New File: `core/proposal_agent/interfaces.py`**
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Type
from pydantic import BaseModel

class LLMService(ABC):
    """A simple, abstract interface for any service that acts like an LLM."""
    @abstractmethod
    async def generate_text(self, prompt: str, context: Dict[str, Any]) -> str:
        pass
    
    @abstractmethod
    async def generate_structured(self, prompt: str, context: Dict[str, Any], output_schema: Type[BaseModel]) -> BaseModel:
        pass

class DocumentSearchService(ABC):
    """A simple, abstract interface for any service that searches documents."""
    @abstractmethod
    async def query_documents(self, collection_name: str, query: str) -> Dict[str, Any]:
        pass
```

### Phase 2: Adapt Existing Services to the Interfaces

We make minimal, non-disruptive changes to our existing service files so they officially implement the new interfaces. This leverages all your existing code.

**File: `core/proposal_agent/modern_parrot_services.py` (Minor Additions)**
```python
# ... other imports
from .interfaces import LLMService, DocumentSearchService
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json

class ModernParrotChatOllama(BaseLanguageModel, Runnable, LLMService): # <-- Implements LLMService
    # ... (all existing code for __init__, _load_scenario, _get_default_scenario, _call, etc. remains the same)

    # --- Add methods to satisfy the LLMService interface ---
    async def generate_text(self, prompt: str, context: Dict[str, Any]) -> str:
        """Adapts the parrot's logic to the simple text generation interface."""
        messages = [HumanMessage(content=prompt.format(**context))]
        response_json_str = self._call(messages)
        try:
            return json.loads(response_json_str).get("content", response_json_str)
        except json.JSONDecodeError:
            return response_json_str

    async def generate_structured(self, prompt: str, context: Dict[str, Any], output_schema: Type[BaseModel]) -> BaseModel:
        """Adapts the parrot's logic to the simple structured generation interface."""
        from .parrot_services import PydanticLikeWrapper
        messages = [HumanMessage(content=prompt.format(**context))]
        response_json_str = self._call(messages)
        # Your parrot's response is already a JSON string matching the desired output
        return PydanticLikeWrapper(json.loads(response_json_str))

class ModernParrotPaperQAService(DocumentSearchService): # <-- Implements DocumentSearchService
    # The existing `query_documents` method already matches the interface. No changes needed.
    # ... (all existing code remains the same)

def get_modern_parrot_services(scenario: str = "happy_path") -> tuple[LLMService, DocumentSearchService]:
    """Factory now returns instances that conform to our new interfaces."""
    llm = ModernParrotChatOllama(scenario)
    paperqa = ModernParrotPaperQAService(scenario)
    # The single LLMService interface handles both text and json, simplifying what we pass around.
    return llm, paperqa
```

### Phase 3: Refactor Nodes and the Graph Builder

With the interfaces in place, we can now create clean, testable node classes and a simple factory to build them.

**New Directory & Files: `core/proposal_agent/nodes/`**
*(This directory will contain individual, focused node classes like `QueryGeneratorNode`, `LiteratureReviewerNode`, etc. They will be simple and depend only on the service interfaces.)*

**New File: `core/proposal_agent/node_factory.py` (Simple and Focused)**
```python
from typing import Dict, Any
from .interfaces import BaseAgentNode, LLMService, DocumentSearchService
from .nodes import QueryGeneratorNode, LiteratureReviewerNode # etc.

class NodeFactory:
    """Creates node instances, injecting the correct service dependency."""
    
    def __init__(self, llm_service: LLMService, doc_service: DocumentSearchService):
        self.llm_service = llm_service
        self.doc_service = doc_service
        self.node_map = {
            "query_generator_base": QueryGeneratorNode,
            "literature_reviewer_local": LiteratureReviewerNode,
            # ... map other node names from agent_config.json to their respective classes
        }

    def create_node(self, node_name: str, node_config: Dict[str, Any]) -> BaseAgentNode:
        node_class = self.node_map.get(node_name)
        if not node_class:
            raise ValueError(f"Unknown node name configured: {node_name}")

        # Simple dependency injection based on the node's needs
        if issubclass(node_class, LiteratureReviewerNode):
             return node_class(node_name, node_config, self.doc_service)
        else:
             return node_class(node_name, node_config, self.llm_service)
```

**File: `core/proposal_agent/modern_graph_builder.py` (Simplified Refactor)**
```python
# ... other imports
from .node_factory import NodeFactory
from .services import OllamaLLMService, PaperQADocumentService # These will be new files housing the real services
from .modern_parrot_services import get_modern_parrot_services
from .graph import USE_PARROT_SERVICES # <-- Continue using the existing flag for simplicity

class ModernWorkflowGraphBuilder:
    def __init__(self, workflow_name: str, scenario: str = "happy_path"):
        # ... (load workflow/team/node configs as before)
        
        # --- Simplified Service Selection Logic ---
        if USE_PARROT_SERVICES:
            print(f"--- Using PARROT services for scenario: '{scenario}' ---")
            llm_service, doc_service = get_modern_parrot_services(scenario)
        else:
            print("--- Using REAL services ---")
            llm_service = OllamaLLMService()       # New class implementing LLMService
            doc_service = PaperQADocumentService() # New class implementing DocumentSearchService
        
        # Inject the chosen services into the factory
        self.node_factory = NodeFactory(llm_service, doc_service)

    def build_graph(self) -> StateGraph:
        # ... (graph building logic remains the same, but now uses the factory to get node instances)
```

## 4. Simplified Implementation Plan

This plan focuses on incremental changes that deliver value quickly.

*   **Phase 1: Interfaces & Adaptation (1 Week)**
    1.  Create `interfaces.py`.
    2.  Create the `nodes/` directory and the `BaseAgentNode` class within it.
    3.  Make the minimal additions to `modern_parrot_services.py` so its classes implement the new interfaces.
    4.  Create `services.py` and move the real `Ollama` and `PaperQA` service logic there, adapting them to the interfaces.

*   **Phase 2: Node Migration & Testing (1-2 Weeks)**
    1.  Create `node_factory.py`.
    2.  One by one, migrate the logic for each agent from the `if/elif` block in `graph.py` into its own dedicated `BaseAgentNode` subclass in the `nodes/` directory.
    3.  For each new node class, write a simple unit test file in `tests/nodes/` that passes in a mocked version of the `LLMService` or `DocumentSearchService` to verify its logic.

*   **Phase 3: Integration & Cleanup (2 Days)**
    1.  Rewrite the `__init__` method of `ModernWorkflowGraphBuilder` to use the new simple service selection logic and the `NodeFactory`.
    2.  **Delete the `create_llm_node` function from `graph.py`**.
    3.  Run the application with `USE_PARROT_SERVICES = True` and confirm your existing test scenarios still work.

## 5. Benefits of This Simplified Approach

-   **Achieves Core Goals**: You get testable, single-responsibility nodes without fundamentally changing the architecture.
-   **Minimal Disruption**: Your excellent testing workflow with `modern_parrot_services.py` is fully preserved and requires almost no changes.
-   **Significantly Less Code**: We avoid adding complex factories, new service management layers, or rewriting parrot logic.
-   **Low Risk**: The changes are focused and incremental, making them easy to implement and debug.
-   **High Velocity**: This plan gets you back to iterating on your research prototype quickly with a much cleaner and more reliable codebase. 