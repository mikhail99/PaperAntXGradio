# PocketFlow Implementation Patterns

## Discovered Patterns from Phase 1 Analysis

### **Core API Structure**

```python
from pocketflow import Flow, Node  # Main classes

# Node is actually BaseNode with retry functionality
class Node(BaseNode):
    def __init__(self, max_retries=1, wait=0): ...
    def prep(self, shared): ...      # Data preparation
    def exec(self, prep_res): ...    # Core logic (isolated)
    def post(self, shared, prep_res, exec_res): ...  # State updates + routing

# Flow orchestrates nodes
class Flow(BaseNode):
    def __init__(self, start=None): ...
    def run(self, shared): ...       # Execute the entire flow
```

### **1. Node 3-Phase Pattern**

Every node follows strict separation of concerns:

```python
class MyNode(Node):
    def prep(self, shared):
        """
        ONLY read from shared state, prepare inputs for exec()
        - Extract needed data from shared dictionary
        - Transform/serialize data for processing
        - Return prep_result for exec()
        """
        return shared["topic"], shared.get("queries", [])
    
    def exec(self, prep_res):
        """
        PURE computation - NO access to shared state
        - Idempotent (safe for retries)
        - Use only prep_res as input
        - Return exec_result for post()
        """
        topic, queries = prep_res
        # Do LLM call, API call, computation
        return computed_result
    
    def post(self, shared, prep_res, exec_res):
        """
        Update shared state and determine routing
        - Write exec_res back to shared state
        - Update state fields
        - Return action string for routing
        """
        shared["search_queries"] = exec_res
        return "default"  # or "approved", "regenerate", etc.
```

### **2. Flow Routing (Action-Based)**

Routing is entirely based on the string returned by `post()`:

```python
# Basic default transition
node_a >> node_b  # If node_a.post() returns "default" or None

# Named action transitions  
router - "approved" >> process_payment
router - "needs_revision" >> revise_report
router - "rejected" >> finish_process

# Flow definition
flow = Flow(start=router)
flow.run(shared)  # Executes until no next node found
```

### **3. Shared State (Plain Dictionary)**

PocketFlow uses a simple dictionary for all state:

```python
shared = {
    "topic": "AI in Healthcare", 
    "search_queries": [],
    "literature_summaries": [],
    # ... any other state
}

# Nodes read and write to this shared dict
# No validation - nodes must handle missing keys gracefully
```

### **4. Error Handling & Retries**

Built into Node class:

```python
class MyNode(Node):
    def __init__(self):
        super().__init__(max_retries=3, wait=1.0)
    
    def exec_fallback(self, prep_res, exc):
        """Called if exec() fails after all retries"""
        return {"error": str(exc)}
```

### **5. Conditional Transitions**

The `- "action"` syntax creates conditional transitions:

```python
# This creates a _ConditionalTransition object
router - "approved" >> payment_node
router - "rejected" >> finish_node

# Equivalent to:
# if router.post() returns "approved" → go to payment_node
# if router.post() returns "rejected" → go to finish_node
```

## **Implementation Guidelines**

### **State Management Best Practices**

1. **Always check for missing keys**:
```python
def prep(self, shared):
    queries = shared.get("search_queries", [])  # Safe default
    topic = shared["topic"]  # Will raise KeyError if missing
```

2. **Use consistent field names**:
```python
# Consistent naming across all nodes
shared["search_queries"]      # not "queries" or "search_list"
shared["literature_summaries"] # not "summaries" or "lit_review"
```

3. **Document state dependencies**:
```python
class LiteratureReviewNode(Node):
    """
    Requires: search_queries (List[str])
    Produces: literature_summaries (List[str])
    """
```

### **Router Node Pattern**

Router nodes are decision points:

```python
class QueryProcessingRouter(Node):
    def prep(self, shared):
        user_input = shared.get("_last_user_input", "").strip().lower()
        return user_input
    
    def exec(self, prep_res):
        # Pure decision logic
        user_input = prep_res
        if user_input == "!regenerate":
            return "regenerate_queries"
        elif user_input.startswith("edit:"):
            return "edit_queries"  
        return "queries_approved"
    
    def post(self, shared, prep_res, exec_res):
        # Handle any side effects
        if exec_res == "edit_queries":
            # Parse edited queries from user input
            edited = shared["_last_user_input"][5:]  # Remove "edit:"
            shared["search_queries"] = edited.split(",")
        return exec_res  # Return the routing decision
```

### **HITL (Human-in-the-Loop) Pattern**

For pausing and waiting for user input:

```python
class PauseForReviewNode(Node):
    def prep(self, shared):
        chat_queue = shared["chat_queue"]
        data_to_review = shared["search_queries"]
        return chat_queue, data_to_review
    
    def exec(self, prep_res):
        chat_queue, data = prep_res
        # Send pause message to UI
        chat_queue.put({
            "type": "pause_for_review",
            "message": "Please review the generated queries",
            "data": data
        })
        # Signal end of flow messages
        chat_queue.put(None)
        return "paused"
    
    def post(self, shared, prep_res, exec_res):
        # Set flag that we're waiting for user input
        shared["_waiting_for_input"] = True
        return "default"  # Continue to next node (which waits)
```

## **Key Insights**

1. **Simple but Powerful**: 100-line framework but handles complex workflows
2. **Pure Functions**: `exec()` methods must be pure (no shared state access)
3. **Action-Based Routing**: All flow control via string actions from `post()`
4. **No Built-in Validation**: Nodes must handle missing/invalid state gracefully
5. **Copy on Execute**: Nodes are copied during execution (enables parallelism)
6. **Dictionary State**: Simple dict-based state (no classes required)

## **Migration Strategy**

1. **Keep our typed state** but add conversion methods
2. **Adapt nodes** to 3-phase pattern (prep/exec/post)
3. **Map routing logic** to action strings in post() methods
4. **Implement HITL** with queue-based communication
5. **Preserve validation** by adapting our flow validator to PocketFlow's structure

This analysis confirms our migration plan is solid and PocketFlow's patterns align well with our existing architecture! 