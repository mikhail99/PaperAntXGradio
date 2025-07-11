# State Management in PocketFlow Demo

## Overview

The PocketFlow demo implements a sophisticated state management system that handles both **real-time shared state** (for inter-node communication) and **persistent session state** (for conversation continuity). This document explains how state flows through the system and how different components interact with it.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Shared State  │◄──►│ Session Storage │◄──►│  Node Execution │
│  (Runtime)      │    │  (Persistent)   │    │   (Workers)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 1. Shared State Structure

The **shared state** is a dictionary passed to every node's `prep()`, `exec()`, and `post()` methods. It contains runtime information needed for coordination between nodes.

### Core Components

```python
shared = {
    "conversation_id": str,        # Unique identifier for this conversation
    "history": List[dict],         # Chat message history
    "query": str,                  # Current user input
    "queue": Queue,                # Chat output queue (user-facing messages)
    "flow_queue": Queue,           # Flow log queue (internal thoughts)
}
```

### Key Features

- **Ephemeral**: Lives only during flow execution
- **Thread-safe**: Uses Queue objects for safe inter-node communication
- **Immutable keys**: Core structure doesn't change, but values can be updated
- **Scoped access**: Each node gets the same shared state reference

## 2. Session Storage System

The **session storage** provides persistence across conversation turns and system restarts.

### Implementation

```python
# core/pocketflow_demo/utils/conversation.py
conversation_cache = {}

def load_conversation(conversation_id: str):
    return conversation_cache.get(conversation_id, {})

def save_conversation(conversation_id: str, session: dict):
    conversation_cache[conversation_id] = session
```

### Session Structure

```python
session = {
    "last_action": Action,           # Previously executed action
    "action_result": str,            # Result from last action
    "params": {                      # Action-specific parameters
        Action.do_generate_queries: {
            "topic": "machine learning"
        },
        Action.do_literature_review: {
            "search_query": "ML education"
        }
    },
    "waiting_for_feedback": str,     # Human-in-the-loop state
}
```

### Persistence Features

- **Conversation continuity**: State survives between user interactions
- **Parameter storage**: Action inputs are preserved for re-execution
- **Flow control**: Tracks where the workflow stopped/paused
- **Feedback tracking**: Manages human-in-the-loop review states

## 3. Node State Interaction Patterns

### Standard Worker Node Pattern

Every worker node follows the same state interaction pattern:

```python
class ExampleWorker(Node):
    def prep(self, shared):
        # 1. Load persistent session
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        
        # 2. Extract required parameters
        params = session["params"][self.action_type()]
        return params["required_value"]

    def exec(self, prep_res):
        # 3. Process the input (stateless computation)
        input_value = prep_res
        return f"Processed: {input_value}"

    def post(self, shared, prep_res, exec_res):
        # 4. Update shared state (queues)
        flow_log = shared["flow_queue"]
        flow_log.put(f"⬅️ Completed work: {exec_res}")
        
        # 5. Persist results to session
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        session["action_result"] = exec_res
        save_conversation(conversation_id, session)
        
        return "default"  # Continue to next node
```

### Human-in-the-Loop (HITL) Pattern

Review nodes pause execution and wait for human feedback:

```python
class ReviewNode(Node):
    def prep(self, shared):
        # 1. Signal end of automated flow
        flow_log = shared["flow_queue"]
        flow_log.put(None)  # Stop flow thoughts
        
        # 2. Prepare review message
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        result = session.get("action_result", "")
        
        review_message = f"Please review: {result}"
        return review_message, shared["queue"]

    def exec(self, prep_res):
        # 3. Send message to user interface
        review_message, queue = prep_res
        queue.put(review_message)
        queue.put(None)  # Signal message completion
        return review_message

    def post(self, shared, prep_res, exec_res):
        # 4. Set waiting state and return "done"
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        session["waiting_for_feedback"] = "review_type"
        save_conversation(conversation_id, session)
        
        return "done"  # Pause execution
```

## 4. Router State Management

The `ResearchAgentRouter` is the central state coordinator:

### State Decision Logic

```python
def compute_next_action(history, query, last_action, last_action_result):
    # State-based decision tree
    if last_action is None:
        return Action.do_generate_queries
    elif last_action == Action.review_queries:
        feedback = check_feedback_in_message(query)
        if feedback == "approved":
            return Action.do_literature_review
        elif feedback == "rejected":
            return Action.do_generate_queries  # Retry
        else:
            return Action.do_follow_up  # Ask for clarification
    # ... more decision logic
```

### Parameter Preparation

```python
def prepare_for_next_action(session, decision, flow_log):
    """Generic parameter validation using node self-declaration"""
    next_action = decision["action"]
    
    # Get required parameters from node class
    node_class = ACTION_TO_NODE[next_action]
    required_params = node_class.required_params()
    
    # Validate and store parameters
    params = {}
    for param in required_params:
        if param in decision:
            params[param] = decision[param]
        else:
            # Fallback to follow-up for missing params
            return handle_missing_params(session, flow_log, next_action, [param])
    
    session["params"][next_action] = params
    return next_action
```

## 5. Action Registry System

The `ACTION_TO_NODE` registry enables dynamic node instantiation:

```python
# router.py
ACTION_TO_NODE = {
    Action.do_generate_queries: GenerateQueries,
    Action.do_literature_review: LiteratureReview,
    Action.review_queries: ReviewQueries,
    Action.review_report: ReviewReport,
    # ... more mappings
}
```

### Self-Declaration Pattern

Each node declares its requirements:

```python
class GenerateQueries(Node):
    @staticmethod
    def required_params():
        return ["topic"]  # What this node needs
    
    @staticmethod
    def action_type():
        return Action.do_generate_queries  # How to identify this node
```

## 6. Queue-Based Communication

### Flow Queue (Internal Thoughts)

```python
flow_log = shared["flow_queue"]
flow_log.put("🤔 Analyzing user request...")
flow_log.put("⬅️ Generated 3 search queries")
flow_log.put(None)  # Signal completion
```

### Chat Queue (User Messages)

```python
chat_queue = shared["queue"]
chat_queue.put("Here are your search queries:")
chat_queue.put("Please review and approve.")
chat_queue.put(None)  # Signal completion
```

## 7. State Lifecycle

### Conversation Start
1. User sends first message
2. Shared state created with conversation_id
3. Empty session loaded (or new one created)
4. Router determines first action
5. Parameters prepared and stored in session

### Action Execution
1. Node prep() loads session data
2. Node exec() processes input (stateless)
3. Node post() updates both shared state and session
4. Flow continues to next node or pauses

### Human Review Points
1. Review node pauses execution (`return "done"`)
2. Session marked as `waiting_for_feedback`
3. User provides feedback
4. Router detects feedback and resumes flow
5. Next action determined based on approval/rejection

### Conversation End
1. `ResultNotification` delivers final output
2. Session state cleared (`last_action = None`)
3. Conversation marked as complete

## 8. Error Handling & Fallbacks

### Missing Parameters
- Router detects missing required parameters
- Automatically falls back to `FollowUp` action
- Asks user for missing information
- Retries original action once provided

### Invalid States
- Unknown actions fall back to `FollowUp`
- Malformed session data gets re-initialized
- Queue errors are caught and logged

### Session Recovery
- Conversation cache survives system restarts (in-memory)
- Session integrity checked on load
- Missing fields get default values

## 9. Best Practices

### For Node Developers
1. **Always use the conversation utilities**: `load_conversation()` and `save_conversation()`
2. **Implement self-declaration methods**: `required_params()` and `action_type()`
3. **Handle None gracefully**: Check for missing session keys
4. **Log important transitions**: Use flow_queue for debugging

### For State Management
1. **Keep shared state minimal**: Only runtime coordination data
2. **Use session for persistence**: Store action results and parameters
3. **Validate parameters early**: In router's `prepare_for_next_action()`
4. **Clean up after completion**: Clear temporary state in final nodes

## 10. Future Improvements

- **Database persistence**: Replace in-memory cache with database
- **State versioning**: Handle breaking changes to session structure
- **Distributed execution**: Share state across multiple processes
- **State debugging tools**: Better introspection and monitoring 