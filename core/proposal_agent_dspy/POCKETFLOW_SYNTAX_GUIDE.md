# PocketFlow-Style Syntax Guide for DSPy Orchestrator

## Overview

This guide demonstrates how to use the new PocketFlow-inspired pipeline syntax for defining DSPy agent workflows. The new syntax makes workflows more readable, maintainable, and easier to understand.

## Basic Concepts

### Operators

- **`>>`** - Pipeline operator for sequential flow (continue transitions)
- **`-`** - Branch operator for conditional transitions
- **`FlowEnd()`** - Terminal marker to end a workflow branch

### Core Classes

- **`Node`** - Base class for workflow steps
- **`Flow`** - Container for the complete workflow graph
- **`FlowEngine`** - Executes flows and manages state
- **`FlowValidator`** - Validates flow integrity

## Quick Start Example

```python
from core.proposal_agent_dspy.orchestrator import Node, Flow, FlowEnd, FlowAction

# 1. Define your nodes
start_node = StartProcessingNode()
decision_node = DecisionNode()
success_node = SuccessNode()
failure_node = FailureNode()

# 2. Connect them with operators
start_node >> decision_node
decision_node - "success" >> success_node >> FlowEnd()
decision_node - "failure" >> failure_node >> FlowEnd()

# 3. Create the flow
flow = Flow.from_start_node(start_node, name="example_flow")

# 4. Validate and visualize
issues = flow.validate()
if not issues:
    flow.print_flow()
```

## Detailed Syntax Guide

### 1. Sequential Flow (Pipeline)

Use `>>` to connect nodes in sequence:

```python
# Simple linear flow
node_a >> node_b >> node_c

# This creates:
# node_a --continue--> node_b --continue--> node_c
```

### 2. Conditional Branching

Use `-` followed by `>>` for conditional transitions:

```python
router = RouterNode()
success_handler = SuccessNode()
error_handler = ErrorNode()

# Branch based on router's decision
router - "success" >> success_handler
router - "error" >> error_handler

# This creates:
# router --[success]--> success_handler
# router --[error]--> error_handler
```

### 3. Complex Workflows

Combine operators for sophisticated flows:

```python
# Mixed sequential and conditional flow
start >> preprocessor >> validator
validator - "valid" >> processor >> success_handler >> FlowEnd()
validator - "invalid" >> error_handler >> FlowEnd()
validator - "retry" >> preprocessor  # Cycle back for retry
```

### 4. Flow Termination

Always end branches with `FlowEnd()`:

```python
# Proper termination
final_node >> FlowEnd()

# Multiple endpoints
router - "approve" >> FlowEnd()
router - "reject" >> FlowEnd()
```

## Real-World Example: Proposal Generation

Here's how the actual proposal generation workflow is defined:

```python
def create_proposal_flow(use_parrot: bool = False) -> Flow:
    """Defines the proposal generation workflow using PocketFlow syntax."""
    
    # 1. Node definitions with dependency injection
    doc_service = MockPaperQAService() if use_parrot else PaperQAService()
    
    generate_queries = GenerateQueriesNode(QueryGenerator())
    pause_for_query_review = PauseForQueryReviewNode()
    user_input_router = UserInputRouterNode()
    literature_review = LiteratureReviewNode(doc_service)
    synthesize_knowledge = SynthesizeKnowledgeNode(KnowledgeSynthesizer())
    write_proposal = WriteProposalNode(ProposalWriter())
    review_proposal = ReviewProposalNode(ProposalReviewer())

    # 2. Main workflow pipeline
    generate_queries >> pause_for_query_review >> user_input_router

    # 3. Router branches
    user_input_router - "queries_approved" >> literature_review
    user_input_router - "regenerate_queries" >> generate_queries

    # 4. Research and writing pipeline
    literature_review >> synthesize_knowledge >> write_proposal >> review_proposal >> user_input_router

    # 5. Final decision branches
    user_input_router - "revision_requested" >> write_proposal
    user_input_router - "approved" >> FlowEnd()

    # 6. Build the flow
    return Flow.from_start_node(generate_queries, name="proposal_generation")
```

## Node Implementation Patterns

### Basic Node Template

```python
class MyCustomNode(Node):
    def __init__(self, custom_param=None):
        super().__init__("my_custom_node")  # Explicit name
        self.custom_param = custom_param
    
    async def execute(self, state: WorkflowState) -> FlowAction:
        # Your business logic here
        result = self.process_state(state)
        
        # Update state
        state.update("key", result)
        
        # Return flow action
        if result.success:
            return FlowAction(type="continue")
        else:
            return FlowAction(type="branch:error")
```

### Router Node Pattern

```python
class RouterNode(Node):
    async def execute(self, state: WorkflowState) -> FlowAction:
        # Make routing decision
        if state.condition_a:
            return FlowAction(type="branch:path_a")
        elif state.condition_b:
            return FlowAction(type="branch:path_b")
        else:
            return FlowAction(type="continue")  # Default path
```

### Pause Node Pattern

```python
class PauseForInputNode(Node):
    async def execute(self, state: WorkflowState) -> FlowAction:
        # Set interrupt context
        state.last_interrupt_type = "user_input"
        
        # Pause for human interaction
        return FlowAction(
            type="pause",
            data={
                "interrupt_type": "user_input",
                "message": "Please provide input",
                "context": {"current_data": state.relevant_data}
            }
        )
```

## Flow Validation and Debugging

### Validation

Always validate your flows before use:

```python
flow = Flow.from_start_node(start_node)

# Check for issues
issues = flow.validate()
if issues:
    print("Flow validation failed:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✅ Flow validation passed!")
```

Common validation issues:
- **Unreachable nodes**: Nodes not connected to the main flow
- **Missing transitions**: References to non-existent target nodes
- **Dangling references**: Transitions pointing to undefined nodes

### Debugging and Visualization

```python
# Human-readable flow structure
flow.print_flow()

# Generate Mermaid diagram for documentation
mermaid_code = flow.to_mermaid()
print(mermaid_code)

# Check node details
for name, node in flow.nodes.items():
    print(f"{name}: {node}")  # Shows successors
```

## Best Practices

### 1. Clear Node Naming

```python
# Good: Descriptive names
process_payment = ProcessPaymentNode()
validate_input = ValidateInputNode()

# Avoid: Generic names
node1 = SomeNode()
handler = AnotherNode()
```

### 2. Logical Flow Organization

```python
# Group related operations
preprocessing >> validation >> processing

# Separate concerns clearly
router - "success" >> success_path
router - "error" >> error_path
```

### 3. Dependency Injection

```python
def create_workflow(use_test_mode: bool = False) -> Flow:
    # Inject dependencies at flow creation
    service = MockService() if use_test_mode else RealService()
    node = ProcessingNode(service)
    # ... rest of flow definition
```

### 4. Always Validate

```python
def create_and_validate_flow() -> Flow:
    flow = Flow.from_start_node(start_node)
    
    issues = flow.validate()
    if issues:
        raise ValueError(f"Flow validation failed: {issues}")
    
    return flow
```

## Testing Your Flows

### Unit Testing Nodes

```python
def test_my_node():
    node = MyCustomNode()
    state = MockState()
    
    action = await node.execute(state)
    
    assert action.type == "continue"
    assert state.updated_field == expected_value
```

### Integration Testing Flows

```python
def test_flow_structure():
    flow = create_my_flow()
    
    # Test structure
    assert flow.start_node == "expected_start"
    assert len(flow.nodes) == expected_count
    
    # Test validation
    issues = flow.validate()
    assert len(issues) == 0
```

### End-to-End Testing

```python
async def test_full_workflow():
    orchestrator = MyOrchestrator(use_parrot=True)
    config = {"test": "config"}
    
    async for step in orchestrator.start_agent(config):
        # Verify each step
        assert "error" not in step
        
        if step.get("step") == "human_input_required":
            # Simulate user input
            async for next_step in orchestrator.continue_agent(
                step["thread_id"], "test input"
            ):
                # Continue verification
                pass
```

## Migration from Old Syntax

### Before (Method Chaining)

```python
def old_flow():
    return Flow("my_flow", "start") \
        .add_node(StartNode()) \
        .add_node(ProcessNode()) \
        .add_node(EndNode()) \
        .on_continue("start", "process") \
        .on_continue("process", "end")
```

### After (PocketFlow Syntax)

```python
def new_flow():
    start = StartNode()
    process = ProcessNode()
    end = EndNode()
    
    start >> process >> end >> FlowEnd()
    
    return Flow.from_start_node(start, name="my_flow")
```

## Common Patterns

### Retry Pattern

```python
processor = ProcessorNode()
validator = ValidatorNode()

processor >> validator
validator - "success" >> FlowEnd()
validator - "retry" >> processor  # Retry loop
validator - "failed" >> error_handler >> FlowEnd()
```

### Fan-out/Fan-in (Future Feature)

```python
# This pattern will be available in advanced features
router >> [worker1, worker2, worker3] >> aggregator
```

## Troubleshooting

### Common Issues

1. **"Unreachable node" validation error**
   - Ensure all nodes are connected to the main flow
   - Check for typos in branch conditions

2. **"Missing transition" error**
   - Verify all branch conditions match FlowAction types
   - Ensure continue paths are properly defined

3. **Flow doesn't execute as expected**
   - Use `flow.print_flow()` to visualize structure
   - Check node execute() methods return correct FlowAction types

### Debug Checklist

- [ ] All nodes connected with `>>` or `- "condition" >>`
- [ ] All branches end with `FlowEnd()`
- [ ] Flow validation passes (`flow.validate()`)
- [ ] Node names are unique and descriptive
- [ ] FlowAction types match transition conditions

## Conclusion

The PocketFlow-style syntax makes DSPy workflows:
- **More readable**: Visual pipeline structure
- **Easier to maintain**: Local node connections
- **Self-documenting**: Clear flow logic
- **Better supported**: IDE autocomplete and refactoring

Start with simple linear flows and gradually add complexity as needed. Always validate your flows and use the visualization tools to ensure correctness. 