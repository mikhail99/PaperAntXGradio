import pytest
import asyncio
from typing import Dict, Any, Optional

# Adjust imports to match your project structure
from core.proposal_agent_dspy.orchestrator import Node, Flow, FlowEnd, FlowAction
from core.proposal_agent_dspy.state import WorkflowState

# Mock nodes for testing purposes
class MockState(WorkflowState):
    def __init__(self, topic="test", collection_name="test"):
        super().__init__(topic, collection_name)
        self.path = []

    def update(self, key, value):
        if key == "path":
            self.path.append(value)
        else:
            super().update(key, value)

class StartNode(Node):
    async def execute(self, state: MockState) -> FlowAction:
        state.update("path", self.name)
        return FlowAction(type="continue")

class NodeA(Node):
    async def execute(self, state: MockState) -> FlowAction:
        state.update("path", self.name)
        return FlowAction(type="branch:to_b")

class NodeB(Node):
    async def execute(self, state: MockState) -> FlowAction:
        state.update("path", self.name)
        return FlowAction(type="continue")

class NodeC(Node):
    async def execute(self, state: MockState) -> FlowAction:
        state.update("path", self.name)
        return FlowAction(type="continue")

class EndNode(Node):
     async def execute(self, state: MockState) -> FlowAction:
        state.update("path", self.name)
        return FlowAction(type="complete")

# --- Test Cases ---

def test_node_initialization():
    """Tests that nodes are named correctly."""
    start_node = StartNode()
    custom_name_node = NodeA(name="CustomA")
    assert start_node.name == "Start"  # "Node" suffix removed
    assert custom_name_node.name == "CustomA"
    assert "Node(Start" in repr(start_node)

def test_simple_pipeline_operator():
    """Tests the '>>' operator for a linear flow."""
    start = StartNode()
    node_a = NodeA()
    node_b = NodeB()
    
    start >> node_a >> node_b
    
    assert "continue" in start.successors
    assert start.successors["continue"] == node_a
    assert "continue" in node_a.successors
    assert node_a.successors["continue"] == node_b
    assert not node_b.successors

def test_conditional_branch_operator():
    """Tests the '-' operator for branching."""
    router = NodeA(name="Router")
    branch_b = NodeB()
    branch_c = NodeC()

    router - "to_b" >> branch_b
    router - "to_c" >> branch_c

    assert "branch:to_b" in router.successors
    assert router.successors["branch:to_b"] == branch_b
    assert "branch:to_c" in router.successors
    assert router.successors["branch:to_c"] == branch_c

def test_flow_end_marker():
    """Tests that FlowEnd correctly terminates a branch."""
    start = StartNode()
    end = EndNode()

    start >> end >> FlowEnd()

    assert "continue" in end.successors
    assert isinstance(end.successors["continue"], FlowEnd)

def test_flow_auto_discovery_from_start_node():
    """Tests the Flow.from_start_node() classmethod."""
    # 1. Define nodes
    start_node = StartNode()
    router_node = NodeA(name="Router")
    node_b = NodeB()
    node_c = NodeC()
    end_node = EndNode()

    # 2. Define flow using operators
    start_node >> router_node
    router_node - "to_b" >> node_b >> end_node >> FlowEnd()
    router_node - "to_c" >> node_c >> end_node

    # 3. Create flow from the start node
    flow = Flow.from_start_node(start_node, name="test_flow")

    # 4. Assertions
    assert flow.name == "test_flow"
    assert flow.start_node == "Start"  # "Node" suffix removed
    
    expected_nodes = {"Start", "Router", "B", "C", "End"}  # Updated expected names
    assert set(flow.nodes.keys()) == expected_nodes

    expected_transitions = {
        "Start": {"continue": "Router"},
        "Router": {"branch:to_b": "B", "branch:to_c": "C"},
        "B": {"continue": "End"},
        "C": {"continue": "End"},
        "End": {"continue": "complete"}
    }
    assert flow.transitions == expected_transitions

def test_flow_with_cycle():
    """Tests that a flow with a cycle is correctly represented."""
    start = StartNode()
    node_a = NodeA()
    node_b = NodeB()

    start >> node_a
    node_a - "to_b" >> node_b
    node_b >> node_a  # Cycle back to node_a

    flow = Flow.from_start_node(start)

    expected_nodes = {"Start", "A", "B"}  # Updated expected names
    assert set(flow.nodes.keys()) == expected_nodes

    expected_transitions = {
        "Start": {"continue": "A"},
        "A": {"branch:to_b": "B"},
        "B": {"continue": "A"}
    }
    assert flow.transitions == expected_transitions

def test_improved_node_naming():
    """Tests that Node suffix is automatically removed for cleaner names."""
    # Test automatic naming with our mock nodes
    start = StartNode()
    node_a = NodeA()
    
    assert start.name == "Start"
    assert node_a.name == "A"
    
    # Test that explicit names still work (real nodes set their own names)
    from core.proposal_agent_dspy.orchestrator import GenerateQueriesNode, PauseForQueryReviewNode
    
    # These nodes explicitly set their names, so they won't use auto-naming
    generate_queries = GenerateQueriesNode(None)  # Pass None for dspy_module
    pause_review = PauseForQueryReviewNode()
    
    assert generate_queries.name == "generate_queries"  # Explicitly set name
    assert pause_review.name == "pause_for_query_review"  # Explicitly set name

def test_node_repr_shows_successors():
    """Tests that Node __repr__ shows successor information for debugging."""
    start = StartNode()
    node_a = NodeA()
    node_b = NodeB()
    
    start >> node_a
    start - "branch_to_b" >> node_b
    
    repr_str = repr(start)
    assert "Start" in repr_str  # Updated expected name
    assert "continue" in repr_str
    assert "branch:branch_to_b" in repr_str

def test_flow_validation_valid_flow():
    """Tests flow validation on a valid flow."""
    start = StartNode()
    end = EndNode()
    
    start >> end >> FlowEnd()
    flow = Flow.from_start_node(start)
    
    issues = flow.validate()
    assert len(issues) == 0

def test_flow_validation_unreachable_node():
    """Tests flow validation detects unreachable nodes."""
    start = StartNode()
    end = EndNode()
    orphan = NodeA(name="Orphan")
    
    start >> end >> FlowEnd()
    # Manually add orphan node to simulate unreachable node
    flow = Flow.from_start_node(start)
    flow.nodes["Orphan"] = orphan
    
    issues = flow.validate()
    assert any("Unreachable node: Orphan" in issue for issue in issues)

def test_flow_print_method():
    """Tests the flow print method produces readable output."""
    start = StartNode()
    router = NodeA(name="Router")
    end = EndNode()
    
    start >> router
    router - "success" >> end >> FlowEnd()
    
    flow = Flow.from_start_node(start, name="test_print")
    
    # Capture print output
    import io
    import contextlib
    
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        flow.print_flow()
    
    printed = output.getvalue()
    assert "Flow: test_print" in printed
    assert "Start: Start" in printed  # Updated expected name
    assert "→" in printed
    assert "→[success]" in printed

def test_mermaid_generation():
    """Tests Mermaid diagram generation."""
    start = StartNode()
    router = NodeA(name="Router")
    end = EndNode()
    
    start >> router
    router - "success" >> end >> FlowEnd()
    
    flow = Flow.from_start_node(start)
    mermaid = flow.to_mermaid()
    
    assert "graph TD" in mermaid
    assert "Start" in mermaid  # Updated expected name
    assert "Router" in mermaid
    assert "End([Complete])" in mermaid
    assert "|success|" in mermaid

def test_dependency_injection_flow_creation():
    """Tests that create_proposal_flow works with dependency injection."""
    from core.proposal_agent_dspy.orchestrator import create_proposal_flow
    
    # Test with parrot=False (should work without external dependencies in test)
    flow_real = create_proposal_flow(use_parrot=False)
    assert flow_real.name == "proposal_generation"
    assert len(flow_real.nodes) > 0
    
    # Test with parrot=True
    flow_parrot = create_proposal_flow(use_parrot=True)
    assert flow_parrot.name == "proposal_generation"
    assert len(flow_parrot.nodes) > 0
    
    # Both flows should have the same structure
    assert set(flow_real.nodes.keys()) == set(flow_parrot.nodes.keys())
    assert flow_real.transitions == flow_parrot.transitions

def test_full_parrot_integration():
    """Tests complete parrot integration for fast testing."""
    from core.proposal_agent_dspy.orchestrator import DSPyOrchestrator
    import asyncio
    
    async def run_parrot_test():
        # Create orchestrator in parrot mode
        orchestrator = DSPyOrchestrator(use_parrot=True)
        
        # Verify flow structure
        assert orchestrator.flow.name == "proposal_generation"
        issues = orchestrator.flow.validate()
        assert len(issues) == 0
        
        # Test agent startup
        config = {"topic": "AI testing", "collection_name": "test_collection"}
        
        step_count = 0
        async for step in orchestrator.start_agent(config):
            step_count += 1
            
            # Verify each step has required fields
            assert "step" in step
            assert "thread_id" in step
            
            # Stop after first few steps to avoid full workflow
            if step_count >= 3:
                break
        
        assert step_count >= 3
        return True
    
    # Run the async test
    result = asyncio.run(run_parrot_test())
    assert result

def test_flow_visualization_integration():
    """Tests that visualization works with real proposal flow."""
    from core.proposal_agent_dspy.orchestrator import create_proposal_flow
    
    flow = create_proposal_flow(use_parrot=True)
    
    # Test print_flow works
    import io
    import contextlib
    
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        flow.print_flow()
    
    printed = output.getvalue()
    assert "proposal_generation" in printed
    assert "generate_queries" in printed
    assert "→" in printed
    
    # Test mermaid generation
    mermaid = flow.to_mermaid()
    assert "graph TD" in mermaid
    assert "generate_queries" in mermaid
    assert "End([Complete])" in mermaid 