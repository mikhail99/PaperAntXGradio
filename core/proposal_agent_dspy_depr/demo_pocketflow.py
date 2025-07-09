#!/usr/bin/env python3
"""
PocketFlow-Style DSPy Orchestrator Demo

This script demonstrates the new PocketFlow-inspired syntax for defining
DSPy agent workflows. Run this to see the clean, readable workflow definition
in action.

Usage:
    python core/proposal_agent_dspy/demo_pocketflow.py
"""

import asyncio
from core.proposal_agent_dspy.orchestrator import (
    Node, Flow, FlowEnd, FlowAction, DSPyOrchestrator,
    create_proposal_flow
)

# Example 1: Simple Linear Workflow
def demo_simple_workflow():
    """Demonstrates basic pipeline operators."""
    print("🔄 Demo 1: Simple Linear Workflow")
    print("=" * 50)
    
    class ProcessNode(Node):
        async def execute(self, state) -> FlowAction:
            print(f"  Processing in {self.name}...")
            return FlowAction(type="continue")
    
    # Define nodes
    start = ProcessNode("start")
    middle = ProcessNode("middle") 
    end = ProcessNode("end")
    
    # Connect with clean syntax
    start >> middle >> end >> FlowEnd()
    
    # Build and visualize flow
    flow = Flow.from_start_node(start, name="simple_demo")
    
    print("\nFlow Structure:")
    flow.print_flow()
    
    print(f"\nValidation: {'✅ Passed' if not flow.validate() else '❌ Failed'}")
    print()

# Example 2: Conditional Branching
def demo_conditional_workflow():
    """Demonstrates conditional branching operators."""
    print("🔀 Demo 2: Conditional Branching")
    print("=" * 50)
    
    class RouterNode(Node):
        async def execute(self, state) -> FlowAction:
            # Simulate routing decision
            return FlowAction(type="branch:success")
    
    class HandlerNode(Node):
        async def execute(self, state) -> FlowAction:
            print(f"  Handling in {self.name}...")
            return FlowAction(type="continue")
    
    # Define nodes
    router = RouterNode("decision_router")
    success = HandlerNode("success_handler")
    error = HandlerNode("error_handler")
    
    # Connect with branching syntax
    router - "success" >> success >> FlowEnd()
    router - "error" >> error >> FlowEnd()
    
    # Build and visualize
    flow = Flow.from_start_node(router, name="branching_demo")
    
    print("\nFlow Structure:")
    flow.print_flow()
    
    print(f"\nValidation: {'✅ Passed' if not flow.validate() else '❌ Failed'}")
    
    # Generate Mermaid diagram
    print("\nMermaid Diagram:")
    print(flow.to_mermaid())
    print()

# Example 3: Complex Workflow with Cycles
def demo_complex_workflow():
    """Demonstrates complex workflow patterns."""
    print("🔄 Demo 3: Complex Workflow with Retry Logic")
    print("=" * 50)
    
    class ProcessorNode(Node):
        async def execute(self, state) -> FlowAction:
            return FlowAction(type="continue")
    
    class ValidatorNode(Node):
        async def execute(self, state) -> FlowAction:
            # Simulate validation result
            return FlowAction(type="branch:retry")  # For demo purposes
    
    class SuccessNode(Node):
        async def execute(self, state) -> FlowAction:
            return FlowAction(type="continue")
    
    # Define nodes
    processor = ProcessorNode("processor")
    validator = ValidatorNode("validator")
    success = SuccessNode("success")
    error_handler = ProcessorNode("error_handler")
    
    # Complex flow with retry pattern
    processor >> validator
    validator - "success" >> success >> FlowEnd()
    validator - "retry" >> processor  # Retry cycle
    validator - "error" >> error_handler >> FlowEnd()
    
    # Build and visualize
    flow = Flow.from_start_node(processor, name="complex_demo")
    
    print("\nFlow Structure:")
    flow.print_flow()
    
    print(f"\nValidation: {'✅ Passed' if not flow.validate() else '❌ Failed'}")
    print()

# Example 4: Real Proposal Generation Workflow
def demo_real_workflow():
    """Demonstrates the actual proposal generation workflow."""
    print("📝 Demo 4: Real Proposal Generation Workflow")
    print("=" * 50)
    
    # Create the real workflow in parrot mode for demo
    flow = create_proposal_flow(use_parrot=True)
    
    print(f"Workflow: {flow.name}")
    print(f"Total Nodes: {len(flow.nodes)}")
    print(f"Node Names: {list(flow.nodes.keys())}")
    
    print("\nFlow Structure:")
    flow.print_flow()
    
    validation_issues = flow.validate()
    print(f"\nValidation: {'✅ Passed' if not validation_issues else f'❌ Failed: {validation_issues}'}")
    
    print("\nMermaid Diagram (first 10 lines):")
    mermaid_lines = flow.to_mermaid().split('\n')
    for line in mermaid_lines[:10]:
        print(f"  {line}")
    print("  ... (truncated)")
    print()

# Example 5: Live Integration Test
async def demo_live_integration():
    """Demonstrates live integration with DSPy orchestrator."""
    print("🚀 Demo 5: Live Integration Test")
    print("=" * 50)
    
    # Create orchestrator in parrot mode for fast demo
    orchestrator = DSPyOrchestrator(use_parrot=True)
    
    print(f"Orchestrator Flow: {orchestrator.flow.name}")
    print(f"Engine Type: {type(orchestrator.engine).__name__}")
    
    # Test agent startup
    config = {
        "topic": "Machine Learning in Healthcare",
        "collection_name": "demo_collection"
    }
    
    print("\n🎬 Starting agent workflow...")
    step_count = 0
    
    try:
        async for step in orchestrator.start_agent(config):
            step_count += 1
            step_name = step.get('step', 'unknown')
            thread_id = step.get('thread_id', 'unknown')
            
            print(f"  Step {step_count}: {step_name} (thread: {thread_id[:8]}...)")
            
            # Stop after a few steps for demo
            if step_count >= 5:
                print("  ... (stopping demo after 5 steps)")
                break
                
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        return
    
    print(f"\n✅ Successfully executed {step_count} workflow steps!")
    print()

def main():
    """Run all PocketFlow syntax demonstrations."""
    print("🎭 PocketFlow-Style DSPy Orchestrator Demo")
    print("=" * 60)
    print("Showcasing clean, readable workflow definition syntax\n")
    
    # Run demonstrations
    demo_simple_workflow()
    demo_conditional_workflow() 
    demo_complex_workflow()
    demo_real_workflow()
    
    # Run async demo
    asyncio.run(demo_live_integration())
    
    print("🎉 Demo completed! The PocketFlow syntax provides:")
    print("  ✅ Clean, readable workflow definitions")
    print("  ✅ Visual pipeline structure (>> operator)")
    print("  ✅ Intuitive conditional branching (- operator)")
    print("  ✅ Automatic flow discovery and validation")
    print("  ✅ Enhanced debugging and visualization tools")
    print("  ✅ Seamless integration with existing DSPy components")
    print("\nReady for production use! 🚀")

if __name__ == "__main__":
    main() 