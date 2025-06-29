"""
Modern ProposalAgentService using LangGraph's Command pattern.
This completely replaces manual pause detection with clean interrupt handling.
"""

import uuid
import json
from pathlib import Path
from typing import Dict, Any, AsyncGenerator

from langgraph.types import Command
from core.proposal_agent.state import ProposalAgentState
from core.proposal_agent.modern_graph_builder import build_modern_graph


class ModernProposalAgentService:
    """Modern service layer using interrupt() and Command patterns."""
    
    def __init__(self, workflow_name: str = "research_proposal_workflow"):
        self.workflow_name = workflow_name
        self.graph = build_modern_graph(workflow_name)
        self.hil_config = self._load_hil_config()
        
    def _load_hil_config(self) -> Dict[str, Any]:
        """Load HIL configuration for interrupt handling."""
        config_path = Path(__file__).parent / "hil_configurations.json"
        with open(config_path, "r") as f:
            return json.load(f)
        
    async def start_agent(self, config: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Starts a new agent run with modern interrupt pattern."""
        thread_id = str(uuid.uuid4())
        
        initial_state = ProposalAgentState(
            topic=config.get("topic", ""),
            collection_name=config.get("collection_name", ""),
            local_papers_only=config.get("local_papers_only", True),
            search_queries=[],
            literature_summaries=[],
            knowledge_gap={},
            proposal_draft="",
            review_team_feedback={},
            final_review={},
            proposal_revision_cycles=0,
            current_literature=""
        )
        
        async for result in self._stream_graph(thread_id, initial_state):
            yield result

    async def continue_agent(self, thread_id: str, user_input: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Continues execution using modern Command pattern."""
        async for result in self._stream_graph(thread_id, Command(resume=user_input)):
            yield result
            
    async def _stream_graph(self, thread_id: str, graph_input: Any) -> AsyncGenerator[Dict[str, Any], None]:
        """Clean streaming using LangGraph's automatic interrupt handling."""
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            async for step in self.graph.astream(graph_input, config=config):
                step_name = list(step.keys())[0] if step else "unknown"
                step_output = step.get(step_name, {}) if step else {}
                
                # Check for interrupt - LangGraph handles this automatically
                if "__interrupt__" in step:
                    interrupt_data = step["__interrupt__"][0]
                    interrupt_payload = interrupt_data.value
                    
                    # Extract structured interrupt data
                    interrupt_type = interrupt_payload.get("type", "unknown")
                    message = interrupt_payload.get("message", "Please provide input")
                    context = interrupt_payload.get("context", {})
                    
                    # Get UI configuration for this interrupt type
                    ui_config = self.hil_config["interrupt_types"].get(interrupt_type, {})
                    
                    yield {
                        "step": "human_input_required",
                        "interrupt_type": interrupt_type,
                        "message": message,
                        "context": context,
                        "ui_config": ui_config,
                        "thread_id": thread_id,
                        "raw_interrupt_data": interrupt_payload
                    }
                    
                    # Break here to wait for user input
                    break
                else:
                    # Regular step output
                    yield {
                        "step": step_name,
                        "state": step_output,
                        "thread_id": thread_id,
                        "node_type": "processing"
                    }
                    
        except Exception as e:
            yield {
                "step": "error",
                "error": str(e),
                "thread_id": thread_id,
                "node_type": "error"
            }
            
    def get_interrupt_configuration(self, interrupt_type: str) -> Dict[str, Any]:
        """Get UI configuration for a specific interrupt type."""
        return self.hil_config["interrupt_types"].get(interrupt_type, {})
        
    def validate_user_input(self, interrupt_type: str, user_input: str) -> Dict[str, Any]:
        """Validate user input according to interrupt type rules."""
        validation_rules = self.hil_config["validation_rules"]
        interrupt_config = self.hil_config["interrupt_types"].get(interrupt_type, {})
        rule_name = interrupt_config.get("validation", "")
        
        if not rule_name or rule_name not in validation_rules:
            return {"valid": True, "message": "No validation rules defined"}
            
        rules = validation_rules[rule_name]
        
        # Check if continue is allowed
        if user_input.strip().lower() == "continue":
            return {"valid": True, "message": "Continue approved"}
            
        # Validate length
        if len(user_input) < rules.get("min_length", 0):
            return {
                "valid": False, 
                "message": f"Input too short. Minimum {rules.get('min_length', 0)} characters required."
            }
            
        if len(user_input) > rules.get("max_length", 1000):
            return {
                "valid": False,
                "message": f"Input too long. Maximum {rules.get('max_length', 1000)} characters allowed."
            }
            
        # Check forbidden characters
        forbidden_chars = rules.get("forbidden_chars", [])
        for char in forbidden_chars:
            if char in user_input:
                return {
                    "valid": False,
                    "message": f"Forbidden character '{char}' found in input."
                }
                
        return {"valid": True, "message": "Input validated successfully"}
        
    async def get_current_state(self, thread_id: str) -> Dict[str, Any]:
        """Get the current state of the graph."""
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state_snapshot = self.graph.get_state(config)
            return {
                "state": state_snapshot.values,
                "next_nodes": state_snapshot.next,
                "checkpoint": state_snapshot.config,
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }


def create_modern_service(workflow_name: str = "research_proposal_workflow") -> ModernProposalAgentService:
    """Factory function to create a modern service instance."""
    return ModernProposalAgentService(workflow_name)


# For testing
if __name__ == "__main__":
    import asyncio
    
    async def test_service():
        service = create_modern_service()
        
        config = {
            "topic": "AI safety testing",
            "collection_name": "test_collection"
        }
        
        print("Starting modern agent service...")
        async for result in service.start_agent(config):
            print(f"Step: {result.get('step', 'unknown')}")
            if result.get("step") == "human_input_required":
                print(f"Interrupt Type: {result.get('interrupt_type')}")
                print(f"Message: {result.get('message')}")
                break
                
        print("Test completed!")
        
    asyncio.run(test_service()) 