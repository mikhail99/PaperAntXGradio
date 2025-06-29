import asyncio
from typing import AsyncGenerator, Dict, Any
import uuid

from langgraph.graph import StateGraph
from core.proposal_agent.graph import graph
from core.proposal_agent.state import ProposalAgentState
from core.paperqa_service import PaperQAService

class ProposalAgentService:
    """
    A service to manage and run the research proposal agent graph.
    It separates the logic for starting and continuing a conversation.
    """
    def __init__(self):
        self.graph: StateGraph = graph
        self.paperqa_service = PaperQAService()
        self.HIL_NODES = {"human_query_review_node", "human_insight_review_node", "human_review_node"}

    async def _stream_graph(self, thread_id: str, graph_input: Any):
        """Helper method to stream the graph and yield results."""
        config = {"configurable": {"thread_id": thread_id}}
        final_state_for_saving = {}
        last_step_name = ""

        try:
            print(f"--- Service: Starting graph astream loop (Thread: {thread_id}) ---")
            async for step in self.graph.astream(graph_input, config=config):
                step_name = list(step.keys())[0]
                last_step_name = step_name
                step_output = step.get(step_name, {})
                
                print(f"--- Service: Yielding step '{step_name}' ---")
                
                # Don't yield the interrupt step itself to the client
                if step_name == "__interrupt__":
                    continue

                yield {
                    "step": step_name,
                    "state": step_output
                }

                if isinstance(step_output, dict):
                    final_state_for_saving.update(step_output)
            print("--- Service: Graph astream loop finished. ---")

        except Exception as e:
            import traceback
            print(f"--- Service: EXCEPTION in astream loop: {e} ---")
            print(traceback.format_exc())
            raise e
        
        finally:
            print(f"--- Service: In finally block. Last step was '{last_step_name}'. ---")
            paused_node = final_state_for_saving.get("paused_on")

            if paused_node and paused_node in self.HIL_NODES:
                print(f"--- Service: Detected pause on '{paused_node}'. Yielding 'human_input_required'. ---")
                yield {
                    "step": "human_input_required",
                    "state": final_state_for_saving,
                    "paused_on": paused_node,
                    "thread_id": thread_id
                }
            else:
                if final_state_for_saving:
                    print("--- Service: Detected natural finish. Yielding 'done'. ---")
                yield {
                    "step": "done",
                    "state": final_state_for_saving,
                    "thread_id": thread_id
                }
            print("--- Service: Final message yielded. Exiting service method. ---")

    async def start_agent(self, config: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Starts a new agent run with a fresh state."""
        thread_id = str(uuid.uuid4())
        print(f"--- New Conversation Started with Thread ID: {thread_id} ---")
        
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
            human_feedback=None,
            paused_on=None,
            proposal_revision_cycles=0,
            current_literature=""
        )
        
        async for result in self._stream_graph(thread_id, initial_state):
            yield result

    async def continue_agent(self, thread_id: str, human_feedback: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Continues an existing agent run that was paused for human input."""
        print(f"--- Resuming Conversation with Thread ID: {thread_id} ---")

        # Update the state with the human's feedback *before* resuming
        # This is a synchronous call.
        self.graph.update_state(
            {"configurable": {"thread_id": thread_id}},
            {"human_feedback": human_feedback},
        )
        
        # Resume the graph by passing None as the input.
        # The checkpointer will load the state automatically.
        async for result in self._stream_graph(thread_id, None):
            yield result 