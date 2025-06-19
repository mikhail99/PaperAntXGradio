import asyncio
from typing import AsyncGenerator, Dict, Any

from langgraph.graph import StateGraph
from core.proposal_agent.graph import graph
from core.proposal_agent.state import ProposalAgentState
from core.paperqa_service import PaperQAService

class ProposalAgentService:
    """
    A service to manage and run the research proposal agent graph.
    """
    def __init__(self):
        self.graph: StateGraph = graph
        self.paperqa_service = PaperQAService()

    async def run_agent(
        self,
        collection_id: str,
        question: str,
        local_papers_only: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Runs the proposal agent graph and yields the state at each step.

        Args:
            collection_id: The ID of the collection to research.
            question: The user's research question/direction.
            local_papers_only: If True, only search within the local ChromaDB.

        Yields:
            A dictionary representing the current state of the agent's process.
        """
        # Keep track of the full state as the agent runs
        current_state = {
            "topic": question,
            "collection_id": collection_id,
            "local_papers_only": local_papers_only,
            "search_queries": [],
            "literature_summaries": [],
            "research_plan": [],
            "novelty_assessment": []
        }

        # Stream the graph execution
        async for step in self.graph.astream(current_state):
            # step is a dictionary where the key is the node name
            # and the value is the updated state.
            step_name = list(step.keys())[0]
            update_diff = step[step_name]
            
            # Manually merge the lists for accumulating fields
            for key, value in update_diff.items():
                if isinstance(value, list) and key in current_state:
                    current_state[key].extend(value)
                else:
                    current_state[key] = value
            
            yield {
                "step": step_name,
                "state": current_state
            }
            # Add a small sleep to allow the UI to update
            await asyncio.sleep(0.1) 