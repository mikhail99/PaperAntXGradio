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
        collection_name: str,
        question: str,
        local_papers_only: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Runs the proposal agent graph and yields the state at each step.

        Args:
            collection_name: The ID of the collection to research.
            question: The user's research question/direction.
            local_papers_only: If True, only search within the local ChromaDB.

        Yields:
            A dictionary representing the current state of the agent's process.
        """
        # Define the initial state for the new structure
        initial_state = ProposalAgentState(
            topic=question,
            collection_name=collection_name,
            local_papers_only=local_papers_only,
            search_queries=[],
            literature_summaries=[],
            knowledge_gap={},
            proposal_draft="",
            review_team_feedback={},
            final_review={},
            proposal_revision_cycles=0
        )

        # Stream the graph execution
        async for step in self.graph.astream(initial_state):
            step_name = list(step.keys())[0]
            # The state is now managed by LangGraph, so we can yield the full
            # state from the step directly.
            current_state = step[step_name]
            
            yield {
                "step": step_name,
                "state": current_state
            }
            # Add a small sleep to allow the UI to update
            await asyncio.sleep(0.1) 