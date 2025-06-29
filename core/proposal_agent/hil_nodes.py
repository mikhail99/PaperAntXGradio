"""
Modern Human-in-the-Loop nodes using LangGraph's interrupt() pattern.
This completely replaces the old interrupt_after approach.
"""

from langgraph.types import interrupt
from core.proposal_agent.state import ProposalAgentState
from typing import Dict, Any


def human_query_review_node(state: ProposalAgentState) -> Dict[str, Any]:
    """Human review of generated queries using modern interrupt() pattern."""
    queries = state.get('search_queries', [])
    query_list = "\n".join(f"- `{q}`" for q in queries)
    topic = state.get("topic", "")
    
    # Modern interrupt pattern - clean and ergonomic
    user_input = interrupt({
        "type": "query_review",
        "message": f"✅ **Query Generated.** The agent plans to search for:\n{query_list}\n\nProvide a new query or type 'continue' to approve.",
        "queries": queries,
        "context": {
            "stage": "query_generation",
            "topic": topic,
            "query_count": len(queries)
        }
    })
    
    # Handle user input
    if user_input.strip().lower() == 'continue':
        return {}  # No state changes needed
    else:
        # User provided a custom query - replace with their input
        return {"search_queries": [user_input]}


def human_insight_review_node(state: ProposalAgentState) -> Dict[str, Any]:
    """Human review of literature synthesis using interrupt() pattern."""
    knowledge_gap_data = state.get('knowledge_gap', {})
    knowledge_gap = knowledge_gap_data.get('knowledge_gap', 'No gap identified') if isinstance(knowledge_gap_data, dict) else str(knowledge_gap_data)
    
    user_input = interrupt({
        "type": "insight_review", 
        "message": f"✅ **Literature Synthesized.** Knowledge gap identified:\n\n*'{knowledge_gap}'*\n\nProvide a refined gap or type 'continue' to approve.",
        "knowledge_gap": knowledge_gap,
        "context": {
            "stage": "literature_review",
            "summaries_count": len(state.get('literature_summaries', [])),
            "topic": state.get("topic", "")
        }
    })
    
    if user_input.strip().lower() == 'continue':
        return {}
    else:
        # Update the knowledge gap with user input
        current_kg = state.get('knowledge_gap', {})
        if isinstance(current_kg, dict):
            current_kg['knowledge_gap'] = user_input
        else:
            current_kg = {"knowledge_gap": user_input, "synthesized_summary": str(current_kg)}
        return {"knowledge_gap": current_kg}


def human_review_node(state: ProposalAgentState) -> Dict[str, Any]:
    """Final human review using interrupt() pattern."""
    proposal = state.get('proposal_draft', 'No proposal generated')
    reviews = state.get('review_team_feedback', {})
    revision_cycle = state.get('proposal_revision_cycles', 0)
    
    # Create a summary of AI reviews for context
    review_summary = ""
    if reviews:
        for reviewer, feedback in reviews.items():
            if isinstance(feedback, dict):
                score = feedback.get('score', 'N/A')
                justification = feedback.get('justification', 'No justification provided')
                review_summary += f"**{reviewer}**: Score {score} - {justification}\n"
    
    user_input = interrupt({
        "type": "final_review",
        "message": f"✅ **AI Review Complete** (Revision #{revision_cycle}).\n\n**AI Feedback:**\n{review_summary}\n\nPlease provide feedback or type 'continue' to approve the AI's suggestions.",
        "proposal_draft": proposal,
        "ai_reviews": reviews,
        "context": {
            "stage": "final_review",
            "revision_cycle": revision_cycle,
            "topic": state.get("topic", "")
        }
    })
    
    if user_input.strip().lower() == 'continue':
        return {}
    else:
        # Store human feedback for next revision - this gets picked up by proposal creation
        return {"current_literature": f"Human feedback: {user_input}"}


def workflow_complete_node(state: ProposalAgentState) -> Dict[str, Any]:
    """Marks workflow completion - replaces legacy clear_pause_state_node."""
    return {"proposal_revision_cycles": state.get('proposal_revision_cycles', 0)} 