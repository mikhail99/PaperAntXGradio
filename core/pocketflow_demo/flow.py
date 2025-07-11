from pocketflow import Flow
from core.pocketflow_demo.nodes.workers import (
    FlowEntry,
    PauseForFeedback,
    GenerateQueries,
    LiteratureReview,
    SynthesizeGap,
    ReportGeneration,
    FollowUp,
    ResultNotification,
)
from core.pocketflow_demo.nodes.hitl import ReviewQueries, ReviewReport
from core.pocketflow_demo.types import SharedState
from queue import Queue
from typing import List, Dict, Any, Optional

def create_flow():
    """
    Create a simplified research flow with smart entry point for resume handling.
    Uses declarative connections and node-driven dynamic routing.
    """
    
    # Create all nodes
    flow_entry = FlowEntry()
    pause_for_feedback = PauseForFeedback()
    generate_queries = GenerateQueries()
    review_queries = ReviewQueries()
    literature_review = LiteratureReview()
    synthesize_gap = SynthesizeGap()
    report_generation = ReportGeneration()
    review_report = ReviewReport()
    follow_up = FollowUp()
    result_notification = ResultNotification()

    # Smart entry routing
    flow_entry - "start_new_flow" >> generate_queries
    flow_entry - "resume_query_review" >> review_queries
    flow_entry - "resume_report_review" >> review_report

    # Main research pipeline - mostly linear
    generate_queries >> review_queries
    
    # Dynamic routing from query review
    review_queries - "approved" >> literature_review
    review_queries - "rejected" >> generate_queries  # Retry query generation
    review_queries - "pause" >> pause_for_feedback   # Pause for human feedback
    
    # Continue linear pipeline
    literature_review >> synthesize_gap
    synthesize_gap >> report_generation
    report_generation >> review_report
    
    # Dynamic routing from report review
    review_report - "approved" >> result_notification
    review_report - "rejected" >> report_generation  # Retry report generation
    review_report - "pause" >> pause_for_feedback    # Pause for human feedback
    
    # Follow-up handling (fallback for unclear feedback)
    review_queries - "default" >> follow_up
    review_report - "default" >> follow_up
    
    # IMPORTANT: Handle "done" responses - this allows the flow to terminate properly when waiting for human input
    # When review nodes return "done", the flow should end and wait for user to provide feedback in a new flow run
    # The FlowEntry node will detect the waiting_for_feedback state and resume at the correct review node

    return Flow(start=flow_entry)

def create_shared_state(
    conversation_id: str,
    query: str, 
    history: List[Dict[str, Any]],
    chat_queue: Queue[str],
    flow_queue: Queue[Optional[str]]
) -> SharedState:
    """
    Create a properly typed shared state object for the research flow.
    
    Args:
        conversation_id: Unique identifier for the conversation
        query: Current user message
        history: Previous conversation history
        chat_queue: Queue for user-facing messages
        flow_queue: Queue for internal flow thoughts (can include None)
        
    Returns:
        Properly typed SharedState object
    """
    return SharedState(
        conversation_id=conversation_id,
        query=query,
        history=history,
        queue=chat_queue,
        flow_queue=flow_queue
    )
