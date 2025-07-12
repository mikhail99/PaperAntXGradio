from pocketflow import Flow
from core.pocketflow_demo.nodes.workers import (
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
    Create a simplified research flow using threading-based HITL.
    Much simpler than the session-based approach.
    """
    
    # Create all nodes
    generate_queries = GenerateQueries()
    review_queries = ReviewQueries()
    literature_review = LiteratureReview()
    synthesize_gap = SynthesizeGap()
    report_generation = ReportGeneration()
    review_report = ReviewReport()
    follow_up = FollowUp()
    result_notification = ResultNotification()

    # Simple linear pipeline with dynamic routing
    generate_queries >> review_queries
    
    # Dynamic routing from query review (threading handles the blocking)
    review_queries - "approved" >> literature_review
    review_queries - "rejected" >> generate_queries  # Retry query generation
    review_queries - "default" >> follow_up  # Unclear feedback
    
    # Continue linear pipeline
    literature_review >> synthesize_gap
    synthesize_gap >> report_generation
    report_generation >> review_report
    
    # Dynamic routing from report review (threading handles the blocking)
    review_report - "approved" >> result_notification
    review_report - "rejected" >> report_generation  # Retry report generation
    review_report - "default" >> follow_up  # Unclear feedback

    return Flow(start=generate_queries)

def create_shared_state(
    conversation_id: str,
    query: str, 
    history: List[Dict[str, Any]],
    chat_queue: Queue[Optional[str]],
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
