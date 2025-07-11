from pocketflow import Flow
from core.pocketflow_demo.nodes.workers import (
    FlowEntry,
    GenerateQueries,
    LiteratureReview,
    SynthesizeGap,
    ReportGeneration,
    FollowUp,
    ResultNotification,
)
from core.pocketflow_demo.nodes.hitl import ReviewQueries, ReviewReport

def create_research_flow():
    """
    Create a simplified research flow with smart entry point for resume handling.
    Uses declarative connections and node-driven dynamic routing.
    """
    
    # Create all nodes
    flow_entry = FlowEntry()
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
    
    # Continue linear pipeline
    literature_review >> synthesize_gap
    synthesize_gap >> report_generation
    report_generation >> review_report
    
    # Dynamic routing from report review
    review_report - "approved" >> result_notification
    review_report - "rejected" >> report_generation  # Retry report generation
    
    # Follow-up handling (fallback for unclear feedback)
    review_queries - "default" >> follow_up
    review_report - "default" >> follow_up

    return Flow(start=flow_entry)
