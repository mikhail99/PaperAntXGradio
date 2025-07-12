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

def create_research_flow():
    """
    Create a simplified research flow with threading-based HITL.
    Linear pipeline with human review points for queries and reports.
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

    # Linear research pipeline with HITL review points
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

    return Flow(start=generate_queries)
