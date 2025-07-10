from pocketflow import Flow

from core.pocketflow_demo.nodes.router import ResearchAgentRouter
from core.pocketflow_demo.nodes.workers import (
    GenerateQueries,
    LiteratureReview,
    SynthesizeGap,
    FollowUp,
    ReportGeneration,
    ResultNotification,
)
from core.pocketflow_demo.nodes.hitl import ReviewQueries, ReviewReport
from core.pocketflow_demo.nodes.router import ResearchAgentRouter
from core.pocketflow_demo.nodes.actions import Action
from core.pocketflow_demo.nodes.router import ACTION_TO_NODE


def create_research_flow():
    """
    Create and connect the nodes to form a research agent flow with human-in-the-loop reviews.
    Critical steps (query generation and report generation) include human review points.
    """
    # Core research nodes
    research_router = ResearchAgentRouter()
    generate_queries = GenerateQueries()
    review_queries = ReviewQueries()
    literature_review = LiteratureReview()
    synthesize_gap = SynthesizeGap()
    report_generation = ReportGeneration()
    review_report = ReviewReport()
    follow_up = FollowUp()
    result_notification = ResultNotification()

    # Hub-and-spoke connections with review nodes
    
    # Query generation flow with review
    research_router - Action.do_generate_queries >> generate_queries
    generate_queries >> research_router
    research_router - Action.review_queries >> review_queries
    # review_queries pauses for human feedback, then user input restarts flow
    
    # Literature review flow (continues after queries approved)
    research_router - Action.do_literature_review >> literature_review
    literature_review >> research_router

    # Gap analysis flow
    research_router - Action.do_literature_review_gap >> synthesize_gap
    synthesize_gap >> research_router

    # Report generation flow with review
    research_router - Action.do_write_proposal >> report_generation
    report_generation >> research_router
    research_router - Action.review_report >> review_report
    # review_report pauses for human feedback, then user input restarts flow

    # Final result notification (after report approved)
    research_router - Action.do_result_notification >> result_notification
    result_notification >> research_router

    # Follow-up for any issues
    research_router - Action.do_follow_up >> follow_up
    # No return from follow_up as it's a terminal human-in-the-loop step

    return Flow(start=research_router)
