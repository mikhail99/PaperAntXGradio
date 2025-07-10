from pocketflow import Flow

from core.pocketflow_demo.research_nodes import (
    ResearchAgentRouter,
    GenerateQueries,
    LiteratureReview,
    SynthesizeGap,
    FollowUp,
    ReportGeneration,
    ResultNotification,
)
from core.pocketflow_demo.research_state import Action


def create_research_flow():
    """
    Create and connect the nodes to form a research agent flow.
    Phase 1 & 2: Hub-and-spoke model for stability.
    The router is stateful and decides the next step based on session history.
    """
    # Core research nodes
    research_router = ResearchAgentRouter()
    generate_queries = GenerateQueries()
    literature_review = LiteratureReview()
    synthesize_gap = SynthesizeGap()
    follow_up = FollowUp()
    report_generation = ReportGeneration()
    result_notification = ResultNotification()

    # Hub-and-spoke connections - all nodes connect to and from the central router
    research_router - Action.do_generate_queries >> generate_queries
    generate_queries >> research_router

    research_router - Action.do_literature_review >> literature_review
    literature_review >> research_router

    research_router - Action.do_literature_review_gap >> synthesize_gap
    synthesize_gap >> research_router

    research_router - Action.do_write_proposal >> report_generation
    report_generation >> research_router

    research_router - Action.do_result_notification >> result_notification
    result_notification >> research_router

    research_router - Action.do_follow_up >> follow_up
    # No return from follow_up as it's a terminal human-in-the-loop step

    return Flow(start=research_router)
