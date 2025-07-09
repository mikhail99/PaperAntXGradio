"""
Minimal PocketFlow Research Flow

Direct copy of cookbook pattern - simple linear flow, no complexity.
"""

from pocketflow import Flow
from .nodes import (
    GenerateQueries,
    QueryDocuments,
    SynthesizeGap,
    WriteProposal,
    ReviewProposal,
)


def create_research_flow():
    """
    Create and connect the nodes to form a complete research flow.
    
    Simple linear flow like cookbook.
    """
    generate_queries = GenerateQueries()
    query_documents = QueryDocuments()
    synthesize_gap = SynthesizeGap()
    write_proposal = WriteProposal()
    review_proposal = ReviewProposal()

    # Linear flow (like cookbook)
    generate_queries >> query_documents >> synthesize_gap >> write_proposal >> review_proposal

    return Flow(start=generate_queries) 