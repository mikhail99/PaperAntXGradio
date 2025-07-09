import dspy
from typing import List
from .state import KnowledgeGap, Critique

class GenerateQueries(dspy.Signature):
    """Generates a list of search queries based on a research topic."""
    topic: str = dspy.InputField(desc="The main research topic.")
    existing_queries: str = dspy.InputField(desc="A stringified list of queries already tried.")
    queries: List[str] = dspy.OutputField(desc="A list of 3-5 new, focused search queries.")

class SynthesizeKnowledge(dspy.Signature):
    """Synthesizes literature summaries into a coherent overview and identifies a knowledge gap."""
    topic: str = dspy.InputField()
    literature_summaries: str = dspy.InputField(desc="A stringified list of summaries from literature reviews.")
    knowledge_gap: KnowledgeGap = dspy.OutputField()

class WriteProposal(dspy.Signature):
    """Writes a research proposal based on an identified knowledge gap and any prior feedback."""
    knowledge_gap_summary: str = dspy.InputField()
    prior_feedback: str = dspy.InputField(desc="A summary of feedback from previous review cycles, if any.")
    proposal: str = dspy.OutputField(desc="A well-structured research proposal.")

class ReviewProposal(dspy.Signature):
    """Reviews a proposal for a specific quality, like novelty or feasibility."""
    proposal_draft: str = dspy.InputField()
    review_aspect: str = dspy.InputField(desc="The specific aspect to review (e.g., 'technical feasibility', 'novelty').")
    critique: Critique = dspy.OutputField() 