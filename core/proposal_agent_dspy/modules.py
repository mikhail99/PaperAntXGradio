"""
DSPy modules for research proposal generation.
Simple, focused modules for each research step.
"""

import dspy
from typing import List, Optional
from pydantic import BaseModel, Field


class QueryGeneration(dspy.Signature):
    """Generate focused research queries for a given topic"""
    topic: str = dspy.InputField(desc="Research topic or question")
    queries: List[str] = dspy.OutputField(desc="3-5 specific, searchable research queries")


class LiteratureReview(dspy.Signature):
    """Synthesize literature findings from search results"""
    topic: str = dspy.InputField(desc="Original research topic")
    queries: List[str] = dspy.InputField(desc="Search queries used")
    search_results: str = dspy.InputField(desc="Raw search results and papers found")
    literature_summary: str = dspy.OutputField(desc="Comprehensive literature review summary")


class GapAnalysis(dspy.Signature):
    """Identify research gaps from literature review"""
    topic: str = dspy.InputField(desc="Research topic")
    literature_summary: str = dspy.InputField(desc="Literature review findings")
    research_gaps: str = dspy.OutputField(desc="Identified gaps and research opportunities")


class ProposalGeneration(dspy.Signature):
    """Generate research proposal from gaps and literature"""
    topic: str = dspy.InputField(desc="Research topic")
    literature_summary: str = dspy.InputField(desc="Literature review")
    research_gaps: str = dspy.InputField(desc="Identified research gaps")
    proposal: str = dspy.OutputField(desc="Complete research proposal with methodology")


class ProposalReview(dspy.Signature):
    """Review and score a research proposal"""
    proposal: str = dspy.InputField(desc="Research proposal to review")
    feedback: str = dspy.OutputField(desc="Constructive feedback and suggestions")
    score: float = dspy.OutputField(desc="Quality score from 0.0 to 1.0")


# DSPy Module Classes
class QueryGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(QueryGeneration)
    
    def forward(self, topic: str) -> List[str]:
        result = self.generate(topic=topic)
        return result.queries


class LiteratureReviewer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.review = dspy.ChainOfThought(LiteratureReview)
    
    def forward(self, topic: str, queries: List[str], search_results: str) -> str:
        result = self.review(
            topic=topic, 
            queries=queries, 
            search_results=search_results
        )
        return result.literature_summary


class GapAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(GapAnalysis)
    
    def forward(self, topic: str, literature_summary: str) -> str:
        result = self.analyze(topic=topic, literature_summary=literature_summary)
        return result.research_gaps


class ProposalWriter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.write = dspy.ChainOfThought(ProposalGeneration)
    
    def forward(self, topic: str, literature_summary: str, research_gaps: str) -> str:
        result = self.write(
            topic=topic,
            literature_summary=literature_summary,
            research_gaps=research_gaps
        )
        return result.proposal


class ProposalReviewer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.review = dspy.ChainOfThought(ProposalReview)
    
    def forward(self, proposal: str) -> tuple[str, float]:
        result = self.review(proposal=proposal)
        return result.feedback, result.score


# Mock search function (replace with real PaperQA integration)
def mock_literature_search(queries: List[str]) -> str:
    """Mock function for literature search - replace with real PaperQA"""
    return f"""
Literature search results for queries: {', '.join(queries)}

📚 Found 12 relevant papers:

1. "Large Language Models in Educational Technology" (2024)
   - Findings: LLMs show 25% improvement in personalized learning
   - Limitations: Limited evaluation in diverse populations

2. "AI-Powered Tutoring Systems: A Systematic Review" (2023)
   - Findings: Adaptive learning pathways improve retention by 18%
   - Gaps: Need for real-time feedback mechanisms
   
3. "Ethical Considerations in Educational AI" (2024)
   - Findings: Privacy concerns in student data collection
   - Recommendations: Federated learning approaches
   
Key themes: Personalization, adaptive learning, ethical AI, real-time feedback
Emerging trends: Multimodal learning, emotional intelligence in AI tutors
Research gaps: Long-term impact studies, cross-cultural validation
"""
