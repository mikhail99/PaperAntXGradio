import dspy
from .signatures import GenerateQueries, SynthesizeKnowledge, WriteProposal, ReviewProposal

class QueryGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateQueries)

    def forward(self, topic, existing_queries):
        return self.generate(topic=topic, existing_queries=str(existing_queries))

class KnowledgeSynthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.Predict(SynthesizeKnowledge)

    def forward(self, topic, literature_summaries):
        return self.synthesize(topic=topic, literature_summaries=str(literature_summaries))

class ProposalWriter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.write = dspy.Predict(WriteProposal)

    def forward(self, knowledge_gap_summary, prior_feedback):
        return self.write(knowledge_gap_summary=knowledge_gap_summary, prior_feedback=str(prior_feedback))

class ProposalReviewer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.review = dspy.Predict(ReviewProposal)

    def forward(self, proposal_draft, review_aspect):
        return self.review(proposal_draft=proposal_draft, review_aspect=review_aspect) 