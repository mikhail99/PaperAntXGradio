import dspy
from .signatures import GenerateQueries, SynthesizeKnowledge, WriteProposal, ReviewProposal
from .state import KnowledgeGap, Critique

class MockLM(dspy.LM):
    """A mock dspy.LM that returns hardcoded responses for fast testing."""
    def __init__(self):
        super().__init__("mock-model")

    def __call__(self, prompt, **kwargs):
        # The 'dspy.Predict' object gives us the signature via kwargs
        signature = kwargs.get("signature")
        
        if signature == GenerateQueries:
            return [{'queries': ['parrot query 1', 'parrot query 2']}]
        if signature == SynthesizeKnowledge:
            # dspy.TypedPredictor expects a pydantic object
            gap = KnowledgeGap(synthesized_summary="A parrot summary.", knowledge_gap="A parrot gap.", is_novel=True)
            return [gap]
        if signature == WriteProposal:
            return [{'proposal': 'A research proposal written by a parrot.'}]
        if signature == ReviewProposal:
            crit = Critique(score=0.9, justification="This is a fine parrot proposal.")
            return [crit]
            
        return ["Default parrot response."]
        
    def basic_request(self, prompt, **kwargs):
        # Not used by dspy.Predict, but required by the abstract class
        pass

class MockPaperQAService:
    """A mock PaperQAService that returns a fixed response."""
    async def query_documents(self, collection_name: str, query: str) -> dict:
        print(f"🦜 (PaperQA Parrot) Received query: '{query}'")
        return { "answer_text": f"This is a parrot summary for the query: '{query}'." } 