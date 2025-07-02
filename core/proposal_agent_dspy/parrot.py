import dspy
import json
from .signatures import GenerateQueries, SynthesizeKnowledge, WriteProposal, ReviewProposal
from .state import KnowledgeGap, Critique

class MockLM(dspy.LM):
    """A mock dspy.LM that returns hardcoded responses for fast testing."""
    def __init__(self):
        super().__init__("mock-model")

    def __call__(self, messages, **kwargs):
        # In some versions of dspy, the signature is not passed to the LM.
        # We will inspect the prompt content to guess which response to provide.
        prompt_content = str(messages).lower()

        # Heuristics based on unique text from each signature's docstring.
        if "generates a list of search queries" in prompt_content:
            response_data = {'queries': ['parrot query 1', 'parrot query 2']}
            return [{"text": json.dumps(response_data)}]
        
        if "identifies a knowledge gap" in prompt_content:
            gap = KnowledgeGap(synthesized_summary="A parrot summary.", knowledge_gap="A parrot gap.", is_novel=True)
            response_data = {"knowledge_gap": gap.model_dump()}
            return [{"text": json.dumps(response_data)}]

        if "writes a research proposal" in prompt_content:
            response_data = {'proposal': 'A research proposal written by a parrot.'}
            return [{"text": json.dumps(response_data)}]

        if "reviews a proposal" in prompt_content:
            crit = Critique(score=0.9, justification="This is a fine parrot proposal.")
            response_data = {"critique": crit.model_dump()}
            return [{"text": json.dumps(response_data)}]
            
        # A default structured response to avoid parsing errors.
        return [{"text": json.dumps({"error": "MockLM did not recognize the signature from the prompt content."})}]
        
    def basic_request(self, prompt, **kwargs):
        # Not used by dspy.Predict with a ChatAdapter, but required by the abstract class
        pass

class MockPaperQAService:
    """A mock PaperQAService that returns a fixed response."""
    async def query_documents(self, collection_name: str, query: str) -> dict:
        print(f"🦜 (PaperQA Parrot) Received query: '{query}'")
        return { "answer_text": f"This is a parrot summary for the query: '{query}'." } 