# core/proposal_agent/parrot_services.py

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from .state import QueryList, KnowledgeGap, Critique, FinalReview

# --- "Parrot" Mock Services ---

class ParrotPaperQAService:
    """A pure Python, mock-free implementation of the PaperQAService."""
    async def query_documents(self, collection_name: str, question: str) -> dict:
        print(f"🦜 (PaperQA Parrot) Received query: '{question}'")
        return {
            "answer_text": f"Parrot summary for query: '{question}'",
            "context": f"Parrot context for query: '{question}'"
        }

class ParrotStructuredRunnable(Runnable):
    """A pure Python object that mimics a structured LLM call."""
    def __init__(self, schema):
        self.schema = schema
        print(f"🦜 ParrotStructuredRunnable created for schema: {self.schema.__name__}")

    def invoke(self, input_dict: dict, config=None) -> object:
        """Generates a fake Pydantic object based on the schema."""
        print(f"🦜 (LLM Parrot) Generating parrot response for schema: {self.schema.__name__}")

        if self.schema == QueryList:
            # The parrot doesn't need to be smart, just needs to return the right shape.
            return QueryList(queries=["parrot query 1", "parrot query 2"])
        if self.schema == KnowledgeGap:
            return KnowledgeGap(
                knowledge_gap="A parrot knowledge gap.",
                synthesized_summary="A parrot synthesized summary."
            )
        if self.schema == Critique:
            return Critique(is_sound=True, critique="A parrot critique.")
        if self.schema == FinalReview:
            return FinalReview(is_approved=True, final_summary="The parrot proposal is approved.")
        # This fallback should ideally not be hit if the graph is correct.
        raise TypeError(f"Parrot does not know how to handle schema: {self.schema}")

class ParrotChatOllama(Runnable):
    """A pure Python, mock-free fake of ChatOllama."""
    def __init__(self, format: str = "text", **kwargs):
        self.format = format
        print(f"🦜 ParrotChatOllama created with format: '{self.format}'")

    def with_structured_output(self, schema) -> ParrotStructuredRunnable:
        """Returns a separate runnable object that handles structured responses."""
        if self.format != "json":
            raise ValueError("with_structured_output can only be called on a JSON-formatted LLM.")
        return ParrotStructuredRunnable(schema)

    def invoke(self, input_dict: dict, config=None) -> AIMessage:
        """Handles simple text generation."""
        if self.format == "json":
            # The JSON model should not be invoked directly without a schema.
            raise ValueError("JSON-formatted LLM was invoked without a schema.")
        return AIMessage(content="Parrot proposal draft.")


def get_parrot_services():
    """Returns a tuple of configured, pure-Python parrot services."""
    print("--- Using PURE PYTHON PARROT services for application ---")
    
    # These are now our own fake classes, with no MagicMock involved.
    parrot_json_llm = ParrotChatOllama(format="json")
    parrot_text_llm = ParrotChatOllama(format="text")
    parrot_paperqa = ParrotPaperQAService()
    
    return parrot_json_llm, parrot_text_llm, parrot_paperqa 