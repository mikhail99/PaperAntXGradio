"""
LLM calling utility adapted to call Ollama directly, bypassing DSPy.
This follows the pattern from core/proposal_agent_depr/graph.py
"""
from langchain_ollama import ChatOllama

# --- Lazy Initializer for Ollama LLM ---
_text_llm = None

def get_llm_instance():
    """Lazily initializes and returns a ChatOllama instance."""
    global _text_llm
    if _text_llm is None:
        print("--- Initializing ChatOllama for PocketFlow Demo ---")
        # Using a small, fast model with low temperature for structured output
        _text_llm = ChatOllama(
            model="gemma3:4b",
            temperature=0.1,
        )
    return _text_llm

def call_llm(message: str):
    """Call Ollama directly using langchain."""
    # The prompt is very large, so we don't print it.
    print(f"Calling Ollama with a structured prompt...")

    try:
        llm = get_llm_instance()
        response = llm.invoke(message)
        # The response object from ChatOllama has a `content` attribute
        return response.content
    except Exception as e:
        print(f"Ollama call failed: {e}")
        # Fallback response to avoid silent failures
        return f"I apologize, but I'm having trouble processing your request right now. Error: {str(e)}"

if __name__ == "__main__":
    # Simple test for the direct Ollama call
    print(call_llm("Hello, how are you? Respond in one sentence."))
