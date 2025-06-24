"""
LLM Interface for Paper2ImplementationDoc
Uses the official ollama Python library for local LLM calls (https://github.com/ollama/ollama-python)
Assumes ollama is installed and available.
"""

import logging
from typing import Dict, Any, Optional, List
import ollama

logger = logging.getLogger(__name__)

# Default model for this project
DEFAULT_MODEL = "gemma3:4b"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4000

class OllamaInterface:
    """Interface for calling ollama models locally using the official Python library."""
    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def is_available(self) -> bool:
        return True  # Always True in this version

    def list_models(self) -> List[str]:
        try:
            models = ollama.list().get('models', [])

            return [m['model'] for m in models]
        except Exception as e:
            logger.error(f"Failed to list ollama models: {str(e)}")
            return []

    def call_llm(self, prompt: str, model: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Call ollama.generate with the given prompt and parameters.
        Returns a dict with 'success', 'response', 'model', etc.
        """
        model = model or self.model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        try:
            result = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            response_text = result.get("response", "")
            return {
                "success": True,
                "response": response_text,
                "model": model,
                "prompt_length": len(prompt),
                "response_length": len(response_text)
            }
        except Exception as e:
            logger.error(f"Ollama call failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": "",
                "model": model
            }

# Global interface instance
_llm_interface = None

def get_llm_interface(model: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> OllamaInterface:
    """Get or create the Ollama interface."""
    global _llm_interface
    if _llm_interface is None:
        _llm_interface = OllamaInterface(model or DEFAULT_MODEL, temperature or DEFAULT_TEMPERATURE, max_tokens or DEFAULT_MAX_TOKENS)
        logger.info(f"Initialized ollama interface with model '{_llm_interface.model}'")
    return _llm_interface

def call_llm(prompt: str, model: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, **kwargs) -> Dict[str, Any]:
    """Convenience function to call LLM."""
    interface = get_llm_interface(model, temperature, max_tokens)
    return interface.call_llm(prompt, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)

def test_llm_interface():
    """Test the LLM interface."""
    print("🧪 Testing LLM Interface")
    # Test ollama interface
    print("\n📋 Testing Ollama Interface...")
    ollama_iface = get_llm_interface()
    print(f"✅ Ollama is assumed available. Default model: {ollama_iface.model}")
    models = ollama_iface.list_models()
    print(f"📚 Available models: {models}")
    # Test call
    result = ollama_iface.call_llm("What is machine learning?")
    if result["success"]:
        print(f"✅ LLM call successful")
        print(f"📝 Response preview: {result['response'][:100]}...")
    else:
        print(f"❌ LLM call failed: {result['error']}")

if __name__ == "__main__":
    test_llm_interface() 