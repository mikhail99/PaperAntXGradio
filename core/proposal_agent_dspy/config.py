"""
DSPy configuration for research proposal agent.
Supports both local Ollama and remote LLM providers.
"""

import dspy
import os
from typing import Optional


def configure_dspy_for_research(
    model_name: str = "gemma3:12b",
    provider: str = "ollama",
    ollama_base_url: str = "http://localhost:11434",
    openai_api_key: Optional[str] = None
):

    # Configure for local Ollama
    ollama_lm = dspy.LM(
        f'ollama_chat/{model_name}',
        api_base=ollama_base_url,
        api_key=''  # Ollama doesn't require API key
    )
    
    dspy.settings(lm=ollama_lm)
    print(f"✅ DSPy configured for Ollama model '{model_name}' at {ollama_base_url}")

