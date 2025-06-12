import os
import dotenv
import google.generativeai as genai
import openai
import anthropic
import json
from typing import List, Dict, Optional, Generator

dotenv.load_dotenv()

class LLMService:
    def __init__(self):
        self.reload()

    def reload(self):
        """Reloads API keys and default provider from .env file."""
        print("Reloading LLM service configuration...")
        dotenv.load_dotenv(override=True)
        
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "gemini").lower()

        self.openai_client = None
        self.anthropic_client = None

        if self.gemini_api_key and self.gemini_api_key != 'your_gemini_api_key_here':
            try:
                genai.configure(api_key=self.gemini_api_key)
            except Exception as e:
                print(f"Warning: Failed to configure Gemini: {e}")

        if self.openai_api_key and self.openai_api_key != 'your_openai_api_key_here':
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)

        if self.anthropic_api_key and self.anthropic_api_key != 'your_anthropic_api_key_here':
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)

    def call_llm(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None, provider: Optional[str] = None) -> Generator[str, None, None]:
        provider = (provider or self.default_provider).lower()

        try:
            if provider == "gemini":
                yield from self._call_gemini(messages, system_prompt)
            elif provider == "openai":
                yield from self._call_openai(messages, system_prompt)
            elif provider == "anthropic":
                yield from self._call_anthropic(messages, system_prompt)
            else:
                yield f"Error: Unsupported LLM provider '{provider}'. Supported providers are 'gemini', 'openai', 'anthropic'."
        except Exception as e:
            yield f"Error calling LLM provider '{provider}': {e}"

    def _call_gemini(self, messages: List[Dict[str, str]], system_prompt: Optional[str]) -> Generator[str, None, None]:
        if not self.gemini_api_key or self.gemini_api_key == 'your_gemini_api_key_here':
            yield "Error: GEMINI_API_KEY not found or is a placeholder. Please set it in your .env file."
            return
        
        model_name = 'gemini-1.5-flash'
        model = genai.GenerativeModel(model_name=model_name, system_instruction=system_prompt)
        
        gemini_messages = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_messages.append({"role": role, "parts": [msg["content"]]})

        print(f"--- Calling Gemini (model: {model_name}) ---")
        print(f"System Prompt: {system_prompt}")
        print(f"Messages: {json.dumps(gemini_messages, indent=2)}")
        print("------------------------------------------")

        response_stream = model.generate_content(gemini_messages, stream=True)
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    def _call_openai(self, messages: List[Dict[str, str]], system_prompt: Optional[str]) -> Generator[str, None, None]:
        if not self.openai_client:
            yield "Error: OPENAI_API_KEY not found, is a placeholder, or client not initialized. Please set it in your .env file."
            return
            
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        
        model_name = "gpt-4o"
        print(f"--- Calling OpenAI (model: {model_name}) ---")
        print(f"Messages: {json.dumps(full_messages, indent=2)}")
        print("-----------------------------------------")

        stream = self.openai_client.chat.completions.create(
            model=model_name,
            messages=full_messages,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def _call_anthropic(self, messages: List[Dict[str, str]], system_prompt: Optional[str]) -> Generator[str, None, None]:
        if not self.anthropic_client:
            yield "Error: ANTHROPIC_API_KEY not found, is a placeholder, or client not initialized. Please set it in your .env file."
            return
        
        model_name = "claude-3-haiku-20240307"
        print(f"--- Calling Anthropic (model: {model_name}) ---")
        print(f"System Prompt: {system_prompt}")
        print(f"Messages: {json.dumps(messages, indent=2)}")
        print("---------------------------------------------")

        with self.anthropic_client.messages.stream(
            model=model_name,
            system=system_prompt,
            messages=messages,
            max_tokens=2048,
        ) as stream:
            for text in stream.text_stream:
                yield text
