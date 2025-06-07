"""
LLM calling utilities for Paper2ImplementationDoc
Based on PocketFlow architecture
"""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class LLMClient:
    """Generic LLM client for Paper2ImplementationDoc"""
    
    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key or self._get_api_key()
        self._client = None
        
        if not self.api_key:
            logger.warning(f"⚠️  No API key found for {self.provider}. LLM features will be disabled.")
        else:
            self._initialize_client()
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment variables"""
        if self.provider == "gemini":
            return os.getenv("GEMINI_API_KEY")
        elif self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        else:
            return None
    
    def _initialize_client(self):
        """Initialize the LLM client"""
        try:
            if self.provider == "gemini":
                # Placeholder for Gemini client initialization
                logger.info("🤖 Gemini client would be initialized here")
                # import google.generativeai as genai
                # self._client = genai.Client(api_key=self.api_key)
                
            elif self.provider == "openai":
                # Placeholder for OpenAI client initialization
                logger.info("🤖 OpenAI client would be initialized here")
                # import openai
                # self._client = openai.OpenAI(api_key=self.api_key)
                
            elif self.provider == "anthropic":
                # Placeholder for Anthropic client initialization
                logger.info("🤖 Anthropic client would be initialized here")
                # import anthropic
                # self._client = anthropic.Anthropic(api_key=self.api_key)
                
            logger.info(f"✅ LLM client initialized: {self.provider}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.provider} client: {str(e)}")
            self._client = None
    
    def call_llm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Call the LLM with a prompt and return the response
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Dict with response and metadata
        """
        if not self._client:
            logger.warning("⚠️  LLM client not available. Returning placeholder response.")
            return {
                "response": f"[PLACEHOLDER] LLM response for prompt: {prompt[:100]}...",
                "success": False,
                "error": "LLM client not initialized",
                "provider": self.provider
            }
        
        try:
            # Placeholder LLM call
            logger.info(f"🤖 Would call {self.provider} LLM with prompt length: {len(prompt)}")
            
            # TODO: Implement actual LLM calls
            response_text = f"[PLACEHOLDER] This would be the {self.provider} response to: {prompt[:50]}..."
            
            return {
                "response": response_text,
                "success": True,
                "provider": self.provider,
                "prompt_length": len(prompt),
                "response_length": len(response_text)
            }
            
        except Exception as e:
            logger.error(f"❌ LLM call failed: {str(e)}")
            return {
                "response": "",
                "success": False,
                "error": str(e),
                "provider": self.provider
            }
    
    def test_connection(self) -> bool:
        """Test if the LLM connection is working"""
        if not self._client:
            return False
        
        try:
            # Simple test call
            result = self.call_llm("Test prompt")
            return result.get("success", False)
        except Exception:
            return False


# Default client instance
_default_client = None

def get_llm_client(provider: str = "gemini") -> LLMClient:
    """Get or create the default LLM client"""
    global _default_client
    
    if _default_client is None or _default_client.provider != provider:
        _default_client = LLMClient(provider)
    
    return _default_client

def call_llm(prompt: str, provider: str = "gemini", **kwargs) -> Dict[str, Any]:
    """Convenience function to call LLM"""
    client = get_llm_client(provider)
    return client.call_llm(prompt, **kwargs)

def main():
    """Test the LLM client"""
    print("🧪 Testing Paper2ImplementationDoc LLM Client")
    
    # Test different providers
    for provider in ["gemini", "openai", "anthropic"]:
        print(f"\n📋 Testing {provider}...")
        client = LLMClient(provider)
        
        if client.test_connection():
            print(f"✅ {provider} connection successful")
        else:
            print(f"❌ {provider} connection failed")
        
        # Test call
        result = client.call_llm("Hello, this is a test prompt.")
        print(f"📝 Response: {result['response'][:100]}...")

if __name__ == "__main__":
    main() 