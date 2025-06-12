# Placeholder for AI copilot/LLM integration logic
import json
import os
from typing import Dict, List, Optional, Any, Tuple

from .data_models import Article, Collection
from .collections_manager import CollectionsManager
from .article_manager import ArticleManager
from .llm_service import LLMService

class CopilotService:
    def __init__(self, collections_manager: CollectionsManager, article_manager: ArticleManager, llm_service: LLMService) -> None:
        """Initialize CopilotService with references to the managers"""
        self.collections_manager = collections_manager
        self.article_manager = article_manager
        self.llm_service = llm_service
        self.agents: Dict[str, Any] = self._load_agents()

    def reload(self) -> List[str]:
        """Reloads agent configs and LLM service settings."""
        self.llm_service.reload()
        self.agents = self._load_agents()
        print("Reloaded agents and LLM configurations.")
        return self.get_agent_list()

    def _load_agents(self) -> Dict[str, Any]:
        """Load agent configurations from the agents directory."""
        agents = {}
        agents_dir = os.path.join(os.path.dirname(__file__), 'copilot_agents')
        if not os.path.exists(agents_dir):
            print(f"Warning: Agents directory not found at {agents_dir}")
            return {}
        for filename in os.listdir(agents_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(agents_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    try:
                        agent_config = json.load(f)
                        if 'name' in agent_config:
                            agents[agent_config['name']] = agent_config
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from {filename}")
        return agents

    def get_agent_list(self) -> List[str]:
        """Returns a list of available agent names."""
        return sorted(list(self.agents.keys()))

    def get_agent_details(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Returns the configuration for a specific agent."""
        return self.agents.get(agent_name)

    def chat_with_agent(self, agent_name: str, message: str, chat_history: List[List[str]]):
        """
        Handles a chat interaction with a specific agent using the LLMService.
        This is now a generator function to support streaming.
        """
        agent_config = self.get_agent_details(agent_name)
        if not agent_config:
            yield "Error: Agent not found."
            return

        system_prompt = agent_config.get('model_prompt', None)
        
        # Convert Gradio history to LLM-compatible format
        messages = []
        for user_msg, assistant_msg in chat_history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": message})

        # Get default provider from env, but can be overridden by agent config later
        provider = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
        
        response_generator = self.llm_service.call_llm(
            messages=messages,
            system_prompt=system_prompt,
            provider=provider
        )

        yield from response_generator
