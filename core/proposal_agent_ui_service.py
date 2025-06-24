import json
import re
from pathlib import Path
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, create_model

# --- Configuration & Initialization ---

# Load configs once
CONFIG_PATH = Path(__file__).parent / "proposal_agent"
with open(CONFIG_PATH / "agent_config.json", "r") as f:
    agent_config = json.load(f)
with open(CONFIG_PATH / "prompts.json", "r") as f:
    prompts = json.load(f)

# Initialize models
json_llm = ChatOllama(model="gemma3:4b", format="json", temperature=0)
text_llm = ChatOllama(model="gemma3:4b", temperature=0.7)


class ProposalAgentUIService:
    """
    A service to support a UI for debugging individual proposal agent nodes.
    This is distinct from the main ProposalAgentService that runs the whole graph.
    """

    def get_teams_config(self) -> Dict[str, Any]:
        """Returns the 'teams' part of the agent configuration."""
        return agent_config.get("teams", {})

    def get_agent_details(self, agent_name: str) -> Dict[str, Any]:
        """
        Returns the full details for a single agent node, including the prompt text.
        """
        details = agent_config.get("nodes", {}).get(agent_name)
        if not details:
            return None
        
        # Inject the actual prompt text into the details
        details["prompt_text"] = prompts.get(details["prompt_key"], "Prompt not found.")
        return details

    def get_prompt_variables(self, agent_name: str) -> List[str]:
        """
        Parses a prompt template to find all required input variables.
        """
        agent_details = self.get_agent_details(agent_name)
        if not agent_details or not agent_details.get("prompt_text"):
            return []
        
        prompt_text = agent_details["prompt_text"]
        # Find all occurrences of {variable_name}
        variables = re.findall(r"\{(\w+)\}", prompt_text)
        return list(dict.fromkeys(variables)) # Return unique variables

    def chat_with_agent(self, agent_name: str, context: Dict[str, Any], user_message: str) -> str:
        """
        Runs a single agent/node with the given context and message.
        """
        agent_details = self.get_agent_details(agent_name)
        if not agent_details:
            return "Error: Agent not found."

        prompt_template = agent_details["prompt_text"]
        
        # Append the user's message to the context under a generic key
        # so it can be part of the prompt if desired.
        full_context = {**context, "user_message": user_message}

        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # For now, we assume all debuggable nodes are simple text output.
        # A more complex implementation could handle structured outputs.
        chain = prompt | text_llm
        
        response = chain.invoke(full_context)
        
        return response.content 