import json
import os
from typing import Dict, List, Optional, Any, Generator
#from .llm_service import LLMService
from .query_agent import QueryAgent
import dspy
import json

import asyncio

from gradio import ChatMessage
from queue import Queue
import threading
import gradio as gr

# Configure DSPy

class CopilotProjectProposalService:
    def __init__(self) -> None:
        """Initialize CopilotService with agent modules"""
        #self.llm_service = llm_service
        self.agents : Dict[str, dspy.Module] = self._create_agents()
    
    def _create_agents(self) -> Dict[str, dspy.Module]:
        """Create agent instances"""
        return {
            "Generate Research Questions": QueryAgent(),  #(self.llm_service),
            "Generate Project Ideas": QueryAgent(),  #(self.llm_service),
            "Generate Literature Review": QueryAgent(),  #(self.llm_service),
            "Generate Project Proposal": QueryAgent(),  #(self.llm_service),
            "Generate Project Review": QueryAgent()  #(self.llm_service),
        }
    
    def get_agent_list(self) -> List[str]:
        """Returns a list of available agent names."""
        return sorted(list(self.agents.keys()))
    
    def get_agent_details(self, agent_name: str = None) -> Dict[str, str]:
        """Returns the configuration for a specific agent or all agents."""
        all_details = {
            "Generate Research Questions": {
                "short_description": "An assistant that randomly looks at abstracts and generates questions.",
                "full_description": "An assistant that randomly looks at abstracts and generates questions.",
                "tools": [{"name": "abstract_retriever", "description": "access to chroma db."}]
            },
            "Generate Project Ideas": {
                "short_description": "Runs evolution algoritm that is design to generate project ideas based on abstracts.",
                "full_description": "Runs evolution algoritm that is design to generate project ideas based on abstracts.",
                "tools": [{"name": "abstract_retriever",  "description": "access to chroma db."}, 
                          {"name": "abstract_evolution",  "description": "evolution algorithm ge generate to abstracts."}]
            },
            "Generate Literature Review": {
                "short_description": "A query research assistant that write literature review on a topic.",
                "full_description": "A query research assistant that write literature review on a topic.",
                "tools": [{"name": "paper_qa", "description": "A tool to generate literature review for the query research assistant."}]
            },
            "Generate Project Proposal": {
                "short_description": "A query research assistant that can answer questions about the stock market.",
                "full_description": "A query research assistant that can answer questions about the stock market.",
                "tools": [{"name": "query_tools", "description": "A tool to generate queries for the query research assistant."}]
            },
            "Generate Project Review": {
                "short_description": "A query research assistant that can answer questions about the stock market.",
                "full_description": "A query research assistant that can answer questions about the stock market.",
                "tools": [{"name": "query_tools", "description": "A tool to generate queries for the query research assistant."}]
            },
        }
        
        if agent_name:
            return all_details.get(agent_name)
        return all_details

    def get_quick_actions(self, agent_name: str) -> List[Dict[str, str]]:
        """Returns quick action buttons for the specified agent."""
        if not agent_name:
            return []
        
        # Dummy implementation - different actions per agent
        actions_map = {
            "Generate Research Questions": [
                {"label": "Topic Questions", "icon": "❓"},
                {"label": "Hypothesis Ideas", "icon": "💡"},
                {"label": "Research Gaps", "icon": "🔍"}
            ],
            "Generate Project Ideas": [
                {"label": "Brainstorm Ideas", "icon": "🧠"},
                {"label": "Innovation Areas", "icon": "⚡"},
                {"label": "Problem Solving", "icon": "🎯"}
            ],
            "Generate Literature Review": [
                {"label": "Find Papers", "icon": "📚"},
                {"label": "Review Structure", "icon": "📋"},
                {"label": "Citation Analysis", "icon": "🔗"}
            ],
            "Generate Project Proposal": [
                {"label": "Proposal Outline", "icon": "📝"},
                {"label": "Budget Planning", "icon": "💰"},
                {"label": "Timeline Creation", "icon": "📅"}
            ],
            "Generate Project Review": [
                {"label": "Review Criteria", "icon": "✅"},
                {"label": "Evaluation Framework", "icon": "📊"},
                {"label": "Feedback Analysis", "icon": "📈"}
            ]
        }
        
        return actions_map.get(agent_name, [])

    def reload(self) -> None:
        """Reload agent configurations - placeholder for UI compatibility."""
        print(f"Reloading {self.__class__.__name__} - agents recreated")
        self.agents = self._create_agents()
    
    def chat_with_agent(self, agent_name: str, message: str, llm_history: List[Dict[str, Any]], provider: str = "ollama") -> Generator[Dict, None, None]:
        """Route chat to the appropriate agent module"""
        
        agent : dspy.Module = self.agents.get(agent_name)
        ## Add the new user message to the conversation history.
        #messages: List[Dict[str, Any]] = llm_history + [{"role": "user", "content": message}]
    
        # Delegate to the specific agent
        past_messages = " \n ".join([h["role"] + ": " + h["content"] for h in llm_history ][-5:])
        topic = "LLM for Math" 
       
        #assistant_message = {"role": "assistant", "content": answer}
        #messages.append(assistant_message)

        agent.query_tools.flow_log = []
        answer = agent(message, past_messages, topic) # ,llm_history, provider)
        return answer, agent.query_tools.flow_log

