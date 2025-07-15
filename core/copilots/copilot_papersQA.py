import json
import os
from typing import Dict, List, Optional, Any, Generator
#from .llm_service import LLMService
import dspy
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from dspy.adapters.types.tool import Tool
import json
import yfinance as yf
import asyncio
import pandas as pd
from gradio import ChatMessage
from queue import Queue
import threading
import gradio as gr


class AbstractQueryAgent(dspy.Module):
    """ReAct agent for financial analysis using Yahoo Finance data."""

    def __init__(self):
        super().__init__()

    def forward(self, user_request: str, past_user_requests: str, question_topic: str):
        return "Hello, world!"
    
class CollectionQAAgent(dspy.Module):
    """ReAct agent for financial analysis using Yahoo Finance data."""

    def __init__(self):
        super().__init__()

    def forward(self, user_request: str, past_user_requests: str, question_topic: str):
        return "Hello, world!"
    
class PaperQAAgent(dspy.Module):
    """ReAct agent for financial analysis using Yahoo Finance data."""

    def __init__(self):
        super().__init__()

    def forward(self, user_request: str, past_user_requests: str, question_topic: str):
        return "Hello, world!"
    
class CopilotPaperQAService:
    def __init__(self) -> None:
        """Initialize CopilotService with agent modules"""
        #self.llm_service = llm_service
        self.agents : Dict[str, dspy.Module] = self._create_agents()
    
    def _create_agents(self) -> Dict[str, dspy.Module]:
        """Create agent instances"""
        return {
            "Library Abstract QA Assistant": AbstractQueryAgent(),  #(self.llm_service),
            "Collection QA Assistant": CollectionQAAgent(),  #(self.llm_service)
            "Paper QA Assistant": PaperQAAgent()  #(self.llm_service)
        }
    
    def get_agent_list(self) -> List[str]:
        """Returns a list of available agent names."""
        return sorted(list(self.agents.keys()))
    
    def get_agent_details(self, agent_name: str = None) -> Dict[str, str]:
        """Returns the configuration for a specific agent or all agents."""
        all_details = {
            "Library Abstract QA Assistant": {
                "short_description": "A QA assistant for abstracts in the library.",
                "full_description": "A QA assistant for abstracts in the library.",
                "tools": [{"name": "abstract_search", "description": "retuns abstract for a given query"}]
            },
            "Collection QA Assistant": {
                "short_description": "A QA assistant for the selected collection using full texts.",
                "full_description": " A QA assistant for the selected collection using full texts.",
                "tools": [{"name": "paperQA", "description": "A tool to answer questions about the selected collection."}]
            },
            "Paper QA Assistant": {
                "short_description": "A QA assistant for the selected paper using full text.",
                "full_description": "A QA assistant for the selected paper using full text.",
                "tools": [{"name": "paperQA", "description": "A tool to answer questions about the selected paper."}]
            }
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
            "Library Abstract QA Assistant": [
                {"label": "Search Abstracts", "icon": "🔍"},
                {"label": "Topic Analysis", "icon": "📊"},
                {"label": "Key Insights", "icon": "💡"}
            ],
            "Collection QA Assistant": [
                {"label": "Collection Overview", "icon": "📚"},
                {"label": "Document Summary", "icon": "📄"},
                {"label": "Knowledge Extract", "icon": "🧠"}
            ],
            "Paper QA Assistant": [
                {"label": "Paper Analysis", "icon": "📝"},
                {"label": "Citation Check", "icon": "🔗"},
                {"label": "Methodology Review", "icon": "🔬"},
                {"label": "Results Summary", "icon": "📈"}
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

