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
    
    def get_agent_details(self) -> Dict[str, str]:
        """Returns the configuration for a specific agent."""
        return {
            "Library Abstract QA Assistant": {
                "short_description": "A QA assistant for abstracts in the library.",
                "full_description": "A QA assistant for abstracts in the library.",
                "tools": [{"name": "abstract_search", "description": "retuns abstract for a given query"}]
            },
            "Collection QA Assistantt": {
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

