import json
import os
from typing import Dict, List, Optional, Any, Generator
#from .llm_service import LLMService
import dspy
from dspy.adapters.types.tool import Tool
import json
import asyncio
import pandas as pd
from gradio import ChatMessage
from queue import Queue
import threading
import gradio as gr
from core.collections_manager import CollectionsManager

class AbstractQueryAgent(dspy.Module):
    """ReAct agent for financial analysis using Yahoo Finance data."""

    def __init__(self):
        super().__init__()
        self.flow_log = []
        self.manager = CollectionsManager()
        self.collection_name = "HuggingFaceDailyPapers" #TODO: make it dynamic
        
        self.QueryBuilder = dspy.Predict("user_request, past_user_requests, context -> query")
        self.AnswerBuilder = dspy.Predict("user_request, past_user_requests, context, articles -> answer")



    def forward(self, user_request: str, past_user_requests: str, question_topic: str):
        '''
        This function is used to build the query and answer for the user request.
        It uses the QueryBuilder to build the query and the AnswerBuilder to build the answer.
        It uses the CollectionsManager to search the articles.
        It returns the answer and the flow log.
        PROGRESS: 60%
        '''
        query_text = self.QueryBuilder(user_request, past_user_requests, question_topic)
        print(f"Query: {query_text}")
        #TODO: implement query
        #TODO: implement query
        articles = self.manager.search_articles(self.collection_name, query_text, limit = 10)

        answer = self.AnswerBuilder(user_request, past_user_requests, question_topic, articles)
        
        return answer.answer
    

    
class PaperQAAgent(dspy.Module):
    """"""

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
            "Abstract QA Assistant": AbstractQueryAgent(),  #(self.llm_service),
            "Literature Review Assistant": PaperQAAgent()  #(self.llm_service)
        }
    
    def get_agent_list(self) -> List[str]:
        """Returns a list of available agent names."""
        return sorted(list(self.agents.keys()))
    
    def get_agent_details(self, agent_name: str = None) -> Dict[str, str]:
        """Returns the configuration for a specific agent or all agents."""
        all_details = {
            "Abstract QA Assistant": {
                "short_description": "A QA assistant for abstracts in the library.",
                "full_description": "A QA assistant for abstracts in the library.",
                "tools": [{"name": "abstract_search", "description": "retuns abstract for a given query"}]
            },
            "Literature Review Assistant": {
                "short_description": "A QA assistant for the selected collection using full texts.",
                "full_description": " A QA assistant for the selected collection using full texts.",
                "tools": [{"name": "paperQA", "description": "A tool to answer questions about the selected collection."}]
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
            "Abstract QA Assistant": [
                {"label": "Search Abstracts", "icon": "🔍"},
                {"label": "Topic Analysis", "icon": "📊"},
                {"label": "Key Insights", "icon": "💡"}
            ],
            "Literature Review Assistant": [
                {"label": "Paper Analysis", "icon": "📝"},
                {"label": "Citation Check", "icon": "🔗"},
                {"label": "Methodology Review", "icon": "📚"},
                {"label": "Results Summary", "icon": "📈"}
            ]
        }
        
        return actions_map.get(agent_name, [])

    def reload(self) -> None:
        """Reload agent configurations - placeholder for UI compatibility."""
        print(f"Reloading {self.__class__.__name__} - agents recreated")
        self.agents = self._create_agents()
    
    def chat_with_agent(self, agent_name: str, message: str, llm_history: List[Dict[str, Any]], provider: str = "ollama") -> Generator[Dict, None, None]:
        '''
        This function is used to chat with the agent.
        It uses the agent to build the query and answer for the user request.
        It uses the CollectionsManager to search the articles.
        It returns the answer and the flow log.
        PROGRESS: 60%
        '''
        """Route chat to the appropriate agent module"""
        

        agent : dspy.Module = self.agents.get(agent_name)
        ## Add the new user message to the conversation history.
        #messages: List[Dict[str, Any]] = llm_history + [{"role": "user", "content": message}]
    
        # Delegate to the specific agent
        past_messages = " \n ".join([h["role"] + ": " + h["content"] for h in llm_history ][-5:])
        topic = "Diffusion Models" 
       
        #assistant_message = {"role": "assistant", "content": answer}
        #messages.append(assistant_message)

        #agent.query_tools.flow_log = []
        agent.flow_log = [] 
        answer = agent(message, past_messages, topic) # ,llm_history, provider)
        return answer, agent.flow_log

