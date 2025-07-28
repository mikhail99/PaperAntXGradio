import json
import os
from typing import Dict, List, Optional, Any, Generator
#from .llm_service import LLMService

import dspy
import json

import asyncio

from gradio import ChatMessage
from queue import Queue
import threading
import gradio as gr
from core.collections_manager import CollectionsManager, Article
from core.copilots.project_proposal.idea_generator.evolution import simplified_evolutionary_abstracts, Abstract, Candidate
# Configure DSPy

class QueryAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.query_tools = []

    def forward(self, user_request: str, past_messages: str, topic: str, **kwargs):
        print("STUB")
        return "STUB"

class IdeaGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.query_tools = []
    
    def forward(self, user_request: str, past_messages: str, topic: str, **kwargs):
        print(f"IdeaGenerator: {user_request}, {past_messages}, {topic}")

        winners = self._evolution_algorithm(topic)
        winners_str = "\n".join([f"Idea {i+1}: {winner.idea}" for i, winner in enumerate(winners)])
        return winners_str

    def _evolution_algorithm(self, topic: str):
        '''
        This function is used to generate project ideas based on abstracts.
        It uses the evolutionary algorithm to generate project ideas.
        It returns a list of project ideas.
        It uses the collections manager to get the abstracts.
        It uses the evolutionary algorithm to generate project ideas.
        It returns a list of project ideas.
        PROGRESS: 80%
        '''
        

        manager = CollectionsManager()
        collection_name = "HuggingFaceDailyPapers" #TODO: make it dynamic
        collection = manager.get_collection_by_name(collection_name)
        print(list(collection.articles.values())[0])

        selected_articles : List[Article] = manager.search_articles(collection_name, topic, limit = 100)
        all_abstracts: List[Abstract] = [Abstract(id=article.id, text=article.abstract) for article in selected_articles]
        print(f"Number of abstracts in collection: {len(all_abstracts)}")

        winners: List[Candidate] = simplified_evolutionary_abstracts(all_abstracts, context) #TODO path context
        print(f"Number of winners: {len(winners)}")
        return winners


class CopilotProjectProposalService:
    def __init__(self) -> None:
        """Initialize CopilotService with agent modules"""
        #self.llm_service = llm_service
        self.agents : Dict[str, dspy.Module] = self._create_agents()
    
    def _create_agents(self) -> Dict[str, dspy.Module]:
        """Create agent instances"""
        return {
            "Generate Project Ideas": IdeaGenerator(),  #(self.llm_service),
            "Generate Project Proposal": QueryAgent(),  #(self.llm_service),
            "Generate Project Review": QueryAgent()  #(self.llm_service),
        }
    
    def get_agent_list(self) -> List[str]:
        """Returns a list of available agent names."""
        return sorted(list(self.agents.keys()))
    
    def get_agent_details(self, agent_name: str = None) -> Dict[str, str]:
        """Returns the configuration for a specific agent or all agents."""
        all_details = {

            "Generate Project Ideas": {
                "short_description": "Runs evolution algoritm that is design to generate project ideas based on abstracts.",
                "full_description": "Runs evolution algoritm that is design to generate project ideas based on abstracts.",
                "tools": [{"name": "abstract_retriever",  "description": "access to chroma db."}, 
                          {"name": "abstract_evolution",  "description": "evolution algorithm ge generate to abstracts."}]
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
            "Generate Project Ideas": [
                {"label": "Brainstorm Ideas", "icon": "🧠"},
                {"label": "Innovation Areas", "icon": "⚡"},
                {"label": "Problem Solving", "icon": "🎯"}
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
    
    def chat_with_agent(self, agent_name: str, message: str, llm_history: List[Dict[str, Any]]) -> Generator[Dict, None, None]:
        """Route chat to the appropriate agent module"""
        
        agent : dspy.Module = self.agents.get(agent_name)
        ## Add the new user message to the conversation history.
        #messages: List[Dict[str, Any]] = llm_history + [{"role": "user", "content": message}]
    
        # Delegate to the specific agent
        past_messages = " \n ".join([h["role"] + ": " + h["content"] for h in llm_history ][-5:])
        topic = "diffusion for video generation" #TODO: make it dynamic
       
        #assistant_message = {"role": "assistant", "content": answer}
        #messages.append(assistant_message)

        #agent.query_tools.flow_log = []
        agent.flow_log = []
        answer = agent(message, past_messages, topic) # ,llm_history, provider)
        return answer, agent.flow_log

