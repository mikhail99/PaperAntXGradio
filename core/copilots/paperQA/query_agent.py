import dspy
import asyncio
import json
import re
import os

from core.collections_manager import CollectionsManager, Article
from core.paperqa_service import PaperQAService
from typing import List, Tuple


# --- Orchestrator Agent ---
class QueryAgent(dspy.Module):
    """
    The main orchestrator agent.
    It uses a MemoryAgent to create a plan and a SearchAgent to execute it.
    This replaces the previous implementation.
    """
    def __init__(self, collection_name):
        super().__init__()
        #self.memory_agent = MemoryAgent(user_id=user_id)
        #self.search_agent = SearchAgent()
        self.collections_manager : CollectionsManager = CollectionsManager()
        self.paperqa_service = PaperQAService()
        self.collection_name = collection_name

        self.QueryBuilder = dspy.Predict("user_request, context -> query")
        self.AnswerBuilder = dspy.Predict("context, question -> answer")

    def forward(self, user_request: str, **kwargs):
        """
        Orchestrates the agent workflow.
        **kwargs are included for compatibility with Gradio UI calls that may pass extra arguments.
        """
        dspy.configure(lm=dspy.LM('ollama_chat/qwen3:30b', api_base='http://localhost:11434', api_key=''))

        query = self.QueryBuilder(user_request, context)
        context, ids = self.abstract_search(query, self.collection_name)
        answer = self.AnswerBuilder(context, user_request)

        return answer

        

    def abstract_search(self, query: str, collection_name: str) -> Tuple[str, str]:
        """
        Performs a fast search on document abstracts and synthesizes an answer.
        Use this for quick, general questions.
        """
        print(f"--- Running Tool: abstract_search ---")
        print(f"Query: '{query}', Collection: '{collection_name}'")

        articles : List[Article] = self.collections_manager.search_articles(
            collection_name=collection_name, 
            query=query, 
            limit=3
        )
        
        # Extract abstracts from the returned Article objects.
        abstracts = [article.abstract for article in articles if article.abstract]
        ids = [article.id for article in articles]


        context_str = "\n\n".join(abstracts)
        ids_str = ", ".join(ids)


        return context_str, ids_str
