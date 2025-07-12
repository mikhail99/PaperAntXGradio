import json
import os
from typing import Dict, List, Optional, Any, Generator
from .llm_service import LLMService
import dspy
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from dspy.adapters.types.tool import Tool
import json
import yfinance as yf
import asyncio

# Configure DSPy

# Convert LangChain Yahoo Finance tool to DSPy
yahoo_finance_tool = YahooFinanceNewsTool()
finance_news_tool = Tool.from_langchain(yahoo_finance_tool)

print(finance_news_tool.args)
print(yahoo_finance_tool.invoke("NVDA"))

class FinancialAnalysisAgent(dspy.Module):
    """ReAct agent for financial analysis using Yahoo Finance data."""

    def __init__(self):
        super().__init__()

        # Combine all tools
        self.tools = [
            finance_news_tool,  # LangChain Yahoo Finance News
        ]

        # Initialize ReAct
        self.react = dspy.ReAct(
            signature="financial_query -> analysis_response",
            tools=self.tools,
            max_iters=6
        )

    def forward(self, financial_query: str):
        # HACK
        dspy.configure(lm=dspy.LM('ollama_chat/qwen3:4b', api_base='http://localhost:11434', api_key=''))
        print(f"Financial query: {financial_query}")
        try:
            answer = self.react(financial_query=financial_query)
            print(f"Answer: {answer}")
            return answer
        except Exception as e:
            print(f"Error: {e}")
            return f"Error: {e}"
        
class BusinessStrategyAgent(dspy.Module):
    """ReAct agent for financial analysis using Yahoo Finance data."""

    def __init__(self):
        super().__init__()

        # Combine all tools
        self.tools = [
            finance_news_tool,  # LangChain Yahoo Finance News
        ]

        # Initialize ReAct
        self.react = dspy.ReAct(
            signature="financial_query -> analysis_response",
            tools=self.tools,
            max_iters=6
        )

    def forward(self, financial_query: str):
        output = asyncio.run(self.react.acall(financial_query=financial_query))
        return output


class QueyTools:
    """Tools for interacting with the Mem0 memory system."""

    def __init__(self):
        self.memory : Dict[str, bool|None] = {}

    def add_query(self, query : str) -> str:
        """Generate a query for a given topic."""
        self.memory[query]= None
        return f"Query added: {query}"
    
    def add_query_review(self, query : str, approved : bool) -> str:
        """Review a query and approve or reject it."""
        self.memory[query] = approved
        return f"Query reviewed: {query} {approved}"

    def search_approve_queries(self) -> str:
        """Search for relevant memories."""
        return ", ".join([query for query, approved in self.memory.items() if approved is not None and approved])
    
    def search_rejected_queries(self) -> str:
        """Search for relevant memories."""
        return ", ".join([query for query, approved in self.memory.items() if approved is not None and not approved])
    
    def search_pending_queries(self) -> str:
        """Search for relevant memories."""
        return ", ".join([query for query, approved in self.memory.items() if approved is None])
    

class QueryHelper(dspy.Signature):
    """
    You're a helpful assistant to help with recording and reviewing queries and have access to query memory method.
    Whenever you answer a user's input you should either add the query to the memory or add a review to query or provide information about the memory.
    """
    user_input: str = dspy.InputField()
    response: str = dspy.OutputField()

class QueryAgent(dspy.Module):
    """ReAct agent for financial analysis using Yahoo Finance data."""

    def __init__(self):
        super().__init__()
        self.query_tools = QueyTools()
        # Combine all tools
        self.tools = [
            self.query_tools.add_query,
            self.query_tools.add_query_review,
            self.query_tools.search_approve_queries,
            self.query_tools.search_rejected_queries,
            self.query_tools.search_pending_queries
        ]

        # Initialize ReAct
        dspy.configure(lm=dspy.LM('ollama_chat/qwen3:4b', api_base='http://localhost:11434', api_key=''))
        self.react = dspy.ReAct(
            signature=QueryHelper,
            tools=self.tools,
            max_iters=1
        )

    def forward(self, user_input: str):
        result = self.react(user_input=user_input)
        return result.response

class CopilotBusinessService:
    def __init__(self, llm_service: LLMService) -> None:
        """Initialize CopilotService with agent modules"""
        self.llm_service = llm_service
        self.agents : Dict[str, dspy.Module] = self._create_agents()
    
    def _create_agents(self) -> Dict[str, dspy.Module]:
        """Create agent instances"""
        return {
            "Finance News Assistant": FinancialAnalysisAgent(),  #(self.llm_service),
            "Query Research Assistant": QueryAgent()  #(self.llm_service)
        }
    
    def get_agent_list(self) -> List[str]:
        """Returns a list of available agent names."""
        return sorted(list(self.agents.keys()))
    
    def get_agent_details(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Returns the configuration for a specific agent."""
        agent : dspy.Module = self.agents.get(agent_name)
        if not agent:
            return None
        
        return {
            "name": agent_name,
            "description": "STUB",
            "model_prompt": "STUB"
        }
    
    def chat_with_agent(self, agent_name: str, message: str, llm_history: List[Dict[str, Any]], provider: str = "ollama") -> Generator[Dict, None, None]:
        """Route chat to the appropriate agent module"""
        agent : dspy.Module = self.agents.get(agent_name)
        if not agent:
            yield {"type": "error", "content": "Agent not found."}
            return
        # Add the new user message to the conversation history.
        messages: List[Dict[str, Any]] = llm_history + [{"role": "user", "content": message}]
    
        # Delegate to the specific agent
        answer = agent(message) # ,llm_history, provider)
        assistant_message = {"role": "assistant", "content": answer}
        messages.append(assistant_message)
        return answer # ,llm_history, provider)
