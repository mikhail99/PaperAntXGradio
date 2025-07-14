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
import pandas as pd
from gradio import ChatMessage
from queue import Queue
import threading
import gradio as gr

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

    def __init__(self, query_generator : dspy.Module, context : str):
        #self.memory : Dict[str, bool|None] = {}
        self.table = pd.DataFrame(columns=["query", "approved"])
        #self.query_index : int = 0
        self.query_generator = query_generator
        self.context = context

        self.flow_log = []


    
    def generate_query(self, user_request : str ) -> str:
        """Generate a query from a user query request."""

        self.flow_log.append(ChatMessage(
            role="assistant",
            content=f"Generating query for: {user_request}",
            metadata={"title": "🛠️ Tool Use: Query Generator"}
        ))
        res = self.query_generator(user_request=user_request, context=self.context)
        result = self._add_query(res.query, pd.NA)
        return result
    
    def add_user_query(self, query : str) -> str:
        """Add a user query to the memory."""
        print(f"add_user_query: {query}")
        self._add_query(query,True)
 
        return f"Query added: {query}, index: {len(self.table)-1}"


    def _add_query(self, query : str, approved : bool) -> str:
        """Generate a query for a given topic."""
        print(f"add_query: {query}")
        #if query in self.memory:
        #    return f"Query already exists: {query}"
        self.table.loc[len(self.table)] = [query, approved]
        #self.memory[query]= None
        return f"Query added: {query}, index: {len(self.table)-1} needs to be reviewed"

  
    def add_query_review(self, query_index : int, approved : bool) -> str:
        """Review a query and approve or reject it."""
        print(f"add_query_review: {query_index}, {approved}")
        self.table.loc[query_index, "approved"] = approved
        return f"Query reviewed: {query_index} {approved}"

    def list_all_queries(self) -> gr.ChatMessage:
        """Returns all queries."""
        self.flow_log.append(ChatMessage(
            role="assistant",
            content="Listing all queries",
            metadata={"title": "🛠️ Tool Use: List All Queries"}
        ))
        print(f"search_approve_queries")
        html_table = self.table.to_html(index=False, escape=True)
        return ChatMessage(
            role="assistant",
            content=html_table,
        )
    
    def list_approved_queries(self) -> gr.HTML:
        """Returns approved queries."""
        print(f"search_approve_queries")
        approved_df = self.table[self.table["approved"] == True][["query"]]
        html_table = approved_df.to_html(index=False, escape=True)
        return gr.HTML(html_table)
    
    def list_rejected_queries(self) -> str:
        """Returns rejected queries."""
        print(f"search_rejected_queries")
        return self.table[self.table["approved"] == False]["query"].to_string()
    
    def list_pending_queries(self) -> str:
        """Returns pending queries."""
        print(f"search_pending_queries")
        return self.table[self.table["approved"].isna()]["query"].to_string()
    
    def save(self) -> str:
        """Save the table to a file."""

        self.flow_log.append(ChatMessage(
            role="assistant",
            content=f"Saving table to queries.csv",
            metadata={"title": "💾 Save Queries"}
        ))
        print(f"save: {self.table}")
        self.table.to_csv("queries.csv", index=False)
        return "Table saved to queries.csv"
    

class QueryHelper(dspy.Signature):
    """
    You're a helpful assistant to help with recording and reviewing queries.
    Your task unclude:
    - Generate a search query from user request.
    - Add a query to the memory.
    - Add a user review to a query.
    - Search for relevant queries. 
    """
    user_request: str = dspy.InputField()
    past_user_requests: str = dspy.InputField()
    question_topic: str = dspy.InputField()
    response: str = dspy.OutputField()

class QueryAgent(dspy.Module):
    """ReAct agent for financial analysis using Yahoo Finance data."""

    def __init__(self):
        super().__init__()

        self.query_generator = dspy.Predict("user_request: str , context: str -> query: str") 
        self.query_tools = QueyTools(self.query_generator, context="LLM agents")  
        self.table = pd.DataFrame(columns=["query", "approved"])    

        self.tools = [
            self.query_tools.generate_query,
            self.query_tools.add_user_query,
            self.query_tools.add_query_review,
            self.query_tools.list_approved_queries,
            self.query_tools.list_rejected_queries,
            self.query_tools.list_pending_queries,
            self.query_tools.list_all_queries,
            self.query_tools.save
        ]

        # Initialize ReAct
        dspy.configure(lm=dspy.LM('ollama_chat/qwen3:4b', api_base='http://localhost:11434', api_key=''))
        self.react = dspy.ReAct(
            signature=QueryHelper,
            tools=self.tools,
            max_iters=1
        )

    def forward(self, user_request: str, past_user_requests: str, question_topic: str):
        result = self.react(user_request=user_request, past_user_requests=past_user_requests,question_topic=question_topic)
        return result.response 

class CopilotProjectProposalService:
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
    
    def get_agent_details(self) -> Dict[str, str]:
        """Returns the configuration for a specific agent."""
        return {
            "Finance News Assistant": {
                "short_description": "A finance news assistant that can answer questions about the stock market.",
                "full_description": "A finance news assistant that can answer questions about the stock market.",
                "tools": [{"name": "yahoo_finance_news", "description": "A tool to get the latest news about a stock."}]
            },
            "Query Research Assistant": {
                "short_description": "A query research assistant that can answer questions about the stock market.",
                "full_description": "A query research assistant that can answer questions about the stock market.",
                "tools": [{"name": "query_tools", "description": "A tool to generate queries for the query research assistant."}]
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

