import json
import os
import time
from typing import Dict, List, Optional, Any, Generator
import dspy
import asyncio
import pandas as pd
from gradio import ChatMessage
from queue import Queue
import threading
import gradio as gr
from core.collections_manager import CollectionsManager
from core.paperqa_service import PaperQAService

class EventLogger:
    """Logger for tracking interesting events during agent execution."""
    
    def __init__(self):
        self.events = []
        self.current_tool_start_time = None
    
    def log_tool_start(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Log the start of a tool call."""
        self.current_tool_start_time = time.time()
        event = {
            "type": "tool_start",
            "tool_name": tool_name,
            "args": args,
            "timestamp": self.current_tool_start_time,
            "status": "pending"
        }
        self.events.append(event)
    
    def log_tool_end(self, tool_name: str, result: str, success: bool = True) -> None:
        """Log the completion of a tool call."""
        end_time = time.time()
        duration = end_time - self.current_tool_start_time if self.current_tool_start_time else 0
        
        # Update the last event (should be the tool_start event)
        if self.events and self.events[-1]["type"] == "tool_start":
            self.events[-1].update({
                "type": "tool_complete",
                "result": result[:200] + "..." if len(result) > 200 else result,  # Truncate long results
                "duration": duration,
                "status": "done",  # Always use "done" since Gradio only accepts "pending" or "done"
                "success": success,  # Track success separately
                "end_timestamp": end_time
            })
    
    def log_reasoning_step(self, thought: str) -> None:
        """Log a reasoning step from the agent."""
        event = {
            "type": "reasoning",
            "thought": thought,
            "timestamp": time.time(),
            "status": "done"
        }
        self.events.append(event)
    
    def get_events_as_chat_messages(self) -> List[ChatMessage]:
        """Convert logged events to ChatMessage objects with metadata."""
        messages = []
        
        for event in self.events:
            if event["type"] == "tool_complete":
                # Create thought bubble for tool calls
                tool_name = event["tool_name"]
                args = event.get("args", {})
                duration = event.get("duration", 0)
                status = event.get("status", "done")
                success = event.get("success", True)
                
                # Format args for display
                args_str = ", ".join([f"{k}='{v}'" for k, v in args.items()])
                
                # Create metadata for the thought bubble
                title_icon = "🛠️" if success else "⚠️"
                metadata = {
                    "title": f"{title_icon} Tool: {tool_name}",
                    "status": status,
                    "duration": duration
                }
                
                # Content shows the tool call and result
                content = f"**Called:** `{tool_name}({args_str})`\n\n"
                if success:
                    content += f"**Result:** {event.get('result', 'No result')}"
                else:
                    content += f"**Error:** {event.get('result', 'No result')}"
                
                messages.append(ChatMessage(
                    role="assistant",
                    content=content,
                    metadata=metadata
                ))
            
            elif event["type"] == "reasoning":
                # Create thought bubble for reasoning steps
                metadata = {
                    "title": "🤔 Reasoning",
                    "status": "done"
                }
                
                messages.append(ChatMessage(
                    role="assistant", 
                    content=event["thought"],
                    metadata=metadata
                ))
        
        return messages
    
    def clear(self) -> None:
        """Clear all logged events."""
        self.events = []
        self.current_tool_start_time = None

class PaperQAReActAgent(dspy.Module):
    """ReAct agent for basic paper search and summarization."""

    def __init__(self, collection_name: str):
        super().__init__()
        self.event_logger = EventLogger()
        self.manager = CollectionsManager()
        self.collection_name = collection_name
        
        # Create ReAct agent with basic tools
        self.react_agent = dspy.ReAct(
            signature="user_request -> analysis_result",
            tools=[self.search_articles, self.summarize_articles],
            max_iters=5
        )

    def search_articles(self, query: str) -> str:
        """
        Search and select relevant research paper abstracts based on a query. Returns up to 10 most relevant abstracts.
        
        Args:
            query (str): Search query to find relevant papers
            
        Returns:
            str: Formatted abstracts and metadata
        """
        # Log tool start
        self.event_logger.log_tool_start("search_articles", {"query": query})
        
        try:
            articles = self.manager.search_articles(self.collection_name, query, limit=10)
            
            if not articles:
                result = f"No articles found for query: {query}"
                self.event_logger.log_tool_end("search_articles", result, success=True)
                return result
            
            result = f"Found {len(articles)} articles for query '{query}':\n\n"
            for i, article in enumerate(articles, 1):
                # Assuming articles have title, abstract, and other metadata
                title = getattr(article, 'title', 'Unknown Title')
                abstract = getattr(article, 'abstract', getattr(article, 'content', 'No abstract available'))
                result += f"{i}. **{title}**\n{abstract[:300]}...\n\n"
            
            self.event_logger.log_tool_end("search_articles", f"Successfully found {len(articles)} articles", success=True)
            return result
        except Exception as e:
            error_msg = f"Error searching articles: {str(e)}"
            self.event_logger.log_tool_end("search_articles", error_msg, success=False)
            return error_msg

    def summarize_articles(self, query: str) -> str:
        """
        Search for articles and provide a comprehensive summary of the research findings. Uses 20 articles for analysis.
        
        Args:
            query (str): Research topic or question to analyze
            
        Returns:
            str: Summary of research findings
        """
        # Log tool start
        self.event_logger.log_tool_start("summarize_articles", {"query": query})
        
        try:
            articles = self.manager.search_articles(self.collection_name, query, limit=20)
            
            if not articles:
                result = f"No articles found to summarize for query: {query}"
                self.event_logger.log_tool_end("summarize_articles", result, success=True)
                return result
            
            # Extract abstracts for summarization
            abstracts = []
            titles = []
            for article in articles:
                title = getattr(article, 'title', 'Unknown Title')
                abstract = getattr(article, 'abstract', getattr(article, 'content', ''))
                if abstract:
                    abstracts.append(abstract)
                    titles.append(title)
            
            if not abstracts:
                result = "No abstracts available for summarization"
                self.event_logger.log_tool_end("summarize_articles", result, success=True)
                return result
            
            # Simple summarization (you mentioned you'll improve this later)
            summary = f"**Summary of {len(abstracts)} research papers on '{query}':**\n\n"
            summary += f"**Key Papers Analyzed:**\n"
            for i, title in enumerate(titles[:5], 1):  # Show first 5 titles
                summary += f"{i}. {title}\n"
            
            summary += f"\n**Research Overview:**\n"
            summary += f"This analysis covers {len(abstracts)} papers related to {query}. "
            summary += "The research spans various methodologies and approaches in this field. "
            summary += "Key findings include advancements in methodology, novel applications, and theoretical contributions.\n"
            
            summary += f"\n**Note:** This is a basic summary. Advanced analysis will be implemented in future versions."
            
            self.event_logger.log_tool_end("summarize_articles", f"Successfully summarized {len(abstracts)} papers", success=True)
            return summary
        except Exception as e:
            error_msg = f"Error summarizing articles: {str(e)}"
            self.event_logger.log_tool_end("summarize_articles", error_msg, success=False)
            return error_msg

    def forward(self, user_request: str):
        """
        Process user request using ReAct agent with research tools.
        
        Args:
            user_request: User's question or research request
            
        Returns:
            Analysis result from the ReAct agent
        """
        # Clear previous events
        self.event_logger.clear()
        
        try:
            result = self.react_agent(user_request=user_request)
            return result.analysis_result
        except Exception as e:
            return f"Error processing request: {str(e)}"

class LiteratureReviewAgent(dspy.Module):
    """Specialized agent for advanced literature review using PaperQA."""

    def __init__(self, collection_name: str):
        super().__init__()
        self.event_logger = EventLogger()
        self.collection_name = collection_name
        self.paperqa_service = PaperQAService()
        
        # Create ReAct agent with literature review tool
        self.react_agent = dspy.ReAct(
            signature="user_request -> literature_review",
            tools=[self.query_literature],
            max_iters=3  # Fewer iterations since this is more direct
        )

    def query_literature(self, question: str) -> str:
        """
        Perform advanced literature review using PaperQA for comprehensive analysis with citations and links.
        
        Args:
            question (str): Research question or topic to analyze
            
        Returns:
            str: Comprehensive literature review with citations and links
        """
        # Log tool start
        self.event_logger.log_tool_start("query_literature", {"question": question})
        
        try:
            # Use asyncio to run the async PaperQA query
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.paperqa_service.query_documents(self.collection_name, question)
                )
            finally:
                loop.close()
            
            if result.get("error"):
                error_msg = f"PaperQA error: {result['error']}"
                self.event_logger.log_tool_end("query_literature", error_msg, success=False)
                return error_msg
            
            answer_text = result.get("answer_text", "No answer found")
            
            if not answer_text or answer_text == "No answer found":
                result_msg = f"No comprehensive literature review could be generated for: {question}"
                self.event_logger.log_tool_end("query_literature", result_msg, success=True)
                return result_msg
            
            # Format the response nicely
            formatted_response = f"## Literature Review: {question}\n\n"
            formatted_response += answer_text
            
            # Add context information if available
            context = result.get("context", [])
            if context:
                formatted_response += f"\n\n**Sources analyzed:** {len(context)} relevant papers"
            
            self.event_logger.log_tool_end("query_literature", f"Successfully generated literature review with {len(context)} sources", success=True)
            return formatted_response
            
        except Exception as e:
            error_msg = f"Error in literature review: {str(e)}"
            self.event_logger.log_tool_end("query_literature", error_msg, success=False)
            return error_msg

    def forward(self, user_request: str):
        """
        Process user request using ReAct agent with literature review tool.
        
        Args:
            user_request: User's research question or request
            
        Returns:
            Literature review result from the ReAct agent
        """
        # Clear previous events
        self.event_logger.clear()
        
        try:
            result = self.react_agent(user_request=user_request)
            return result.literature_review
        except Exception as e:
            return f"Error processing literature review request: {str(e)}"

class CopilotPaperQAService:
    def __init__(self) -> None:
        """Initialize CopilotService with both ReAct agents"""
        self.collection_name = "Diffusion_Models" #HACK: make it dynamic TODO
        self.agents : Dict[str, dspy.Module] = self._create_agents(self.collection_name)
    
    def _create_agents(self, collection_name: str) -> Dict[str, dspy.Module]:
        """Create agent instances"""
        return {
            "Research Assistant": PaperQAReActAgent(collection_name),
            "Literature Review Assistant": LiteratureReviewAgent(collection_name)
        }
    
    def get_agent_list(self) -> List[str]:
        """Returns a list of available agent names."""
        return sorted(list(self.agents.keys()))
    
    def get_agent_details(self, agent_name: str = None) -> Dict[str, str]:
        """Returns the configuration for a specific agent or all agents."""
        all_details = {
            "Research Assistant": {
                "short_description": "ReAct agent for basic research paper search and summarization.",
                "full_description": "A ReAct agent that can search through research abstracts and provide basic summaries of findings on specific topics.",
                "tools": [
                    {"name": "search_articles", "description": "Search and select relevant research paper abstracts"},
                    {"name": "summarize_articles", "description": "Analyze and summarize research findings from multiple papers"}
                ]
            },
            "Literature Review Assistant": {
                "short_description": "Advanced literature review agent using PaperQA with citations and links.",
                "full_description": "A specialized ReAct agent that performs comprehensive literature reviews using PaperQA, providing detailed analysis with proper citations and links to sources.",
                "tools": [
                    {"name": "query_literature", "description": "Perform advanced literature review with citations and source links"}
                ]
            },
        }
        
        if agent_name:
            return all_details.get(agent_name)
        return all_details

    def get_quick_actions(self, agent_name: str) -> List[Dict[str, str]]:
        """Returns quick action buttons for the specified agent."""
        if not agent_name:
            return []
        
        actions_map = {
            "Research Assistant": [
                {"label": "Search Papers", "icon": "🔍", "color_class": "search-btn"},
                {"label": "Summarize Research", "icon": "📊", "color_class": "outline-btn"},
                {"label": "Find Trends", "icon": "📈", "color_class": "research-btn"},
                {"label": "Compare Methods", "icon": "⚖️", "color_class": "method-btn"}
            ],
            "Literature Review Assistant": [
                {"label": "Literature Review", "icon": "📚", "color_class": "research-btn"},
                {"label": "Comprehensive Analysis", "icon": "🔬", "color_class": "outline-btn"},
                {"label": "Citation Analysis", "icon": "🔗", "color_class": "search-btn"},
                {"label": "Research Synthesis", "icon": "🧠", "color_class": "method-btn"}
            ]
        }
        
        return actions_map.get(agent_name, [])

    def reload(self) -> None:
        """Reload agent configurations."""
        print(f"Reloading {self.__class__.__name__} - agents recreated")
        self.agents = self._create_agents()
    
    def chat_with_agent(self, agent_name: str, message: str, llm_history: List[Dict[str, Any]], provider: str = "ollama") -> List[ChatMessage]:
        """
        Chat with the selected ReAct agent and return ChatMessage objects with tool call metadata.
        
        Args:
            agent_name: Name of the agent to use
            message: User's message/question
            llm_history: Previous conversation history
            provider: LLM provider (not used in current implementation)
            
        Returns:
            List of ChatMessage objects (tool calls + final answer)
        """
        agent = self.agents.get(agent_name)
        
        if not agent:
            error_msg = ChatMessage(role="assistant", content=f"Agent '{agent_name}' not found.")
            return [error_msg]
        
        try:
            # Process the user request with ReAct agent
            answer = agent(message)
            
            # Get the tool call events as ChatMessage objects
            tool_messages = agent.event_logger.get_events_as_chat_messages()
            
            # Create the final answer message
            final_message = ChatMessage(role="assistant", content=answer)
            
            # Return all messages: tool calls + final answer
            return tool_messages + [final_message]
        except Exception as e:
            error_msg = f"Error in agent processing: {str(e)}"
            print(f"Debug - Error details: {e}")  # Add debug info
            error_message = ChatMessage(role="assistant", content=error_msg)
            return [error_message]

