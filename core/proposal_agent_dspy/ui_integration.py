"""
Integration with existing Gradio UI.
Simple adapter that works with your current chat interface.
"""

import asyncio
import threading
from typing import Dict, Any
from queue import Queue

from .orchestrator import create_research_orchestrator


# Global storage for active research sessions
active_research_sessions: Dict[str, Any] = {}


def start_research_workflow(conversation_id: str, topic: str, chat_queue: Queue, flow_queue: Queue):
    """Start a new research workflow in background thread"""
    
    orchestrator = create_research_orchestrator(chat_queue, flow_queue)
    active_research_sessions[conversation_id] = orchestrator
    
    # Run async workflow in thread
    def run_async_workflow():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(orchestrator.run_research_workflow(topic))
        finally:
            loop.close()
            # Clean up session
            if conversation_id in active_research_sessions:
                del active_research_sessions[conversation_id]
    
    thread = threading.Thread(target=run_async_workflow, daemon=True)
    thread.start()


def send_human_feedback(conversation_id: str, feedback: str) -> bool:
    """Send human feedback to waiting research workflow"""
    if conversation_id in active_research_sessions:
        orchestrator = active_research_sessions[conversation_id]
        orchestrator.set_human_response(feedback)
        return True
    return False


def is_research_active(conversation_id: str) -> bool:
    """Check if research workflow is active for conversation"""
    return conversation_id in active_research_sessions
