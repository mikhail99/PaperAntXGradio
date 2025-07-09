"""
Minimal PocketFlow Research Orchestrator

Direct copy of cookbook pattern - no async, no abstractions, no complexity.
Uses existing DSPy configuration to avoid threading conflicts.
"""

import uuid
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Dict, Any, Generator

from .flow import create_research_flow

# Global thread pool (like cookbook)
research_thread_pool = ThreadPoolExecutor(
    max_workers=3,
    thread_name_prefix="research_worker"
)

# Global storage for active workflows (like cookbook)
active_workflows = {}


def research_fn(topic: str, collection: str, use_parrot: bool = False) -> Generator[Dict[str, Any], None, None]:
    """
    Main research function (like cookbook chat_fn).
    
    Exactly copies cookbook pattern - no session management, no complexity.
    Uses existing DSPy configuration (no reconfiguration to avoid thread conflicts).
    """
    
    # Note: We don't configure DSPy here to avoid thread conflicts
    # The app should configure DSPy once at startup
    
    # Generate conversation ID (like cookbook)
    conversation_id = str(uuid.uuid4())
    
    # Initialize queues (like cookbook)
    chat_queue = Queue()
    flow_queue = Queue()
    
    # Create shared context (like cookbook)
    shared = {
        "conversation_id": conversation_id,
        "topic": topic,
        "collection": collection,
        "queue": chat_queue,
        "flow_queue": flow_queue,
        "use_parrot": use_parrot,  # Pass to nodes
        "user_input": None,  # Store user input when resumed
    }
    
    # Store active workflow (like cookbook)
    active_workflows[conversation_id] = {
        "shared": shared,
        "chat_queue": chat_queue,
        "flow_queue": flow_queue,
        "thread": None
    }
    
    # Create and run flow (like cookbook)
    research_flow = create_research_flow()
    thread = research_thread_pool.submit(research_flow.run, shared)
    active_workflows[conversation_id]["thread"] = thread
    
    # Start with initial status
    yield {
        "thread_id": conversation_id,
        "step": "starting",
        "state": shared,
        "message": "🚀 Starting research..."
    }
    
    # Monitor flow queue for progress (like cookbook)
    import time
    
    while True:
        # Check if workflow completed
        if thread.done():
            try:
                thread.result()  # Check for exceptions
            except Exception as e:
                print(f"Workflow error: {e}")
            
            yield {
                "thread_id": conversation_id,
                "step": "workflow_complete_node", 
                "state": shared,
                "message": "Research completed"
            }
            # Cleanup
            active_workflows.pop(conversation_id, None)
            break
        
        try:
            message = flow_queue.get(timeout=0.1)
            
            # Check for pause signal
            if message == "PAUSED":
                # Check chat queue for HITL message
                try:
                    hitl_message = chat_queue.get(timeout=0.1)
                    if hitl_message:
                        # Parse structured messages from nodes
                        try:
                            import json
                            parsed = json.loads(hitl_message)
                            interrupt_type = parsed.get("type", "query_review")
                            msg_text = parsed.get("message", hitl_message)
                            context = {k: v for k, v in parsed.items() if k not in ["type", "message"]}
                        except (json.JSONDecodeError, TypeError):
                            # Fallback for plain text messages
                            interrupt_type = "query_review"
                            msg_text = str(hitl_message)
                            context = {}
                        
                        yield {
                            "thread_id": conversation_id,
                            "step": "human_input_required",
                            "state": shared,
                            "interrupt_type": interrupt_type,
                            "message": msg_text,
                            "context": context
                        }
                        return  # Stop monitoring until resumed
                except:
                    pass
            else:
                yield {
                    "thread_id": conversation_id,
                    "step": "progress",
                    "state": shared,
                    "message": message
                }
                
        except:
            pass
        
        time.sleep(0.1)


def continue_workflow(thread_id: str, user_input: str) -> Generator[Dict[str, Any], None, None]:
    """Continue a paused workflow with user input (like cookbook)."""
    
    if thread_id not in active_workflows:
        yield {
            "thread_id": thread_id,
            "step": "error",
            "state": {},
            "message": "Workflow not found or already completed"
        }
        return
    
    # Get the workflow's context
    workflow = active_workflows[thread_id]
    shared = workflow["shared"]
    flow_queue = workflow["flow_queue"]
    thread = workflow["thread"]
    
    # Store user input and signal the waiting node
    shared["user_input"] = user_input
    
    # Resume the paused node by setting the threading event
    if "hitl_event" in shared:
        shared["hitl_event"].set()
    
    yield {
        "thread_id": thread_id,
        "step": "input_received",
        "state": shared,
        "message": f"Received: {user_input}"
    }
    
    # Continue monitoring the workflow (same as research_fn)
    import time
    
    while True:
        # Check if workflow completed
        if thread.done():
            try:
                thread.result()  # Check for exceptions
            except Exception as e:
                print(f"Workflow error: {e}")
            
            yield {
                "thread_id": thread_id,
                "step": "workflow_complete_node",
                "state": shared,
                "message": "Research completed"
            }
            # Cleanup
            active_workflows.pop(thread_id, None)
            break
        
        try:
            message = flow_queue.get(timeout=0.1)
            
            # Check for pause signal
            if message == "PAUSED":
                # Check chat queue for HITL message
                chat_queue = workflow["chat_queue"]
                try:
                    hitl_message = chat_queue.get(timeout=0.1)
                    if hitl_message:
                        # Parse structured messages from nodes
                        try:
                            import json
                            parsed = json.loads(hitl_message)
                            interrupt_type = parsed.get("type", "query_review")
                            msg_text = parsed.get("message", hitl_message)
                            context = {k: v for k, v in parsed.items() if k not in ["type", "message"]}
                        except (json.JSONDecodeError, TypeError):
                            # Fallback for plain text messages
                            interrupt_type = "query_review"
                            msg_text = str(hitl_message)
                            context = {}
                        
                        yield {
                            "thread_id": thread_id,
                            "step": "human_input_required",
                            "state": shared,
                            "interrupt_type": interrupt_type,
                            "message": msg_text,
                            "context": context
                        }
                        return  # Stop monitoring until resumed again
                except:
                    pass
            else:
                yield {
                    "thread_id": thread_id,
                    "step": "progress",
                    "state": shared,
                    "message": message
                }
                
        except:
            pass
        
        time.sleep(0.1)


# Factory function for compatibility
def create_research_service(use_parrot: bool = False):
    """Factory function - returns synchronous research service."""
    
    # Mock object for compatibility
    class ResearchService:
        def __init__(self):
            self.use_parrot = use_parrot
        
        def start_agent(self, config):
            return research_fn(
                topic=config["topic"],
                collection=config["collection_name"],
                use_parrot=use_parrot
            )
        
        def continue_agent(self, thread_id, user_input):
            return continue_workflow(thread_id, user_input)
    
    return ResearchService() 