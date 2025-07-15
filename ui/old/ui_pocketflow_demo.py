import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from typing import List, Dict, Any, Optional, Generator, Union

import gradio as gr
from gradio import ChatMessage

from core.pocketflow_demo.flow import create_flow
from core.pocketflow_demo.types import SharedState

# create global thread pool
chatflow_thread_pool = ThreadPoolExecutor(
    max_workers=5,
    thread_name_prefix="chatflow_worker",
)

# Global storage for active conversations
active_conversations: Dict[str, SharedState] = {}

def chat_fn(message: str, history: List[Dict[str, Any]], uuid_val: str) -> Generator[Union[ChatMessage, List[ChatMessage]], None, None]:
    """
    Main chat function that handles the conversation flow and message processing.
    
    Args:
        message: The current user message
        history: Previous conversation history
        uuid_val: Unique identifier for the conversation
    
    Yields:
        ChatMessage or List[ChatMessage]: Streams of thought process and chat responses
    """
    # Enhanced logging for debugging
    print(f"DEBUG: Conversation ID: {uuid_val}")
    print(f"DEBUG: Message: '{message}'")
    print(f"DEBUG: Active conversations: {list(active_conversations.keys())}")
    print(f"DEBUG: Is continuation: {uuid_val in active_conversations}")
    
    # Additional debugging for conversation state
    if uuid_val in active_conversations:
        existing_shared = active_conversations[uuid_val]
        print(f"DEBUG: Found existing conversation, has hitl_event: {'hitl_event' in existing_shared}")
    
    print(f"---")
    
    # Check if this is a continuation of an existing conversation (HITL feedback)
    if uuid_val in active_conversations:
        print(f"🔄 Continuing existing conversation with feedback: {message}")
        existing_shared = active_conversations[uuid_val]
        
        # Set user input and trigger the waiting event
        existing_shared["user_input"] = message  # type: ignore  # Dynamic field access
        if "hitl_event" in existing_shared:
            existing_shared["hitl_event"].set()  # type: ignore  # Dynamic field access
            print(f"DEBUG: Triggered hitl_event for conversation {uuid_val}")
        
        # Provide immediate feedback that input was received
        feedback_response = ChatMessage(
            content=f"✅ Received your feedback: **{message}**\n\nProcessing...",
            metadata={"title": "Feedback Received", "id": int(time.time())}
        )
        yield feedback_response
        
        # Continue monitoring the existing conversation's queues without starting a new flow
        chat_queue = existing_shared["queue"]
        flow_queue = existing_shared["flow_queue"]
        
        # Process any remaining flow thoughts first
        while True:
            try:
                thought = flow_queue.get(timeout=0.1)  # Short timeout to check for immediate messages
                if thought is None:
                    break
                # Create a flow log update
                flow_update = ChatMessage(
                    content=f"- {thought}",
                    metadata={"title": "Flow Log Update", "id": int(time.time())}
                )
                yield flow_update
                flow_queue.task_done()
            except Empty:
                break  # No more immediate flow messages
        
        # Now process chat messages
        while True:
            try:
                msg = chat_queue.get(timeout=5.0)  # Wait up to 5 seconds for response
                if msg is None:
                    break
                final_response = [feedback_response, ChatMessage(content=msg)]
                yield final_response
                chat_queue.task_done()
            except Empty:
                # Timeout - the flow might still be processing
                processing_update = [feedback_response, ChatMessage(content="⏳ Still processing your feedback...")]
                yield processing_update
                continue
        
        return
    
    # New conversation - create fresh state and start flow
    print(f"🆕 Starting new conversation for UUID: {uuid_val}")
    
    # Initialize queues for chat messages and flow thoughts with proper typing
    chat_queue: Queue[Optional[str]] = Queue()
    flow_queue: Queue[Optional[str]] = Queue()
    
    # Create properly typed shared context for the flow
    shared: SharedState = {
        "conversation_id": uuid_val,
        "query": message,
        "history": history,
        "queue": chat_queue,
        "flow_queue": flow_queue,
    }
    
    # Store for future HITL interactions
    active_conversations[uuid_val] = shared
    print(f"DEBUG: Stored conversation {uuid_val} in active_conversations")
    print(f"DEBUG: Active conversations now: {list(active_conversations.keys())}")
    
    # Create and run the chat flow in a separate thread (keeps running for HITL)
    chat_flow = create_flow()
    print(f"DEBUG: Starting flow for conversation {uuid_val}")
    chatflow_thread_pool.submit(chat_flow.run, shared)

    # Initialize thought response tracking
    start_time = time.time()
    thought_response = ChatMessage(
        content="", metadata={"title": "Flow Log", "id": 0, "status": "pending"}
    )
    yield thought_response

    # Process and accumulate thoughts from the flow queue
    accumulated_thoughts = ""
    while True:
        thought = flow_queue.get()
        if thought is None:
            break
        accumulated_thoughts += f"- {thought}\n\n"
        thought_response.content = accumulated_thoughts.strip()
        yield thought_response
        flow_queue.task_done()

    # Mark thought processing as complete and record duration
    thought_response.metadata["status"] = "done"
    thought_response.metadata["duration"] = time.time() - start_time
    yield thought_response

    # Process and yield chat messages from the chat queue
    while True:
        msg = chat_queue.get()
        if msg is None:
            break
        chat_response = [thought_response, ChatMessage(content=msg)]
        yield chat_response
        chat_queue.task_done()
    
    # Don't remove conversation from active_conversations here 
    # Let it stay for potential HITL feedback
    print(f"DEBUG: Flow completed for {uuid_val}, keeping conversation active for HITL")


def clear_fn():
    print("DEBUG: Clearing ALL conversations manually")
    new_uuid = uuid.uuid4()
    # Clean up any active conversation data
    active_conversations.clear()
    print(f"DEBUG: Generated new UUID: {new_uuid}")
    return new_uuid


def create_pocketflow_demo_tab():
    """Create the PocketFlow Demo tab for research workflow"""
    
    with gr.Tab("Research Demo"):
        gr.Markdown("## 🔬 Research Proposal Assistant")
        gr.Markdown("This is a working demonstration of Human-in-the-Loop research workflow using PocketFlow. Try asking about:")
        gr.Markdown("- 📚 **Research Topics**: 'I want to research LLMs in education'")
        gr.Markdown("- 🤖 **AI Applications**: 'Help me write a proposal about AI safety'")
        gr.Markdown("- 🧪 **Any Scientific Domain**: 'Research proposal for quantum computing applications'")
        gr.Markdown("- ✅ **Approval Process**: The system will ask for your feedback at key steps!")
        
        uuid_state = gr.State(uuid.uuid4())
        
        chatbot = gr.Chatbot(type="messages", scale=1)
        chatbot.clear(clear_fn, outputs=[uuid_state])

        gr.ChatInterface(
            fn=chat_fn,
            type="messages",
            additional_inputs=[uuid_state],
            chatbot=chatbot,
            title="Research Proposal Assistant",
        ) 