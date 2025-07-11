import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
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
    # Log conversation details
    print(f"Conversation ID: {uuid_val}\nHistory: {history}\nQuery: {message}\n---")
    
    # Initialize queues for chat messages and flow thoughts with proper typing
    chat_queue: Queue[str] = Queue()
    flow_queue: Queue[Optional[str]] = Queue()
    
    # Create properly typed shared context for the flow
    shared: SharedState = {
        "conversation_id": uuid_val,
        "query": message,
        "history": history,
        "queue": chat_queue,
        "flow_queue": flow_queue,
    }
    
    # Create and run the chat flow in a separate thread
    chat_flow = create_flow()
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


def clear_fn():
    print("Clearing conversation")
    return uuid.uuid4()


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