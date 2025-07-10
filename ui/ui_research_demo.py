import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import gradio as gr
from gradio import ChatMessage

from core.pocketflow_demo.research_flow import create_research_flow

# create global thread pool
research_thread_pool = ThreadPoolExecutor(
    max_workers=5,
    thread_name_prefix="research_worker",
)


def research_chat_fn(message, history, uuid):
    """
    Main chat function that handles the research conversation flow and message processing.
    
    Args:
        message (str): The current user message
        history (list): Previous conversation history
        uuid (UUID): Unique identifier for the conversation
    
    Yields:
        ChatMessage: Streams of thought process and chat responses
    """
    # Log conversation details
    print(f"Research Conversation ID: {str(uuid)}\nHistory: {history}\nQuery: {message}\n---")
    
    # Initialize queues for chat messages and flow thoughts
    chat_queue = Queue()
    flow_queue = Queue()
    
    # Create shared context for the flow
    shared = {
        "conversation_id": str(uuid),
        "query": message,
        "history": history,
        "queue": chat_queue,
        "flow_queue": flow_queue,
    }
    
    # Create and run the research flow in a separate thread
    research_flow = create_research_flow()
    research_thread_pool.submit(research_flow.run, shared)

    # Initialize thought response tracking
    start_time = time.time()
    thought_response = ChatMessage(
        content="", metadata={"title": "Research Flow Log", "id": 0, "status": "pending"}
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


def handle_like_dislike(data: gr.LikeData):
    """
    Handle like/dislike events from the chatbot for demo purposes.
    Logs the feedback - in a full implementation this could trigger flow updates.
    
    Args:
        data: LikeData from Gradio containing like/dislike info
    """
    if data.liked:
        print("✅ User LIKED step - would proceed to next step")
        print(f"   Liked message: {data.value}")
    else:
        print("❌ User DISLIKED step - would retry current step") 
        print(f"   Disliked message: {data.value}")


def clear_research_fn():
    print("Clearing research conversation")
    return uuid.uuid4()


def create_research_demo_tab():
    """Create the Research Demo tab using the research flow with human-in-the-loop"""
    
    with gr.Tab("Research Demo"):
        gr.Markdown("## Research Proposal Agent Demo with Human-in-the-Loop")
        gr.Markdown("This demonstration shows a research proposal workflow with critical review points:")
        gr.Markdown("- 🔬 **Research Topics**: 'I want to research LLMs in education'")
        gr.Markdown("- 🧠 **AI Research**: 'Help me research machine learning for healthcare'")
        gr.Markdown("- 📚 **Any Academic Topic**: 'Research climate change impact on agriculture'")
        
        gr.Markdown("### Human-in-the-Loop Research Pipeline:")
        gr.Markdown("1. **Topic Input** → **Query Generation** → **👍👎 Human Review** → **Literature Review**")
        gr.Markdown("2. **Gap Analysis** → **Report Generation** → **👍👎 Human Review** → **Final Result**")
        gr.Markdown("3. **For Review Steps**: Type 'approved' or 'proceed' to continue")
        gr.Markdown("4. **Or**: Type 'rejected' or 'retry' to redo the step")
        gr.Markdown("5. **👍👎 Like buttons** log feedback (demo feature)")
        
        uuid_state = gr.State(uuid.uuid4())
        
        # Create custom chatbot with like/dislike functionality
        chatbot = gr.Chatbot(
            type="messages", 
            scale=1,
            show_copy_button=True,
            placeholder="<strong>Research Assistant</strong><br>Ask me to help with your research proposal!"
        )
        
        # Attach like/dislike handler to chatbot (logs feedback for demo)
        chatbot.like(handle_like_dislike, None, None)
        chatbot.clear(clear_research_fn, outputs=[uuid_state])

        gr.ChatInterface(
            fn=research_chat_fn,
            type="messages",
            additional_inputs=[uuid_state],
            chatbot=chatbot,
            title="Research Proposal Agent with Human Review",
            examples=[
                ["I want to research LLMs in education"],
                ["Help me research machine learning for healthcare"], 
                ["Research climate change impact on agriculture"],
                ["Develop AI for personalized learning"]
            ]
        ) 