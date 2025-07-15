import gradio as gr

def get_shared_state():
    """
    Initializes the shared state for the application.
    This state is passed to all top-level tab creation functions.
    It should only contain information that needs to be accessed across different tabs.
    """
    return {
        "selected_collection_name": gr.State(None),
        "selected_article_id": gr.State(None),
        # Agent selection states for each copilot tab
        "selected_agent_proposal": gr.State(None),
        "selected_agent_business": gr.State(None),
        "selected_agent_library_qa": gr.State(None),
        "selected_agent_portfolio": gr.State(None),
    } 