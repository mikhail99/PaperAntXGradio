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
    } 