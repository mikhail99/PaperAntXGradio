import gradio as gr

def get_shared_state():
    return {
        "selected_collection_id": gr.State(""),
        "selected_article_id": gr.State(""),
        "copilot_chat_history": gr.State([]),
        "selected_article_title": gr.State(""),
        # Add more as needed
    } 