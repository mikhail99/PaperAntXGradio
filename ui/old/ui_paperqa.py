import gradio as gr
from core.collections_manager import CollectionsManager
import os
import html

from core.paperqa_service import PaperQAService

# --- Initialize PaperQA Service ---
paperqa_service = PaperQAService()

manager = CollectionsManager()

# Helper to get collection options and descriptions
def get_collection_options():
    return [c.name for c in manager.get_all_collections() if not c.archived]

def get_collection_description(collection_name):
    c = manager.get_collection(collection_name)
    return c.description if c else ""

def create_paperqa_tab(state):
    with gr.TabItem("📝 PaperQA"):
        with gr.Row():
            with gr.Column(scale=1):
                collection_dropdown = gr.Dropdown(
                    choices=get_collection_options(),
                    label="Select Collection",
                    value=None,
                )
                collection_desc_md = gr.Markdown("<i>Select a collection to see its description.</i>")
            with gr.Column(scale=1):
                question_box = gr.Textbox(label="Your Question", placeholder="Ask a detailed question about the collection...", lines=2)
                ask_btn = gr.Button("Get Report")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Report")
                answer_card = gr.Markdown("(Report will appear here)", elem_id="answer-card")
                
        loading_md = gr.Markdown("", visible=False)

        def update_collection_desc(collection_name):
            desc = get_collection_description(collection_name)
            return desc or "<i>No description available.</i>"

        async def handle_ask(collection_name, question):
            if not collection_name or not question.strip():
                yield {
                    loading_md: gr.update(visible=False),
                    answer_card: "Please select a collection and enter a question."
                }
                return
            
            collection = manager.get_collection(collection_name)
            if not collection:
                yield {
                    loading_md: gr.update(visible=False),
                    answer_card: "Collection not found."
                }
                return
            collection_name = collection.name

            yield {
                loading_md: gr.update(value="_Querying PaperQA cache..._", visible=True),
                answer_card: "Processing..."
            }

            service_response = await paperqa_service.query_documents(collection_name, question)
            
            final_answer_card_val = ""
            if service_response["error"]:
                error_message = html.escape(service_response["error"])
                final_answer_card_val = f"**Error:**\n{error_message}"
            else:
                answer_text = html.escape(service_response["answer_text"])
                final_answer_card_val = f"**Q:** {html.escape(question)}\n\n**A:** {answer_text}"

            yield {
                loading_md: gr.update(visible=False),
                answer_card: final_answer_card_val
            }

        collection_dropdown.change(update_collection_desc, collection_dropdown, collection_desc_md)
        ask_btn.click(
            handle_ask,
            [collection_dropdown, question_box],
            [loading_md, answer_card]
        ).then(
            lambda: "", None, question_box
        ) 