import gradio as gr
from core.collections_manager import CollectionsManager
from core.data_models import Article # Assuming Article is used or can be imported
# import time # No longer needed
import asyncio
# import tempfile # No longer needed directly in UI
# from pathlib import Path # No longer needed directly in UI
import os
import html

# from paperqa import Docs, Settings # No longer needed directly in UI
# from paperqa.settings import AgentSettings, IndexSettings # No longer needed directly in UI
from core.paperqa_service import PaperQAService # Import the new service

# --- Initialize PaperQA Service ---
paperqa_service = PaperQAService()

manager = CollectionsManager(persist_directory="data/chroma_db_store")

# Helper to get collection options and descriptions
def get_collection_options():
    return [(c.name, c.id) for c in manager.get_all_collections() if not c.archived]

def get_collection_description(collection_id):
    c = manager.get_collection(collection_id)
    return c.description if c else ""

def extract_pdf_path_from_notes(notes: str) -> str | None:
    if not notes:
        return None
    lines = notes.splitlines()
    for line in lines:
        if line.startswith("Local PDF: "):
            return line.replace("Local PDF: ", "").strip()
    return None

def create_paperqa_tab(state):
    with gr.TabItem("📝 PaperQA"):
        with gr.Row():
            with gr.Column():
                collection_dropdown = gr.Dropdown(
                    choices=get_collection_options(),
                    label="Select Collection",
                    value=None,
                )
                collection_desc_md = gr.Markdown("<i>Select a collection to see its description.</i>")
            with gr.Column():
                question_box = gr.Textbox(label="Your Question", placeholder="Ask a detailed question about the collection...", lines=2)
                ask_btn = gr.Button("Get Report")
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Report")
                answer_card = gr.Markdown("(Report will appear here)", elem_id="answer-card")
            with gr.Column(scale=1, elem_classes=["qa-history-card"]):
                gr.Markdown("### Q&A History")
                history_radio = gr.Radio(choices=[], label="History", interactive=True, value=None, elem_classes=["qa-history-radio"])
        loading_md = gr.Markdown("", visible=False)

        history_state = gr.State([])
        selected_history_value = gr.State(None)

        def update_collection_desc(collection_id):
            desc = get_collection_description(collection_id)
            return desc or "<i>No description available.</i>"

        async def handle_ask(collection_id, question, current_history):
            if not collection_id or not question.strip():
                yield {
                    loading_md: gr.update(visible=False),
                    answer_card: "Please select a collection and enter a question.",
                    history_state: current_history,
                    selected_history_value: selected_history_value.value,
                    history_radio: gr.update(choices=history_radio.choices, value=selected_history_value.value)
                }
                return
            
            collection = manager.get_collection(collection_id)
            if not collection:
                yield {
                    loading_md: gr.update(visible=False),
                    answer_card: "Collection not found.",
                    history_state: current_history,
                    selected_history_value: selected_history_value.value,
                    history_radio: gr.update(choices=history_radio.choices, value=selected_history_value.value)
                }
                return

            # Build a list of Article objects with valid local PDFs
            valid_articles = []
            for article in collection.articles.values():
                pdf_path = extract_pdf_path_from_notes(getattr(article, 'notes', ''))
                if pdf_path and os.path.exists(pdf_path):
                    valid_articles.append(article)
                elif pdf_path:
                    print(f"Warning: PDF path found ({pdf_path}) but file does not exist.")

            if not valid_articles:
                yield {
                    loading_md: gr.update(visible=False),
                    answer_card: "No valid PDF documents with existing files found in the selected collection.",
                    history_state: current_history,
                    selected_history_value: selected_history_value.value,
                    history_radio: gr.update(choices=history_radio.choices, value=selected_history_value.value)
                }
                return
            
            yield {
                loading_md: gr.update(value="_Processing with PaperQA... This may take a while._", visible=True),
                answer_card: "Processing..."
            }

            # Pass the list of Article objects to the service
            service_response = await paperqa_service.query_documents(valid_articles[:10], question) # HACK: Limit to 10 articles
            print("HACK: Limit to 10 articles")
            
            new_history_list = list(current_history)
            answer_text_to_store = ""
            evidence_to_store = ""
            final_answer_card_val = ""

            if service_response["error"]:
                error_message = html.escape(service_response["error"])
                final_answer_card_val = f"**Error:**\n{error_message}"
            else:
                answer_text_raw = service_response["answer_text"]
                evidence_raw = service_response["formatted_evidence"]
                answer_text_to_store = html.escape(answer_text_raw)
                evidence_to_store = "".join([
                    f"\n{i + 1}. **Source:** {html.escape(line.split('(Score:')[0].split('Source:')[1].strip())} (Score: {html.escape(line.split('(Score:')[1].split(')')[0])})\n> {html.escape(line.split('>', 1)[1].strip() if '>' in line else line)}\n\n"
                    if ">" in line and "Source:" in line and "(Score:" in line 
                    else html.escape(line) + "\n" 
                    for i, line_group in enumerate(evidence_raw.strip().split("\n\n"))
                    for line in line_group.strip().split("\n") if line.strip()
                ]) if evidence_raw.strip() else "_No specific evidence found by PaperQA._\n"

                final_answer_card_val = f"**Q:** {html.escape(question)}\n\n**A:** {answer_text_to_store}\n\n**Evidence:**\n{evidence_to_store}"
                entry = {"q": question, "a": answer_text_to_store, "evidence": evidence_to_store}
                new_history_list.insert(0, entry)

            current_radio_choices = [
                f"💬 {item['q'][:50]}{'...' if len(item['q']) > 50 else ''}"
                for item in new_history_list
            ]
            
            current_selected_radio_val = selected_history_value.value
            if service_response["error"]:
                pass
            elif current_radio_choices:
                current_selected_radio_val = current_radio_choices[0]
            
            if not any(choice == current_selected_radio_val for choice in current_radio_choices):
                 current_selected_radio_val = None

            yield {
                loading_md: gr.update(visible=False),
                answer_card: final_answer_card_val,
                history_state: new_history_list,
                selected_history_value: current_selected_radio_val,
                history_radio: gr.update(choices=current_radio_choices, value=current_selected_radio_val)
            }

        def handle_history_click(selected_q_short, hist_list):
            if not hist_list or not selected_q_short:
                return "(No Q&A selected)", selected_q_short
            for entry in hist_list:
                entry_q_display = f"💬 {entry['q'][:50]}{'...' if len(entry['q']) > 50 else ''}"
                if entry_q_display == selected_q_short:
                    return f"**Q:** {html.escape(entry['q'])}\n\n**A:** {entry['a']}\n\n**Evidence:**\n{entry['evidence']}", selected_q_short
            return "(Error: Could not find selected history item)", selected_q_short

        collection_dropdown.change(update_collection_desc, collection_dropdown, collection_desc_md)
        ask_btn.click(
            handle_ask,
            [collection_dropdown, question_box, history_state],
            [loading_md, answer_card, history_state, selected_history_value, history_radio]
        ).then(
            lambda: "", None, question_box
        )
        history_radio.change(handle_history_click, [history_radio, history_state], [answer_card, selected_history_value]) 