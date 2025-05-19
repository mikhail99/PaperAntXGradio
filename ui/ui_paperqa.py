import gradio as gr
from core.collections_manager import CollectionsManager
from core.data_models import Collection
import time

manager = CollectionsManager(persist_directory="data/chroma_db_store")

# Helper to get collection options and descriptions
def get_collection_options():
    return [(c.name, c.id) for c in manager.get_all_collections() if not c.archived]

def get_collection_description(collection_id):
    c = manager.get_collection(collection_id)
    return c.description if c else ""

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

        # State for Q&A history and selected Q&A
        history_state = gr.State([])  # List of dicts: {q, a, evidence}
        selected_history_value = gr.State(None)  # Store the selected question short text

        # --- Callbacks ---
        def update_collection_desc(collection_id):
            desc = get_collection_description(collection_id)
            return desc or "<i>No description available.</i>"

        def handle_ask(collection_id, question, history):
            if not collection_id or not question.strip():
                # Defensive: radio should be updated with empty choices and None value
                return gr.update(visible=False), "Please select a collection and enter a question.", history, None, gr.update(choices=[], value=None)
            collection = manager.get_collection(collection_id)
            if not collection:
                return gr.update(visible=False), "Collection not found.", history, None, gr.update(choices=[], value=None)
            # Show loading
            loading = gr.update(value="_Generating answer..._", visible=True)
            # Simulate LLM delay
            time.sleep(0.5)
            # Mock LLM response
            answer = f"**Q:** {question}\n\n**A:** This is a mock answer for your question about collection '{collection.name}'."
            evidence = f"_Evidence: (mock)_ {len(collection.articles)} articles in this collection."
            entry = {"q": question, "a": answer, "evidence": evidence}
            new_history = (history or [])[:]
            new_history.insert(0, entry)  # Most recent on top
            # Prepare radio choices (short question text)
            radio_choices = ["💬 " + (item["q"][:50] + ("..." if len(item["q"]) > 50 else "")) for item in new_history]
            selected_value = radio_choices[0] if radio_choices else None
            # Show most recent Q&A
            answer_card_val = f"{answer}\n\n{evidence}"
            # Use gr.update for atomic update
            return gr.update(visible=False), answer_card_val, new_history, selected_value, gr.update(choices=radio_choices, value=selected_value)

        def handle_history_click(selected_value, history):
            if not history or not selected_value:
                return "(No Q&A selected)", selected_value
            # Find the entry with the matching short text (with emoji)
            for entry in history:
                short_text = "💬 " + (entry["q"][:50] + ("..." if len(entry["q"]) > 50 else ""))
                if short_text == selected_value:
                    return f"{entry['a']}\n\n{entry['evidence']}", selected_value
            return "(No Q&A selected)", selected_value

        # Bindings
        collection_dropdown.change(update_collection_desc, collection_dropdown, collection_desc_md)
        ask_btn.click(
            handle_ask,
            [collection_dropdown, question_box, history_state],
            [loading_md, answer_card, history_state, selected_history_value, history_radio]
        ).then(
            lambda: "", None, question_box  # Auto-clear question box
        )
        history_radio.change(handle_history_click, [history_radio, history_state], [answer_card, selected_history_value]) 