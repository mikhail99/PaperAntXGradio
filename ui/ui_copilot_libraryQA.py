from core.copilots.copilot_papersQA import CopilotPaperQAService
import gradio as gr
from ui.components.copilot import create_copilot
# from ui.components.agent_list import _js_attach_listener  # No longer needed

def create_copilot_tab(state, copilot_service: CopilotPaperQAService):
    with gr.TabItem("📚 Library QA Copilot") as copilot_tab:
        agent_list_display, initial_load_fn, js_listener = create_copilot("Library QA Copilot", copilot_service, "library_qa", state)
        copilot_tab.select(
            initial_load_fn,
            None,
            [agent_list_display],
            js=js_listener
        )
