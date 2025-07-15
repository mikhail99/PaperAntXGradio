from core.copilots.copilot_business_service import CopilotBusinessService
import gradio as gr
from ui.components.copilot import create_copilot
# from ui.components.agent_list import _js_attach_listener  # No longer needed

def create_copilot_tab(state, copilot_service: CopilotBusinessService):
    with gr.TabItem("💼 Business Copilot") as copilot_tab:
        agent_list_display, initial_load_fn, js_listener = create_copilot("💼 Business Copilot", copilot_service, "business", state)
        copilot_tab.select(
            initial_load_fn,
            None,
            [agent_list_display],
            js=js_listener
        )

