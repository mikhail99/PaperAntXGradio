from core.copilots.copilot_project_proposal_service import CopilotProjectProposalService
import gradio as gr
from ui.components.copilot import create_copilot
# from ui.components.agent_list import _js_attach_listener  # No longer needed

def create_copilot_tab(state, copilot_service: CopilotProjectProposalService):
    with gr.TabItem("📝 Project Proposal Copilot") as copilot_tab:
        # Capture the new js_listener return value
        agent_list_display, initial_load_fn, js_listener = create_copilot("Project Proposal Copilot", copilot_service, "business", state)
        copilot_tab.select(
            initial_load_fn,
            None,
            [agent_list_display],
            js=js_listener # Apply the dynamic listener on tab load
        )

