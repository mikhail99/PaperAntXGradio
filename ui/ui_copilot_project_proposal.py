from core.copilot_project_proposal_service import CopilotProjectProposalService
import gradio as gr
from ui.components.copilot import create_copilot
from ui.components.agent_list import _js_attach_listener

def create_copilot_tab(state, copilot_service: CopilotProjectProposalService):
    with gr.TabItem("Project Proposal Copilot") as copilot_tab:
        agent_list_display, initial_load_fn = create_copilot("Project Proposal Copilot", copilot_service, "business")
        copilot_tab.select(
            initial_load_fn,
            None,
            [agent_list_display],
            js=_js_attach_listener
        )

