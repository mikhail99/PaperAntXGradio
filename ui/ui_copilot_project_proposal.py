from core.copilots.project_proposal.copilot_project_proposal_service import CopilotProjectProposalService
import gradio as gr
from ui.components.enhanced_copilot import build_copilot_view

def create_copilot_tab(state, copilot_service: CopilotProjectProposalService, trigger: gr.Textbox):
    with gr.TabItem("📝 Project Proposal Copilot") as copilot_tab:
        agent_list_display, initial_load_fn, js_listener = build_copilot_view(
            tab_title="📝 Project Proposal Copilot",
            copilot_service=copilot_service,
            tab_id_suffix="proposal",
            state=state,
            trigger=trigger
        )
        
        copilot_tab.select(
            fn=initial_load_fn,
            inputs=None,
            outputs=[agent_list_display],
            js=js_listener
        )

