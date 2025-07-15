from core.copilots.copilot_project_proposal_service import CopilotProjectProposalService
import gradio as gr
from ui.components.copilot import create_copilot
from ui.components.quick_actions import create_quick_actions_toolbar, get_research_proposal_actions

def create_copilot_tab(state, copilot_service: CopilotProjectProposalService):
    with gr.TabItem("📝 Project Proposal Copilot") as copilot_tab:
        
        # Create beautiful quick actions toolbar
        actions = get_research_proposal_actions()
        action_buttons, css_component = create_quick_actions_toolbar(
            actions=actions,
            title="🚀 Research Proposal Quick Actions",
            show_title=False,
            actions_per_row=4
        )
        
        # Integrate the standard copilot component (with working JavaScript)
        agent_list_display, initial_load_fn, js_listener = create_copilot(
            "📝 Project Proposal Copilot", 
            copilot_service, 
            "proposal", 
            state
        )
        
        # TODO: Connect quick action buttons to chat interface
        # For now, the buttons are created but not yet functional
        # This can be implemented by connecting them to the chat input
        
        copilot_tab.select(
            initial_load_fn,
            None,
            [agent_list_display],
            js=js_listener
        )

