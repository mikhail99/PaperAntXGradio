import gradio as gr
from core.proposal_agent_ui_service import ProposalAgentUIService

# --- UI Helper Functions (made private to indicate module-level use) ---

def _create_team_selection_ui(service: ProposalAgentUIService):
    """Creates the accordion-based UI for team and agent selection."""
    teams_config = service.get_teams_config()
    agent_radio_buttons = []

    with gr.Column(scale=1, min_width=320):
        gr.Markdown("### Agent Teams")
        
        for team_name, team_info in teams_config.items():
            with gr.Accordion(team_name, open=True):
                member_names = [service.get_agent_details(m)['name'] for m in team_info['members'] if service.get_agent_details(m)]
                # Map display names back to their internal node keys
                member_keys = [m for m in team_info['members']]
                
                # Use a radio button for selection within the team
                radio = gr.Radio(
                    choices=list(zip(member_names, member_keys)),
                    label=f"{team_name} Members",
                    elem_id=f"radio_{team_name}"
                )
                agent_radio_buttons.append(radio)
    
    return agent_radio_buttons


def create_proposal_debugger_tab(service: ProposalAgentUIService):
    """
    Creates the Gradio TabItem for the Proposal Agent Debugger.
    This function is designed to be called from a main app UI.
    """
    with gr.TabItem("🕵️‍♂️ Proposal Agent Debugger") as debugger_tab:
        with gr.Blocks(theme=gr.themes.Soft()):
            max_context_vars = 10 # Maximum number of dynamic context fields supported

            with gr.Row(equal_height=False):
                # --- Left Column: Agent Selection & Details ---
                with gr.Column(scale=1, min_width=350):
                    selected_agent_key = gr.State(None)
                    agent_radios = _create_team_selection_ui(service)

                    with gr.Accordion("Agent Details", open=True):
                        agent_details_display = gr.Markdown("Select an agent to see details.")
                    
                    with gr.Accordion("Required Context", open=True):
                        context_inputs_group = gr.Group(visible=False)
                        with context_inputs_group:
                            gr.Markdown("Please provide the required context for this agent:")
                            # Pre-define a fixed number of placeholders for context variables
                            context_components = []
                            for i in range(max_context_vars):
                                context_components.append(
                                    gr.Textbox(visible=False, elem_id=f"context_var_{i}")
                                )

                # --- Right Column: Chat Interface ---
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        [],
                        elem_id="agent_debugger_chatbot",
                        label="Agent Conversation",
                        show_copy_button=True,
                        height=600
                    )
                    
                    with gr.Row():
                        chat_input = gr.Textbox(
                            scale=4,
                            show_label=False,
                            placeholder="Ask the selected agent a question...",
                            container=False,
                        )
                        send_button = gr.Button("Send", scale=1, variant="primary")

            # --- Event Handlers ---
            
            def on_select_agent(agent_key: str, *all_radios):
                updated_radios = []
                for radio in all_radios:
                    updated_radios.append(gr.update(value=None) if radio != agent_key else gr.update())

                details = service.get_agent_details(agent_key)
                if not details:
                    # Create a list of hidden updates for the context boxes
                    context_updates = [gr.update(visible=False, value="")] * max_context_vars
                    return [gr.Markdown("Agent not found."), gr.update(visible=False), None] + updated_radios + context_updates

                details_md = f"### {details['name']}\n**Description:** {details['description']}\n\n**Prompt Template:**\n```\n{details['prompt_text']}\n```"
                
                variables = service.get_prompt_variables(agent_key)
                
                # Create a list of gr.update objects for the context textboxes
                context_updates = []
                for i in range(max_context_vars):
                    if i < len(variables):
                        context_updates.append(gr.update(label=variables[i], visible=True, value=""))
                    else:
                        context_updates.append(gr.update(visible=False, value=""))

                return [
                    gr.Markdown(details_md),
                    gr.Group(visible=bool(variables)),
                    agent_key
                ] + updated_radios + context_updates

            def on_chat_message(agent_key: str, chat_input_text: str, ui_history: list, *context_values):
                if not agent_key:
                    ui_history.append([chat_input_text, "Please select an agent first."])
                    return ui_history, gr.update()

                # Reconstruct the context dictionary from the dynamic inputs
                variables = service.get_prompt_variables(agent_key)
                context = dict(zip(variables, context_values))

                ui_history.append([chat_input_text, None])
                yield ui_history, gr.update(interactive=False)

                response = service.chat_with_agent(agent_key, context, chat_input_text)
                ui_history[-1][1] = response
                
                yield ui_history, gr.update(interactive=True)

            # --- Wiring ---
            
            for radio in agent_radios:
                radio.input(
                    fn=on_select_agent,
                    inputs=[radio] + agent_radios,
                    outputs=[agent_details_display, context_inputs_group, selected_agent_key] + agent_radios + context_components,
                    show_progress="hidden"
                )

            chat_submit_args = {
                "fn": on_chat_message,
                "inputs": [selected_agent_key, chat_input, chatbot] + context_components,
                "outputs": [chatbot, send_button],
            }
            chat_input.submit(**chat_submit_args)
            send_button.click(**chat_submit_args)

    return debugger_tab 