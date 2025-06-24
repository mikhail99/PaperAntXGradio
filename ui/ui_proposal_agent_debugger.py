import gradio as gr
from core.proposal_agent_ui_service import ProposalAgentUIService
import hashlib
import json
import webbrowser
import os

# --- UI Helper Functions (adapted from ui_copilot.py) ---

_js_debugger_listener = """
() => {
  setTimeout(() => {
    const container = document.getElementById('debugger-agent-list-container');
    if (!container) return;
    if (container.dataset.listenerAttached === 'true') return;
    container.dataset.listenerAttached = 'true';

    container.addEventListener('click', (event) => {
      const agentItem = event.target.closest('.agent-item');
      if (!agentItem) return;
      const agentKey = agentItem.dataset.agentKey;
      if (!agentKey) return;

      const hiddenTextbox = document.querySelector('#debugger_selected_agent_trigger');
      if (!hiddenTextbox) return;
      
      const hiddenInput = hiddenTextbox.querySelector('input[type="text"], textarea');
      if (!hiddenInput) return;

      if (hiddenInput.value !== agentKey) {
        hiddenInput.value = agentKey;
        hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
      }
    });
  }, 150);
}
"""

def _name_to_color(name: str) -> str:
    hash_object = hashlib.md5(name.encode())
    hex_dig = hash_object.hexdigest()
    r = (int(hex_dig[0:2], 16) % 150) + 50
    g = (int(hex_dig[2:4], 16) % 150) + 50
    b = (int(hex_dig[4:6], 16) % 150) + 50
    return f"rgb({r},{g},{b})"

def _generate_agent_list_html(service: ProposalAgentUIService, team_name: str, selected_agent_key: str) -> str:
    teams_config = service.get_teams_config()
    team_info = teams_config.get(team_name, {})
    members = team_info.get('members', [])
    
    html_items = []
    for agent_key in members:
        details = service.get_agent_details(agent_key)
        if not details: continue

        is_selected_class = "selected" if agent_key == selected_agent_key else ""
        color = _name_to_color(details['name'])
        description = details.get('description', 'No description.')
        safe_agent_key = agent_key.replace('"', '&quot;')
        
        html_items.append(f"""
        <div class="agent-item {is_selected_class}" data-agent-key="{safe_agent_key}">
            <div class="agent-icon" style="background-color: {color};"></div>
            <div class="agent-text">
                <div class="agent-name">{details['name']}</div>
                <div class="agent-description">{description}</div>
            </div>
        </div>
        """)
    
    return f"<div class='agent-list-container'>{''.join(html_items)}</div>"


def create_proposal_debugger_tab(service: ProposalAgentUIService):
    """Creates the Gradio TabItem for the Proposal Agent Debugger."""
    with gr.TabItem("🕵️‍♂️ Proposal Agent Debugger") as debugger_tab:
        with gr.Blocks(theme=gr.themes.Soft()):
            max_context_vars = 10 

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=350):
                    selected_agent_key = gr.State(None)
                    selected_collection_name = gr.State(None)

                    collection_names = service.get_collection_names()
                    team_names = list(service.get_teams_config().keys())
                    
                    collection_dropdown = gr.Dropdown(
                        choices=collection_names, 
                        label="Select a Collection (for context)", 
                        value=collection_names[0] if collection_names else None,
                        interactive=True
                    )
                    team_dropdown = gr.Dropdown(
                        choices=team_names, 
                        label="Select a Team", 
                        value=team_names[0] if team_names else None,
                        interactive=True
                    )
                    
                    with gr.Row():
                        gr.HTML("<div style='flex-grow: 1'></div>") # Spacer
                        edit_button = gr.Button("✏️ Edit Team", size="sm")
                        reload_button = gr.Button("🔄 Reload Team", size="sm")

                    agent_list_display = gr.HTML(elem_id="debugger-agent-list-container")
                    
                    selected_agent_trigger = gr.Textbox(visible=False, elem_id="debugger_selected_agent_trigger")

                    with gr.Accordion("Agent Details", open=True):
                        agent_details_display = gr.Markdown("Select an agent to see details.")
                    
                    with gr.Accordion("Required Context", open=True):
                        context_inputs_group = gr.Group(visible=False)
                        with context_inputs_group:
                            gr.Markdown("Please provide the required context for this agent:")
                            context_components = [gr.Textbox(visible=False, elem_id=f"context_var_{i}") for i in range(max_context_vars)]

                with gr.Column(scale=2):
                    chatbot = gr.Chatbot([], elem_id="agent_debugger_chatbot", label="Agent Conversation", show_copy_button=True, height=600)
                    with gr.Row():
                        chat_input = gr.Textbox(scale=4, show_label=False, placeholder="Ask the selected agent a question...", container=False)
                        send_button = gr.Button("Send", scale=1, variant="primary")

            # --- Event Handlers ---
            
            def initial_load():
                """Populates the UI when the tab is first loaded."""
                collections = service.get_collection_names()
                teams = list(service.get_teams_config().keys())
                initial_team = teams[0] if teams else None
                html = _generate_agent_list_html(service, initial_team, None)
                context_updates = [gr.update(visible=False, value="")] * max_context_vars
                return [
                    gr.update(choices=collections, value=collections[0] if collections else None),
                    gr.update(choices=teams, value=initial_team),
                    gr.HTML(html),
                    None,
                    gr.Markdown("Select an agent to see details."),
                    gr.Group(visible=False)
                ] + context_updates

            def on_select_team(team_name):
                # When team changes, clear agent selection and details
                html = _generate_agent_list_html(service, team_name, None)
                context_updates = [gr.update(visible=False, value="")] * max_context_vars
                return [
                    gr.HTML(html),
                    None, # Clear selected agent
                    gr.Markdown("Select an agent to see details."),
                    gr.Group(visible=False)
                ] + context_updates
            
            def on_select_agent(agent_key, team_name):
                html = _generate_agent_list_html(service, team_name, agent_key)
                
                details = service.get_agent_details(agent_key)
                if not details:
                    context_updates = [gr.update(visible=False, value="")] * max_context_vars
                    return [html, gr.Markdown("Agent not found."), gr.update(visible=False), agent_key] + context_updates

                details_md = f"### {details['name']}\n**Description:** {details['description']}\n\n**Prompt Template:**\n```\n{details['prompt_text']}\n```"
                
                mcp_info = details.get('mcp_info', {})
                tools = mcp_info.get('tools', [])
                if tools:
                    details_md += "\n\n**Available Tools:**\n"
                    for tool in tools:
                        details_md += f"- `{tool.get('name')}`: {tool.get('description', 'No description.')}\n"

                variables = service.get_prompt_variables(agent_key)
                context_updates = []
                for i in range(max_context_vars):
                    context_updates.append(gr.update(label=variables[i], visible=True, value="") if i < len(variables) else gr.update(visible=False, value=""))

                return [html, gr.Markdown(details_md), gr.Group(visible=bool(variables)), agent_key] + context_updates

            def on_chat_message(agent_key, collection_name, chat_input_text, ui_history, *context_values):
                if not agent_key:
                    ui_history.append([chat_input_text, "Please select an agent first."])
                    return ui_history, gr.update()
                
                variables = service.get_prompt_variables(agent_key)
                context = dict(zip(variables, context_values))
                # Add collection_name to the context if it's a required variable
                if 'collection_name' in variables:
                    context['collection_name'] = collection_name

                ui_history.append([chat_input_text, None])
                yield ui_history, gr.update(interactive=False)

                response = service.chat_with_agent(agent_key, context, chat_input_text)
                ui_history[-1][1] = response
                
                yield ui_history, gr.update(interactive=True)

            def on_reload(current_team_name, current_collection_name):
                print("\n--- Reloading Proposal Agent Debugger Configuration ---")
                service.reload()
                new_teams = list(service.get_teams_config().keys())
                new_collections = service.get_collection_names()
                
                # Preserve selections if possible
                new_selected_team = current_team_name if current_team_name in new_teams else (new_teams[0] if new_teams else None)
                new_selected_collection = current_collection_name if current_collection_name in new_collections else (new_collections[0] if new_collections else None)
                
                return [
                    gr.update(choices=new_collections, value=new_selected_collection),
                    gr.update(choices=new_teams, value=new_selected_team)
                ]

            def on_edit_config():
                path = service.get_config_file_path()
                try:
                    # Use webbrowser.open for better cross-platform support
                    webbrowser.open(f"file://{os.path.abspath(path)}")
                    print(f"Attempted to open config file: {path}")
                except Exception as e:
                    print(f"Error opening file: {e}")

            # --- Wiring ---
            reload_button.click(
                on_reload,
                inputs=[team_dropdown, collection_dropdown],
                outputs=[collection_dropdown, team_dropdown]
            )

            edit_button.click(on_edit_config, None, None)

            # Link the collection dropdown to its state tracker
            collection_dropdown.change(
                lambda x: x,
                [collection_dropdown],
                [selected_collection_name]
            )

            team_dropdown.change(
                on_select_team,
                inputs=[team_dropdown],
                outputs=[agent_list_display, selected_agent_key, agent_details_display, context_inputs_group] + context_components
            ).then(None, None, None, js=_js_debugger_listener)
            
            selected_agent_trigger.input(
                on_select_agent,
                inputs=[selected_agent_trigger, team_dropdown],
                outputs=[agent_list_display, agent_details_display, context_inputs_group, selected_agent_key] + context_components
            ).then(None, None, None, js=_js_debugger_listener)
            
            chat_submit_args = {"fn": on_chat_message, "inputs": [selected_agent_key, selected_collection_name, chat_input, chatbot] + context_components, "outputs": [chatbot, send_button]}
            chat_input.submit(**chat_submit_args)
            send_button.click(**chat_submit_args)
            
            debugger_tab.select(initial_load, None, [collection_dropdown, team_dropdown, agent_list_display, selected_agent_key, agent_details_display, context_inputs_group] + context_components).then(None, None, None, js=_js_debugger_listener)

    return debugger_tab 