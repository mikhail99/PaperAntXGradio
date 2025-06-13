import gradio as gr
from core.copilot_service import CopilotService
import json
import hashlib

# Defines a Gradio-compliant way to run JavaScript for component interaction.
# The JS is defined as a string and passed to an event listener's `js` parameter.
_js_attach_listener = """
() => {
  // We wrap our code in a setTimeout to ensure it runs after Gradio has
  // finished rendering the HTML from the Python event on the page.
  setTimeout(() => {
    const container = document.getElementById('agent-list-container');
    if (!container) return;

    // Prevent attaching multiple listeners if the component is re-rendered.
    if (container.dataset.listenerAttached === 'true') return;
    container.dataset.listenerAttached = 'true';

    container.addEventListener('click', (event) => {
      const agentItem = event.target.closest('.agent-item');
      if (!agentItem) return;

      const agentName = agentItem.dataset.agentName;
      if (!agentName) return;

      const hiddenTextbox = document.querySelector('#copilot_selected_agent_trigger');
      if (!hiddenTextbox) return;

      // A gr.Textbox can be rendered as either an <input> or a <textarea>.
      // This more robust selector finds whichever one is present.
      const hiddenInput = hiddenTextbox.querySelector('input[type="text"], textarea');
      if (!hiddenInput) return;

      if (hiddenInput.value !== agentName) {
        hiddenInput.value = agentName;
        hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
      }
    });
  }, 150);
}
"""

def name_to_color(name: str) -> str:
    """Generates a deterministic, somewhat pleasant color from a string."""
    hash_object = hashlib.md5(name.encode())
    hex_dig = hash_object.hexdigest()
    # Skew towards softer, pastel-like colors by ensuring components are not too dark
    r = (int(hex_dig[0:2], 16) % 150) + 50
    g = (int(hex_dig[2:4], 16) % 150) + 50
    b = (int(hex_dig[4:6], 16) % 150) + 50
    return f"rgb({r},{g},{b})"

def generate_agent_list_html(copilot_service: CopilotService, selected_agent_name: str) -> str:
    """Generates an HTML block for the agent selection list."""
    agent_list = copilot_service.get_agent_list()
    html_items = []
    for agent_name in agent_list:
        details = copilot_service.get_agent_details(agent_name)
        if not details:
            continue

        is_selected_class = "selected" if agent_name == selected_agent_name else ""
        color = name_to_color(agent_name)
        description = details.get('description', 'No description.')
        # Escape double quotes for safe insertion into the data attribute
        safe_agent_name = agent_name.replace('"', '&quot;')
        
        # The click event is handled by a listener attached via JS, using the `data-agent-name` attribute.
        html_items.append(f"""
        <div class="agent-item {is_selected_class}" data-agent-name="{safe_agent_name}">
            <div class="agent-icon" style="background-color: {color};"></div>
            <div class="agent-text">
                <div class="agent-name">{agent_name}</div>
                <div class="agent-description">{description}</div>
            </div>
        </div>
        """)
    
    return f"<div class='agent-list-container'>{''.join(html_items)}</div>"


def create_copilot_tab(state, copilot_service: CopilotService):
    with gr.TabItem("🤖 AI Copilot") as copilot_tab:
        with gr.Blocks(theme=gr.themes.Soft()):
            with gr.Row(equal_height=False, elem_id="copilot-main-container"):
                with gr.Column(scale=1, min_width=320):
                    with gr.Row():
                        gr.Markdown("### Agents")
                        # This spacer div will grow to push the button to the right.
                        gr.HTML("<div style='flex-grow: 1'></div>")
                        with gr.Column(scale=0, min_width=120): # Wrap button in a non-expanding column
                            reload_button = gr.Button("Reload Agents", size="sm", elem_classes="discrete-reload-button")

                    agent_list = copilot_service.get_agent_list()
                    initial_agent = agent_list[0] if agent_list else None
                    
                    # State to hold the name of the currently selected agent
                    selected_agent_name_state = gr.State(initial_agent)
                    
                    # Hidden Textbox to receive selection events from JS
                    selected_agent_trigger = gr.Textbox(
                        label="selected_agent_trigger",
                        visible=False,
                        elem_id="copilot_selected_agent_trigger"
                    )

                    # UI component to display the HTML list
                    agent_list_display = gr.HTML(elem_id="agent-list-container")

                    with gr.Accordion("Agent Details", open=True):
                        agent_details_display = gr.Markdown("Select an agent to see details.", elem_id="agent-details-display")

                    with gr.Accordion("LLM Settings", open=False):
                        llm_provider_selector = gr.Radio(
                            label="LLM Provider",
                            choices=["gemini", "openai", "anthropic"],
                            value=copilot_service.llm_service.default_provider,
                            interactive=True,
                        )
                        llm_model_textbox = gr.Textbox(
                            label="Model Name",
                            value=getattr(copilot_service.llm_service, f"{copilot_service.llm_service.default_provider}_model"),
                            interactive=True,
                        )


                with gr.Column(scale=3, elem_id="copilot-chat-column"):
                    chatbot = gr.Chatbot(
                        [],
                        elem_id="copilot_chatbot",
                        bubble_full_width=False,
                        # The height is controlled by CSS for responsiveness.
                        label="Copilot Chat",
                        show_copy_button=True
                    )
                    conversation_history_state = gr.State([])
                    
                    with gr.Row():
                        chat_input = gr.Textbox(
                            scale=4,
                            show_label=False,
                            placeholder="Ask the AI copilot anything...",
                            container=False,
                        )
                        send_button = gr.Button("Send", scale=1, variant="primary")

            # --- Event Handlers ---
            def on_select_agent(agent_name, current_selected_agent):
                # If the agent is already selected, do nothing to avoid clearing the chat.
                if agent_name == current_selected_agent:
                    return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                
                agent_details = copilot_service.get_agent_details(agent_name)
                if not agent_details:
                    no_details_md = "Agent details not found."
                    return no_details_md, [], [], agent_name, generate_agent_list_html(copilot_service, agent_name)

                # --- UI Update ---
                details_md = f"**{agent_details.get('name', 'N/A')}**\n\n"
                details_md += f"*{agent_details.get('description', 'No description.')}*\n\n"
                
                if agent_details.get('model_prompt'):
                    details_md += f"**System Prompt:**\n```\n{agent_details.get('model_prompt')}\n```\n"

                mcp_info = agent_details.get('mcp_info', None)
                tools = mcp_info.get('tools', []) if mcp_info else []
                if tools:
                    details_md += "\n**Available Tools:**\n"
                    for tool in tools:
                        details_md += f"- `{tool.get('name')}`: {tool.get('description', 'No description.')}\n"
                
                # --- Console Logging ---
                print(f"\n--- Switched to Agent: {agent_name} ---")
                print(f"System Prompt: {agent_details.get('model_prompt', 'None')}")
                print(f"Tools: {json.dumps(tools, indent=2)}")
                print("------------------------------------------")
                
                return (
                    details_md, 
                    [], 
                    [], 
                    agent_name, 
                    generate_agent_list_html(copilot_service, agent_name)
                )

            def on_provider_change(provider):
                if provider == "gemini":
                    return copilot_service.llm_service.gemini_model
                elif provider == "openai":
                    return copilot_service.llm_service.openai_model
                elif provider == "anthropic":
                    return copilot_service.llm_service.anthropic_model
                return "" # Should not happen

            def on_chat_message(agent_name, message, ui_history, llm_history, provider, model):
                if not message.strip():
                    yield ui_history, gr.update(), gr.update(), llm_history
                    return

                ui_history.append([message, None])
                yield ui_history, gr.update(interactive=False), gr.update(interactive=False), llm_history

                if not agent_name:
                    ui_history[-1][1] = "Please select an agent first."
                    yield ui_history, gr.update(interactive=True), gr.update(interactive=True), llm_history
                    return

                response_generator = copilot_service.chat_with_agent(agent_name, message, llm_history, provider, model)
                
                # State for processing this turn's response
                full_assistant_message = {"role": "assistant", "content": None}
                assembled_tool_calls = []
                is_first_assistant_bubble = True
                last_bubble_was_text = False
                tool_call_ui_map = {}  # Maps tool_call_id to {'index': int, 'call': dict}

                for chunk in response_generator:
                    if chunk['type'] == 'text_chunk':
                        if not last_bubble_was_text:
                            # New text bubble needed after a tool call or at the start
                            if is_first_assistant_bubble:
                                ui_history[-1][1] = chunk['content']
                            else:
                                ui_history.append([None, chunk['content']])
                            is_first_assistant_bubble = False
                            last_bubble_was_text = True
                        else:
                            # Append to the existing text bubble
                            ui_history[-1][1] += chunk['content']
                        
                        # Assemble content for the final LLM history object
                        if full_assistant_message["content"] is None:
                            full_assistant_message["content"] = ""
                        full_assistant_message["content"] += chunk['content']

                    elif chunk['type'] == 'tool_call':
                        last_bubble_was_text = False
                        tool_call = chunk['tool_call']

                        # Assemble tool call for LLM history
                        assembled_tool_calls.append({
                            "id": tool_call["id"], "type": "function", 
                            "function": {"name": tool_call["name"], "arguments": tool_call["arguments"]}
                        })

                        # Create a new, dedicated bubble for this tool call, using HTML for a collapsible section
                        tool_md = f"""<details>
<summary>🛠️ Calling Tool: <code>{tool_call['name']}</code></summary>
<br>

**Arguments:**
```json
{json.dumps(tool_call['arguments'], indent=2)}
```
</details>"""
                        if is_first_assistant_bubble:
                            ui_history[-1][1] = tool_md
                        else:
                            ui_history.append([None, tool_md])
                        
                        is_first_assistant_bubble = False
                        tool_call_ui_map[tool_call['id']] = {'index': len(ui_history) - 1, 'call': tool_call}

                    elif chunk['type'] == 'tool_result':
                        tool_result = chunk['tool_result']
                        tool_call_id = tool_result['tool_call_id']
                        if tool_call_id in tool_call_ui_map:
                            bubble_info = tool_call_ui_map[tool_call_id]
                            bubble_index = bubble_info['index']
                            original_call = bubble_info['call']
                            
                            content = tool_result['content']
                            try:
                                # Try to pretty-print if the content is JSON
                                parsed_content = json.loads(content)
                                content_md = f"```json\n{json.dumps(parsed_content, indent=2)}\n```"
                            except (json.JSONDecodeError, TypeError):
                                content_md = f"```\n{str(content)}\n```"

                            # Re-render the bubble content with the result, making it temporarily expanded
                            tool_md_with_result = f"""<details open>
<summary>✅ Tool Call Succeeded: <code>{original_call['name']}</code></summary>
<br>

**Arguments:**
```json
{json.dumps(original_call['arguments'], indent=2)}
```

**Result:**
{content_md}
<br>🤔 Thinking...
</details>"""
                            ui_history[bubble_index][1] = tool_md_with_result
                    
                    elif chunk['type'] == 'error':
                        if ui_history[-1][1] is None: ui_history[-1][1] = ""
                        ui_history[-1][1] += f"\n\nAn error occurred: {chunk['content']}"
                        yield ui_history, gr.update(interactive=True), gr.update(interactive=True), llm_history
                        return

                    yield ui_history, gr.update(interactive=False), gr.update(interactive=False), llm_history

                # Finalize the full assistant message for LLM history
                if assembled_tool_calls:
                    full_assistant_message["tool_calls"] = assembled_tool_calls
                    if full_assistant_message["content"] is None:
                        full_assistant_message.pop("content")
                
                # Clean up tool bubbles: remove "Thinking..." and collapse details
                for bubble_info in tool_call_ui_map.values():
                    idx = bubble_info['index']
                    if idx < len(ui_history) and ui_history[idx][1]:
                        # Remove the "Thinking..." indicator
                        cleaned_md = ui_history[idx][1].replace("<br>🤔 Thinking...", "")
                        # Collapse the details view for a cleaner final state
                        cleaned_md = cleaned_md.replace("<details open>", "<details>")
                        ui_history[idx][1] = cleaned_md

                # Update the master LLM history state for the next turn
                new_llm_history = llm_history + [
                    {"role": "user", "content": message},
                    full_assistant_message
                ]

                yield ui_history, gr.update(interactive=True), gr.update(interactive=True), new_llm_history

            def on_reload_config(current_selected_agent: str):
                print("\n--- Reloading AI Copilot Configuration ---")
                copilot_service.reload()
                new_agent_list = copilot_service.get_agent_list()
                
                # Preserve selection if possible, otherwise fall back to the first agent.
                if current_selected_agent in new_agent_list:
                    new_selected_agent = current_selected_agent
                else:
                    new_selected_agent = new_agent_list[0] if new_agent_list else None

                details_md = "No agents found after reload."
                if new_selected_agent:
                    agent_details = copilot_service.get_agent_details(new_selected_agent)
                    if agent_details:
                        details_md = f"**{agent_details.get('name', 'N/A')}**\n\n"
                        details_md += f"*{agent_details.get('description', 'No description.')}*\n\n"
                        
                        if agent_details.get('model_prompt'):
                            details_md += f"**System Prompt:**\n```\n{agent_details.get('model_prompt')}\n```\n"

                        mcp_info = agent_details.get('mcp_info', None)
                        tools = mcp_info.get('tools', []) if mcp_info else []
                        if tools:
                            details_md += "\n**Available Tools:**\n"
                            for tool in tools:
                                details_md += f"- `{tool.get('name')}`: {tool.get('description', 'No description.')}\n"
                
                new_html = generate_agent_list_html(copilot_service, new_selected_agent)

                return (
                    new_selected_agent,
                    new_html,
                    details_md,
                    [], # Clear chatbot UI
                    []  # Clear LLM history state
                )
            
            # --- Wiring ---
            
            # Agent selection from JS trigger
            selected_agent_trigger.input(
                fn=on_select_agent,
                inputs=[selected_agent_trigger, selected_agent_name_state],
                outputs=[
                    agent_details_display,
                    chatbot,
                    conversation_history_state,
                    selected_agent_name_state,
                    agent_list_display
                ],
                queue=False
            )

            # Config reload logic
            reload_button.click(
                on_reload_config,
                inputs=[selected_agent_name_state], # Pass the current state in
                outputs=[
                    selected_agent_name_state,
                    agent_list_display,
                    agent_details_display,
                    chatbot,
                    conversation_history_state
                ],
                js=_js_attach_listener # Re-attach listener after HTML is reloaded
            )

            # LLM Provider change logic
            llm_provider_selector.change(
                fn=on_provider_change,
                inputs=[llm_provider_selector],
                outputs=[llm_model_textbox],
                queue=False
            )

            # Chat logic
            chat_submit_event = chat_input.submit(
                on_chat_message,
                [selected_agent_name_state, chat_input, chatbot, conversation_history_state, llm_provider_selector, llm_model_textbox],
                [chatbot, chat_input, send_button, conversation_history_state],
            ).then(
                lambda: gr.update(value=""),
                None,
                [chat_input],
                queue=False,
            )

            send_button.click(
                on_chat_message,
                [selected_agent_name_state, chat_input, chatbot, conversation_history_state, llm_provider_selector, llm_model_textbox],
                [chatbot, chat_input, send_button, conversation_history_state],
            ).then(
                lambda: gr.update(value=""),
                None,
                [chat_input],
                queue=False,
            )
            
            # Initial load for agent details and list
            def initial_load_fn():
                initial_agent_name = agent_list[0] if agent_list else None
                html = generate_agent_list_html(copilot_service, initial_agent_name)
                
                details_md = "Select an agent to see details."
                if initial_agent_name:
                    agent_details = copilot_service.get_agent_details(initial_agent_name)
                    if agent_details:
                        details_md = f"**{agent_details.get('name', 'N/A')}**\n\n"
                        details_md += f"*{agent_details.get('description', 'No description.')}*\n\n"
                        
                        if agent_details.get('model_prompt'):
                            details_md += f"**System Prompt:**\n```\n{agent_details.get('model_prompt')}\n```\n"

                        mcp_info = agent_details.get('mcp_info', None)
                        tools = mcp_info.get('tools', []) if mcp_info else []
                        if tools:
                            details_md += "\n**Available Tools:**\n"
                            for tool in tools:
                                details_md += f"- `{tool.get('name')}`: {tool.get('description', 'No description.')}\n"
                
                return html, details_md

            copilot_tab.select(
                initial_load_fn,
                None,
                [agent_list_display, agent_details_display],
                js=_js_attach_listener # Attach listener when tab is first selected
            )
