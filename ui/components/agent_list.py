import gradio as gr
import hashlib
from typing import Dict


# Defines a Gradio-compliant way to run JavaScript for component interaction.
# The JS is defined as a string and passed to an event listener's `js` parameter.
_js_attach_listener = """
() => {
  // We wrap our code in a setTimeout to ensure it runs after Gradio has
  // finished rendering the HTML from the Python event on the page.
  setTimeout(() => {
    const container = document.getElementById('agent-list-container-business');
    if (!container) return;

    // Prevent attaching multiple listeners if the component is re-rendered.
    if (container.dataset.listenerAttached === 'true') return;
    container.dataset.listenerAttached = 'true';

    container.addEventListener('click', (event) => {
      const agentItem = event.target.closest('.agent-item');
      if (!agentItem) return;

      const agentName = agentItem.dataset.agentName;
      if (!agentName) return;

      const hiddenTextbox = document.querySelector('#copilot_selected_agent_trigger_business');
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

def generate_agent_list_html(agent_details: Dict[str, Dict[str, str]], selected_agent_name: str) -> str:
    """Generates an HTML block for the agent selection list and details of the selected agent."""
    agent_list = agent_details.keys()
    html_items = []
    for agent_name in agent_list:
        short_description = agent_details.get(agent_name).get("short_description")
        is_selected_class = "selected" if agent_name == selected_agent_name else ""
        color = name_to_color(agent_name)
        safe_agent_name = agent_name.replace('"', '&quot;')
        html_items.append(f"""
        <div class="agent-item {is_selected_class}" data-agent-name="{safe_agent_name}">
            <div class="agent-icon" style="background-color: {color};"></div>
            <div class="agent-text">
                <div class="agent-name">{agent_name}</div>
                <div class="agent-description">{short_description}</div>
            </div>
        </div>
        """)

    # Add selected agent details at the bottom
    selected_details = agent_details.get(selected_agent_name, {})
    full_desc = selected_details.get("full_description", "No description available.")
    tools = selected_details.get("tools", [])
    tools_html = ""
    if tools:
        tools_html = "<ul>" + "".join(
            f"<li><b>{tool.get('name')}</b>: {tool.get('description', '')}</li>" for tool in tools
        ) + "</ul>"

    selected_agent_info = f"""
    <div class="selected-agent-info" style="margin-top: 1em; padding: 1em; border-top: 1px solid #eee;">
        <h4>About <span style="color:{name_to_color(selected_agent_name)}">{selected_agent_name}</span></h4>
        <div>{full_desc}</div>
        <div><b>Tools:</b> {tools_html if tools else 'None'}</div>
    </div>
    """

    return f"<div class='agent-list-container'>{''.join(html_items)}{selected_agent_info}</div>"
