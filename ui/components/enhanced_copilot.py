import gradio as gr
from gradio import ChatMessage
from ui.components.agent_list import generate_agent_list_html, create_js_event_listener

def build_copilot_view(tab_title: str, copilot_service, tab_id_suffix: str, state):
    """
    Creates the UI for a copilot tab, including quick actions at the top,
    and wires up all the event handling logic.
    This function must be called from within a gr.Blocks() or gr.TabItem() context.
    """
    
    # --- Define IDs and JS ---
    main_container_id = f"copilot-main-container-{tab_id_suffix}"
    selected_agent_trigger_id = f"copilot_selected_agent_trigger_{tab_id_suffix}"
    agent_list_display_id = f"agent-list-container-{tab_id_suffix}"
    chat_column_id = f"copilot-chat-column-{tab_id_suffix}"
    js_listener = create_js_event_listener(agent_list_display_id, selected_agent_trigger_id)

    # --- Define CSS ---
    # The CSS is simplified as we are removing the outer containers
    css = """
    <style>
    .quick-actions-row {
        gap: 0.5rem !important;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    .quick-action-btn {
        transition: all 0.3s ease !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        border: none !important;
        color: white !important;
    }
    .quick-action-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }
    .research-btn { background: linear-gradient(135deg, #ff6b6b, #ee5a24) !important; }
    .outline-btn { background: linear-gradient(135deg, #4834d4, #686de0) !important; }
    .search-btn { background: linear-gradient(135deg, #00d2d3, #54a0ff) !important; }
    .method-btn { background: linear-gradient(135deg, #5f27cd, #a55eea) !important; }
    .default-btn { background: linear-gradient(135deg, #8854d0, #3742fa) !important; }
    </style>
    """
    gr.HTML(css)
    
    # --- Define State ---
    agent_list = copilot_service.get_agent_list()
    initial_agent = agent_list[0] if agent_list else None
    selected_agent_name_state = gr.State(initial_agent)

    # --- UI LAYOUT ---
    # We render components sequentially. By rendering the main content first,
    # the buttons will now appear at the bottom.
    
    # 1. Main Content Row
    with gr.Row(equal_height=False, elem_id=main_container_id):
        # Left Column: Agent List
        with gr.Column(scale=1, min_width=320):
            with gr.Row():
                gr.Markdown("### Agents")
                gr.HTML("<div style='flex-grow: 1'></div>")
                with gr.Column(scale=0, min_width=120):
                    reload_button = gr.Button("Reload Agents", size="sm", elem_classes="discrete-reload-button")
            selected_agent_trigger = gr.Textbox(label="selected_agent_trigger", visible=False, elem_id=selected_agent_trigger_id)
            agent_list_display = gr.HTML(elem_id=agent_list_display_id)

        # Right Column: Chatbot
        with gr.Column(scale=3, elem_id=chat_column_id):
            chatbot = gr.Chatbot(label="My Chatbot", type="messages", avatar_images=(None, "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png"), scale=1, render_markdown=True, show_copy_button=True, placeholder=f"<strong>{tab_title}</strong><br>Ask me to help with your research proposal!")
            
            def reply(prompt, history, selected_agent_name):
                answer, flow_log = copilot_service.chat_with_agent(selected_agent_name, prompt, history)
                messages = list(flow_log)
                if not isinstance(answer, str):
                    answer = str(answer)
                messages.append(ChatMessage(role="assistant", content=answer))
                return messages

            gr.ChatInterface(fn=reply, type="messages", additional_inputs=[selected_agent_name_state], chatbot=chatbot, title=f"{tab_title} Agent with Human Review", examples=[["I want to research LLMs in education"], ["Help me research machine learning for healthcare"]])
            conversation_history_state = gr.State([])

    # 2. Quick Actions Row (now at the bottom)
    action_buttons = []
    max_buttons = 8
    with gr.Row(elem_classes=["quick-actions-row"]):
        initial_actions = copilot_service.get_quick_actions(initial_agent) if initial_agent else []
        for i in range(max_buttons):
            is_visible = i < len(initial_actions)
            action = initial_actions[i] if is_visible else {"icon": "", "label": "Hidden", "color_class": "default-btn"}
            
            color_class = action.get("color_class", "default-btn")
            
            button = gr.Button(
                f"{action['icon']} {action['label']}",
                size="sm",
                elem_classes=["quick-action-btn", color_class],
                visible=is_visible
            )
            action_buttons.append(button)

    # --- EVENT HANDLING ---
    
    # Agent selection updates buttons and agent details
    def on_select_agent(agent_name, current_selected_agent):
        button_updates = []
        if agent_name != current_selected_agent:
            new_actions = copilot_service.get_quick_actions(agent_name)
            for i in range(max_buttons):
                is_visible = i < len(new_actions)
                action = new_actions[i] if is_visible else {"icon": "", "label": "Hidden", "color_class": "default-btn"}
                # Gradio can't update elem_classes, so we just update label and visibility.
                button_updates.append(gr.update(
                    value=f"{action['icon']} {action['label']}", 
                    visible=is_visible,
                ))
        else:
            button_updates = [gr.update()] * max_buttons
        return [
            [], # chatbot
            [], # history
            agent_name,
            generate_agent_list_html(copilot_service.get_agent_details(), agent_name)
        ] + button_updates

    selected_agent_trigger.input(
        fn=on_select_agent,
        inputs=[selected_agent_trigger, selected_agent_name_state],
        outputs=[chatbot, conversation_history_state, selected_agent_name_state, agent_list_display] + action_buttons,
        queue=False
    )
    
    # Button clicks print their action
    for i, button in enumerate(action_buttons):
        def make_click_handler(btn_index):
            def handle_click(current_agent):
                if current_agent:
                    current_actions = copilot_service.get_quick_actions(current_agent)
                    if btn_index < len(current_actions):
                        print(f"Agent: {current_agent} | Button: {current_actions[btn_index]['label']}")
                return None
            return handle_click
        button.click(fn=make_click_handler(i), inputs=[selected_agent_name_state], outputs=[])

    # Initial load of agent list
    def initial_load_fn():
        return generate_agent_list_html(copilot_service.get_agent_details(), initial_agent)

    return agent_list_display, initial_load_fn, js_listener 