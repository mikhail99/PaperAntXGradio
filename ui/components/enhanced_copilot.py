import gradio as gr
from gradio import ChatMessage
from ui.components.agent_list import generate_agent_list_html, create_js_event_listener

def create_copilot_with_quick_actions(tab_title: str, copilot_service, tab_id_suffix: str, state):
    """
    Creates a copilot with quick action buttons from the agent service.
    """
    
    main_container_id = f"copilot-main-container-{tab_id_suffix}"
    selected_agent_trigger_id = f"copilot_selected_agent_trigger_{tab_id_suffix}"
    agent_list_display_id = f"agent-list-container-{tab_id_suffix}"
    chat_column_id = f"copilot-chat-column-{tab_id_suffix}"

    # Generate the dynamic JavaScript for this specific component instance
    js_listener = create_js_event_listener(agent_list_display_id, selected_agent_trigger_id)

    # CSS for button styling
    css = """
    <style>
    .quick-actions-container {
        padding: 0;
        margin: 0;
    }
    .quick-action-btn {
        margin: 0.1rem !important;
        transition: all 0.2s ease !important;
        height: 32px !important;
    }
    .quick-action-btn:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
    }
    </style>
    """

    with gr.Blocks(theme=gr.themes.Soft(), fill_height=True):
        gr.HTML(css)
        
        # Quick Actions Section - spans full width above everything
        with gr.Row():
            action_buttons_container = gr.Row()
        
        with gr.Row(equal_height=False, elem_id=main_container_id):
            with gr.Column(scale=1, min_width=320):
                with gr.Row():
                    gr.Markdown("### Agents")
                    gr.HTML("<div style='flex-grow: 1'></div>")
                    with gr.Column(scale=0, min_width=120):
                        reload_button = gr.Button("Reload Agents", size="sm", elem_classes="discrete-reload-button")

                agent_list = copilot_service.get_agent_list()
                initial_agent = agent_list[0] if agent_list else None
                selected_agent_name_state = gr.State(initial_agent)
                selected_agent_trigger = gr.Textbox(
                    label="selected_agent_trigger",
                    visible=False,
                    elem_id=selected_agent_trigger_id
                )
                agent_list_display = gr.HTML(elem_id=agent_list_display_id)

            with gr.Column(scale=3, elem_id=chat_column_id):
                chatbot = gr.Chatbot(
                    label="My Chatbot",
                    type="messages",
                    avatar_images=(
                            None,
                            "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png",
                        ),
                    scale=1,
                    render_markdown=True,
                    show_copy_button=True,
                    placeholder=f"<strong>{tab_title}</strong><br>Ask me to help with your research proposal!"
                )

                def reply(prompt, history, selected_agent_name):
                    messages = []
                    
                    # Final answer
                    answer, flow_log = copilot_service.chat_with_agent(selected_agent_name, prompt, history)

                    for msg in flow_log:
                        messages.append(msg)
                        yield messages

                    answer_msg = ChatMessage(
                        role="assistant",
                        content=answer
                    )
                    messages.append(answer_msg)
                    yield messages

                gr.ChatInterface(
                    fn=reply,
                    type="messages",
                    additional_inputs=[selected_agent_name_state],
                    chatbot=chatbot,
                    title=f"{tab_title} Agent with Human Review",
                    examples=[
                        ["I want to research LLMs in education"],
                        ["Help me research machine learning for healthcare"],
                        ["Research climate change impact on agriculture"],
                        ["Develop AI for personalized learning"]
                    ]
                )
                conversation_history_state = gr.State([])

        # Create buttons dynamically based on initial agent
        initial_actions = copilot_service.get_quick_actions(initial_agent) if initial_agent else []
        action_buttons = create_action_buttons(initial_actions, action_buttons_container, copilot_service, selected_agent_name_state)

        # --- Event Handlers ---
        def on_select_agent(agent_name, current_selected_agent):
            if agent_name == current_selected_agent:
                return [gr.update()] * (4 + len(action_buttons))  # No change
            
            # Get new actions for the selected agent
            new_actions = copilot_service.get_quick_actions(agent_name) if agent_name else []
            
            # Update button labels
            button_updates = []
            for i, button in enumerate(action_buttons):
                if i < len(new_actions):
                    # Update button with new action
                    new_label = f"{new_actions[i]['icon']} {new_actions[i]['label']}"
                    button_updates.append(gr.update(value=new_label, visible=True))
                else:
                    # Hide extra buttons if new agent has fewer actions
                    button_updates.append(gr.update(visible=False))
            
            return (
                [],  # chatbot
                [],  # conversation_history_state
                agent_name,  # selected_agent_name_state
                generate_agent_list_html(copilot_service.get_agent_details(), agent_name),  # agent_list_display
                *button_updates  # action button updates
            )

        def on_reload_config(current_selected_agent: str):
            print(f"\n--- Reloading AI Copilot Configuration for {tab_id_suffix} ---")
            copilot_service.reload()
            new_agent_list = copilot_service.get_agent_list()
            if current_selected_agent in new_agent_list:
                new_selected_agent = current_selected_agent
            else:
                new_selected_agent = new_agent_list[0] if new_agent_list else None
            new_html = generate_agent_list_html(copilot_service.get_agent_details(), new_selected_agent)
            
            # Update button labels for reloaded agent
            new_actions = copilot_service.get_quick_actions(new_selected_agent) if new_selected_agent else []
            button_updates = []
            for i, button in enumerate(action_buttons):
                if i < len(new_actions):
                    new_label = f"{new_actions[i]['icon']} {new_actions[i]['label']}"
                    button_updates.append(gr.update(value=new_label, visible=True))
                else:
                    button_updates.append(gr.update(visible=False))
            
            return (
                new_selected_agent,  # selected_agent_name_state
                new_html,  # agent_list_display
                [],  # chatbot
                [],  # conversation_history_state
                *button_updates  # action button updates
            )

        # Agent selection from JS trigger
        selected_agent_trigger.input(
            fn=on_select_agent,
            inputs=[selected_agent_trigger, selected_agent_name_state],
            outputs=[
                chatbot,
                conversation_history_state,
                selected_agent_name_state,
                agent_list_display,
                *action_buttons  # Include all action buttons in outputs
            ],
            queue=False
        )

        # Config reload logic
        reload_button.click(
            on_reload_config,
            inputs=[selected_agent_name_state],
            outputs=[
                selected_agent_name_state,
                agent_list_display,
                chatbot,
                conversation_history_state,
                *action_buttons  # Include all action buttons in outputs
            ],
            js=js_listener
        )

        # Initial load for agent details and list
        def initial_load_fn():
            initial_agent_name = agent_list[0] if agent_list else None
            html = generate_agent_list_html(copilot_service.get_agent_details(), initial_agent_name)
            return html

        # Return the display, the initial load function, and the dynamic JS listener
        return agent_list_display, initial_load_fn, js_listener

def create_action_buttons(actions, container, copilot_service, selected_agent_name_state):
    """Create action buttons from the actions list."""
    buttons = []
    
    # Create a fixed number of buttons (max we might need across all agents)
    max_buttons = 8  # Keep same max, but all in one row
    
    # Create all buttons directly in the container row
    for button_index in range(max_buttons):
        if button_index < len(actions):
            # Show button with action
            action = actions[button_index]
            with container:
                button = gr.Button(
                    f"{action['icon']} {action['label']}",
                    size="sm",
                    variant="secondary",
                    elem_classes=["quick-action-btn"],
                    visible=True
                )
        else:
            # Create hidden placeholder button
            with container:
                button = gr.Button(
                    "Hidden",
                    size="sm",
                    variant="secondary",
                    elem_classes=["quick-action-btn"],
                    visible=False
                )
        
        # Button click handler that gets current agent and button info
        def make_click_handler(btn_index):
            def handle_click(current_agent):
                if current_agent:
                    # Get current actions from service
                    current_actions = copilot_service.get_quick_actions(current_agent)
                    if btn_index < len(current_actions):
                        button_label = current_actions[btn_index]['label']
                        print(f"Agent: {current_agent} | Button: {button_label}")
                    else:
                        print(f"Agent: {current_agent} | Button: [hidden]")
                else:
                    print("No agent selected")
                return None
            return handle_click
        
        button.click(
            fn=make_click_handler(button_index),
            inputs=[selected_agent_name_state],
            outputs=[]
        )
        buttons.append(button)
    
    return buttons 