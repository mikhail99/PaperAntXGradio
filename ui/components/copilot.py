import gradio as gr
from gradio import ChatMessage
from ui.components.agent_list import generate_agent_list_html, create_js_event_listener

def create_copilot(tab_title:str, copilot_service, tab_id_suffix:str, state: gr.State, trigger: gr.Textbox):
    main_container_id = f"copilot-main-container-{tab_id_suffix}"
    selected_agent_trigger_id = trigger.elem_id
    agent_list_display_id = f"agent-list-container-{tab_id_suffix}"
    chat_column_id = f"copilot-chat-column-{tab_id_suffix}"

    # Generate the dynamic JavaScript for this specific component instance
    js_listener = create_js_event_listener(agent_list_display_id, selected_agent_trigger_id)

    with gr.Blocks(theme=gr.themes.Soft(), fill_height=True):
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
                # The trigger is now passed in, so we no longer create it here.
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
                    messages = [] # list(history)  # or however you build your history                    yield messages
                    
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

        # --- Event Handlers ---
        def on_select_agent(agent_name, current_selected_agent):
            if agent_name == current_selected_agent:
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            return (
                [],
                [],
                agent_name,
                generate_agent_list_html(copilot_service.get_agent_details(), agent_name)
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
            return (
                new_selected_agent,
                new_html,
                [],
                []
            )

        # Agent selection from JS trigger
        trigger.input(
            fn=on_select_agent,
            inputs=[trigger, selected_agent_name_state],
            outputs=[
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
            inputs=[selected_agent_name_state],
            outputs=[
                selected_agent_name_state,
                agent_list_display,
                chatbot,
                conversation_history_state
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