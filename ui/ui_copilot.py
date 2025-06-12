import gradio as gr
from core.copilot_service import CopilotService
import json

def create_copilot_tab(state, copilot_service: CopilotService):
    with gr.TabItem("🤖 AI Copilot") as copilot_tab:
        with gr.Blocks(theme=gr.themes.Soft()):
            with gr.Row(equal_height=False, elem_id="copilot-main-container"):
                with gr.Column(scale=1, min_width=250):
                    with gr.Row():
                        gr.Markdown("### Agents")
                        reload_button = gr.Button("🔄", variant="tool", elem_id="copilot-reload-button")

                    agent_list = copilot_service.get_agent_list()
                    
                    selected_agent_name = gr.Radio(
                        agent_list,
                        label="Select Agent",
                        value=agent_list[0] if agent_list else None,
                        elem_id="copilot-agent-selector"
                    )

                    with gr.Accordion("Agent Details", open=True):
                        agent_details_display = gr.Markdown("Select an agent to see details.", elem_id="agent-details-display")

                with gr.Column(scale=3, elem_id="copilot-chat-column"):
                    chatbot = gr.Chatbot(
                        [],
                        elem_id="copilot_chatbot",
                        bubble_full_width=False,
                        # height is now controlled by CSS for responsiveness
                        label="Copilot Chat",
                        show_copy_button=True
                    )
                    
                    with gr.Row():
                        chat_input = gr.Textbox(
                            scale=4,
                            show_label=False,
                            placeholder="Ask the AI copilot anything...",
                            container=False,
                        )
                        send_button = gr.Button("Send", scale=1, variant="primary")

            # --- Event Handlers ---
            def on_select_agent(agent_name):
                agent_details = copilot_service.get_agent_details(agent_name)
                if not agent_details:
                    return "No details available.", []
                
                details_md = f"**{agent_details.get('name', 'N/A')}**\n\n"
                details_md += f"*{agent_details.get('description', 'No description.')}*\n\n"
                if agent_details.get('model_prompt', ""):
                    details_md += f"**System Prompt:**\n```\n{agent_details.get('model_prompt')}\n```"
                
                return details_md, [] # returns details and clears chat history

            def on_chat_message(agent_name, message, history):
                if not message.strip():
                    # Return updates for all outputs to avoid errors
                    yield history, gr.update(), gr.update()
                    return

                history.append([message, None])
                # Disable inputs during processing
                yield history, gr.update(interactive=False), gr.update(interactive=False)
                
                if not agent_name:
                    history[-1][1] = "Please select an agent first."
                    # Re-enable inputs if no agent is selected
                    yield history, gr.update(interactive=True), gr.update(interactive=True)
                    return

                user_message = history[-1][0]
                service_history = [h for h in history[:-1] if h[1] is not None]
                
                # Stream the response
                history[-1][1] = ""
                response_generator = copilot_service.chat_with_agent(agent_name, user_message, service_history)
                for token in response_generator:
                    history[-1][1] += token
                    yield history, gr.update(interactive=False), gr.update(interactive=False)

                # Re-enable inputs after streaming is done
                yield history, gr.update(interactive=True), gr.update(interactive=True)

            def on_reload_config():
                new_agent_list = copilot_service.reload()
                new_selected_agent = new_agent_list[0] if new_agent_list else None
                
                if new_selected_agent:
                    details_md, _ = on_select_agent(new_selected_agent)
                else:
                    details_md = "No agents found after reload."

                return (
                    gr.update(choices=new_agent_list, value=new_selected_agent),
                    details_md,
                    [] # Clear chatbot
                )
            
            # --- Wiring ---
            
            # Agent selection logic
            selected_agent_name.select(
                on_select_agent,
                inputs=[selected_agent_name],
                outputs=[agent_details_display, chatbot]
            )

            # Config reload logic
            reload_button.click(
                on_reload_config,
                inputs=[],
                outputs=[selected_agent_name, agent_details_display, chatbot]
            )

            # Chat logic
            chat_submit_event = chat_input.submit(
                on_chat_message,
                [selected_agent_name, chat_input, chatbot],
                [chatbot, chat_input, send_button],
            ).then(
                lambda: gr.update(value=""),
                None,
                [chat_input],
                queue=False,
            )

            send_button.click(
                on_chat_message,
                [selected_agent_name, chat_input, chatbot],
                [chatbot, chat_input, send_button],
            ).then(
                lambda: gr.update(value=""),
                None,
                [chat_input],
                queue=False,
            )
            
            # Initial load for agent details
            def initial_load_fn():
                if agent_list:
                    return on_select_agent(agent_list[0])[0]
                return "No agents found."
            
            copilot_tab.select(initial_load_fn, None, agent_details_display)
