import gradio as gr
from core.proposal_agent_dspy.orchestrator import create_dspy_service
from core.collections_manager import CollectionsManager
import asyncio

def create_proposal_debugger_tab(service, collections_manager: CollectionsManager):
    """Creates the Gradio TabItem for the new DSPy Proposal Agent."""
    
    # The 'service' is now passed in, so we remove local creation.
    # service = create_dspy_service(use_parrot=True)

    with gr.TabItem("🕵️‍♂️ Proposal Agent (DSPy)") as debugger_tab:
        with gr.Blocks(theme=gr.themes.Soft()):
            # State management for the conversation
            thread_id = gr.State(None)
            
            with gr.Row(equal_height=False):
                # --- Left Column: Configuration ---
                with gr.Column(scale=1, min_width=350):
                    gr.Markdown("### 1. Configure Workflow")
                    collection_names = [c.name for c in collections_manager.get_all_collections()]
                    collection_dropdown = gr.Dropdown(
                        choices=collection_names, 
                        label="Select a Collection", 
                        value=collection_names[0] if collection_names else None,
                        interactive=True
                    )
                    topic_input = gr.Textbox(lines=2, label="Research Topic", placeholder="Enter the core research topic here...")
                    start_button = gr.Button("🚀 Start Proposal Workflow", variant="primary")

                # --- Right Column: Chat and Controls ---
                with gr.Column(scale=2):
                    gr.Markdown("### 2. Agent Conversation")
                    chatbot = gr.Chatbot([], elem_id="dspy_agent_chatbot", label="Agent Conversation", show_copy_button=True, height=600)
                    status_display = gr.Markdown("Status: Idle", elem_id="dspy_agent_status")

                    # --- Human-in-the-Loop (HIL) Controls ---
                    # These are hidden by default and shown when the agent needs input.

                    # HIL Group 1: Query Review
                    with gr.Group(visible=False) as query_review_group:
                        gr.Markdown("#### Action Required: Review Queries")
                        query_review_box = gr.Textbox(lines=5, label="Generated Search Queries")
                        with gr.Row():
                            query_regenerate_btn = gr.Button("♻️ Regenerate Queries")
                            query_accept_btn = gr.Button("👍 Accept & Continue", variant="primary")
                    
                    # HIL Group 2: Final Review
                    with gr.Group(visible=False) as final_review_group:
                        gr.Markdown("#### Action Required: Final Review")
                        with gr.Row():
                            final_revision_btn = gr.Button("👎 Request Revision")
                            final_approve_btn = gr.Button("✅ Approve & Finalize", variant="primary")
                    
                    # HIL Group 3: Revision Feedback Input
                    with gr.Group(visible=False) as revision_feedback_group:
                        revision_feedback_box = gr.Textbox(lines=3, label="Provide your feedback for revision:")
                        revision_submit_btn = gr.Button("Submit Feedback", variant="primary")


            # --- Event Handlers ---

            async def start_workflow(collection, topic):
                """Kicks off a new agent workflow."""
                if not collection or not topic:
                    yield {status_display: gr.update(value="Status: Error - Collection and Topic are required.")}
                    return

                config = {"topic": topic, "collection_name": collection}
                initial_history = [
                    (f"Starting new proposal for topic: **{topic}**", "Okay, I will begin the research process. I'll ask for your input along the way.")
                ]

                # Reset UI and start flow
                yield {
                    chatbot: initial_history,
                    status_display: gr.update(value="Status: Running..."),
                    query_review_group: gr.update(visible=False),
                    final_review_group: gr.update(visible=False),
                    revision_feedback_group: gr.update(visible=False),
                }

                # Run the agent and handle streaming updates
                async for update in service.start_agent(config):
                    yield _handle_agent_update(update, initial_history)


            async def continue_workflow(current_tid, user_input, history):
                """Continues the workflow after a user interaction."""
                if not current_tid:
                    return

                # Hide all interaction panels while the agent is thinking
                yield {
                    chatbot: history,
                    status_display: gr.update(value="Status: Running..."),
                    query_review_group: gr.update(visible=False),
                    final_review_group: gr.update(visible=False),
                    revision_feedback_group: gr.update(visible=False),
                }
                
                async for update in service.continue_agent(current_tid, user_input):
                    yield _handle_agent_update(update, history)


            def _handle_agent_update(update, history):
                """A helper to process updates from the orchestrator and update the UI accordingly."""
                new_tid = update.get("thread_id")
                step = update.get("step")
                state = update.get("state", {})
                
                # Update status and chatbot
                ui_updates = {
                    thread_id: new_tid,
                    status_display: gr.update(value=f"Status: Running: `{step}`...")
                }
                
                # HIL (Human-in-the-Loop) Handling
                if step == "human_input_required":
                    interrupt_type = update.get("interrupt_type")
                    message = update.get("message")
                    context = update.get("context", {})

                    history.append((None, f"**PAUSED: {message}**"))
                    ui_updates[chatbot] = history
                    ui_updates[status_display] = gr.update(value="Status: Paused - Waiting for your input.")
                    
                    if interrupt_type == "query_review":
                        queries_str = ", ".join(context.get("queries", []))
                        ui_updates[query_review_group] = gr.update(visible=True)
                        ui_updates[query_review_box] = gr.update(value=queries_str)
                    
                    elif interrupt_type == "final_review":
                        review_html = f"**AI Review (Revision #{context.get('revision_cycle', 0)})**:<br>"
                        review_html += f"_{context.get('review', {}).get('justification', 'No justification provided.')}_"
                        history.append((None, review_html))
                        ui_updates[chatbot] = history
                        ui_updates[final_review_group] = gr.update(visible=True)

                elif step == "workflow_complete_node":
                    final_proposal = state.get("proposal_draft", "Proposal not available.")
                    completion_message = f"**Workflow Complete!** Here is the final proposal:\n\n---\n\n{final_proposal}"
                    history.append((None, completion_message))
                    ui_updates[chatbot] = history
                    ui_updates[status_display] = gr.update(value="Status: Complete!")
                
                return ui_updates

            def on_request_revision():
                """Shows the feedback input box when user wants to revise."""
                return {
                    final_review_group: gr.update(visible=False),
                    revision_feedback_group: gr.update(visible=True)
                }

            # --- Wiring ---

            # Start Button
            start_button.click(
                start_workflow,
                inputs=[collection_dropdown, topic_input],
                outputs=[chatbot, status_display, thread_id, query_review_group, final_review_group, revision_feedback_group]
            )

            # Query Review Buttons
            query_accept_btn.click(
                continue_workflow,
                inputs=[thread_id, query_review_box, chatbot],
                outputs=[chatbot, status_display, thread_id, query_review_group, final_review_group, revision_feedback_group]
            )
            query_regenerate_btn.click(
                lambda tid, hist: continue_workflow(tid, "!regenerate", hist),
                inputs=[thread_id, chatbot],
                outputs=[chatbot, status_display, thread_id, query_review_group, final_review_group, revision_feedback_group]
            )

            # Final Review Buttons
            final_approve_btn.click(
                lambda tid, hist: continue_workflow(tid, "approve", hist),
                inputs=[thread_id, chatbot],
                outputs=[chatbot, status_display, thread_id, query_review_group, final_review_group, revision_feedback_group]
            )
            final_revision_btn.click(
                on_request_revision,
                inputs=[],
                outputs=[final_review_group, revision_feedback_group]
            )
            
            # Revision Feedback Button
            revision_submit_btn.click(
                continue_workflow,
                inputs=[thread_id, revision_feedback_box, chatbot],
                outputs=[chatbot, status_display, thread_id, query_review_group, final_review_group, revision_feedback_group]
            )
            
        return debugger_tab 