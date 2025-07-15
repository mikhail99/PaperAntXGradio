import gradio as gr
from core.collections_manager import CollectionsManager
from core.proposal_agent_pf_dspy.main import create_research_service as create_dspy_service
from core.analysis_storage_service import AnalysisStorageService
import json

# --- Remove local service instantiation ---
# collections_manager = CollectionsManager()
# proposal_agent_service = create_dspy_service(use_parrot=False) # This is now passed in
analysis_storage_service = AnalysisStorageService()

# --- UI Helper Functions ---
def get_collection_options(collections_manager: CollectionsManager):
    """Returns a list of (name, id) for active collections."""
    return [c.name for c in collections_manager.get_all_collections() if not c.archived]

def get_collection_description(collections_manager: CollectionsManager, collection_name: str):
    """Gets the description of a collection by its ID."""
    c = collections_manager.get_collection(collection_name)
    return c.description if c else ""

# --- Gradio UI Definition ---
def create_research_plan_tab(proposal_agent_service, collections_manager: CollectionsManager):
    with gr.TabItem("🧠 Research Proposal Agent DSPY"):
        gr.Markdown("## Research Proposal Agent")
        gr.Markdown("Select a collection, provide a research direction, and let the agent generate a comprehensive proposal with your guidance.")

        # --- State Management ---
        thread_id_state = gr.State(None)
        # Using a state object for the chatbot history to manage it across generator yields
        chat_history_state = gr.State([])

        with gr.Row(equal_height=False):
            # Left Column: Inputs & Start Controls
            with gr.Column(scale=1, min_width=350):
                gr.Markdown("### 1. Start Here")
                with gr.Group():
                    collection_dropdown = gr.Dropdown(
                        choices=get_collection_options(collections_manager),
                        label="Select Collection for Literature Review",
                        value=None,
                    )
                    collection_desc_md = gr.Markdown("<i>Select a collection to see its description.</i>")
                    topic_input = gr.Textbox(lines=3, label="Research Topic / Direction", placeholder="Describe the core research topic or question...")
                    start_button = gr.Button("🚀 Begin Research Proposal", variant="primary")

                gr.Markdown("### 2. Live Results")
                with gr.Accordion("Literature Review Summary", open=True) as summary_accordion:
                    literature_summary_md = gr.Markdown("*(Waiting for agent...)*")

                with gr.Accordion("Proposed Research Plan", open=True) as plan_accordion:
                    research_plan_md = gr.Markdown("*(Waiting for literature review...)*")

                with gr.Accordion("Novelty Assessment", open=True) as novelty_accordion:
                    novelty_assessment_md = gr.Markdown("*(Waiting for plan...)*")
                
                with gr.Accordion("Full Research Proposal", open=True) as proposal_accordion:
                    final_proposal_md = gr.Markdown("*(Waiting for agent...)*", elem_id="final-proposal-card")

            # Right Column: Chat Interface & HIL Controls
            with gr.Column(scale=2):
                gr.Markdown("### 3. Agent Conversation & Your Actions")
                chatbot = gr.Chatbot(
                    [],
                    label="Agent Conversation",
                    elem_id="proposal-chatbot",
                    height=600
                )
                status_display = gr.Markdown("Status: Idle")

                # --- Human-in-the-Loop (HIL) Controls ---
                with gr.Group(visible=False) as query_review_group:
                    gr.Markdown("#### Action Required: Review Queries")
                    query_review_box = gr.Textbox(lines=5, label="Generated Search Queries")
                    with gr.Row():
                        query_regenerate_btn = gr.Button("♻️ Regenerate Queries")
                        query_accept_btn = gr.Button("👍 Accept & Continue", variant="primary")
                
                with gr.Group(visible=False) as final_review_group:
                    gr.Markdown("#### Action Required: Final Review")
                    with gr.Row():
                        final_revision_btn = gr.Button("👎 Request Revision")
                        final_approve_btn = gr.Button("✅ Approve & Finalize", variant="primary")
                
                with gr.Group(visible=False) as revision_feedback_group:
                    revision_feedback_box = gr.Textbox(lines=3, label="Provide your feedback for revision:")
                    revision_submit_btn = gr.Button("Submit Feedback", variant="primary")


        # --- Event Handler Functions ---
        def update_collection_desc(collection_name):
            desc = get_collection_description(collections_manager, collection_name)
            return gr.update(value=desc or "<i>No description available.</i>")

        def start_workflow(collection, topic, history):
            """Kicks off a new agent workflow."""
            if not collection or not topic:
                gr.Warning("Please select a collection and enter a research topic.")
                yield { status_display: gr.update(value="Status: Paused - Collection and Topic are required.") }
                return

            config = {"topic": topic, "collection_name": collection}
            history = [(f"Starting new proposal for topic: **{topic}**", "Okay, I will begin the research process. I'll ask for your input along the way.")]
            
            # All UI updates are now returned in a dictionary
            yield {
                chatbot: history,
                chat_history_state: history,
                status_display: gr.update(value="Status: Running..."),
                # Hide all HIL controls at the start
                query_review_group: gr.update(visible=False),
                final_review_group: gr.update(visible=False),
                revision_feedback_group: gr.update(visible=False),
                # Reset result panes
                literature_summary_md: "*(Waiting for agent...)*",
                research_plan_md: "*(Waiting for literature review...)*",
                novelty_assessment_md: "*(Waiting for plan...)*",
                final_proposal_md: "*(Waiting for agent...)*",
            }
            
            for update in proposal_agent_service.start_agent(config):
                # We pass all UI components to the handler so it can update them
                yield _handle_agent_update(update, history)

        def continue_workflow(current_tid, user_input, history):
            """Continues the workflow after a user interaction."""
            if not current_tid:
                return

            history.append((user_input, None)) # Show user input immediately
            yield {
                chatbot: history,
                chat_history_state: history,
                status_display: gr.update(value="Status: Running..."),
                # Hide all HIL controls while agent is working
                query_review_group: gr.update(visible=False),
                final_review_group: gr.update(visible=False),
                revision_feedback_group: gr.update(visible=False),
            }
            
            for update in proposal_agent_service.continue_agent(current_tid, user_input):
                yield _handle_agent_update(update, history)

        def _handle_agent_update(update, history):
            """Processes updates from the orchestrator and updates all relevant UI components."""
            new_tid = update.get("thread_id")
            step = update.get("step")
            state = update.get("state", {})
            
            # Start with a base set of UI updates
            ui_updates = {
                thread_id_state: new_tid,
                status_display: gr.update(value=f"Status: Running: `{step}`...")
            }

            # Update result panes based on the latest state
            if "literature_summaries" in state:
                summaries = state["literature_summaries"]
                # Format the summaries as a numbered list with proper spacing
                formatted_summaries = []
                for i, summary in enumerate(summaries):
                    # Use markdown for a clear, numbered list
                    formatted_summaries.append(f"**{i+1}.** {summary.replace('<br>', ' ')}")
                
                ui_updates[literature_summary_md] = "### Literature Summaries<br><br>" + "<br><br>---<br><br>".join(formatted_summaries)
            if "proposal_draft" in state:
                 # Use HTML <br> tags for newlines
                draft_html = state['proposal_draft'].replace('\n', '<br>')
                ui_updates[research_plan_md] = f"### Proposal Draft<br><br>{draft_html}"
            if "review_team_feedback" in state:
                feedback_json = json.dumps(state['review_team_feedback'], indent=2)
                ui_updates[novelty_assessment_md] = f"### Agent Reviews<br><br>```json<br>{feedback_json}<br>```"

            # Handle HIL pauses
            if step == "human_input_required":
                interrupt_type = update.get("interrupt_type")
                message = update.get("message")
                context = update.get("context", {})

                history.append((None, f"**PAUSED: {message}**"))
                ui_updates[status_display] = gr.update(value="Status: Paused - Waiting for your input.")
                
                if interrupt_type == "query_review":
                    queries_str = ", ".join(context.get("queries", []))
                    ui_updates[query_review_group] = gr.update(visible=True)
                    ui_updates[query_review_box] = gr.update(value=queries_str)
                
                elif interrupt_type == "final_review":
                    review = context.get('review', {})
                    rev_cycle = context.get('revision_cycle', 0)
                    review_html = f"**AI Review (Revision #{rev_cycle})**:<br>_{review.get('justification', 'No justification.')}_"
                    history.append((None, review_html))
                    ui_updates[final_review_group] = gr.update(visible=True)

            # Handle workflow completion
            elif step == "workflow_complete_node":
                final_draft_html = state.get("proposal_draft", "Proposal not available.").replace('\n', '<br>')
                completion_message = f"**Workflow Complete!** The final proposal is below."
                history.append((None, completion_message))
                ui_updates[status_display] = gr.update(value="Status: Complete!")
                ui_updates[final_proposal_md] = f"### Final Proposal<br><br>{final_draft_html}"

            ui_updates[chatbot] = history
            ui_updates[chat_history_state] = history
            return ui_updates
        
        def on_request_revision():
            """Shows the feedback input box."""
            return {
                final_review_group: gr.update(visible=False),
                revision_feedback_group: gr.update(visible=True)
            }

        # --- Connect UI Components to Functions ---
        collection_dropdown.change(
            update_collection_desc, 
            inputs=[collection_dropdown], 
            outputs=[collection_desc_md]
        )

        # Define all possible outputs for the event handlers
        all_outputs = [
            chatbot, chat_history_state, status_display, thread_id_state,
            query_review_group, final_review_group, revision_feedback_group,
            query_review_box, literature_summary_md, research_plan_md,
            novelty_assessment_md, final_proposal_md
        ]

        start_button.click(
            start_workflow,
            inputs=[collection_dropdown, topic_input, chat_history_state],
            outputs=all_outputs
        )

        query_accept_btn.click(
            continue_workflow,
            inputs=[thread_id_state, query_review_box, chat_history_state],
            outputs=all_outputs
        )

        # To fix the Gradio value error, we need to ensure the generator function
        # is the top-level callable for the event. We create specialized handlers
        # that call the main continue_workflow logic.

        def handle_query_regenerate(tid, hist):
            """Specialized handler for the query regenerate button."""
            for update in continue_workflow(tid, "!regenerate", hist):
                yield update

        def handle_final_approve(tid, hist):
            """Specialized handler for the final approve button."""
            for update in continue_workflow(tid, "approve", hist):
                yield update
        
        def handle_revision_submit(tid, feedback, hist):
            """Specialized handler for submitting revision feedback."""
            for update in continue_workflow(tid, feedback, hist):
                yield update

        query_regenerate_btn.click(
            handle_query_regenerate,
            inputs=[thread_id_state, chat_history_state],
            outputs=all_outputs
        )

        final_approve_btn.click(
            handle_final_approve,
            inputs=[thread_id_state, chat_history_state],
            outputs=all_outputs
        )
        
        final_revision_btn.click(
            on_request_revision,
            inputs=[],
            outputs=[final_review_group, revision_feedback_group]
        )
        
        revision_submit_btn.click(
            handle_revision_submit,
            inputs=[thread_id_state, revision_feedback_box, chat_history_state],
            outputs=all_outputs
        )

    # This tab is self-contained and doesn't need to return any values. 