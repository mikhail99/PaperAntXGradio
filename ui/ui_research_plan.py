import gradio as gr
from core.collections_manager import CollectionsManager
from core.proposal_agent_service import ProposalAgentService
from core.analysis_storage_service import AnalysisStorageService

# --- Initialize Services ---
collections_manager = CollectionsManager()
proposal_agent_service = ProposalAgentService()
analysis_storage_service = AnalysisStorageService()

# --- UI Helper Functions ---
def get_collection_options():
    """Returns a list of (name, id) for active collections."""
    return [c.name for c in collections_manager.get_all_collections() if not c.archived]

def get_collection_description(collection_name):
    """Gets the description of a collection by its ID."""
    c = collections_manager.get_collection(collection_name)
    return c.description if c else ""

# --- Gradio UI Definition ---
def create_research_plan_tab(state):
    with gr.TabItem("🧠 Research Proposal Agent"):
        gr.Markdown("## Research Proposal Agent")
        gr.Markdown("Select a collection, provide a research direction, and start a conversation to generate a comprehensive proposal.")

        with gr.Row(equal_height=True):
            # Left Column: Inputs & Outputs
            with gr.Column(scale=1):
                with gr.Group():
                    collection_dropdown = gr.Dropdown(
                        choices=get_collection_options(),
                        label="Select Collection for Literature Review",
                        value=None,
                    )
                    collection_desc_md = gr.Markdown("<i>Select a collection to see its description.</i>")

                with gr.Accordion("1. Literature Review Summary", open=False) as summary_accordion:
                    literature_summary_md = gr.Markdown("*(Waiting for agent...)*")

                with gr.Accordion("2. Proposed Research Plan", open=False) as plan_accordion:
                    research_plan_md = gr.Markdown("*(Waiting for literature review...)*")

                with gr.Accordion("3. Novelty Assessment", open=False) as novelty_accordion:
                    novelty_assessment_md = gr.Markdown("*(Waiting for plan...)*")
                
                with gr.Accordion("4. Full Research Proposal", open=False) as proposal_accordion:
                    final_proposal_md = gr.Markdown("*(Waiting for agent...)*", elem_id="final-proposal-card")

            # Right Column: Chat Interface
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    [],
                    label="Agent Conversation",
                    elem_id="proposal-chatbot",
                    height=600
                )
                
                with gr.Row():
                    chat_input_box = gr.Textbox(
                        label="Your Research Direction or Feedback",
                        placeholder="Describe the research topic or question...",
                        lines=3,
                        scale=4
                    )
                    chat_send_btn = gr.Button("Send", variant="primary", scale=1)


        # --- Helper Functions for UI Updates ---
        def update_collection_desc(collection_name):
            desc = get_collection_description(collection_name)
            return gr.update(value=desc or "<i>No description available.</i>")

        async def handle_chat_interaction(collection_name, user_input, chat_history):
            """Main handler to run the proposal agent and stream results to the UI."""
            # Keep existing state for all outputs if validation fails
            outputs = [
                chat_history,
                user_input,
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update()
            ]
            if not collection_name or not user_input.strip():
                gr.Warning("Please select a collection and enter a research direction.")
                yield tuple(outputs)
                return

            chat_history.append((user_input, None))
            
            # Initial yield to show user message and disable inputs
            yield (
                chat_history,
                gr.update(value="", interactive=False),
                gr.update(interactive=False),
                "*(Processing...)*",
                "*(Waiting for literature review...)*",
                "*(Waiting for plan...)*",
                "*(Waiting for plan...)*",
                gr.update(open=True),
                gr.update(open=False),
                gr.update(open=False),
                gr.update(open=False),
            )
            
            research_direction = user_input
            bot_response = ""
            chat_history[-1] = (user_input, bot_response)

            final_state_for_saving = None
            try:
                # Stream results from the agent
                async for step_data in proposal_agent_service.run_agent(collection_name, research_direction):
                    step_name = step_data.get("step")
                    current_state = step_data.get("state")
                    if not step_name or not current_state:
                        continue
                    
                    # --- Prepare Update Values ---
                    updates = {
                        "chatbot": chat_history,
                        "literature_summary": gr.update(), "research_plan": gr.update(),
                        "novelty_assessment": gr.update(), "final_proposal": gr.update(),
                        "summary_accordion": gr.update(), "plan_accordion": gr.update(),
                        "novelty_accordion": gr.update(), "proposal_accordion": gr.update()
                    }

                    # Update chat with status
                    status_message = ""
                    if step_name == "run_single_query":
                        query_count = len(current_state.get('search_queries', []))
                        status_message = f"Query {query_count} complete. Reflecting on summary..."
                    else:
                        status_message = f"Running: {step_name}..."
                    
                    bot_response = f"**Status:** {status_message}"
                    chat_history[-1] = (user_input, bot_response)
                    
                    # --- Update Accordion Content ---
                    if "literature_summaries" in current_state and current_state["literature_summaries"]:
                        summaries = current_state.get("literature_summaries", [])
                        summary_md = ""
                        for i, s in enumerate(summaries, 1):
                            # Display the summary directly as Markdown
                            summary_md += f"**Summary {i}:**\n{s}\n\n---\n\n"
                        updates["literature_summary"] = summary_md if summary_md else "*(No summaries generated yet...)*"
                        if step_name == "synthesize_literature_review":
                            status_message = "Literature Review Complete. Formulating research plan..."
                            updates["plan_accordion"] = gr.update(open=True)
                            chat_history[-1] = (user_input, f"**Status:** {status_message}")

                    if "proposal_draft" in current_state and current_state["proposal_draft"]:
                        draft = current_state["proposal_draft"]
                        # Display draft directly as Markdown
                        updates["research_plan"] = draft
                        status_message = f"Proposal Draft Generated. Assessing..."
                        updates["novelty_accordion"] = gr.update(open=True)
                        chat_history[-1] = (user_input, f"**Status:** {status_message}")

                    if "review_team_feedback" in current_state and current_state["review_team_feedback"]:
                        feedback = current_state.get("review_team_feedback", {})
                        novelty_md = ""
                        for reviewer, assessment in feedback.items():
                            justification = assessment.get('justification', 'N/A')
                            novelty_md += f"### Assessment from: `{reviewer}`\n\n"
                            novelty_md += f"**Score:** {assessment.get('score', 'N/A')}\n\n"
                            # Display justification directly as Markdown
                            novelty_md += f"**Justification:**\n{justification}\n\n"
                            novelty_md += "\n\n---\n\n"
                        updates["novelty_assessment"] = novelty_md
                        
                        status_message = "Review complete. Synthesizing final decision."
                        chat_history[-1] = (user_input, f"**Status:** {status_message}")

                    if "final_review" in current_state and current_state.get("final_review"):
                        final_review = current_state["final_review"]
                        if final_review.get("is_approved"):
                             status_message = f"Proposal Approved. Process complete."
                        else:
                             status_message = f"Proposal Rejected. Looping to generate a new plan..."
                        chat_history[-1] = (user_input, f"**Status:** {status_message}")

                    if "markdown_proposal" in current_state:
                        updates["final_proposal"] = current_state["markdown_proposal"]
                        updates["proposal_accordion"] = gr.update(open=True)
                        status_message = "Proposal Complete!"
                        chat_history[-1] = (user_input, f"**Status:** {status_message}")

                    if current_state:
                        final_state_for_saving = current_state
                    
                    yield (
                        updates["chatbot"],
                        gr.update(), gr.update(),
                        updates["literature_summary"], updates["research_plan"],
                        updates["novelty_assessment"], updates["final_proposal"],
                        updates["summary_accordion"], updates["plan_accordion"],
                        updates["novelty_accordion"], updates["proposal_accordion"]
                    )

                # After the loop finishes successfully, save the final analysis
                if final_state_for_saving:
                    analysis_storage_service.save_analysis(
                        collection_name, research_direction, final_state_for_saving
                    )
                
                yield (
                    chat_history,
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update()
                )

            except Exception as e:
                import traceback
                error_str = traceback.format_exc()
                bot_response = f"**An error occurred.**\n\n```\n{error_str}\n```"
                chat_history[-1] = (user_input, bot_response)
                yield (
                    chat_history,
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update()
                )


        # --- Connect UI Components to Functions ---
        collection_dropdown.change(
            update_collection_desc, 
            inputs=[collection_dropdown], 
            outputs=[collection_desc_md]
        )

        # Create a list of outputs to avoid repetition
        all_outputs = [
            chatbot,
            chat_input_box,
            chat_send_btn,
            literature_summary_md, 
            research_plan_md, 
            novelty_assessment_md, 
            final_proposal_md,
            summary_accordion,
            plan_accordion,
            novelty_accordion,
            proposal_accordion
        ]

        chat_send_btn.click(
            handle_chat_interaction,
            inputs=[collection_dropdown, chat_input_box, chatbot],
            outputs=all_outputs
        )
        
        chat_input_box.submit(
            handle_chat_interaction,
            inputs=[collection_dropdown, chat_input_box, chatbot],
            outputs=all_outputs
        ) 