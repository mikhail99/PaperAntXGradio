import gradio as gr
from core.collections_manager import CollectionsManager
from core.proposal_agent_service import ProposalAgentService
from core.analysis_storage_service import AnalysisStorageService
import json

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


        # --- State Management ---
        thread_id_state = gr.State(None)
        agent_state_store = gr.State({})

        # --- Helper Functions for UI Updates ---
        def update_collection_desc(collection_name):
            desc = get_collection_description(collection_name)
            return gr.update(value=desc or "<i>No description available.</i>")

        async def handle_chat_interaction(collection_name, user_input, chat_history, thread_id, agent_state):
            """Main handler to run the proposal agent and stream results to the UI."""
            
            # --- Initialize Local State for UI ---
            live_agent_state = agent_state.copy()
            summary_md = literature_summary_md.value
            plan_md = research_plan_md.value
            novelty_md = novelty_assessment_md.value
            final_proposal_md_val = final_proposal_md.value

            if (thread_id is None) and (not collection_name or not user_input.strip()):
                gr.Warning("Please select a collection and enter a research direction.")
                yield chat_history, gr.update(), gr.update(), thread_id, live_agent_state, summary_md, plan_md, novelty_md, final_proposal_md_val
                return

            is_resuming = thread_id is not None
            
            chat_history.append((user_input, None))
            
            # On first run, clear any old state
            if not is_resuming:
                live_agent_state = {}

            # Disable inputs during processing
            yield chat_history, gr.update(value="", interactive=False), gr.update(interactive=False), thread_id, live_agent_state, gr.update(), gr.update(), gr.update(), gr.update()
            
            agent_stream = None
            try:
                # Decide which service method to call
                if is_resuming:
                    agent_stream = proposal_agent_service.continue_agent(thread_id, user_input)
                else:
                    start_config = {
                        "topic": user_input,
                        "collection_name": collection_name,
                        "local_papers_only": True # Or get from UI
                    }
                    agent_stream = proposal_agent_service.start_agent(start_config)

                # --- Main Streaming Loop ---
                async for step_data in agent_stream:
                    step_name = step_data.get("step")
                    partial_state = step_data.get("state", {})
                    
                    if not step_name: continue

                    if isinstance(partial_state, dict):
                        live_agent_state.update(partial_state)
                    
                    # Store thread_id when the agent provides it
                    if "thread_id" in step_data:
                        thread_id = step_data["thread_id"]
                    
                    # --- UI UPDATES based on live state ---
                    if "literature_summaries" in live_agent_state:
                        summaries = live_agent_state["literature_summaries"]
                        summary_md = "### Literature Summaries\n\n" + "\n\n---\n\n".join(summaries)
                    
                    if "proposal_draft" in live_agent_state:
                        plan_md = f"### Proposal Draft\n\n{live_agent_state['proposal_draft']}"

                    if "review_team_feedback" in live_agent_state:
                        # This could be more detailed, for now just show it exists
                        novelty_md = f"### Agent Reviews\n\n```json\n{json.dumps(live_agent_state['review_team_feedback'], indent=2)}\n```"

                    if "final_review" in live_agent_state and live_agent_state["final_review"].get("is_approved"):
                         final_proposal_md_val = f"### Final Proposal\n\n{live_agent_state.get('proposal_draft', 'Error: Draft not found.')}"

                    # --- Determine Bot's Response ---
                    bot_display_content = f"**Status:** Running: `{step_name}`..."

                    if step_name == "human_input_required":
                        paused_on_node = step_data.get("paused_on")
                        state_at_pause = step_data.get("state", {})

                        if paused_on_node == "human_query_review_node":
                            queries = state_at_pause.get('search_queries', [])
                            query_str = "\n".join(f"- `{q}`" for q in queries)
                            bot_display_content = f"✅ **Queries Generated.** The agent plans to search for the following:\n{query_str}\n\nPlease provide a new query to use instead, or type 'continue' to approve."
                        elif paused_on_node == "human_insight_review_node":
                            gap = state_at_pause.get('knowledge_gap', {}).get('knowledge_gap', 'N/A')
                            bot_display_content = f"✅ **Literature Synthesized.** The agent identified this knowledge gap:\n\n*'{gap}'*\n\nPlease provide a refined knowledge gap, or type 'continue' to approve."
                        elif paused_on_node == "human_review_node":
                            bot_display_content = "✅ **AI Review Complete.** Please provide your feedback, or type 'continue' to approve."
                        
                        chat_history[-1] = (user_input, bot_display_content)
                        break # Exit loop, re-enable inputs

                    elif step_name == "done":
                        bot_display_content = "✅ **Process Finished.** The final proposal is below."
                        chat_history[-1] = (user_input, bot_display_content)
                        thread_id = None # Clear for next run
                        break # Exit loop, re-enable inputs

                    # --- Display In-Progress Updates ---
                    chat_history[-1] = (user_input, bot_display_content)
                    
                    # Yield partial updates to the chatbot and accordions
                    yield chat_history, gr.update(), gr.update(), thread_id, live_agent_state, summary_md, plan_md, novelty_md, final_proposal_md_val

            except Exception as e:
                import traceback
                error_str = traceback.format_exc()
                chat_history[-1] = (user_input, f"**An error occurred.**\n\n```\n{error_str}\n```")
                thread_id = None
            
            finally:
                # This block ALWAYS runs, ensuring the UI is re-enabled.
                yield chat_history, gr.update(interactive=True), gr.update(interactive=True), thread_id, live_agent_state, summary_md, plan_md, novelty_md, final_proposal_md_val


        # --- Connect UI Components to Functions ---
        collection_dropdown.change(
            update_collection_desc, 
            inputs=[collection_dropdown], 
            outputs=[collection_desc_md]
        )

        chat_submit_args = {
            "fn": handle_chat_interaction,
            "inputs": [collection_dropdown, chat_input_box, chatbot, thread_id_state, agent_state_store],
            "outputs": [
                chatbot, chat_input_box, chat_send_btn, thread_id_state, agent_state_store,
                literature_summary_md, research_plan_md, novelty_assessment_md, final_proposal_md
            ]
        }
        chat_input_box.submit(**chat_submit_args)
        chat_send_btn.click(**chat_submit_args)

    # This tab is self-contained and doesn't need to return any values. 