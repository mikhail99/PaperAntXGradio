import gradio as gr
from core.collections_manager import CollectionsManager
from core.proposal_agent_service import ProposalAgentService

# --- Initialize Services ---
collections_manager = CollectionsManager()
proposal_agent_service = ProposalAgentService()

# --- UI Helper Functions ---
def get_collection_options():
    """Returns a list of (name, id) for active collections."""
    return [(c.name, c.id) for c in collections_manager.get_all_collections() if not c.archived]

def get_collection_description(collection_id):
    """Gets the description of a collection by its ID."""
    c = collections_manager.get_collection(collection_id)
    return c.description if c else ""

# --- Gradio UI Definition ---
def create_research_plan_tab(state):
    with gr.TabItem("🧠 Research Proposal Agent"):
        gr.Markdown("## Research Proposal Agent")
        gr.Markdown("Select a collection and provide a research direction to generate a comprehensive proposal.")

        with gr.Row():
            # Left Column: Inputs
            with gr.Column(scale=1):
                collection_dropdown = gr.Dropdown(
                    choices=get_collection_options(),
                    label="Select Collection for Literature Review",
                    value=None,
                )
                collection_desc_md = gr.Markdown("<i>Select a collection to see its description.</i>")
                
            # Right Column: Inputs
            with gr.Column(scale=1):
                research_direction_box = gr.Textbox(
                    label="Your Research Direction",
                    placeholder="Describe the research topic or question...",
                    lines=3
                )
                start_agent_btn = gr.Button("🚀 Generate Research Proposal", variant="primary")

        # --- Output Area ---
        with gr.Row():
            with gr.Column():
                gr.Markdown("---")
                status_md = gr.Markdown("Status: **Idle**", visible=True)
                
                with gr.Accordion("1. Literature Review Summary", open=False) as summary_accordion:
                    literature_summary_md = gr.Markdown("*(Waiting for agent...)*")
                
                with gr.Accordion("2. Proposed Research Plan", open=False) as plan_accordion:
                    research_plan_md = gr.Markdown("*(Waiting for agent...)*")

                with gr.Accordion("3. Novelty Assessment", open=False) as novelty_accordion:
                    novelty_assessment_md = gr.Markdown("*(Waiting for agent...)*")

                with gr.Accordion("4. Full Research Proposal", open=False) as proposal_accordion:
                    final_proposal_md = gr.Markdown("*(Waiting for agent...)*", elem_id="final-proposal-card")
        
        # --- Helper Functions for UI Updates ---
        def update_collection_desc(collection_id):
            desc = get_collection_description(collection_id)
            return gr.update(value=desc or "<i>No description available.</i>")

        async def handle_generate_proposal(collection_id, research_direction):
            """Main handler to run the proposal agent and stream results to the UI."""
            if not collection_id or not research_direction.strip():
                yield {status_md: gr.update(value="Status: **Error** - Please select a collection and enter a research direction.")}
                return

            # Reset UI components
            yield {
                status_md: gr.update(value="Status: **Starting Agent...**", visible=True),
                literature_summary_md: "*(Processing...)*",
                research_plan_md: "*(Waiting for literature review...)*",
                novelty_assessment_md: "*(Waiting for plan...)*",
                final_proposal_md: "*(Waiting for plan...)*",
                summary_accordion: gr.update(open=True),
                plan_accordion: gr.update(open=False),
                novelty_accordion: gr.update(open=False),
                proposal_accordion: gr.update(open=False),
            }

            try:
                # Stream results from the agent
                async for step_result in proposal_agent_service.run_agent(collection_id, research_direction):
                    updates = {}
                    
                    if "literature_summary" in step_result:
                        updates[literature_summary_md] = step_result["literature_summary"]
                        updates[status_md] = "Status: **Literature Review Complete.** Formulating research plan..."
                        updates[plan_accordion] = gr.update(open=True)

                    if "research_plan" in step_result:
                        updates[research_plan_md] = step_result["research_plan"]
                        updates[status_md] = "Status: **Research Plan Generated.** Assessing novelty..."
                        updates[novelty_accordion] = gr.update(open=True)

                    if "novelty_assessment" in step_result:
                        assessment = step_result["novelty_assessment"]
                        novelty_text = f"**Is the plan novel?** {assessment.is_novel}\n\n"
                        novelty_text += f"**Justification:**\n{assessment.justification}\n\n"
                        if assessment.similar_papers:
                            novelty_text += "**Similar Papers Found:**\n" + "\n".join(f"- {p.get('title', 'N/A')}" for p in assessment.similar_papers)
                        updates[novelty_assessment_md] = novelty_text
                        updates[status_md] = "Status: **Novelty Assessed.** Writing full proposal..."
                        updates[proposal_accordion] = gr.update(open=True)

                    if "markdown_proposal" in step_result:
                        updates[final_proposal_md] = step_result["markdown_proposal"]
                        updates[status_md] = "Status: **Proposal Complete!**"
                    
                    if updates:
                        yield updates

            except Exception as e:
                import traceback
                error_str = traceback.format_exc()
                yield {status_md: f"Status: **An error occurred.**\n\n```\n{error_str}\n```"}


        # --- Connect UI Components to Functions ---
        collection_dropdown.change(
            update_collection_desc, 
            inputs=[collection_dropdown], 
            outputs=[collection_desc_md]
        )

        start_agent_btn.click(
            handle_generate_proposal,
            inputs=[collection_dropdown, research_direction_box],
            outputs=[
                status_md, 
                literature_summary_md, 
                research_plan_md, 
                novelty_assessment_md, 
                final_proposal_md,
                summary_accordion,
                plan_accordion,
                novelty_accordion,
                proposal_accordion
            ]
        ).then(
            lambda: "", 
            None, 
            research_direction_box
        ) 