# Proposal Agent Integration Plan

This document outlines the steps to integrate the `proposal_agent` into the Gradio UI.

## Phase 1: Backend Refactoring

- [ ] **Create `ProposalAgentService`:** Create a new service file, `core/proposal_agent_service.py`, to manage the proposal generation graph and its execution. This will be similar to `core/paperqa_service.py`.

- [ ] **Adapt Agent for `PaperQA`:** Modify the `core/proposal_agent/graph.py`.
    - [ ] Replace the `search_and_summarize` node's logic. It should now call our existing `paperqa_service.query_documents` using the user-selected collection and research direction.
    - [ ] The result from `paperqa_service` will directly become the `literature_summary` in the agent's state.
    - [ ] Remove all ArXiv-related search logic. The agent must only use the specified collection.

- [ ] **Adjust Novelty Assessment:** Modify the `assess_plan_novelty` node in `core/proposal_agent/graph.py`.
    - [ ] Update its paper search to be limited to the user-selected collection, instead of searching globally. The `PaperSearchTool` might need a small adjustment to accept a collection name.

- [x] **Change Output to Markdown:** In `core/proposal_agent/prompts.py`, update the `write_proposal_prompt` to request output in Markdown format instead of LaTeX.

## Phase 2: Frontend Integration

- [x] **Update Research Plan UI:** Modify `ui/ui_research_plan.py`.
    - [x] Import and initialize the new `ProposalAgentService`.
    - [x] Redesign the output area. Instead of a single `gr.Markdown` for the answer, create multiple components to display the different parts of the research proposal (e.g., Summary, Plan, Novelty, Final Proposal).

- [x] **Connect UI to Service:** In `ui/ui_research_plan.py`, rewrite the `handle_ask` function.
    - [x] It should call the `ProposalAgentService` with the collection ID and the user's research question.
    - [x] As the agent runs, stream the intermediate results (like "Generating plan...", "Assessing novelty...") and final outputs to the new UI components. 