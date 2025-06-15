# KAG: Iterative Implementation Plan

This document breaks down the Knowledge-Augmented Generation (KAG) architecture into a series of concrete, verifiable iterations. The goal is to incrementally build the new system, ensuring each component is functional before proceeding to the next.

---

## Iteration 1: Knowledge Base Foundation & `Paper-QA` Integration

**Goal:** Establish the core components of the Knowledge Base (KB) and replace the existing PDF processing pipeline with `Paper-QA` for document ingestion and indexing.

-[x] Create a new directory structure `algorithms/paper2code_kag/` to house the new system.
-[x] Add `paper-qa` to `requirements.txt`.
-[x] Create `utils/knowledge_base.py` to manage the KB components:
    -[x] A class to handle saving/loading the `paperqa.Docs` object.
    -[x] A class for the Abstractions & Connections DB (can use SQLite or a managed JSON file).
    -[x] A class for the Implementation Document Store (can be a directory of markdown files with a manifest).
-[x] Create `0_setup_kb_flow.py` to initialize an empty Knowledge Base.
-[x] Create `1_ingest_paper_flow.py`:
    -[x] A node that takes a PDF path, adds it to the `paperqa.Docs` object, and saves the updated `Docs` object.
-[ ] **Validation:**
    -[ ] Run `0_setup_kb_flow.py`, verify that the empty KB artifacts are created.
    -[ ] Run `1_ingest_paper_flow.py` with a test paper.
    -[ ] Verify the `paperqa.Docs` object is saved and can be loaded.
    -[ ] Programmatically check that the test paper has been added to the `Docs` index.

---

## Iteration 2: Bootstrapping the Abstractions Knowledge Base

**Goal:** Populate the Abstractions & Connections DB by leveraging the existing planning logic on a corpus of seed papers. This will create the initial "collective intelligence."

-[x] Create a "bootstrap" flow: `2_bootstrap_kb_flow.py`.
-[x] This flow iterates through a directory of seed PDFs and, for each one:
    -[x] Ingests it into the `Paper-QA` index using the flow from Iteration 1.
    -[x] Runs the *existing* abstraction (`1.1_abstraction_planning_flow.py`) and connection (`connection_planning_flow.py`) planning flows.
    -[x] A new node, `UpdateAbstractionsDBNode`, parses the resulting `abstraction_plan.json` and `connection_plan.json`.
    -[x] The node then populates the global Abstractions & Connections DB, linking each entry to the paper's ID in the `Paper-QA` index.
-[ ] **Validation:**
    -[ ] Process at least 3-5 seed papers through the bootstrap flow.
    -[ ] Inspect the Abstractions & Connections DB. Verify that it contains concepts from all processed papers.
    -[ ] Verify that abstractions are correctly linked back to their source paper ID.

---

## Iteration 3: Plan Seeding via Knowledge Retrieval (KAG Stages 1 & 2)

**Goal:** Implement the "Retrieve" phase of the KAG workflow. For a new paper, find similar work and seed a draft plan using the Knowledge Base.

-[x] Create `seed_plan_nodes.py`:
    -[x] `FindSimilarPapersNode`: Takes a new paper's abstract, queries the `Paper-QA` index, and returns the top N most similar indexed papers.
    -[x] `RetrieveKnownAbstractionsNode`: For each section of the new paper, performs a semantic search against the `Paper-QA` index to find similar chunks from *all* indexed papers. It then retrieves the known abstractions/connections associated with those chunks from the Abstractions DB.
    -[x] `AssembleDraftPlanNode`: Collates the retrieved concepts into a structured `draft_plan.json`.
-[x] Create `seed_plan_flow.py` to orchestrate these nodes.
-[ ] **Validation:**
    -[ ] Run the flow on a new paper that is conceptually similar to one of the seed papers.
    -[ ] Verify that `FindSimilarPapersNode` correctly identifies the known similar paper(s).
    -[ ] Check that `RetrieveKnownAbstractionsNode` successfully pulls relevant, pre-existing concepts from the DB.
    -[ ] Inspect the output `draft_plan.json` to ensure it's well-formed and contains the seeded concepts.

---

## Iteration 4: LLM-Powered Plan Refinement (KAG Stage 3)

**Goal:** Implement the "Refine" phase. Use an LLM to validate the seeded draft plan against the new paper, correcting it and identifying novel concepts.

-[ ] Create `4_refine_plan_nodes.py`:
    -[ ] `RefinePlanNode`: Takes the new paper's text and the `draft_plan.json`.
    -[ ] This node constructs a detailed prompt for an LLM: *"Here is a new paper and a draft implementation plan based on similar work. Please validate this plan... Correct inaccuracies, fill in missing details, and identify any truly novel concepts."*
    -[ ] The node parses the LLM's response into a final `refined_plan.json`.
-[ ] Create `4_refine_plan_flow.py` to orchestrate the refinement.
-[ ] **Validation:**
    -[ ] Run the flow using a new paper that contains at least one key concept *not* present in the seed papers.
    -[ ] Verify that the LLM preserves the correct parts of the draft plan.
    -[ ] Verify that the LLM successfully identifies and adds the novel concept to the plan.
    -[ ] Check that the output `refined_plan.json` is well-structured and complete.

---

## Iteration 5: Document Generation & KB Reinforcement (KAG Stages 4 & 5)

**Goal:** Generate the final implementation document and "close the loop" by reinforcing the Knowledge Base with the new findings.

-[ ] Create `5_generate_and_reinforce_nodes.py`:
    -[ ] `GenerateDocumentNode`: Reuses/adapts the summarization logic from the original pipeline (Iterations 5-7). It takes the `refined_plan.json` and the paper text to produce the final `.md` document. It should prioritize using existing high-quality summaries from the KB where possible.
    -[ ] `ReinforceKnowledgeBaseNode`:
        -[ ] Parses the `refined_plan.json` and adds any new or updated abstractions/connections to the Abstractions DB.
        -[ ] Saves the final `.md` document to the Implementation Document Store.
-[ ] Create `5_generate_and_reinforce_flow.py`.
-[ ] **Validation:**
    -[ ] Run the entire KAG pipeline end-to-end on a new paper.
    -[ ] Verify the final `.md` document is generated correctly.
    -[ ] Inspect all three components of the Knowledge Base (`Paper-QA` index, Abstractions DB, Document Store) to confirm they have been updated with the information from the new paper.

---

## Iteration 6: Integration, Testing, and Finalization

**Goal:** Integrate all stages into a single master pipeline, add comprehensive testing, and clean up the codebase.

-[ ] Create `main_kag.py` to orchestrate the full 5-stage KAG pipeline.
-[ ] Add comprehensive unit and integration tests for all nodes and flows.
-[ ] Refactor shared utilities in `kag_utils/` for clarity and robustness.
-[ ] Update `README.md` with instructions for the new KAG pipeline.
-[ ] **Validation:**
    -[ ] All tests pass.
    -[ ] The full pipeline runs end-to-end on multiple different test papers.
    -[ ] The "flywheel" effect is demonstrable: processing a second, similar paper is significantly faster and/or produces a higher quality result. 