# PaperAnt DSPy Agent Workflow Examples

This document provides several examples of how the `WorkflowState` evolves as it progresses through the proposal generation pipeline defined in `orchestrator.py`.

## Scenario 1: Happy Path (Straight-through execution)

This example shows a simple, successful run from an initial topic to an approved research proposal without any revisions.

**1. Initial State**

The workflow starts with a topic and a collection name.

```json
{
  "topic": "The Impact of Large Language Models on Scientific Research",
  "collection_name": "llm_in_science_2024",
  "thread_id": "uuid-1",
  "search_queries": [],
  "literature_summaries": [],
  "knowledge_gap": null,
  "proposal_draft": "",
  "review_team_feedback": {},
  "is_approved": false,
  "revision_cycles": 0
}
```

**2. After `generate_queries` node**

The first step generates a set of search queries based on the topic.

```json
{
  "topic": "The Impact of Large Language Models on Scientific Research",
  "collection_name": "llm_in_science_2024",
  "thread_id": "uuid-1",
  "search_queries": [
    "applications of LLMs in automating systematic literature reviews",
    "biases and limitations of LLMs in scientific data analysis",
    "ethical considerations of using large language models for academic writing"
  ],
  "literature_summaries": [],
  "knowledge_gap": null,
  "proposal_draft": "",
  "review_team_feedback": {},
  "is_approved": false,
  "revision_cycles": 0
}
```

**3. After `literature_review` node**

The system uses the queries to search a document collection and produces summaries. (User approves queries in the background).

```json
{
  "topic": "The Impact of Large Language Models on Scientific Research",
  "collection_name": "llm_in_science_2024",
  "thread_id": "uuid-1",
  "search_queries": [
    "applications of LLMs in automating systematic literature reviews",
    "biases and limitations of LLMs in scientific data analysis",
    "ethical considerations of using large language models for academic writing"
  ],
  "literature_summaries": [
    "Summary 1: LLMs like GPT-3 can significantly accelerate literature screening for systematic reviews, but require human oversight to ensure accuracy...",
    "Summary 2: Current research highlights that LLMs may perpetuate existing biases present in their training data, leading to skewed interpretations in scientific analysis...",
    "Summary 3: The use of LLMs in academic writing raises concerns about plagiarism, authorship, and the potential for generating convincing but fabricated results..."
  ],
  "knowledge_gap": null,
  "proposal_draft": "",
  "review_team_feedback": {},
  "is_approved": false,
  "revision_cycles": 0
}
```

**4. After `synthesize_knowledge` node**

The summaries are synthesized to identify a novel knowledge gap.

```json
{
  "topic": "The Impact of Large Language Models on Scientific Research",
  // ... other fields remain the same
  "knowledge_gap": {
    "synthesized_summary": "While LLMs show promise in automating literature reviews and data analysis, their application is hampered by concerns about inherent biases and ethical issues like plagiarism. The current focus is on identifying these problems.",
    "knowledge_gap": "There is a lack of established frameworks for validating the outputs of LLMs in scientific contexts to ensure reproducibility and reliability, especially for sensitive domains like medical research.",
    "is_novel": true
  },
  "proposal_draft": "",
  "review_team_feedback": {},
  "is_approved": false,
  "revision_cycles": 0
}
```

**5. After `write_proposal` node**

A research proposal draft is generated to address the identified knowledge gap.

```json
{
  // ... other fields remain the same
  "proposal_draft": "Title: A Framework for the Validation of Large Language Model Outputs in Scientific Research..."
}
```

**6. After `review_proposal` node**

An AI agent reviews the draft. This populates the feedback field.

```json
{
  // ... other fields remain the same
  "review_team_feedback": {
    "ai_reviewer": {
      "score": 0.9,
      "justification": "The proposal clearly addresses a critical and novel gap. The methodology is sound, but could benefit from specifying the benchmark datasets."
    }
  }
}
```

**7. Final State (After user approval)**

The user approves the proposal, completing the workflow.

```json
{
  // ... other fields remain the same
  "is_approved": true
}
```

---

## Scenario 2: User Regenerates Queries

This example shows the flow when the user is not satisfied with the initial set of queries.

**1. Initial State => `generate_queries`**

The flow starts as before, generating an initial set of queries.

```json
{
  "topic": "Renewable Energy Storage Solutions",
  "collection_name": "energy_storage_2024",
  "thread_id": "uuid-2",
  "search_queries": [
    "advancements in lithium-ion battery technology",
    "cost-effectiveness of pumped-hydro storage",
    "hydrogen fuel cells for grid-scale energy storage"
  ],
  // ... other fields are empty/default
}
```

**2. User Input: `!regenerate`**

The workflow pauses for user review. The user decides the queries are too broad and inputs `!regenerate`. The `user_input_router` node directs the flow back to `generate_queries`.

**3. After second `generate_queries` call**

The `generate_queries` node runs again. `dspy.Suggest` ensures it produces a different output.

```json
{
  "topic": "Renewable Energy Storage Solutions",
  "collection_name": "energy_storage_2024",
  "thread_id": "uuid-2",
  "search_queries": [
    "solid-state batteries for improved safety and energy density",
    "compressed air energy storage (CAES) in underground caverns",
    "role of thermal energy storage in balancing solar power intermittency"
  ],
  // ... other fields are empty/default
}
```

**4. Execution Continues...**

The user is now satisfied with the new queries and allows the workflow to proceed to the `literature_review` step and beyond.

---

## Scenario 3: Proposal Revision Cycle

This example shows the flow when the user requests a revision after the AI review.

**1. State before User Review**

The workflow has proceeded to the point where a draft is written and an AI review is complete. The workflow pauses for the user's final decision.

```json
{
  "topic": "AI in Drug Discovery",
  "collection_name": "pharma_ai_2024",
  "thread_id": "uuid-3",
  // ... fields are populated
  "proposal_draft": "Title: Using Graph Neural Networks to Predict Protein-Ligand Binding Affinity...",
  "review_team_feedback": {
    "ai_reviewer": {
      "score": 0.8,
      "justification": "Strong proposal, but the novelty compared to existing GNN models is not sufficiently highlighted."
    }
  },
  "is_approved": false,
  "revision_cycles": 0
}
```

**2. User Input: Revision Request**

The user reads the AI's review and their own assessment and provides feedback for revision: `"The proposal is good, but please add a section comparing this approach to the top 3 existing methods and emphasize our unique contribution."`

The `user_input_router` processes this.

**3. State after `user_input_router`**

The router updates the state and sends the flow back to the `write_proposal` node.

```json
{
  // ... other fields are the same
  "proposal_draft": "Title: Using Graph Neural Networks to Predict Protein-Ligand Binding Affinity...",
  "review_team_feedback": {
    "user_review": {
      "score": 0.5,
      "justification": "The proposal is good, but please add a section comparing this approach to the top 3 existing methods and emphasize our unique contribution."
    }
  },
  "is_approved": false,
  "revision_cycles": 1
}
```

**4. After the second `write_proposal` call**

The `write_proposal` node now has access to the user's feedback in the `prior_feedback` field of its signature. It generates a new, improved draft.

```json
{
  // ...
  "proposal_draft": "Title: A Novel Graph Neural Network Architecture for Predicting Protein-Ligand Binding Affinity...\n\n... (new section on comparison with existing methods) ...",
  "review_team_feedback": {
    "user_review": { ... } // from previous step
  },
  "is_approved": false,
  "revision_cycles": 1
}
```

**5. Execution Continues...**

The revised draft then goes through the `review_proposal` node again, and the user gets another chance to approve it. This cycle can continue until the proposal is approved. 