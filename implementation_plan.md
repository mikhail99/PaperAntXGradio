# Implementation Plan: Team-Based, Configurable Proposal Agent (V2)

This document outlines a plan to refactor the `proposal_agent`. The goal is to create a "semi-static" graph where the main workflow is defined in Python, but the composition of agent "teams" and their individual behaviors (prompts, tools) are loaded from external JSON files.

## Guiding Principles

-   **Configuration over Code**: The members of each agent team and their specific prompts and toolsets will be defined in JSON files.
-   **Standardized Agent Definitions**: Each agent node will be defined in a structured format, similar to a standalone agent, for clarity and consistency.
-   **Static Workflow, Dynamic Teams**: The high-level graph structure (e.g., query generation -> literature review -> proposal review) will be defined in Python for simplicity. The agents that compose the teams within this workflow will be dynamically loaded from config.

## Step 1: JSON-based Configuration

We will use two JSON files in `core/proposal_agent/` to manage the agent's configuration.

### `prompts.json`

This file will externalize all prompt templates. It remains unchanged from the previous plan.

**`core/proposal_agent/prompts.json` (Example)**
```json
{
  "generate_query_base": "You are a research assistant. Generate 3 search queries for: {topic}",
  "literature_review_local": "Summarize the key findings from the following text regarding {topic}. Text: {literature}",
  "review_proposal_feasibility": "Review this proposal for its technical feasibility. Proposal: {proposal_draft}",
  "synthesize_proposal_review": "Synthesize the following feedback into a final pass/fail grade. Feedback: {review_feedbacks}"
}
```

### `agent_config.json`

This is the core configuration file. It defines each potential node as a structured "agent" and then groups them into teams.

**`core/proposal_agent/agent_config.json` (Example)**
```json
{
  "nodes": {
    "query_generator_base": {
      "name": "Base Query Generator",
      "description": "Generates a standard set of search queries based on the research topic.",
      "prompt_key": "generate_query_base",
      "output_schema": "QueryList",
      "mcp_info": { "server_id": "proposal_agent_tools", "tools": [] }
    },
    "literature_reviewer_local": {
      "name": "Local Literature Reviewer",
      "description": "Runs a query against the local document database to find relevant information and synthesize an answer.",
      "prompt_key": "literature_review_local",
      "output_schema": "string",
      "mcp_info": {
        "server_id": "proposal_agent_tools",
        "tools": [
          {
            "name": "paperqa_service_query",
            "description": "Queries the local document database to find relevant papers and get answers.",
            "inputSchema": {
              "type": "object",
              "properties": {
                "collection_name": { "type": "string" },
                "query": { "type": "string" }
              },
              "required": ["collection_name", "query"]
            }
          }
        ]
      }
    },
    "review_feasibility": {
       "name": "Feasibility Reviewer",
       "description": "Reviews the proposal draft for technical feasibility.",
       "prompt_key": "review_proposal_feasibility",
       "output_schema": "Critique",
       "mcp_info": { "server_id": "proposal_agent_tools", "tools": [] }
    },
    "synthesize_review": {
       "name": "Review Synthesizer",
       "description": "Aggregates feedback from multiple reviewers into a final decision.",
       "prompt_key": "synthesize_proposal_review",
       "output_schema": "FinalReview",
       "mcp_info": { "server_id": "proposal_agent_tools", "tools": [] }
    }
  },
  "teams": {
    "query_generation_team": {
      "members": ["query_generator_base"],
      "aggregator": "deduplicate_queries_node" 
    },
    "literature_review_team": {
      "members": ["literature_reviewer_local"],
      "aggregator": "synthesize_literature_review_node" 
    },
    "proposal_review_team": {
      "members": ["review_feasibility"],
      "aggregator": "synthesize_review"
    }
  }
}
```
*(Note: Aggregator nodes can be simple Python functions for tasks like list deduplication, or they can be full LLM-based nodes defined in the `nodes` section).*


## Step 2: Update Agent State (`state.py`)

The `ProposalAgentState` will be updated to handle the outputs from the teams. This remains largely the same as the previous plan.

**`core/proposal_agent/state.py` (Proposed Changes)**
```python
class Critique(TypedDict):
    score: float 
    justification: str

class FinalReview(TypedDict):
    is_approved: bool
    final_critique: str

class ProposalAgentState(TypedDict):
    # ... existing fields ...
    search_queries: Annotated[List[str], operator.add]
    literature_summaries: Annotated[List[str], operator.add]
    aggregated_summary: str
    proposal_draft: str
    review_team_feedback: dict 
    final_review: FinalReview
```

## Step 3: Refactor Graph Builder (`graph.py`)

The main `graph.py` will be refactored to build the graph in a semi-static way.

1.  The Python code will define the static sequence of operations: `START` -> `QUERY_TEAM` -> `LITERATURE_TEAM` -> `PROPOSAL_TEAM` -> `END`.
2.  For each team step (e.g., `QUERY_TEAM`), the builder will read `agent_config.json` to get the list of members (e.g., `["query_generator_base"]`).
3.  It will dynamically add the nodes for those members, telling `langgraph` to run them in parallel. It will use the new structured definitions from the config to create each node's runnable (with its specific prompt and tools).
4.  It will then add the team's specified aggregator node.
5.  Finally, it will connect the aggregator of one team to the member nodes of the next team.

This is the most robust design, combining stability, clarity, and configuration flexibility. 