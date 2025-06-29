print("--- Importing: core.proposal_agent.graph ---")
import json
from pathlib import Path
from typing import Literal, List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from core.paperqa_service import PaperQAService
from core.proposal_agent.state import ProposalAgentState, QueryList, KnowledgeGap, Critique, FinalReview

# --- Configuration ---

# Load JSON configs
CONFIG_PATH = Path(__file__).parent
with open(CONFIG_PATH / "agent_config.json", "r") as f:
    agent_config = json.load(f)
with open(CONFIG_PATH / "prompts.json", "r") as f:
    prompts = json.load(f)

# Map schema names from config to actual Python classes
OUTPUT_SCHEMAS = {
    "QueryList": QueryList,
    "KnowledgeGap": KnowledgeGap,
    "Critique": Critique,
    "FinalReview": FinalReview,
    "string": None # For simple text outputs
}

MAX_PROPOSAL_REVISIONS = 3

# --- Lazy Service and Model Initializer ---
# We store the instances here so they are only created once.
_json_llm = None
_text_llm = None
_paperqa_service = None

def get_services():
    """Lazily initializes and returns service instances."""
    global _json_llm, _text_llm, _paperqa_service
    if _json_llm is None:
        print("--- Initializing json_llm ---")
        _json_llm = ChatOllama(
            model="gemma3:4b", 
            format="json", 
            temperature=0.7,
            base_url="http://localhost:11434" # Force use of main daemon
        )
    if _text_llm is None:
        print("--- Initializing text_llm ---")
        _text_llm = ChatOllama(
            model="gemma3:4b", 
            temperature=0.7,
            base_url="http://localhost:11434" # Force use of main daemon
        )
    if _paperqa_service is None:
        print("--- Initializing PaperQAService ---")
        _paperqa_service = PaperQAService()
    return _json_llm, _text_llm, _paperqa_service

# --- Node Factory (Creates node functions from config) ---

def create_llm_node(node_name: str, node_config: Dict[str, Any]):
    """Factory function to create a generic LLM-based graph node."""
    
    prompt_template = prompts[node_config["prompt_key"]]
    prompt = ChatPromptTemplate.from_template(prompt_template)
    output_schema = OUTPUT_SCHEMAS.get(node_config["output_schema"])

    async def _node(state: ProposalAgentState) -> Dict[str, Any]:
        print(f"--- Running node: {node_name} ---")
        
        # Lazily get services on first node run
        json_llm, text_llm, paperqa_service = get_services()

        # Use a structured output model if a schema is specified
        if output_schema:
            llm = json_llm.with_structured_output(output_schema)
        else:
            llm = text_llm
            
        chain = prompt | llm
        
        # This is a single, unified block to handle special logic for different nodes.
        if node_name == "literature_reviewer_local":
            # This node now directly uses the output from PaperQA to preserve citations.
            human_input = state.get("human_feedback")
            query_to_run_list = state.get('search_queries', [])
            query_to_run = query_to_run_list[-1] if query_to_run_list else state.get("topic")

            if human_input and human_input.strip().lower() != 'continue':
                query_to_run = human_input
                print(f"--- Overriding search with human-provided query: '{query_to_run}' ---")
            else:
                print(f"--- Proceeding with approved query: '{query_to_run}' ---")

            collection_name = state['collection_name']
            
            print(f"--- Node '{node_name}' is calling a tool: paperqa_service.query_documents ---")
            response = await paperqa_service.query_documents(collection_name, query_to_run)
            
            # Directly use the answer with citations, bypassing the extra LLM call.
            summary_with_citations = response.get("answer_text", f"PaperQA did not return an answer for query: {query_to_run}")
            
            return {
                "literature_summaries": state.get("literature_summaries", []) + [summary_with_citations],
                "human_feedback": None # Clear feedback after use
            }
        elif node_name == "review_novelty":
            # This node needs a specific part of the state (the aggregated summary)
            input_data = {
                "proposal_draft": state["proposal_draft"],
                "aggregated_summary": state["knowledge_gap"]["synthesized_summary"]
            }
            print(f"--- Input for {node_name}: {input_data} ---")
            result = chain.invoke(input_data)
        elif node_name == "synthesize_review":
            # This node needs the dictionary of feedback.
            input_data = {"review_feedbacks": state["review_team_feedback"]}
            print(f"--- Input for {node_name}: {input_data} ---")
            result = chain.invoke(input_data)
        elif node_name == "query_generator_base":
            # This node needs the topic and any prior queries or feedback.
            topic = state.get("topic", "")
            feedback = state.get("human_feedback") or ""

            # If human feedback is just 'continue', ignore it and stick to the original topic.
            # Otherwise, the feedback can override the topic for a new query direction.
            if feedback.strip().lower() == 'continue':
                final_topic = topic
            else:
                final_topic = feedback if feedback else topic

            input_data = {
                "topic": final_topic,
                "search_queries": state.get("search_queries", []),
                "human_feedback": feedback
            }
            print(f"--- Input for {node_name}: {input_data} ---")
            result = chain.invoke(input_data)
        elif node_name == "formulate_plan":
            # This node is used for both initial creation and revision.

            # --- Truncation Fix ---
            # This node combines multiple potentially large inputs, which can crash the LLM.
            # We convert complex objects to strings and truncate them before sending to the prompt.
            TRUNCATION_LIMIT = 1500

            kg_str = str(state.get("knowledge_gap", {}))
            if len(kg_str) > TRUNCATION_LIMIT:
                print(f"--- Truncating knowledge_gap from {len(kg_str)} to {TRUNCATION_LIMIT} chars ---")
                kg_str = kg_str[:TRUNCATION_LIMIT] + "..."

            draft_str = state.get("proposal_draft", "")
            if len(draft_str) > TRUNCATION_LIMIT:
                print(f"--- Truncating proposal_draft from {len(draft_str)} to {TRUNCATION_LIMIT} chars ---")
                draft_str = draft_str[:TRUNCATION_LIMIT] + "..."

            reviews_str = str(state.get("review_team_feedback", {}))
            if len(reviews_str) > TRUNCATION_LIMIT:
                print(f"--- Truncating review_team_feedback from {len(reviews_str)} to {TRUNCATION_LIMIT} chars ---")
                reviews_str = reviews_str[:TRUNCATION_LIMIT] + "..."

            input_data = {
                "knowledge_gap": kg_str,
                "proposal_draft": draft_str,
                "review_team_feedback": reviews_str,
                "human_feedback": state.get("human_feedback") or ""
            }
            print(f"--- Input for {node_name} (with truncated inputs) ---")
            result = chain.invoke(input_data)
        elif node_name == "synthesize_literature_review":
            # This node only needs the topic and the collected summaries.
            summaries = state.get("literature_summaries", [])
            # Join summaries into a single block of text for the prompt.
            full_summary_text = "\n\n---\n\n".join(summaries)

            # --- Truncation Fix ---
            TRUNCATION_LIMIT = 8000
            if len(full_summary_text) > TRUNCATION_LIMIT:
                print(f"--- Truncating summaries from {len(full_summary_text)} to {TRUNCATION_LIMIT} chars ---")
                full_summary_text = full_summary_text[:TRUNCATION_LIMIT]

            input_data = {
                "topic": state.get("topic", ""),
                "literature_summaries": full_summary_text
            }
            print(f"--- Input for {node_name} (summaries length: {len(full_summary_text)}) ---")
            result = chain.invoke(input_data)
        else:
            # Fallback for nodes that don't need special input handling (e.g., synthesize_literature_review)
            print(f"--- Input for {node_name}: (full state) ---")
            result = chain.invoke(state)

        print(f"--- Raw output from {node_name}: {result} ---")

        # This logic routes the output of the LLM call to the correct key in the state
        return_value = {}
        if node_name == "query_generator_base":
             return_value = {"search_queries": result['queries']}
        elif node_name == "literature_reviewer_local":
            # This logic is now handled inside the node's main body to avoid complexity.
            # The node now returns directly.
            return_value = {} # Should not be reached
        elif node_name == "synthesize_literature_review":
             return_value = {"knowledge_gap": result}
        elif node_name == "formulate_plan":
            content = result.content if hasattr(result, 'content') else result['content']
            current_cycles = state.get("proposal_revision_cycles", 0)
            return_value = {
                "proposal_draft": content,
                "human_feedback": None, # Clear feedback after use
                "proposal_revision_cycles": current_cycles + 1
             }
        elif node_name == "review_novelty" or node_name == "review_feasibility":
            # These are review nodes, their results are critiques.
            return_value = {"review_team_feedback": {node_name: result}}
        elif node_name == "synthesize_review":
             return_value = {"final_review": result}
        else:
             return_value = {}

        print(f"--- Returning from {node_name}: {return_value} ---")
        return return_value

    return _node

# --- Special Aggregator & Non-LLM Nodes ---

def deduplicate_queries_node(state: ProposalAgentState) -> Dict[str, List[str]]:
    """A simple utility node to deduplicate queries from parallel runs."""
    print("--- Running node: deduplicate_queries_node ---")
    all_queries = state.get('search_queries', [])
    unique_queries = list(dict.fromkeys(all_queries))
    # MUST return the full state, not just the part that changed.
    # But since the state has reducers, we only need to return the changed key.
    return {"search_queries": unique_queries}

def human_query_review_node(state: ProposalAgentState) -> dict:
    """A placeholder node that serves as a breakpoint for human query review."""
    print("--- Paused for Human Query Review ---")
    return {"paused_on": "human_query_review_node"}

def human_insight_review_node(state: ProposalAgentState) -> dict:
    """A placeholder node that serves as a breakpoint for human insight review."""
    print("--- Paused for Human Insight Review ---")
    return {"paused_on": "human_insight_review_node"}

def human_review_node(state: ProposalAgentState) -> dict:
    """
    A placeholder node that serves as a breakpoint for human review.
    The graph will pause here.
    """
    print("--- Paused for Human Review ---")
    # This node can optionally clear the human feedback after it's been "consumed"
    # by the planner, to prevent it from being used in subsequent loops.
    return {"paused_on": "human_review_node", "human_feedback": None}

# --- Conditional Logic ---

def is_proposal_approved(state: ProposalAgentState) -> Literal["approved", "revise", "max_revisions_reached"]:
    """Checks the final review to decide if the proposal is done or needs revision."""
    print("--- Running node: is_proposal_approved ---")
    
    # Stop condition 1: max revisions reached
    if state.get("proposal_revision_cycles", 0) >= MAX_PROPOSAL_REVISIONS:
        print(f"--- Max proposal revisions ({MAX_PROPOSAL_REVISIONS}) reached. ENDING. ---")
        return "max_revisions_reached"

    # Stop condition 2: proposal is approved
    if state.get('final_review') and state['final_review'].get('is_approved'):
        print("--- Proposal approved. ENDING. ---")
        return "approved"
        
    # Otherwise, continue to revise
    print("--- Proposal needs revision. Looping back. ---")
    return "revise"

# --- Graph Construction ---

def build_graph() -> StateGraph:
    """Builds the LangGraph workflow from the JSON configuration."""
    
    # By passing the state definition here, we tell the graph to use the Annotated
    # reducers within the TypedDict instead of its default "replace" behavior.
    # This is the crucial fix for the state overwriting issue.
    builder = StateGraph(ProposalAgentState)
    
    # 1. Add all nodes defined in the config to the graph
    for node_name, node_config in agent_config["nodes"].items():
        # All nodes are currently LLM-based, some with special logic.
        node_func = create_llm_node(node_name, node_config)
        builder.add_node(node_name, node_func)

    # Add special utility nodes
    builder.add_node("deduplicate_queries_node", deduplicate_queries_node)
    builder.add_node("human_query_review_node", human_query_review_node)
    builder.add_node("human_insight_review_node", human_insight_review_node)
    builder.add_node("human_review_node", human_review_node)

    # 2. Define the static workflow using teams
    
    # Entry Point -> Query Generation Team
    query_team_config = agent_config["teams"]["query_generation_team"]
    for member in query_team_config["members"]:
        builder.add_edge(START, member)
    
    # Query Team Members -> Aggregator
    for member in query_team_config["members"]:
        builder.add_edge(member, query_team_config["aggregator"])

    # Query Aggregator -> HIL #1
    builder.add_edge(query_team_config["aggregator"], "human_query_review_node")

    # HIL #1 -> Literature Review Team
    lit_team_config = agent_config["teams"]["literature_review_team"]
    for member in lit_team_config["members"]:
        builder.add_edge("human_query_review_node", member)

    # Literature Review Team Members -> Aggregator
    for member in lit_team_config["members"]:
        builder.add_edge(member, lit_team_config["aggregator"])

    # Literature Aggregator -> HIL #2
    builder.add_edge(lit_team_config["aggregator"], "human_insight_review_node")

    # HIL #2 -> Formulate Plan
    builder.add_edge("human_insight_review_node", "formulate_plan")

    # Formulate Plan -> Proposal Review Team
    review_team_config = agent_config["teams"]["proposal_review_team"]
    for member in review_team_config["members"]:
        builder.add_edge("formulate_plan", member)

    # Proposal Review Team Members -> Final Review Aggregator
    for member in review_team_config["members"]:
        builder.add_edge(member, review_team_config["aggregator"])
    
    # Final Review Aggregator -> Human Review Breakpoint
    builder.add_edge(review_team_config["aggregator"], "human_review_node")
    
    # 3. Add conditional logic for the revision loop
    builder.add_conditional_edges(
        "human_review_node", # The decision point is now AFTER human review
        is_proposal_approved,
        {
            "approved": END,
            "max_revisions_reached": END,
            "revise": "formulate_plan" # If rejected, go back to the planning stage
        }
    )

    # Create a memory checkpointer for persistence
    memory = MemorySaver()
    
    return builder.compile(
        checkpointer=memory,
        interrupt_after=[
            "human_query_review_node",
            "human_insight_review_node",
            "human_review_node"
        ]
    )

# --- Export the compiled graph ---
graph = build_graph() 