import json
from pathlib import Path
from typing import Literal, List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from core.paperqa_service import PaperQAService
from core.proposal_agent.state import ProposalAgentState, QueryList, KnowledgeGap, Critique, FinalReview

# --- Configuration ---

# Load JSON configs
CONFIG_PATH = Path(__file__).parent
with open(CONFIG_PATH / "agent_config.json", "r") as f:
    agent_config = json.load(f)
with open(CONFIG_PATH / "prompts.json", "r") as f:
    prompts = json.load(f)

# Initialize models and services
json_llm = ChatOllama(model="gemma2:9b", format="json", temperature=0.7)
text_llm = ChatOllama(model="gemma2:9b", temperature=0.7)
paperqa_service = PaperQAService()

# Map schema names from config to actual Python classes
OUTPUT_SCHEMAS = {
    "QueryList": QueryList,
    "KnowledgeGap": KnowledgeGap,
    "Critique": Critique,
    "FinalReview": FinalReview,
    "string": None # For simple text outputs
}

MAX_PROPOSAL_REVISIONS = 3

# --- Node Factory (Creates node functions from config) ---

def create_llm_node(node_name: str, node_config: Dict[str, Any]):
    """Factory function to create a generic LLM-based graph node."""
    
    prompt_template = prompts[node_config["prompt_key"]]
    prompt = ChatPromptTemplate.from_template(prompt_template)
    output_schema = OUTPUT_SCHEMAS.get(node_config["output_schema"])

    def _node(state: ProposalAgentState) -> Dict[str, Any]:
        print(f"--- Running node: {node_name} ---")
        
        # Use a structured output model if a schema is specified
        if output_schema:
            llm = json_llm.with_structured_output(output_schema)
        else:
            llm = text_llm
            
        chain = prompt | llm
        
        # The result from an LLM node is a single object (or string).
        # We need to return a dictionary to update the state.
        # The key under which it's stored depends on the node's function in the graph.
        result = chain.invoke(state)

        # This is a simplification. The actual state update is handled
        # by the graph structure itself when the node returns.
        # For team members, the output is collected. For aggregators, it updates a specific field.
        # This function just produces the core output.
        return result

    return _node

def create_tool_node(node_name: str, node_config: Dict[str, Any]):
    """Factory function for a tool-using node (currently specific to paperqa)."""
    
    async def _node(state: ProposalAgentState) -> Dict[str, Any]:
        print(f"--- Running node: {node_name} ---")
        # This is built specifically for the literature review tool for now
        query = state['search_queries'][-1] # Assumes one query at a time
        collection_name = state['collection_name']
        
        print(f"Running query: '{query}'")
        response = await paperqa_service.query_documents(collection_name, query)
        
        summary = response.get("answer_text", f"Error processing query: {query}")
        return {"literature_summaries": [summary]}

    return _node

# --- Special Aggregator & Non-LLM Nodes ---

def deduplicate_queries_node(state: ProposalAgentState) -> Dict[str, List[str]]:
    """A simple utility node to deduplicate queries from parallel runs."""
    print("--- Running node: deduplicate_queries_node ---")
    all_queries = state.get('search_queries', [])
    unique_queries = list(dict.fromkeys(all_queries))
    return {"search_queries": unique_queries}

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
    
    builder = StateGraph(ProposalAgentState)
    
    # 1. Add all nodes defined in the config to the graph
    for node_name, node_config in agent_config["nodes"].items():
        # This part needs to be more robust. For now, we assume llm or tool node.
        # In a real scenario, you'd have a mapping of types to factory functions.
        if "prompt_key" in node_config:
            node_func = create_llm_node(node_name, node_config)
        else: # Assumes tool node
            node_func = create_tool_node(node_name, node_config)
        builder.add_node(node_name, node_func)

    # Add special utility nodes
    builder.add_node("deduplicate_queries_node", deduplicate_queries_node)

    # 2. Define the static workflow using teams
    
    # Entry Point -> Query Generation Team
    query_team_config = agent_config["teams"]["query_generation_team"]
    builder.add_edge(START, query_team_config["members"])
    
    # Query Team -> Literature Review Team
    # The aggregator for the query team is the entry point for the literature review
    lit_team_config = agent_config["teams"]["literature_review_team"]
    builder.add_edge(query_team_config["aggregator"], lit_team_config["members"])

    # Literature Review Team -> Formulate Plan
    # This team's aggregator output (knowledge gap) feeds into the planner.
    builder.add_edge(lit_team_config["aggregator"], "formulate_plan")

    # Formulate Plan -> Proposal Review Team
    review_team_config = agent_config["teams"]["proposal_review_team"]
    builder.add_edge("formulate_plan", review_team_config["members"])

    # Proposal Review Team -> Final Review Aggregator
    builder.add_edge(review_team_config["members"], review_team_config["aggregator"])
    
    # 3. Add conditional logic for the revision loop
    builder.add_conditional_edges(
        review_team_config["aggregator"],
        is_proposal_approved,
        {
            "approved": END,
            "max_revisions_reached": END,
            "revise": "formulate_plan" # If rejected, go back to the planning stage
        }
    )

    return builder.compile()

# --- Export the compiled graph ---
graph = build_graph() 