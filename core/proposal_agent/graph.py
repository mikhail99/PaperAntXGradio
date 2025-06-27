import json
from pathlib import Path
from typing import Literal, List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START

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
json_llm = ChatOllama(model="gemma3:4b", format="json", temperature=0.7)
text_llm = ChatOllama(model="gemma3:4b", temperature=0.7)
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

    async def _node(state: ProposalAgentState) -> Dict[str, Any]:
        print(f"--- Running node: {node_name} ---")
        
        # Use a structured output model if a schema is specified
        if output_schema:
            llm = json_llm.with_structured_output(output_schema)
        else:
            llm = text_llm
            
        chain = prompt | llm
        
        # HACK: Special handling for the reviewer. This node needs to use a tool FIRST, then an LLM.
        if node_name == "literature_reviewer_local":
            query = state['search_queries'][-1]
            collection_name = state['collection_name']
            
            print(f"--- Node '{node_name}' is calling a tool: paperqa_service.query_documents ---")
            print(f"Running query: '{query}' on collection '{collection_name}'")
            response = await paperqa_service.query_documents(collection_name, query)
            literature_context = response.get("context", f"Could not find relevant literature for query: {query}")

            input_data = {"topic": state["topic"], "literature": literature_context}
            result = chain.invoke(input_data)
        elif node_name == "review_novelty":
            # This node needs a specific part of the state (the aggregated summary)
            input_data = {
                "proposal_draft": state["proposal_draft"],
                "aggregated_summary": state["knowledge_gap"]["synthesized_summary"]
            }
            result = chain.invoke(input_data)
        elif node_name == "synthesize_review":
            # This node needs the dictionary of feedback.
            input_data = {"review_feedbacks": state["review_team_feedback"]}
            result = chain.invoke(input_data)
        else:
            result = chain.invoke(state)

        # This logic needs to be much smarter.
        # It should know which state key to update based on the node's purpose.
        if node_name == "query_generator_base":
             return {"search_queries": result['queries']}
        elif node_name == "literature_reviewer_local":
             # The reviewer's output is a summary string. We add it to the list.
             # The aggregator will later combine all summaries.
             return {"literature_summaries": state.get("literature_summaries", []) + [result.content]}
        elif node_name == "synthesize_literature_review":
             return {"knowledge_gap": result}
        elif node_name == "formulate_plan":
             return {"proposal_draft": result.content}
        elif node_name in ["review_feasibility", "review_novelty"]:
             # The output of a review is a Critique dict. We collect them in a dict.
             # The reducer we defined in the state will merge the results from the parallel nodes.
             return {"review_team_feedback": {node_name: result}}
        elif node_name == "synthesize_review":
             return {"final_review": result}

        return {} # Should not be reached for most nodes

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
        # All nodes are currently LLM-based, some with special logic.
        node_func = create_llm_node(node_name, node_config)
        builder.add_node(node_name, node_func)

    # Add special utility nodes
    builder.add_node("deduplicate_queries_node", deduplicate_queries_node)

    # 2. Define the static workflow using teams
    
    # Entry Point -> Query Generation Team
    query_team_config = agent_config["teams"]["query_generation_team"]
    for member in query_team_config["members"]:
        builder.add_edge(START, member)
    
    # Query Team Members -> Aggregator
    for member in query_team_config["members"]:
        builder.add_edge(member, query_team_config["aggregator"])

    # Query Team Aggregator -> Literature Review Team
    lit_team_config = agent_config["teams"]["literature_review_team"]
    for member in lit_team_config["members"]:
        builder.add_edge(query_team_config["aggregator"], member)

    # Literature Review Team Members -> Aggregator
    for member in lit_team_config["members"]:
        builder.add_edge(member, lit_team_config["aggregator"])

    # Literature Review Aggregator -> Formulate Plan
    builder.add_edge(lit_team_config["aggregator"], "formulate_plan")

    # Formulate Plan -> Proposal Review Team
    review_team_config = agent_config["teams"]["proposal_review_team"]
    for member in review_team_config["members"]:
        builder.add_edge("formulate_plan", member)

    # Proposal Review Team Members -> Final Review Aggregator
    for member in review_team_config["members"]:
        builder.add_edge(member, review_team_config["aggregator"])
    
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