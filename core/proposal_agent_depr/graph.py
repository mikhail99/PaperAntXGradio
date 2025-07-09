print("--- Importing: core.proposal_agent.graph ---")
import json
from pathlib import Path
from typing import Literal, List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from core.paperqa_service import PaperQAService
from core.proposal_agent.state import ProposalAgentState

# --- Configuration ---

# Load JSON configs
CONFIG_PATH = Path(__file__).parent
with open(CONFIG_PATH / "agent_config.json", "r") as f:
    agent_config = json.load(f)
with open(CONFIG_PATH / "prompts.json", "r") as f:
    prompts = json.load(f)

# Map schema names from config to actual Python classes
from core.proposal_agent.state import QueryList, KnowledgeGap, Critique, FinalReview

OUTPUT_SCHEMAS = {
    "QueryList": QueryList,
    "KnowledgeGap": KnowledgeGap,
    "Critique": Critique,
    "FinalReview": FinalReview,
    "string": None # For simple text outputs
}

MAX_PROPOSAL_REVISIONS = 3
USE_PARROT_SERVICES = True # <-- SET THIS TO True TO USE MOCKS

# --- Lazy Service and Model Initializer ---
# We store the instances here so they are only created once.
_json_llm = None
_text_llm = None
_paperqa_service = None

def get_services():
    """Lazily initializes and returns service instances."""
    global _json_llm, _text_llm, _paperqa_service

    if USE_PARROT_SERVICES:
        from .parrot_services import get_parrot_services as get_mock
        if _json_llm is None: # Check so we only initialize once
            _json_llm, _text_llm, _paperqa_service = get_mock()
        return _json_llm, _text_llm, _paperqa_service

    # --- Original Service Initialization ---
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
            query_to_run_list = state.get('search_queries', [])
            query_to_run = query_to_run_list[-1] if query_to_run_list else state.get("topic")

            collection_name = state['collection_name']
            
            print(f"--- Node '{node_name}' is calling a tool: paperqa_service.query_documents ---")
            response = await paperqa_service.query_documents(collection_name, query_to_run)
            
            # Directly use the answer with citations, bypassing the extra LLM call.
            summary_with_citations = response.get("answer_text", f"PaperQA did not return an answer for query: {query_to_run}")
            
            return {
                "literature_summaries": [summary_with_citations]
            }
        elif node_name == "review_feasibility":
            # This node needs the proposal draft to review
            input_data = {"proposal_draft": state["proposal_draft"]}
            print(f"--- Input for {node_name}: {input_data} ---")
            result = chain.invoke(input_data)
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
            # Return the structured result in the correct field
            return {"final_review": result.dict()}
        elif node_name == "query_generator_base":
            # This node needs the topic and any prior queries.
            topic = state.get("topic", "")
            
            input_data = {
                "topic": topic,
                "search_queries": state.get("search_queries", [])
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

            # Check for human feedback in current_literature field (modern HIL pattern)
            human_feedback = ""
            current_lit = state.get("current_literature", "")
            if current_lit.startswith("Human feedback: "):
                human_feedback = current_lit.replace("Human feedback: ", "")

            input_data = {
                "knowledge_gap": kg_str,
                "proposal_draft": draft_str,
                "review_team_feedback": reviews_str,
                "human_feedback": human_feedback
            }
            print(f"--- Input for {node_name} (with truncated inputs) ---")
            result = chain.invoke(input_data)
        elif node_name == "synthesize_literature_review":
            # This node only needs the topic and the collected summaries.
            summaries = state.get("literature_summaries", [])
            input_data = {"topic": state["topic"], "literature_summaries": summaries}
            print(f"--- Input for {node_name}: {input_data} ---")
            result = chain.invoke(input_data)
            # Return the structured result directly
            return {"knowledge_gap": result.dict()}
        else:
            # For generic nodes, use all the input fields they expect
            state_inputs = node_config.get("state_inputs", [])
            input_data = {field: state.get(field, "") for field in state_inputs}
            print(f"--- Input for {node_name}: {input_data} ---")
            result = chain.invoke(input_data)

        # Handle different result types based on output schema
        if node_name == "formulate_plan":
            current_cycles = state.get("proposal_revision_cycles", 0)
            # The result is a message object; we need to extract its content.
            draft_content = result.content if hasattr(result, 'content') else str(result)
            return {
                "proposal_draft": draft_content,
                "proposal_revision_cycles": current_cycles + 1
            }
        elif node_name == "query_generator_base":
            # The output is {'queries': [...]}, but the state field is 'search_queries'.
            # This remaps the output to the correct state field.
            queries_list = result.get('queries', [])
            return {"search_queries": queries_list}
        elif node_name in ["review_feasibility", "review_novelty"]:
            # These are review nodes, their results are critiques that need to be merged.
            return {"review_team_feedback": {node_name: result.dict()}}
        
        if isinstance(result, dict):
            # Structured output - return directly
            return result
        elif isinstance(result, str):
            # Simple string - map to the appropriate output field
            output_field = node_config.get("output_field", "output")
            return {output_field: result}
        else:
            # Pydantic model - convert to dict
            return result.dict() if hasattr(result, 'dict') else {"output": str(result)}

    return _node

# --- Utility Nodes ---

def deduplicate_queries_node(state: ProposalAgentState) -> Dict[str, List[str]]:
    """
    Remove duplicate queries from the search_queries list.
    This node ensures that we don't search for the same thing multiple times.
    """
    queries = state.get('search_queries', [])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_queries = []
    for query in queries:
        if query.lower() not in seen:
            seen.add(query.lower())
            unique_queries.append(query)
    
    print(f"--- Deduplicated queries: {len(queries)} -> {len(unique_queries)} ---")
    return {"search_queries": unique_queries}


def is_proposal_approved(state: ProposalAgentState) -> Literal["approved", "revise", "max_revisions_reached"]:
    """
    Determines if the proposal should be approved, revised, or if max revisions are reached.
    This is used as a conditional edge function.
    """
    print("--- APPROVAL DEBUG: Evaluating proposal approval ---")
    
    final_review = state.get('final_review', {})
    revision_cycles = state.get('proposal_revision_cycles', 0)
    
    # Extract approval decision from final review
    is_approved = final_review.get('is_approved', False) if isinstance(final_review, dict) else False
    
    print(f"--- APPROVAL DEBUG: is_approved: {is_approved}, revision_cycles: {revision_cycles} ---")
    
    if is_approved:
        return "approved"
    elif revision_cycles >= MAX_PROPOSAL_REVISIONS:
        print(f"--- Max revisions ({MAX_PROPOSAL_REVISIONS}) reached. Stopping. ---")
        return "max_revisions_reached"
    else:
        return "revise" 