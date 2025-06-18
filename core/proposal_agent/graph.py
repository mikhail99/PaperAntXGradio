from langgraph.graph import StateGraph, END, START
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from core.proposal_agent.state import ProposalAgentState, Reflection, NoveltyAssessment, ExperimentProtocol, ProposalCritique, QueryList
from core.proposal_agent.tools import PaperSearchTool
import core.proposal_agent.prompts as prompts
from core.paperqa_service import PaperQAService

# --- Agent Configuration ---
MAX_RESEARCH_PAPERS = 30
MAX_REFLECTION_PAPERS = 10
MAX_LIT_REVIEW_LOOPS = 2

# --- LLM and Tool Initialization ---
json_llm = ChatOllama(model="gemma3:4b", format="json", temperature=0.0)
text_llm = ChatOllama(model="gemma3:4b", temperature=0.0) # For text generation
paper_search_tool = PaperSearchTool()
paperqa_service = PaperQAService()

# --- Graph Nodes ---

def generate_queries(state: ProposalAgentState) -> ProposalAgentState:
    """The entrypoint to the agent. Gets the user's question and generates search queries."""
    print("--- Running node: generate_queries ---")
    # Get the user's question from the latest message
    question = state['messages'][-1][1]
    
    prompt = ChatPromptTemplate.from_template(prompts.generate_queries_prompt)
    structured_llm = json_llm.with_structured_output(QueryList)
    chain = prompt | structured_llm
    # Pass the question to the prompt
    result = chain.invoke({"topic": question, "num_queries": 3})
    # Initialize the query index
    return {"search_queries": result.queries, "query_index": 0}

async def run_single_query(state: ProposalAgentState) -> ProposalAgentState:
    """Runs a single PaperQA query and appends the result to the literature summary."""
    print("--- Running node: run_single_query ---")
    
    # Get the current query
    query_index = state['query_index']
    search_queries = state['search_queries']
    query = search_queries[query_index]
    collection_id = state['collection_id']
    
    # Get existing summaries
    literature_summaries = state.get('literature_summaries', [])
    
    # Run the query
    print(f"Running query {query_index + 1}/{len(search_queries)}: '{query}'")
    response = await paperqa_service.query_documents(collection_id, query)
    
    if response and not response.get("error"):
        new_summary = response.get("answer_text", "")
    else:
        new_summary = f"Error processing query: {query}. Details: {response.get('error', 'Unknown error')}"
        
    # Update state
    all_summaries = literature_summaries + [new_summary]
    accumulated_summary = "\n\n---\n\n".join([s for s in all_summaries if s])
    
    return {
        "literature_summaries": all_summaries,
        "literature_summary": accumulated_summary,
        "query_index": query_index + 1,  # Increment index for the next loop
        "search_queries": search_queries # --- FIX: Pass queries through
    }

def reflect_on_summary(state: ProposalAgentState) -> ProposalAgentState:
    print("--- Running node: reflect_on_summary ---")
    # After all queries are run, increment the literature review loop counter
    literature_review_loops = state.get('literature_review_loops', 0)
    
    prompt = ChatPromptTemplate.from_template(prompts.reflect_on_literature_prompt)
    chain = prompt | json_llm.with_structured_output(Reflection)
    reflection = chain.invoke({"literature_summary": state['literature_summary']})
    
    # Reset query index for the next potential research loop
    return {
        "reflection": reflection, 
        "literature_review_loops": literature_review_loops + 1, 
        "query_index": 0,
        "search_queries": state['search_queries'] # --- FIX: Pass queries through
    }

def formulate_plan(state: ProposalAgentState) -> ProposalAgentState:
    print("--- Running node: formulate_plan ---")
    prompt = ChatPromptTemplate.from_template(prompts.formulate_plan_prompt)
    chain = prompt | text_llm
    plan = chain.invoke({
        "knowledge_gap": state['reflection'].knowledge_gap,
        "literature_summary": state['literature_summary']
    })
    return {"research_plan": plan.content}

def assess_plan_novelty(state: ProposalAgentState) -> ProposalAgentState:
    print("--- Running node: assess_plan_novelty ---")
    current_reflection_count = state.get('reflection_papers_count', 0)
    collection_id = state.get('collection_id')
    
    if current_reflection_count >= MAX_REFLECTION_PAPERS:
        print(f"Reflection paper limit reached ({MAX_REFLECTION_PAPERS}). Assuming plan is novel.")
        assessment = NoveltyAssessment(
            is_novel=True,
            justification=f"Paper search limit reached ({MAX_REFLECTION_PAPERS} papers). Proceeding with assumption of novelty."
        )
        return {"novelty_assessment": assessment}
    
    # Find similar papers
    similar_papers = paper_search_tool.find_similar_papers(
        state['research_plan'], 
        n_results=min(5, MAX_REFLECTION_PAPERS - current_reflection_count),
        collection_id=collection_id
    )
    
    # Get assessment from LLM (without asking it to return the papers)
    prompt = ChatPromptTemplate.from_template(prompts.assess_novelty_prompt)
    chain = prompt | json_llm.with_structured_output(NoveltyAssessment)
    assessment = chain.invoke({
        "research_plan": state['research_plan'],
        "similar_papers": "\n".join([f"- {p['title']}: {p['url']}" for p in similar_papers])
    })
    
    # Manually add the papers to the assessment object
    assessment.similar_papers = similar_papers
    
    # Update state
    new_reflection_count = current_reflection_count + len(similar_papers)
    return {
        "novelty_assessment": assessment,
        "reflection_papers_count": new_reflection_count
    }

def design_experiments(state: ProposalAgentState) -> ProposalAgentState:
    print("--- Running node: design_experiments ---")
    prompt = ChatPromptTemplate.from_template(prompts.design_experiments_prompt)
    chain = prompt | json_llm.with_structured_output(ExperimentProtocol)
    protocol = chain.invoke({"research_plan": state['research_plan']})
    return {"experiment_protocol": protocol}

def write_proposal(state: ProposalAgentState) -> ProposalAgentState:
    print("--- Running node: write_proposal ---")
    prompt = ChatPromptTemplate.from_template(prompts.write_proposal_prompt)
    chain = prompt | text_llm
    proposal = chain.invoke({
        "literature_summary": state['literature_summary'],
        "research_plan": state['research_plan'],
        "experiment_protocol": state['experiment_protocol'].model_dump_json()
    })
    return {"markdown_proposal": proposal.content}

def review_proposal(state: ProposalAgentState) -> ProposalAgentState:
    print("--- Running node: review_proposal ---")
    prompt = ChatPromptTemplate.from_template(prompts.review_proposal_prompt)
    chain = prompt | json_llm.with_structured_output(ProposalCritique)
    critique = chain.invoke({"markdown_proposal": state['markdown_proposal']})
    return {"critique": critique}

def plan_followup_research(state: ProposalAgentState) -> ProposalAgentState:
    """Takes the follow-up queries from the reflection and sets them as the new
    search queries for the next research loop."""
    print("--- Running node: plan_followup_research ---")
    follow_up_queries = state["reflection"].follow_up_queries
    return {"search_queries": follow_up_queries}

# --- Conditional Edges ---

def should_run_query(state: ProposalAgentState) -> str:
    """Determines whether to run another query or move on to reflection."""
    if state['query_index'] < len(state['search_queries']):
        return "run_single_query"
    else:
        return "reflect_on_summary"

def should_continue_research(state: ProposalAgentState) -> str:
    loops = state.get('literature_review_loops', 0)
    if state['reflection'].is_sufficient:
        return "formulate_plan"
    if loops >= MAX_LIT_REVIEW_LOOPS:
        return "formulate_plan"
    return "plan_followup_research"

def is_plan_novel(state: ProposalAgentState) -> str:
    return "design_experiments" if state['novelty_assessment'].is_novel else "formulate_plan"

def should_revise_proposal(state: ProposalAgentState) -> str:
    # Simplified: for now, we just end. A more complex logic could route back.
    if state['critique'].revision_required:
        print("Revision required. Ending process for now.")
        return END
    return END

# --- Graph Definition ---
builder = StateGraph(ProposalAgentState)

builder.add_node("generate_queries", generate_queries)
builder.add_node("run_single_query", run_single_query)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("formulate_plan", formulate_plan)
builder.add_node("assess_plan_novelty", assess_plan_novelty)
builder.add_node("design_experiments", design_experiments)
builder.add_node("write_proposal", write_proposal)
builder.add_node("review_proposal", review_proposal)
builder.add_node("plan_followup_research", plan_followup_research)

builder.add_edge(START, "generate_queries")
builder.add_conditional_edges(
    "generate_queries",
    should_run_query
)
builder.add_conditional_edges(
    "run_single_query",
    should_run_query
)

builder.add_conditional_edges(
    "reflect_on_summary",
    should_continue_research,
    {
        "formulate_plan": "formulate_plan",
        "plan_followup_research": "plan_followup_research"
    }
)
builder.add_conditional_edges(
    "plan_followup_research", 
    should_run_query
)

builder.add_edge("formulate_plan", "assess_plan_novelty")
builder.add_conditional_edges(
    "assess_plan_novelty", 
    is_plan_novel,
    {
        "design_experiments": "design_experiments",
        "formulate_plan": "formulate_plan"
    }
)

builder.add_edge("design_experiments", "write_proposal")
builder.add_edge("write_proposal", "review_proposal")
builder.add_conditional_edges("review_proposal", should_revise_proposal)

graph = builder.compile()

