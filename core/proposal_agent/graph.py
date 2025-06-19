from langgraph.graph import StateGraph, END, START
from typing import Literal
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from core.proposal_agent.state import ProposalAgentState, SummaryReflection, NoveltyAssessment, SingleQuery
from core.proposal_agent.tools import PaperSearchTool
import core.proposal_agent.prompts as prompts
from core.paperqa_service import PaperQAService
from core.collections_manager import CollectionsManager
# --- Agent Configuration ---

MOCK_MODE = False # Set to True to run in mock mode without real API calls

MAX_LIT_REVIEW_LOOPS = 5
MAX_NOVELTY_LOOPS = 5

# --- LLM and Tool Initialization ---
json_llm = ChatOllama(model="gemma3:4b", format="json", temperature=0.7)
text_llm = ChatOllama(model="gemma3:4b", temperature=0.7) # For text generation
paper_search_tool = CollectionsManager()
paperqa_service = PaperQAService()

# --- Graph Nodes ---

def generate_query(state: ProposalAgentState) -> ProposalAgentState:
    """Generates a single, novel search query based on reflection."""
    print("--- Running node: generate_query ---")
    topic = state['topic']
    previous_queries = state.get('search_queries', [])
    reflection = state.get('reflection')
    knowledge_gap = reflection.knowledge_gap if reflection else ""

    prompt = ChatPromptTemplate.from_template(prompts.generate_query_prompt)

    if MOCK_MODE:
        formatted_prompt = prompt.format(
            topic=topic, 
            previous_queries="\\n".join(previous_queries),
            knowledge_gap=knowledge_gap
        )
        return {"search_queries": [f"--- MOCK QUERY ---\n{formatted_prompt}"]}

    structured_llm = json_llm.with_structured_output(SingleQuery)
    chain = prompt | structured_llm
    
    # Invoke with topic and previous queries
    result = chain.invoke({
        "topic": topic, 
        "previous_queries": "\n".join(previous_queries),
        "knowledge_gap": knowledge_gap
    })
    
    new_query = result.query
    
    # The new query is added to the history via operator.add in the state
    return {
        "search_queries": [new_query] 
    }

async def run_single_query(state: ProposalAgentState) -> ProposalAgentState:
    """Runs a single PaperQA query and appends the result to the literature summary."""
    print("--- Running node: run_single_query ---")
    
    query = state['search_queries'][-1]
    
    if MOCK_MODE:
        return {"literature_summaries": [f"--- MOCK SUMMARY for query: '{query}' ---"]}

    collection_name = state['collection_name']
    
    # Run the query
    print(f"Running query: '{query}'")
    response = await paperqa_service.query_documents(collection_name, query)
    
    if response and not response.get("error"):
        new_summary = response.get("answer_text", "")
    else:
        new_summary = f"Error processing query: {query}. Details: {response.get('error', 'Unknown error')}"
        

    print(f"New summary: {new_summary}")
    return {"literature_summaries": [new_summary]}

def reflect_on_summary(state: ProposalAgentState) -> ProposalAgentState:
    print("--- Running node: reflect_on_summary ---")
    
    if MOCK_MODE:
        loops = len(state.get("literature_summaries", []))
        is_sufficient = loops >= 2 # Mock stop condition
        mock_reflection = SummaryReflection(
            is_sufficient=is_sufficient,
            knowledge_gap="Mock: The summary is too short, need more details." if not is_sufficient else "Mock: The summary is now sufficient.",
            follow_up_queries=[] 
        )
        return {"reflection": mock_reflection}

    # We now reflect on the entire history of summaries
    all_summaries = "\n\n---\n\n".join(state.get('literature_summaries', []))

    prompt = ChatPromptTemplate.from_template(prompts.reflect_on_literature_prompt)
    chain = prompt | json_llm.with_structured_output(SummaryReflection)
    reflection = chain.invoke({
        "literature_summary": all_summaries,
        "previous_queries": state.get('search_queries', [])
    })
    
    return {"reflection": reflection}

def formulate_plan(state: ProposalAgentState) -> ProposalAgentState:
    print("--- Running node: formulate_plan ---")
    
    if MOCK_MODE:
        mock_plan = f"--- MOCK PLAN based on knowledge gap: {state['reflection'].knowledge_gap} ---"
        return {"research_plan": [mock_plan]}
    
    all_summaries = "\n\n---\n\n".join(state.get('literature_summaries', []))

    prompt = ChatPromptTemplate.from_template(prompts.formulate_plan_prompt)
    chain = prompt | text_llm
    plan = chain.invoke({
        "knowledge_gap": state['reflection'].knowledge_gap,
        "literature_summary": all_summaries
    })
    return {"research_plan": [plan.content]}

def assess_plan_novelty(state: ProposalAgentState) -> ProposalAgentState:
    print("--- Running node: assess_plan_novelty ---")

    if MOCK_MODE:
        loops = len(state.get("novelty_assessment", []))
        is_novel = loops >= 1 # First plan is not novel, subsequent are
        mock_assessment = NoveltyAssessment(
            is_novel=is_novel,
            justification="Mock: Plan is not novel, needs refinement." if not is_novel else "Mock: Plan is novel.",
            similar_papers=[]
        )
        return {"novelty_assessment": [mock_assessment]}

    collection_name = state.get('collection_name')
    
    # Assess the novelty of the most recent research plan
    latest_plan = state['research_plan'][-1]
    
    # Find similar papers
    similar_papers = paper_search_tool.search_articles(
        query=latest_plan, 
        limit=3, # Simplified for now
        collection_name=collection_name
    )
    
    # Get assessment from LLM
    prompt = ChatPromptTemplate.from_template(prompts.assess_novelty_prompt)
    chain = prompt | json_llm.with_structured_output(NoveltyAssessment)
    assessment = chain.invoke({
        "research_plan": latest_plan,
        "similar_papers": "\n".join([f"- {p['title']}: {p['url']}" for p in similar_papers])
    })
    
    # Manually add the papers to the assessment object
    assessment.similar_papers = similar_papers
    
    return {"novelty_assessment": [assessment]}
'''
def design_experiments(state: ProposalAgentState) -> ProposalAgentState:
    print("--- Running node: design_experiments ---")

    if MOCK_MODE:
        mock_protocol = ExperimentProtocol(
            steps=["Mock experiment step 1", "Mock experiment step 2"],
            success_criteria="Mock success criteria: Get a mock result."
        )
        return {"experiment_protocol": mock_protocol}

    prompt = ChatPromptTemplate.from_template(prompts.design_experiments_prompt)
    chain = prompt | json_llm.with_structured_output(ExperimentProtocol)
    protocol = chain.invoke({"research_plan": state['research_plan'][-1]})
    return {"experiment_protocol": protocol}

def write_proposal(state: ProposalAgentState) -> ProposalAgentState:
    print("--- Running node: write_proposal ---")

    if MOCK_MODE:
        mock_proposal = f"""# --- MOCK PROPOSAL ---
## Research Plan
{state['research_plan'][-1]}
## Experiments
{state['experiment_protocol'].model_dump_json(indent=2)}
"""
        return {"markdown_proposal": mock_proposal}

    prompt = ChatPromptTemplate.from_template(prompts.write_proposal_prompt)
    chain = prompt | text_llm
    proposal = chain.invoke({
        "literature_summary": state['literature_summary'],
        "research_plan": state['research_plan'][-1],
        "experiment_protocol": state['experiment_protocol'].model_dump_json()
    })
    return {"markdown_proposal": proposal.content}

def review_proposal(state: ProposalAgentState) -> ProposalAgentState:
    print("--- Running node: review_proposal ---")

    if MOCK_MODE:
        mock_critique = ProposalCritique(
            critique="This is a mock critique. The proposal looks fine for a mock proposal.",
            suggestions=["Mock suggestion: Add more mocks."]
        )
        return {"critique": mock_critique}

    prompt = ChatPromptTemplate.from_template(prompts.review_proposal_prompt)
    chain = prompt | json_llm.with_structured_output(ProposalCritique)
    critique = chain.invoke({"markdown_proposal": state['markdown_proposal']})
    return {"critique": critique}
'''
# --- Conditional Edges ---

def is_summary_sufficient(state: ProposalAgentState) -> Literal["is_sufficient", "reached_max_review_loops", "not_sufficient"]:
    """Determines whether to continue the research loop or formulate a plan."""
   
 
    # Stop condition 1: literature is sufficient
    if state['reflection'].is_sufficient:
        print("--- Literature sufficient. Proceeding to formulate plan. ---")
        return "is_sufficient"
        
    # Stop condition 2: max loops reached
    loops = len(state["literature_summaries"])
    if loops >= MAX_LIT_REVIEW_LOOPS:
        print(f"--- Max literature review loops ({MAX_LIT_REVIEW_LOOPS}) reached. Proceeding to formulate plan. ---")
        return "reached_max_review_loops"
        
    # Otherwise, continue research
    print("--- Continuing research. Generating new query. ---")
    return "not_sufficient"

def is_plan_novel(state: ProposalAgentState) -> Literal["is_novel", "not_novel", "reached_max_novelty_loops"]:
    loops = len(state["novelty_assessment"])
    if loops >= MAX_NOVELTY_LOOPS:
        print(f"--- Max novelty loops ({MAX_NOVELTY_LOOPS}) reached. Proceeding to formulate plan. ---")
        return "reached_max_novelty_loops"
    return "is_novel" if state['novelty_assessment'][-1].is_novel else "not_novel"



# --- Graph Definition ---
builder = StateGraph(ProposalAgentState)

builder.add_node("generate_query", generate_query)
builder.add_node("run_single_query", run_single_query)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("formulate_plan", formulate_plan)
builder.add_node("assess_plan_novelty", assess_plan_novelty)
#builder.add_node("review_proposal", review_proposal)

builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "run_single_query")
builder.add_edge("run_single_query", "reflect_on_summary")

builder.add_conditional_edges(
    "reflect_on_summary",
    is_summary_sufficient,
    {
        "is_sufficient": "formulate_plan",
        "reached_max_review_loops":  "formulate_plan",
        "not_sufficient": "generate_query"
    }
)

builder.add_edge("formulate_plan", "assess_plan_novelty")
builder.add_conditional_edges(
    "assess_plan_novelty", 
    is_plan_novel,
    {
        "is_novel": END,
        "not_novel": "generate_query",
        "reached_max_novelty_loops": END
    }
)

#builder.add_edge("design_experiments", "write_proposal")
#builder.add_edge("write_proposal", END)
#builder.add_edge("write_proposal", "review_proposal")
#builder.add_conditional_edges("review_proposal", should_revise_proposal)

graph = builder.compile()

