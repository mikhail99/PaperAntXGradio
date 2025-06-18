from langgraph.graph import StateGraph, END, START
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from core.proposal_agent.state import ProposalAgentState, Reflection, NoveltyAssessment, ExperimentProtocol, ProposalCritique, QueryList
from core.proposal_agent.tools import PaperSearchTool
import core.proposal_agent.prompts as prompts

# --- Agent Configuration ---
MAX_RESEARCH_PAPERS = 30
MAX_REFLECTION_PAPERS = 10

# --- LLM and Tool Initialization ---
json_llm = ChatOllama(model="gemma3:4b", format="json", temperature=0.0)
text_llm = ChatOllama(model="gemma3:4b", temperature=0.0) # For text generation
paper_search_tool = PaperSearchTool()

# --- Graph Nodes ---

def generate_queries(state: ProposalAgentState) -> ProposalAgentState:
    """The entrypoint to the agent. Gets the user's question and generates search queries."""
    # Get the user's question from the latest message
    question = state['messages'][-1].content
    
    prompt = ChatPromptTemplate.from_template(prompts.generate_queries_prompt)
    structured_llm = json_llm.with_structured_output(QueryList)
    chain = prompt | structured_llm
    # Pass the question to the prompt
    result = chain.invoke({"topic": question, "num_queries": 5})
    return {"search_queries": result.queries}

def search_and_summarize(state: ProposalAgentState) -> ProposalAgentState:
    current_count = state.get('research_papers_count', 0)
    
    if current_count >= MAX_RESEARCH_PAPERS:
        print(f"Research paper limit reached ({MAX_RESEARCH_PAPERS}). Using existing papers for summary.")
        # Use only existing papers from ChromaDB for summary
        all_relevant_papers = []
        for query in state['search_queries']:
            db_papers = paper_search_tool.get_relevant_papers_from_db(query, n_results=5)
            all_relevant_papers.extend(db_papers)
        
        # Remove duplicates by ID
        seen_ids = set()
        unique_papers = []
        for paper in all_relevant_papers:
            if paper.id not in seen_ids:
                unique_papers.append(paper)
                seen_ids.add(paper.id)
        
        summary_text = "\n".join([f"Title: {p.title}\nSummary: {p.summary}" for p in unique_papers])
        
        prompt = ChatPromptTemplate.from_template(prompts.summarize_papers_prompt)
        chain = prompt | text_llm
        summary = chain.invoke({"papers_summary": summary_text})
        
        return {"literature_summary": summary.content}
    
    # Get papers from previous steps
    papers_from_state = state.get('papers', [])
    
    all_new_papers = []
    all_relevant_papers = []
    
    # Check if we should only use local papers
    local_only = state.get('local_papers_only', False)
    
    for query in state['search_queries']:
        if not local_only:
            # Check if we're still under the limit before searching
            if current_count + len(all_new_papers) >= MAX_RESEARCH_PAPERS:
                print(f"Approaching research paper limit. Stopping search at {current_count + len(all_new_papers)} papers.")
                break
                
            # Get new papers from ArXiv
            print(f"Searching ArXiv for: '{query}'")
            found_papers = paper_search_tool.search_arxiv(query)
            all_new_papers.extend(found_papers)
        
        # Also get relevant papers from our database
        print(f"Searching ChromaDB for: '{query}'")
        db_papers = paper_search_tool.get_relevant_papers_from_db(query, n_results=5)
        all_relevant_papers.extend(db_papers)
    
    # Combine and deduplicate all papers for the summary
    combined_papers = papers_from_state + all_new_papers + all_relevant_papers
    
    seen_ids = set()
    unique_papers_for_summary = []
    for paper in combined_papers:
        if paper.id not in seen_ids:
            unique_papers_for_summary.append(paper)
            seen_ids.add(paper.id)
            
    # Papers to add to the state are the ones newly found in this step
    papers_to_add_to_state = all_new_papers + all_relevant_papers

    # Create summary from all unique papers
    summary_text = "\n".join([f"Title: {p.title}\nSummary: {p.summary}" for p in unique_papers_for_summary])
    
    prompt = ChatPromptTemplate.from_template(prompts.summarize_papers_prompt)
    chain = prompt | text_llm
    summary = chain.invoke({"papers_summary": summary_text})
    
    # Update the count based on the number of unique papers added in this step
    newly_added_count = len(papers_to_add_to_state)
    new_total_count = current_count + newly_added_count
    
    return {
        "papers": papers_to_add_to_state, 
        "literature_summary": summary.content,
        "research_papers_count": new_total_count
    }

def reflect_on_summary(state: ProposalAgentState) -> ProposalAgentState:
    prompt = ChatPromptTemplate.from_template(prompts.reflect_on_literature_prompt)
    chain = prompt | json_llm.with_structured_output(Reflection)
    reflection = chain.invoke({"literature_summary": state['literature_summary']})
    return {"reflection": reflection}

def formulate_plan(state: ProposalAgentState) -> ProposalAgentState:
    prompt = ChatPromptTemplate.from_template(prompts.formulate_plan_prompt)
    chain = prompt | text_llm
    plan = chain.invoke({
        "knowledge_gap": state['reflection'].knowledge_gap,
        "literature_summary": state['literature_summary']
    })
    return {"research_plan": plan.content}

def assess_plan_novelty(state: ProposalAgentState) -> ProposalAgentState:
    current_reflection_count = state.get('reflection_papers_count', 0)
    
    if current_reflection_count >= MAX_REFLECTION_PAPERS:
        print(f"Reflection paper limit reached ({MAX_REFLECTION_PAPERS}). Assuming plan is novel.")
        # Force novelty to be true when limit is reached
        assessment = NoveltyAssessment(
            is_novel=True,
            similar_papers=[],
            justification=f"Paper search limit reached ({MAX_REFLECTION_PAPERS} papers). Proceeding with assumption of novelty."
        )
        return {"novelty_assessment": assessment}
    
    similar_papers = paper_search_tool.find_similar_papers(state['research_plan'], n_results=min(5, MAX_REFLECTION_PAPERS - current_reflection_count))
    prompt = ChatPromptTemplate.from_template(prompts.assess_novelty_prompt)
    chain = prompt | json_llm.with_structured_output(NoveltyAssessment)
    assessment = chain.invoke({
        "research_plan": state['research_plan'],
        "similar_papers": "\n".join([f"- {p['title']}: {p['url']}" for p in similar_papers])
    })
    
    new_reflection_count = current_reflection_count + len(similar_papers)
    return {
        "novelty_assessment": assessment,
        "reflection_papers_count": new_reflection_count
    }

def design_experiments(state: ProposalAgentState) -> ProposalAgentState:
    prompt = ChatPromptTemplate.from_template(prompts.design_experiments_prompt)
    chain = prompt | json_llm.with_structured_output(ExperimentProtocol)
    protocol = chain.invoke({"research_plan": state['research_plan']})
    return {"experiment_protocol": protocol}

def write_proposal(state: ProposalAgentState) -> ProposalAgentState:
    prompt = ChatPromptTemplate.from_template(prompts.write_proposal_prompt)
    chain = prompt | text_llm
    proposal = chain.invoke({
        "literature_summary": state['literature_summary'],
        "research_plan": state['research_plan'],
        "experiment_protocol": state['experiment_protocol'].model_dump_json()
    })
    return {"markdown_proposal": proposal.content}

def review_proposal(state: ProposalAgentState) -> ProposalAgentState:
    prompt = ChatPromptTemplate.from_template(prompts.review_proposal_prompt)
    chain = prompt | json_llm.with_structured_output(ProposalCritique)
    critique = chain.invoke({"markdown_proposal": state['markdown_proposal']})
    return {"critique": critique}

def plan_followup_research(state: ProposalAgentState) -> ProposalAgentState:
    """Takes the follow-up queries from the reflection and sets them as the new
    search queries for the next research loop."""
    follow_up_queries = state["reflection"].follow_up_queries
    return {"search_queries": follow_up_queries}

# --- Conditional Edges ---

def should_continue_research(state: ProposalAgentState) -> str:
    current_count = state.get('research_papers_count', 0)
    
    # Force progression if we've hit the paper limit
    if current_count >= MAX_RESEARCH_PAPERS:
        print(f"Research paper limit reached ({MAX_RESEARCH_PAPERS}). Proceeding to plan formulation.")
        return "formulate_plan"
    
    if state['reflection'].is_sufficient:
        return "formulate_plan"
    else:
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
builder.add_node("search_and_summarize", search_and_summarize)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("formulate_plan", formulate_plan)
builder.add_node("assess_plan_novelty", assess_plan_novelty)
builder.add_node("design_experiments", design_experiments)
builder.add_node("write_proposal", write_proposal)
builder.add_node("review_proposal", review_proposal)
builder.add_node("plan_followup_research", plan_followup_research)

builder.add_edge(START, "generate_queries")
builder.add_edge("generate_queries", "search_and_summarize")
builder.add_edge("search_and_summarize", "reflect_on_summary")
builder.add_conditional_edges(
    "reflect_on_summary",
    should_continue_research,
    {
        "formulate_plan": "formulate_plan",
        "plan_followup_research": "plan_followup_research"
    }
)
builder.add_edge("plan_followup_research", "search_and_summarize")

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
