# Prompts for the Research Proposal Agent

# 1. Query Generation
generate_query_prompt = """
You are an expert at generating academic search queries. Your goal is to generate one novel query at a time to explore a research topic.
Based on the research topic "{topic}", and the identified knowledge gap (if any), generate a single new search query for the ArXiv search engine. Avoid generating queries that are similar to the ones already tried.

Research Topic: {topic}
Identified Knowledge Gap: {knowledge_gap}
Previously tried queries:
{previous_queries}

Your task is to generate the next query. 
- If an "Identified Knowledge Gap" is provided, formulate a query that specifically addresses this gap.
- If the knowledge gap is empty, generate a new, broader query based on the "Research Topic".
- The query MUST be different from the "Previously tried queries".

Format the output as a JSON object with a single key "query" containing the new query string.

Example:
Topic: "The impact of transformers on natural language processing"
Knowledge Gap: "The performance of transformer models on low-resource languages is not well understood."
Previously tried queries: ["transformer architecture in NLP", "self-attention mechanism for language models"]
```json
{{
    "query": "transformer models for low-resource language translation"
}}
```
"""

# 2. Literature Summary
summarize_papers_prompt = """
You are a research assistant. Based on the following paper summaries, create a concise and coherent literature review.
The review should synthesize the key findings and identify the general trend of the research.
Do not include information not present in the provided summaries.

Papers:
{papers_summary}
"""

# 3. Reflection on Literature
reflect_on_literature_prompt = """
You are a critical and efficient research strategist. Your goal is to formulate a research plan with the minimum necessary information.
Analyze the provided literature summary. Determine if it provides a reasonable basis for a research plan.

The following queries have already been run:
{previous_queries}

Only if the summary is clearly and critically insufficient, set "is_sufficient" to false and generate a maximum of 2-3 essential **and new** follow-up questions that are **different** from the ones already run.
Otherwise, set "is_sufficient" to true.

Format the output as a JSON object with the keys "is_sufficient", "knowledge_gap", and "follow_up_queries".

Literature Summary:
{literature_summary}
"""

# 4. Plan Formulation
formulate_plan_prompt = """
You are a Postdoctoral researcher, a strategist in research.
Based on the literature summary and the identified knowledge gap, formulate a high-level research plan.
The plan should outline a novel contribution, a clear hypothesis, and a proposed approach.

Knowledge Gap: {knowledge_gap}
Literature Summary: {literature_summary}
"""

# 5. Novelty Assessment
assess_novelty_prompt = """
You are a critical reviewer. Assess the novelty of the proposed research plan by comparing it against the summaries of similar papers found in the database.
Is the plan sufficiently different to be considered novel? Provide a justification for your assessment.

Format the output as a JSON object with the keys "is_novel" and "justification".

Proposed Research Plan:
{research_plan}

Similar Papers:
{similar_papers}
"""

# 6. Experiment Design
design_experiments_prompt = """
You are an ML Engineer. Translate the high-level research plan into a detailed, reproducible experimental protocol.
Specify the methodology, datasets, evaluation metrics, and provide pseudocode for the core algorithm.

Format the output as a JSON object with the keys "methodology", "datasets", "evaluation_metrics", and "pseudocode".

Research Plan:
{research_plan}
"""

# 7. Proposal Writing
write_proposal_prompt = """
You are an automated scientific writer. Synthesize the following artifacts into a single, coherent, and professionally formatted research proposal in markdown.
The proposal should have sections for Introduction, Related Work (from the literature summary), Proposed Method (from the research plan), and Experimental Setup (from the experiment protocol).

Literature Summary:
{literature_summary}

Research Plan:
{research_plan}

Experiment Protocol:
{experiment_protocol}
"""

# 8. Proposal Review
review_proposal_prompt = """
You are a peer reviewer. Critically evaluate the generated markdown research proposal.
Identify its strengths and weaknesses. Determine if a revision is required and provide specific, actionable suggestions for improvement.

Format the output as a JSON object with the keys "strengths", "weaknesses", "revision_required", and "suggested_changes".

Markdown Proposal:
{markdown_proposal}
"""
