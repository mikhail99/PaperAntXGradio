# Prompts for the Research Proposal Agent

# 1. Query Generation
generate_queries_prompt = """
You are an expert at generating academic search queries.
Based on the research topic "{topic}", generate a list of {num_queries} diverse and specific search queries for the ArXiv search engine.
Format the output as a JSON object with a single key "queries" containing a list of strings.

Example:
Topic: "The impact of transformers on natural language processing"
```json
{{
    "queries": [
        "transformer architecture in NLP",
        "self-attention mechanism for language models",
        "BERT and its variants performance",
        "GPT-3 few-shot learning capabilities",
        "efficient transformers for long sequences"
    ]
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
You are an expert research strategist. Analyze the provided literature summary to identify knowledge gaps.
Based on the summary, determine if the information is sufficient to formulate a novel research plan.
If not, generate a list of follow-up questions to address the gaps.

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
Is the plan sufficiently different to be considered novel?
Provide a justification for your assessment and a list of URLs for the most similar papers.

Format the output as a JSON object with the keys "is_novel", "justification", and "similar_papers".

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
