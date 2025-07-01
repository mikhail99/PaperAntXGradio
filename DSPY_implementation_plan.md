# DSPy Integration Plan for Research Proposal Agent

## 1. Objective

This document outlines a phased approach to integrate the DSPy framework into the Research Proposal Agent. The primary goal is to enhance the quality, reliability, and maintainability of the agent's Language Model (LLM) components by replacing hard-coded, static prompts with compilable, optimized DSPy modules.

This integration will leverage LangGraph for high-level workflow orchestration (managing agent states, teams, and human-in-the-loop interactions) while delegating low-level prompt engineering and LLM interaction to DSPy.

---

## 2. Guiding Principles

- **Incremental Adoption:** Introduce DSPy one component at a time to minimize risk and ensure stability.
- **Maintain Core Logic:** The existing LangGraph workflow, state management, and human-in-the-loop (HIL) systems will remain unchanged.
- **Testability:** Each change must be verifiable through the existing test suite, adapted as necessary.
- **Measurable Improvement:** The final phase will focus on demonstrating qualitative and quantitative improvements from DSPy's optimization capabilities.

---

## 3. Phase 1: Setup & Foundational Integration

**Goal:** Replace a single LLM node (`query_generator_base`) with a non-optimized DSPy module to establish the integration pattern.

- **Task 1.1: Environment & Configuration**
    - **Action:** Add `dspy-ai` to the project's dependency file (e.g., `requirements.txt`).
    - **Action:** Create a new file `core/proposal_agent/dspy_config.py` to centralize DSPy's configuration. This will handle setting the global `dspy.LM` (e.g., `dspy.OpenAI`, `dspy.Ollama`).

- **Task 1.2: Create First DSPy Module**
    - **Action:** Create `core/proposal_agent/dspy_modules.py`.
    - **Action:** Define the first module, `GenerateQueries(dspy.Module)`, within this new file.
    - **Details:**
        - It will use a simple signature like `dspy.Signature("research_topic -> search_queries")`.
        - The module will initially wrap a `dspy.ChainOfThought` or `dspy.Predict` module.

- **Task 1.3: Integrate Module into the Graph**
    - **Action:** Modify the node creation logic in `core/proposal_agent/graph.py`.
    - **Details:** The function responsible for the `query_generator_base` node will now instantiate `GenerateQueries`, invoke it with the topic from the agent state, and place the resulting queries back into the state. It will no longer reference `prompts.json`.

- **Task 1.4: Update Parrot Services for Testing**
    - **Action:** Adapt `core/proposal_agent/modern_parrot_services.py`.
    - **Details:** The parrot service must now mock the behavior of a DSPy module call, returning a `dspy.Prediction` object with the expected output fields (e.g., `search_queries`). This ensures integration tests can run without live LLM calls.

- **Task 1.5: Verification**
    - **Action:** Execute the full test suite.
    - **Expected Outcome:** The Research Proposal Agent completes its workflow successfully. The only internal change is that query generation is now powered by a foundational DSPy module.

---

## 4. Phase 2: Full Component Conversion

**Goal:** Convert all remaining LLM-based nodes to DSPy modules, fully deprecating the old prompt management system.

- **Task 2.1: Convert All LLM Nodes**
    - **Action:** Systematically create DSPy modules in `dspy_modules.py` for every LLM call defined in `agent_config.json` and `prompts.json`.
    - **Modules to Create:** `SynthesizeLiteratureReview`, `FormulatePlan`, `ReviewFeasibility`, `ReviewNovelty`, and `SynthesizeReview`.
    - **Details:** For modules requiring structured output (e.g., critiques with a score and justification), use Pydantic models as type hints in the `dspy.Signature` to guide the LLM's output.

- **Task 2.2: Finalize Graph Integration**
    - **Action:** Update `core/proposal_agent/graph.py` to ensure all corresponding nodes now use their respective DSPy modules.

- **Task 2.3: Deprecate Old System**
    - **Action:** Once all modules are converted and tests are passing, safely delete `core/proposal_agent/prompts.json` and the now-unused `core/proposal_agent/prompts.py`.

- **Task 2.4: Verification**
    - **Action:** Run the end-to-end test suite.
    - **Expected Outcome:** The agent functions identically to the end-user, but its internal logic is now fully based on DSPy modules.

---

## 5. Phase 3: Optimization & Evaluation

**Goal:** Leverage DSPy's core strength by compiling modules with training data to improve performance.

- **Task 3.1: Curate Training Datasets**
    - **Action:** Create a new directory: `core/proposal_agent/dspy_training_data/`.
    - **Action:** For each DSPy module, create a small (`.py` or `.jsonl`) training file containing 5-10 high-quality `dspy.Example` objects. An example consists of the input(s) and the ideal output(s).

- **Task 3.2: Build the Optimization Pipeline**
    - **Action:** Create a new script: `core/proposal_agent/optimize_modules.py`.
    - **Details:** This script will:
        1.  Load the training data for a specific module.
        2.  Define a simple evaluation metric (e.g., output format correctness, keyword matching).
        3.  Instantiate a DSPy optimizer, like `dspy.BootstrapFewShot`.
        4.  Run the `optimizer.compile()` process.
        5.  Save the resulting optimized module to `dspy_compiled_modules.json`.

- **Task 3.3: Load Optimized Modules**
    - **Action:** Modify the node logic in `graph.py`. Before using a DSPy module, it should first attempt to load the optimized state from `dspy_compiled_modules.json`.
    - **Example:** `my_module.load("path/to/compiled.json")`.

- **Task 3.4: Evaluate Performance**
    - **Action:** Conduct qualitative A/B testing by running the agent with both the default (zero-shot) and the compiled (few-shot) modules.
    - **Expected Outcome:** Observe a measurable improvement in the quality of generated content, such as more relevant search queries, more coherent summaries, and more insightful critiques.

---

## 6. Phase 4: Intelligent HIL Routing (Advanced Feature)

**Goal:** Transform from fixed human checkpoints to intelligent routing based on complexity and confidence.

- **Task 4.1: Router Architecture**
    - **Action:** Create `core/proposal_agent/routing_modules.py` with DSPy-based routing components.
    - **Details:** Implement `QueryComplexityRouter`, `LiteratureComplexityRouter`, and `ProposalComplexityRouter` as DSPy modules that predict when human intervention is needed.

- **Task 4.2: Feature Engineering**
    - **Action:** Develop feature extraction for routing decisions.
    - **Details:** Extract features like topic clarity, domain familiarity, AI reviewer agreement, and user expertise level to inform routing decisions.

- **Task 4.3: Training Data Collection**
    - **Action:** Collect historical data on when human intervention was actually valuable.
    - **Details:** Label past proposal runs with whether the human feedback materially changed the outcome, creating training data for routing optimization.

- **Task 4.4: Smart Graph Integration**
    - **Action:** Modify `modern_graph_builder.py` to include conditional routing nodes.
    - **Details:** Add router nodes between each stage and HIL checkpoint, creating paths for `auto_continue`, `human_review`, and `expert_escalation`.

- **Task 4.5: Routing Configuration**
    - **Action:** Create `core/proposal_agent/routing_config.json`.
    - **Details:** Define confidence thresholds, feature weights, and user profile adjustments for routing decisions.

- **Task 4.6: Evaluation & Tuning**
    - **Action:** A/B test fixed HIL vs. intelligent routing.
    - **Expected Outcome:** Reduce human review burden from 100% to 20-30% while maintaining quality, achieving 3-4x processing throughput improvement.

---

## 7. New & Modified Files Summary

- **New:**
    - `DSPY_implementation_plan.md`
    - `core/proposal_agent/dspy_config.py`
    - `core/proposal_agent/dspy_modules.py`
    - `core/proposal_agent/optimize_modules.py`
    - `core/proposal_agent/dspy_training_data/` (Directory)
    - `dspy_compiled_modules.json` (Generated artifact)
    - `core/proposal_agent/routing_modules.py`
    - `core/proposal_agent/routing_config.json`

- **Modified:**
    - `requirements.txt`
    - `core/proposal_agent/graph.py`
    - `core/proposal_agent/modern_parrot_services.py`
    - Associated testing files.

- **Deprecated:**
    - `core/proposal_agent/prompts.json`
    - `core/proposal_agent/prompts.py` 