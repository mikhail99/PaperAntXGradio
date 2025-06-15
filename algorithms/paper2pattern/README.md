# Paper to Implementation Blueprint Pipeline

This project provides a multi-stage pipeline for extracting structured information from scientific papers. It uses a local LLM, orchestrated by the DSPy library, to intelligently analyze papers, assess the importance of different sections, extract domain-specific terminology, and generate high-level "implementation blueprints" for the core concepts.

## Technology Stack

-   **Orchestration**: [DSPy](https://github.com/stanfordnlp/dspy)
-   **LLM Serving**: [Ollama](https://ollama.com/)
-   **HTML Parsing**: [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/)
-   **Core Libraries**: PyTorch, Transformers, Sentence-Transformers

## Project Structure

-   `/papers/[paper_id]/paper.html`: The input HTML file for a given paper.
-   `/outputs/[paper_id]/`: Directory for all generated files.
    -   `sections.json`: The paper content, split into structured sections.
    -   `importance.json`: The assessment of which sections are important for implementation.
    -   `dictionary.json`: A dictionary of key terms and definitions, per section.
    -   `blueprint.json`: The final implementation blueprints, per important section.
-   `/src/`: All Python source code.
    -   `html_section_splitter.py`: Logic for splitting the HTML paper into sections.
    -   `dspy_config.py`: Configuration for DSPy and the LLM.
    -   `signatures.py`: DSPy signatures that define the tasks for the LLM.
    -   `modules.py`: DSPy modules that execute the tasks.
-   `main.py`: The main script to run the pipeline.
-   `requirements.txt`: Python dependencies.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure LLM:**
    This project is configured to use a local Ollama model. Ensure you have Ollama installed and have pulled a model (e.g., `ollama pull gemma3:12b`). The model name is configured in `src/dspy_config.py`.

3.  **Add a Paper:**
    -   Create a directory inside `/papers` named after the paper's ID (e.g., `/papers/arxiv:2504.0541v2`).
    -   Place the HTML version of the paper inside this directory as `paper.html`.

## How to Use the Pipeline

The pipeline is designed to be run as a sequence of discrete, resumable steps. Each command processes files from the `/papers` directory and saves its output in the `/outputs` directory, allowing you to stop and restart the process without losing progress.

### Step 1: Split Paper into Sections

First, you need to preprocess the HTML paper into a structured JSON file containing the paper's sections. This command reads the `paper.html` file and creates `sections.json`.

**Command:**
```bash
python main.py <paper_id> split
```
**Example:**
```bash
python main.py arxiv:2504.0541v2 split
```
**Output:** This will create `outputs/[paper_id]/sections.json`.

### Step 2: Assess Section Importance

This step uses the LLM to determine which sections of the paper are relevant for creating an implementation blueprint. It intelligently filters out sections like "References", "Conclusion", and "Introduction".

**Command:**
```bash
python main.py <paper_id> assess
```
**Example:**
```bash
python main.py arxiv:2504.0541v2 assess
```
**Output:** This will create `outputs/[paper_id]/importance.json`.

### Step 3: Extract Domain Dictionary

Using the importance assessment from the previous step, this command extracts key terms and definitions from the most relevant sections.

**Command:**
```bash
python main.py <paper_id> dictionary
```
**Example:**
```bash
python main.py arxiv:2504.0541v2 dictionary
```
**Output:** This will create `outputs/[paper_id]/dictionary.json`.

### Step 4: Generate Implementation Blueprints

This is the final step. It uses the paper's content, the importance assessment, and the domain dictionary to generate a detailed, free-text implementation blueprint for each important section.

**Command:**
```bash
python main.py <paper_id> blueprint
```
**Example:**
```bash
python main.py arxiv:2504.0541v2 blueprint
```
**Output:** This will create `outputs/[paper_id]/blueprint.json`.

## Recommended Workflow

The entire pipeline can be executed by running the following commands in order.

1.  Place your `paper.html` in the correct `/papers/[paper_id]/` directory.
2.  Run `python main.py <paper_id> split` to create `sections.json`.
3.  Run `python main.py <paper_id> assess` to create `importance.json`.
4.  Run `python main.py <paper_id> dictionary` to create `dictionary.json`.
5.  Run `python main.py <paper_id> blueprint` to create `blueprint.json`.
6.  Inspect the final results in `outputs/[paper_id]/blueprint.json`. 