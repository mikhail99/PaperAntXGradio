# Paper2ImplementationDoc: Modular Pipeline Implementation Sketch (Stage-based Naming)

This document outlines a **stage-based, modular file structure** for the pipeline, inspired by best practices from seed projects and the **two-stage hybrid approach** discussed. Each stage has its own nodes and flow definition for clarity and extensibility.

The pipeline takes arxiv article (title, abstract, pdf) and generates summaries of the article (algorithms, workflows, ...) 

## **Pipeline Overview: Two-Stage Architecture**

**Stage I: Extended Planning** - Comprehensive up-front planning with explicit traceability  
**Stage II: Guided Summarization** - Detailed analysis using planned abstractions and connections

---

## **Stage 0: PDF Processing**

- **0_pdf_process_nodes.py**  
  _Nodes for PDF validation, download, and text extraction (e.g., from arXiv)._  
  _Handles extracting raw text and metadata from PDFs._

- **0_pdf_process_flow.py**  
  _Defines the PocketFlow pipeline for PDF processing stage._

---

## **Stage I: Extended Planning**

### **Stage 1: Section Planning**

- **1_section_planning_nodes.py**  
  _Nodes for section splitting (regex), section selection (LLM), and saving selected sections._  
  _Flow: split_sections >> select_sections >> save_sections_

- **1_section_planning_flow.py**  
  _Defines the PocketFlow pipeline for section planning stage._

### **Stage 2: Abstraction Planning**

- **2_abstraction_planning_nodes.py**  
  _Nodes for identifying and categorizing abstractions (algorithms, methods, datasets, workflows)._  
  _Uses hybrid rule-based + LLM approach for parameterizable abstraction detection._  
  _Flow: identify_abstractions >> categorize_abstractions >> save_abstractions_

- **2_abstraction_planning_flow.py**  
  _Defines the PocketFlow pipeline for abstraction planning stage._

### **Stage 3: Connection Planning**

- **3_connection_planning_nodes.py**  
  _Nodes for mapping relationships between abstractions, dependencies, and workflow connections._  
  _Flow: analyze_dependencies >> map_connections >> save_connections_

- **3_connection_planning_flow.py**  
  _Defines the PocketFlow pipeline for connection planning stage._

---

## **Stage II: Guided Summarization**

### **Stage 4: Component Summarization**

- **4_summarization_nodes.py**  
  _Nodes for detailed LLM-based summarization using planned abstractions and connections._  
  _Leverages planning artifacts for focused, structured analysis._

- **4_summarization_flow.py**  
  _Defines the PocketFlow pipeline for component summarization stage._

### **Stage 5: Review & QA**

- **5_review_nodes.py**  
  _Nodes for reviewing, validating, and (optionally) redoing LLM summaries. Includes QA nodes for user queries._

- **5_review_flow.py**  
  _Defines the PocketFlow pipeline for the review and QA stage._

### **Stage 6: Documentation Generation**

- **6_docgen_nodes.py**  
  _Nodes for combining summaries into a final implementation guide (Markdown, diagrams, etc.)._

- **6_docgen_flow.py**  
  _Defines the PocketFlow pipeline for documentation generation stage._

---

## **Shared/Utility Modules**

- **utils/pdf_processor.py**  
  _PDF validation and text extraction._

- **utils/section_detector.py**  
  _Regex-based section splitting._

- **utils/abstraction_detector.py**  
  _Hybrid rule-based + LLM extraction of algorithms, methods, datasets, and workflows._  
  _Parameterizable abstraction types with structured output._

- **utils/component_analyzer.py**  
  _Rule-based extraction of obvious candidates (code blocks, equations, figures)._

- **utils/connection_mapper.py**  
  _Analysis of dependencies and relationships between abstractions._

- **utils/llm_interface.py**  
  _Centralized utility for calling LLMs._

- **utils/arxiv_api.py**  
  _Fetches arXiv metadata and downloads PDFs._

---

## **Planning Artifacts (Output Structure)**

- **output/section_plan.json**  
  _Selected sections with metadata and reasoning._

- **output/abstraction_plan.json**  
  _Identified abstractions categorized by type (algorithms, methods, datasets, workflows)._

- **output/connection_plan.json**  
  _Mapped relationships, dependencies, and workflow connections._

- **output/summaries/**  
  _Detailed summaries for each planned component._

- **output/final_implementation_guide.md**  
  _Combined final documentation._

---

## **Other Supporting Files**

- **requirements.txt**  
  _Python dependencies._

- **README.md**  
  _Project documentation and usage instructions._

- **output/**  
  _Directory for generated documentation and artifacts._

- **test_paper.pdf, test_research_paper.pdf, ...**  
  _Test input files._

---

## **Optional/Advanced**

- **utils/html_processor.py**  
  _For future: HTML/LaTeX input support._

- **utils/embedding_search.py**  
  _For future: Embedding-based retrieval for QA or section selection._

- **utils/diagram_generator.py**  
  _For future: Automatic workflow diagram generation from connections._

- **test/**  
  _Unit and integration tests for pipeline components._

---

## **Key Design Benefits**

- **Explicit Planning:** All decisions are planned up-front with clear traceability
- **Separation of Concerns:** Planning logic separated from summarization logic  
- **Modularity:** Each stage is independent and can be debugged/optimized separately
- **Reusability:** Planning artifacts can be reused for different summarization approaches
- **Extensibility:** Easy to add new abstraction types or connection patterns
- **Robustness:** Hybrid rule-based + LLM approach provides fallbacks

---

## **Summary**

- **Two-stage architecture with extended planning and guided summarization**
- **Each stage has its own nodes and flow file for clarity**
- **Planning artifacts provide explicit traceability and reusability**
- **Hybrid approaches maximize both accuracy and robustness**
- **Ready for iterative implementation!** 