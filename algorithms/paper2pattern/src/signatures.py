import dspy

class DomainDictionarySignature(dspy.Signature):
    """Extract key domain terms and their definitions from a given paper section, using the paper's title and abstract for context."""

    title = dspy.InputField(
        desc="The title of the scientific paper."
    )
    abstract = dspy.InputField(
        desc="The abstract of the scientific paper."
    )
    paper_section = dspy.InputField(
        desc="A section of a scientific paper from which to extract terms."
    )
    domain_dictionary = dspy.OutputField(
        desc='A JSON string representing a list of objects, where each object has a "term" and a "definition" key. Example: [{"term": "T1", "definition": "D1"}, {"term": "T2", "definition": "D2"}]'
    )

class FieldSignature(dspy.Signature):
    """Extract a specific field of a design pattern from a scientific paper."""

    paper_context = dspy.InputField(
        desc="The full text of the paper, providing context for extraction."
    )
    field_name = dspy.InputField(
        desc="The specific design pattern field to extract (e.g., Intent, Motivation, Applicability)."
    )
    field_value = dspy.OutputField(
        desc="The extracted value or description for the specified field."
    )

class MermaidSignature(dspy.Signature):
    """Generate a Mermaid.js diagram representing the structure of a design pattern from a scientific paper."""

    paper_context = dspy.InputField(
        desc="The full text of the paper, providing context for generating the diagram."
    )
    field_name = dspy.InputField(desc="Must be 'Structure'.")
    
    mermaid_diagram = dspy.OutputField(
        desc="A string containing a Mermaid.js diagram representing the pattern's structure."
    )

class ImplementationBlueprintSignature(dspy.Signature):
    """From a paper's title, abstract, and a key section, generate a concise implementation blueprint."""

    title = dspy.InputField(
        desc="The title of the scientific paper."
    )
    abstract = dspy.InputField(
        desc="The abstract of the scientific paper."
    )
    paper_section = dspy.InputField(
        desc="A key section of the paper (e.g., Introduction or Methodology) that describes the core contribution."
    )
    implementation_blueprint = dspy.OutputField(
        desc=("A detailed, free-text description that explains how to implement the core ideas of the paper. "
              "It should cover the problem being solved, the core technical approach, key components or algorithms, "
              "and the main steps required for implementation.")
    )

class SectionImportanceSignature(dspy.Signature):
    """Assess the importance of a paper section for understanding the core implementation."""

    section_title = dspy.InputField(desc="The title of the paper section.")
    content_preview = dspy.InputField(desc="The first 100 words of the section's content.")
    
    assessment = dspy.OutputField(
        desc="A single word: 'Important' or 'Unimportant'. Important sections typically include Introduction, Methods, Results, or core technical descriptions. Unimportant sections include References, Acknowledgements, Appendices, or boilerplate."
    ) 