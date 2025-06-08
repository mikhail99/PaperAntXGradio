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
        desc='A string containing a list of "Term: Definition" pairs, with each pair on a new line.'
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