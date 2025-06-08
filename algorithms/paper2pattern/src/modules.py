import dspy
from .signatures import DomainDictionarySignature, FieldSignature, MermaidSignature

class DictionaryExtractor(dspy.Module):
    """A dspy.Module for extracting a domain dictionary from a paper section."""

    def __init__(self):
        super().__init__()
        # Initialize the ChainOfThought module with our custom signature
        self.generate_dictionary = dspy.ChainOfThought(DomainDictionarySignature)

    def forward(self, title, abstract, paper_section):
        """
        Executes the dictionary extraction.

        Args:
            title (str): The title of the paper.
            abstract (str): The abstract of the paper.
            paper_section (str): The content of the paper section.

        Returns:
            dspy.Prediction: An object containing the extracted 'domain_dictionary'.
        """
        prediction = self.generate_dictionary(
            title=title,
            abstract=abstract,
            paper_section=paper_section
        )
        return prediction

class StandalonePatternExtractor(dspy.Module):
    """A module to extract a full design pattern from a paper, field by field."""

    def __init__(self):
        super().__init__()
        self.fields_to_extract = [
            "Intent", "Motivation", "Applicability", 
            "Participants", "Collaborations", "Consequences",
            "Implementation", "Sample Code", "Known Uses", "Related Patterns"
        ]
        
        # Predictor for standard text fields
        self.field_extractor = dspy.ChainOfThought(FieldSignature)
        
        # Specialized predictor for the structure diagram
        self.structure_extractor = dspy.ChainOfThought(MermaidSignature)

    def forward(self, paper_context):
        """
        Executes the full pattern extraction process.

        Args:
            paper_context (str): The full text of the paper.

        Returns:
            dict: A dictionary containing all the extracted pattern fields.
        """
        extracted_pattern = {}

        # Extract standard fields
        for field_name in self.fields_to_extract:
            print(f"Extracting field: {field_name}...")
            prediction = self.field_extractor(
                paper_context=paper_context,
                field_name=field_name
            )
            extracted_pattern[field_name] = prediction.field_value
        
        # Extract the structure as a Mermaid diagram
        print("Extracting field: Structure...")
        prediction = self.structure_extractor(
            paper_context=paper_context,
            field_name="Structure"
        )
        extracted_pattern["Structure"] = prediction.mermaid_diagram
        
        print("Pattern extraction complete.")
        return extracted_pattern 