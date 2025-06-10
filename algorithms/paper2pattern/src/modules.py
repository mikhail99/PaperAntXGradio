import dspy
import json
from .signatures import (
    DomainDictionarySignature, 
    ImplementationBlueprintSignature,
    SectionImportanceSignature
)

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

class BlueprintExtractor(dspy.Module):
    """A module to generate an Implementation Blueprint from a paper."""

    def __init__(self):
        super().__init__()
        self.generate_blueprint = dspy.ChainOfThought(ImplementationBlueprintSignature)

    def forward(self, title, abstract, paper_section):
        """
        Generates the implementation blueprint.

        Args:
            title (str): The title of the paper.
            abstract (str): The abstract of the paper.
            paper_section (str): The content of a key paper section.

        Returns:
            dspy.Prediction: An object containing the 'implementation_blueprint'.
        """
        prediction = self.generate_blueprint(
            title=title,
            abstract=abstract,
            paper_section=paper_section
        )
        return prediction

class ImportanceAssessor(dspy.Module):
    """A module to assess the importance of a paper section."""

    def __init__(self):
        super().__init__()
        self.assess_importance = dspy.ChainOfThought(SectionImportanceSignature)

    def forward(self, section_title, content_preview):
        """
        Assesses the importance of a paper section.

        Args:
            section_title (str): The title of the section.
            content_preview (str): A preview of the section's content.

        Returns:
            bool: True if the section is 'Important', False otherwise.
        """
        result = self.assess_importance(
            section_title=section_title,
            content_preview=content_preview
        )
        
        # The output is now a simple string: 'Important' or 'Unimportant'.
        return result.assessment.strip() == 'Important' 