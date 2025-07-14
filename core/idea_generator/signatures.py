import dspy
from typing import Literal
from models import IdeaTemplate

class IdeaTemplateSignature(dspy.Signature):
    """
    You're a responsible and helpful reasearch assistant helping to record paper abstract in a standardized format.
    """
    abstract: str = dspy.InputField(desc="Paper abstract.")
    idea_template: IdeaTemplate = dspy.OutputField()

class IdeaCompetition(dspy.Signature):
    """
    You're a bright and helpful reasearch assistant helping with research project idea assessment.
    You will be given two ideas A and B and you will need to judge which one is better.
    Important criteria:
    - Clarity
    - Feasibility
    - Novelty
    - Impact
    You will need to return "A" if idea A is better and "B" if idea B is better.
    """
    idea_A: str = dspy.InputField(desc="Idea A to compare.")
    idea_B: str = dspy.InputField(desc="Idea B to compare.")
    winner: Literal["A", "B"] = dspy.OutputField(description="The better project idea.")

class IdeaEvolution(dspy.Signature):
    """
    You're a bright and helpful reasearch assistant helping with research project ideation.
    You will be given two ideas and you will need write a new project idea that is dervied from the combination of the two.
    The new project idea should combine the strengths of the two parent ideas and satisfy the following criteria:
    - Clarity
    - Feasibility
    - Novelty
    - Impact
    """
    idea_A: IdeaTemplate = dspy.InputField(desc="Idea A.")
    idea_B: IdeaTemplate = dspy.InputField(desc="Idea B.")
    new_idea: IdeaTemplate = dspy.OutputField()