import dspy
from typing import Literal


class IdeaTemplateSignature(dspy.Signature):
    """
    You're a responsible and helpful reasearch assistant helping to record paper abstract in a standardized format.
    You will be given a paper abstract and you will need to record the abstract in a standardized format.
    The standardized format is as follows:
    - Research question : <research_question>
    - Hypothesis: <hypothesis>
    - Methodology: <methodology>
    - Data: <data used in the study>
    - Applications: <applications>
    If some fields are not applicable, you can leave them blank. Do not include benchmark results of the paper.
    """
    abstract: str = dspy.InputField(desc="Paper abstract.")
    idea_template: str = dspy.OutputField()

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
    You will be given two paper summaries and a context of the field.  
    You will need write a new project idea that is dervied from the combination of the two paper summaries (potentially using some of the context).
    The new project idea should satisfy the following criteria:
    - Clarity
    - Feasibility
    - Novelty
    - Impact
    Do not just pile up the ideas, search for an interesting combination of a few ideas.
    You will need to return the new idea in a standardized format.
    The standardized format is as follows:
    - Research question: <research_question>
    - Hypothesis: <hypothesis>
    - Methodology: <methodology>
    - Data: <data used in the study>
    - Applications: <applications>
    If some fields are not applicable, you can leave them blank.
    The explanation of the new idea should be self-contained assuming no knowledge from the reader about the parent ideas.
    """
    idea_A: str = dspy.InputField(desc="Idea A.")
    idea_B: str = dspy.InputField(desc="Idea B.")
    aha_information: str = dspy.InputField(desc="Key ideas from similar papers.")
    new_idea: str = dspy.OutputField()


class AHAMomentSignature(dspy.Signature):
    """
    You're a bright and helpful reasearch assistant helping with abstractassessment.
    You will be given an abstract and you will need to judge if there is an AHA information: something that is very surprising, and very important.
    AHA information could be a new finding, a new method, a new library, an new dataset, etc. that is new to you and is likely to generate many new research questions.
    In other words you are predicting if the paper is a game changer with high citation potential.
    Example of AHA information:
    - Vision transformers
    - Diffusion models
    - PyG 2.0 is released
    Be critical, most of the papers to not have any AHA information, respose "None" if you don't find any AHA information.
    If you find an AHA information, write its description in a self-contained way.
    """
    abstract: str = dspy.InputField(desc="Paper abstract.")
    AHA_information: str = dspy.OutputField()


class AHAKeyIdeas(dspy.Signature):
    """
    You're a bright and helpful reasearch assistant helping with abstract assessment.
    You will be given a list of ideas/information extracted from a collection of papers.
    You need to extract the top 10 ideas/information from the list.
    The best ideas/information are the most surprising and important information, game changer that will generate many new research questions.
    """
    idea_information: str = dspy.InputField(desc="AHA information.")
    key_ideas: str = dspy.OutputField()