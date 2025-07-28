import random
# Assuming dspy is your LLM interface
import dspy
#evolution algotithm go generate new abstracts 
from typing import Literal, List
# --- Parameters ---
POP_SIZE = 16 # must be a power of 2
WIN_THRESHOLD = 2
N_GENERATIONS = 3

from core.copilots.project_proposal.idea_generator.models import Candidate
from core.copilots.project_proposal.idea_generator.signatures import IdeaEvolution, IdeaCompetition, IdeaTemplateSignature, AHAMomentSignature, AHAKeyIdeas
from pydantic import BaseModel

class Abstract(BaseModel):
    id: str
    text: str


def abstract_to_aha(abstract:Abstract)->str:
    aha_detector = dspy.ChainOfThought(AHAMomentSignature)
    try:
        print(f"Abstract => AHA information")
        aha_information = aha_detector(abstract=abstract.text).AHA_information
        if len(aha_information) >  20:
            return aha_information
        else:
            return ""
    except Exception as e:
        print(f"Error converting abstract {abstract.id} to candidate: {e}")
        # Return a dummy or fallback candidate, or handle error as needed
        return ""

def abstract_to_candidate(abstract:Abstract)->Candidate:
    converter = dspy.Predict(IdeaTemplateSignature)
    try:
        print(f"Abstract => Idea template")
        idea_template = converter(abstract=abstract.text).idea_template

        # The output from the LLM is a string representation of the Pydantic model.
        # We need to parse it into an actual IdeaTemplate object.
        # This assumes the LLM returns a JSON-compatible string.
        return Candidate(id=abstract.id, idea=idea_template, win_count=0)
    except Exception as e:
        print(f"Error converting abstract {abstract.id} to candidate: {e}")
        # Return a dummy or fallback candidate, or handle error as needed
        return None


# --- LLM-based generation and evaluation (placeholders) ---
def generate_new_candidate(parent1:Candidate, parent2:Candidate,context: str, temperature:float) -> Candidate:
    generator = dspy.ChainOfThought(IdeaEvolution)
    try:
        print(f"Parent1, Parent2 => Child")
        dspy.settings.configure(temperature=temperature)
        result = generator(idea_A=parent1.idea, idea_B=parent2.idea, context=context)
        new_idea_str : str = result.new_idea
        #new_idea = IdeaTemplate.model_validate_json(new_idea_str)
        new_id = f"{parent1.id}-{parent2.id}-{random.randint(1000, 9999)}"
        return Candidate(id=new_id, idea=new_idea_str, win_count=0)
    except Exception as e:
        print(f"Error generating new candidate from {parent1.id} and {parent2.id}: {e}")
        return None

def judge_abstracts(candidate1:Candidate, candidate2:Candidate)->int :
    selector = dspy.ChainOfThought(IdeaCompetition)
    print(f"Candidate1, Candidate2 => Winner")
    result = selector(idea_A=candidate1.idea, idea_B=candidate2.idea)
    print(f"Judge result: {result}")
    if "A" in result.winner:
        return 0
    else:
        return 1


def get_random_abstracts_from_collection(all_abstracts:List[Abstract], num_abstracts:int)-> List[Abstract]:
    
    return random.sample(all_abstracts, num_abstracts)

def get_new_abstract(all_abstracts:List[Abstract]) -> Abstract:
    '''
    randomly draws 1 abstract from the collection
    '''
    random_abstracts = get_random_abstracts_from_collection(all_abstracts, num_abstracts=1)

    return random_abstracts[0]

def get_new_candidate(all_abstracts:List[Abstract]) -> Candidate:
    '''
    Randomly draws 1 abstract from the collection and converts it to a Candidate.
    '''
    new_abstract = get_new_abstract(all_abstracts)
    return abstract_to_candidate(new_abstract)

def initialize_population(all_abstracts:List[Abstract], population_size:int)->List[Candidate]:

    initial_candidates_abstracts = get_random_abstracts_from_collection(all_abstracts, population_size)
    initial_candidates = [abstract_to_candidate(abstract) for abstract in initial_candidates_abstracts]
    return initial_candidates

def get_population_aha(all_abstracts:List[Abstract])->str:
    aha_information = [abstract_to_aha(abstract) for abstract in all_abstracts]
    summarizer = dspy.Predict(AHAKeyIdeas)
    print(f"AHA Summary")
    key_ideas = summarizer(idea_information="\n".join(aha_information)).key_ideas
    return key_ideas

def pairwise_competition(population: List[Candidate]) -> List[Candidate]:
    # Shuffle and pair up for competitions
    random.shuffle(population)
    survivors = []
    for i in range(0, len(population)-1, 2):
        a1, a2 = population[i], population[i+1]
        winner_idx = judge_abstracts(a1, a2)
        winner = a1 if winner_idx == 0 else a2
        winner.win_count += 1
        survivors.append(winner)
    return survivors

def simplified_evolutionary_abstracts(all_abstracts: List[Abstract], context: str) -> Candidate:
    # Start with initial population
    population = initialize_population(all_abstracts, POP_SIZE)  # or any power of 2
    
    while len(population) > 1:
        print(f"Generation: {len(population)} candidates")
        
        # Pair up and generate children
        new_population = []
        for i in range(0, len(population), 2):
            if i + 1 < len(population):
                p1, p2 = population[i], population[i + 1]
                
                # Generate 2 children
                c1 = generate_new_candidate(p1, p2, context=context, temperature=0.3)
                c2 = generate_new_candidate(p2, p1, context=context, temperature=0.7)
                
                # Keep only the better child
                winner = judge_abstracts(c1, c2)
                selected_child = c1 if winner == 0 else c2
                new_population.append(selected_child)
            else:
                # Odd number - keep the last one
                new_population.append(population[i])
        
        population = new_population
    
    return population[0]  # Final winner


from core.collections_manager import CollectionsManager
if __name__ == "__main__":

    dspy.configure(lm=dspy.LM('ollama_chat/qwen3:4b', api_base='http://localhost:11434', api_key=''))

    manager = CollectionsManager()
    collection_name = "HuggingFaceDailyPapers" #TODO: change this to the collection you want to use
    collection = manager.get_collection_by_name(collection_name)
    print(list(collection.articles.values())[0])

    all_abstracts = [Abstract(id=article.id, text=article.abstract) for article in collection.articles.values()]
    print(f"Number of abstracts in collection: {len(all_abstracts)}")

    aha_information = get_population_aha(all_abstracts[:100])
    print(f"AHA information: {aha_information}")

    winner = simplified_evolutionary_abstracts(all_abstracts[:100], aha_information)
    print(f"Final winner: {winner.id}")
    print(winner)





    