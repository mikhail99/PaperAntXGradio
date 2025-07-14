import random
# Assuming dspy is your LLM interface
import dspy
#evolution algotithm go generate new abstracts 
from typing import Literal, List
# --- Parameters ---
POP_SIZE = 10
WIN_THRESHOLD = 2
N_GENERATIONS = 3
from models import Candidate, IdeaTemplate
from signatures import IdeaEvolution, IdeaCompetition, IdeaTemplateSignature
from pydantic import BaseModel

class Abstract(BaseModel):
    id: str
    text: str


def abstract_to_candidate(abstract:Abstract)->Candidate:
    converter = dspy.ChainOfThought(IdeaTemplateSignature)
    try:
        idea_template = converter(abstract=abstract.text).idea_template

        print(f"Idea template: {idea_template}")
        print(f"Idea template type: {type(idea_template)}")
        # The output from the LLM is a string representation of the Pydantic model.
        # We need to parse it into an actual IdeaTemplate object.
        # This assumes the LLM returns a JSON-compatible string.
        return Candidate(id=abstract.id, idea=idea_template, win_count=0)
    except Exception as e:
        print(f"Error converting abstract {abstract.id} to candidate: {e}")
        # Return a dummy or fallback candidate, or handle error as needed
        return None


# --- LLM-based generation and evaluation (placeholders) ---
def generate_new_candidate(parent1:Candidate, parent2:Candidate,temperature:float=0.5) -> Candidate:
    generator = dspy.ChainOfThought(IdeaEvolution)
    try:
        result = generator(idea_A=parent1.idea.model_dump_json(), idea_B=parent2.idea.model_dump_json(), temperature=temperature)
        new_idea_str : str = result.new_idea
        new_idea = IdeaTemplate.model_validate_json(new_idea_str)
        new_id = f"{parent1.id}-{parent2.id}-{random.randint(1000, 9999)}"
        return Candidate(id=new_id, idea=new_idea, win_count=0)
    except Exception as e:
        print(f"Error generating new candidate from {parent1.id} and {parent2.id}: {e}")
        return None

def judge_abstracts(candidate1:Candidate, candidate2:Candidate)->int :
    selector = dspy.ChainOfThought(IdeaCompetition)
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

def evolutionary_abstracts(all_abstracts:List[Abstract]) -> List[Candidate]:
    # 1. Initialize a population of Candidates
    population = initialize_population(all_abstracts, POP_SIZE)
    winners = []

    for gen in range(N_GENERATIONS):
        print(f"\n--- Generation {gen+1} ---")
        
        # 2. Generate new candidates (offspring)
        offspring = []
        for _ in range(POP_SIZE // 2):
            p1, p2 = random.sample(population, 2)
            # Generate two children from each pair of parents
            child1 = generate_new_candidate(p1, p2, temperature=0.2)
            child2 = generate_new_candidate(p2, p1, temperature=0.6)
            if child1: offspring.append(child1)
            if child2: offspring.append(child2)

        # 3. Combine current population and offspring for competition
        combined_population = population + offspring

        # 4. Pairwise competitions to select survivors
        survivors = pairwise_competition(combined_population)

        # 5. Process survivors: move winners and build the next generation's pool
        next_population = []
        for candidate in survivors:
            if candidate.win_count >= WIN_THRESHOLD:
                print(f"🏆 Candidate {candidate.id} promoted to winners!")
                winners.append(candidate)
            else:
                next_population.append(candidate)

        # 6. Repopulate to maintain population size
        while len(next_population) < POP_SIZE:
            new_candidate = get_new_candidate(all_abstracts)
            if new_candidate:
                next_population.append(new_candidate)

        population = next_population[:POP_SIZE] # Ensure size constraint

        print(f"Population: {len(population)}, Winners: {len(winners)}")
        # Print top 3 ideas in the current population for observability
        population.sort(key=lambda c: c.win_count, reverse=True)
        print("--- Top 3 ideas in population ---")
        for cand in population[:3]:
            print(cand)


    return winners


from core.collections_manager import CollectionsManager
if __name__ == "__main__":

    dspy.configure(lm=dspy.LM('ollama_chat/qwen3:4b', api_base='http://localhost:11434', api_key=''))

    manager = CollectionsManager()
    collection_name = "LLM_Reasoning_Agents"
    collection = manager.get_collection_by_name(collection_name)
    print(list(collection.articles.values())[0])

    all_abstracts = [Abstract(id=article.id, text=article.abstract) for article in collection.articles.values()]
    print(f"Number of abstracts in collection: {len(all_abstracts)}")

    winners = evolutionary_abstracts(all_abstracts)
    print(f"Number of winners: {len(winners)}")
    for winner in winners:
        print(winner)





    