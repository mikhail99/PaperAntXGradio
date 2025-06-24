from openai_agents.agents import Agent
from openai_agents.utils import unwrap

# This is a manual implementation of the agentic system for TravelPlanner
# discovered by SwarmAgentic, as described in https://arxiv.org/pdf/2506.15672

# 1. Define each agent with specific instructions based on the paper
transportation_planner = Agent(
    "Transportation Planner",
    "Create a transportation schedule detailing the mode of transport for each leg of the journey.",
)

accommodation_coordinator = Agent(
    "Accommodation Coordinator",
    "Given a transportation schedule and user preferences, create and verify an accommodation plan. Handle constraints like room type, minimum night stays, and user-specific requirements.",
)

restaurant_advisor = Agent(
    "Restaurant Advisor",
    "Given an accommodation plan and user cuisine preferences, recommend restaurants for each non-travel day.",
)

attraction_specialist = Agent(
    "Attraction Specialist",
    "Given an accommodation plan, recommend attractions for each day of the trip.",
)

qa_specialist = Agent(
    "Quality Assurance Specialist",
    "Verify all components of a travel plan (transportation, accommodation, dining, attractions), ensuring all constraints are met and the plan is coherent.",
)

travel_plan_integrator = Agent(
    "Travel Plan Integrator",
    "Compile all components (transportation schedule, accommodation plan, restaurant and attraction recommendations, and QA verification) into a single, comprehensive travel plan document.",
)


# 2. Create the orchestration function that calls the agents in sequence
def run_travel_planner(task_description: str):
    """
    Orchestrates a team of agents to create a comprehensive travel plan.
    This follows the code-based orchestration pattern from the openai-agents SDK.
    """
    print("--- Starting Travel Plan Generation ---")

    # Step 1: Transportation Planner creates a transportation schedule.
    print("1. Planning transportation...")
    transportation_schedule = unwrap(transportation_planner(
        prompt=task_description,
    ))
    print(f"   - Result: {transportation_schedule}")

    # Step 2 & 3 & 4: Accommodation Coordinator creates and refines the plan.
    # The paper shows multiple calls; we can simulate this with a focused prompt.
    print("2. Coordinating accommodation...")
    accommodation_context = f"Transportation Schedule:\n{transportation_schedule}\n\nTask:\nCreate a detailed accommodation plan including user preferences and verified details."
    accommodation_plan_final = unwrap(accommodation_coordinator(
        prompt=accommodation_context,
    ))
    print(f"   - Result: {accommodation_plan_final}")

    # Step 5: Restaurant Advisor recommends restaurants.
    print("3. Recommending restaurants...")
    restaurant_context = f"Accommodation Plan:\n{accommodation_plan_final}\n\nTask:\nRecommend restaurants for each non-travel day based on the plan and user cuisine preferences mentioned in the original request: {task_description}"
    restaurant_recommendations = unwrap(restaurant_advisor(
        prompt=restaurant_context,
    ))
    print(f"   - Result: {restaurant_recommendations}")

    # Step 6: Attraction Specialist recommends attractions.
    print("4. Recommending attractions...")
    attraction_context = f"Accommodation Plan:\n{accommodation_plan_final}\n\nTask:\nRecommend attractions for each day of the trip."
    attraction_recommendations = unwrap(attraction_specialist(
        prompt=attraction_context,
    ))
    print(f"   - Result: {attraction_recommendations}")

    # Step 7: Quality Assurance Specialist verifies all components.
    print("5. Performing quality assurance...")
    qa_context = f"""
        Original Request: {task_description}
        Transportation Schedule: {transportation_schedule}
        Accommodation Plan: {accommodation_plan_final}
        Restaurant Recommendations: {restaurant_recommendations}
        Attraction Recommendations: {attraction_recommendations}

        Task: Verify all components, ensuring all constraints from the original request are met.
    """
    qa_verification = unwrap(qa_specialist(
        prompt=qa_context,
    ))
    print(f"   - Result: {qa_verification}")


    # Step 8: Travel Plan Integrator compiles the final plan.
    print("6. Integrating final travel plan...")
    integrator_context = f"""
        Transportation Schedule: {transportation_schedule}
        Accommodation Plan: {accommodation_plan_final}
        Restaurant Recommendations: {restaurant_recommendations}
        Attraction Recommendations: {attraction_recommendations}
        QA Verification: {qa_verification}

        Task: Compile all of the above components into a comprehensive, easy-to-read travel plan.
    """
    comprehensive_travel_plan = unwrap(travel_plan_integrator(
        prompt=integrator_context,
    ))

    print("--- Travel Plan Generation Complete ---")
    return comprehensive_travel_plan


# 3. Example Usage
if __name__ == "__main__":
    # Ensure you have OPENAI_API_KEY set in your environment variables
    # (e.g., in a .env file)
    import os
    from dotenv import load_dotenv

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set. Please set it in your environment.")
    else:
        # This is the high-level task description that kicks off the process
        example_task = "Plan a 5-day trip to Paris for two people. They are on a medium budget, love art museums, and prefer Italian food. They need to fly from New York."
        final_plan = run_travel_planner(example_task)
        print("\n\n===== FINAL TRAVEL PLAN =====")
        print(final_plan) 