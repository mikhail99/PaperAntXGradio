import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# --- Core Data Structures for an Agentic System (a "Particle") ---

class AgentDefinition(BaseModel):
    """Defines a single agent's role in the system."""
    name: str = Field(..., description="The unique name of the agent, e.g., 'TransportationPlanner'.")
    role_description: str = Field(..., description="A detailed description of what this agent does.")

class OrchestrationStep(BaseModel):
    """Defines a single step in the workflow, specifying which agent acts and with what inputs."""
    step_number: int = Field(..., description="The sequential order of this step.")
    agent_name: str = Field(..., description="The name of the agent to execute this step.")
    input_variables: List[str] = Field(..., description="List of output variables from previous steps to be used as input.")
    prompt_template: str = Field(..., description="The template for the prompt to be given to the agent. It can use placeholders for input variables.")
    output_variable: str = Field(..., description="The name of the variable to store the output of this step.")

class AgenticSystem(BaseModel):
    """
    Represents a single "particle" in the swarm. It is a complete, executable
    agentic system with a defined set of agents and a workflow.
    """
    system_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this system version.")
    agents: List[AgentDefinition] = Field(..., description="The list of all agents available in this system.")
    orchestration: List[OrchestrationStep] = Field(..., description="The sequence of steps that define the workflow.")
    
    # --- PSO-related attributes ---
    score: Optional[float] = Field(None, description="The performance score of this system on a given task.")
    
    # Storing the best version of itself
    personal_best_score: Optional[float] = Field(None, description="The best score this particle has achieved so far.")
    personal_best_system_definition: Optional[Dict[str, Any]] = Field(None, description="The dictionary definition of the system that achieved the personal best score.")

    def execute(self, initial_task: str, verbose: bool = False) -> str:
        """
        Dynamically executes the defined workflow.
        This is a simplified interpreter for the orchestration plan.
        """
        if verbose:
            print(f"--- Executing System ID: {self.system_id} ---")
            
        # Context stores the outputs of each step
        context: Dict[str, Any] = {"initial_task": initial_task}
        
        # We'd need to instantiate openai_agents here, this is a placeholder
        # for the execution logic which we will flesh out later.
        
        for step in sorted(self.orchestration, key=lambda s: s.step_number):
            if verbose:
                print(f"\n=> Step {step.step_number}: Running Agent '{step.agent_name}'")

            # In a real implementation, you would instantiate the agent:
            # agent_def = next((a for a in self.agents if a.name == step.agent_name), None)
            # llm_agent = openai_agents.Agent(agent_def.name, agent_def.role_description)
            
            # Gather inputs from the context
            inputs = {k: context.get(k) for k in step.input_variables}
            
            # Format the prompt
            try:
                prompt = step.prompt_template.format(**inputs)
            except KeyError as e:
                return f"Execution failed: Missing variable {e} for prompt at step {step.step_number}."

            if verbose:
                print(f"   - Prompt: {prompt[:150]}...")

            # --- Placeholder for actual LLM call ---
            # In a real run, this would be:
            # result = unwrap(llm_agent(prompt=prompt))
            result = f"Mock result for {step.output_variable} from {step.agent_name}"
            # -----------------------------------------

            if verbose:
                print(f"   - Output stored in: '{step.output_variable}'")
            
            context[step.output_variable] = result

        final_output_variable = self.orchestration[-1].output_variable
        return context.get(final_output_variable, "Execution finished with no final output.")

# --- Example of what a "particle" would look like ---
def get_example_travel_planner_system() -> AgenticSystem:
    """Returns a manually defined AgenticSystem for the TravelPlanner task."""
    travel_planner_agents = [
        AgentDefinition(name="TransportationPlanner", role_description="Creates a transportation schedule."),
        AgentDefinition(name="AccommodationCoordinator", role_description="Creates an accommodation plan."),
        AgentDefinition(name="FinalIntegrator", role_description="Compiles all parts into a final plan.")
    ]
    
    travel_planner_orchestration = [
        OrchestrationStep(
            step_number=1,
            agent_name="TransportationPlanner",
            input_variables=["initial_task"],
            prompt_template="Based on the request '{initial_task}', create a transportation schedule.",
            output_variable="transport_plan"
        ),
        OrchestrationStep(
            step_number=2,
            agent_name="AccommodationCoordinator",
            input_variables=["transport_plan"],
            prompt_template="Given the transport plan '{transport_plan}', find suitable accommodation.",
            output_variable="accommodation_plan"
        ),
        OrchestrationStep(
            step_number=3,
            agent_name="FinalIntegrator",
            input_variables=["transport_plan", "accommodation_plan"],
            prompt_template="Combine the transport plan '{transport_plan}' and accommodation plan '{accommodation_plan}' into a final document.",
            output_variable="final_plan"
        )
    ]
    
    return AgenticSystem(
        agents=travel_planner_agents,
        orchestration=travel_planner_orchestration
    )

if __name__ == '__main__':
    # This demonstrates how to create and execute a system
    example_system = get_example_travel_planner_system()
    
    print("--- System Definition ---")
    print(example_system.model_dump_json(indent=2))
    
    print("\n\n--- Running Mock Execution ---")
    final_result = example_system.execute(
        initial_task="Plan a 5-day trip to Paris for two people.",
        verbose=True
    )
    
    print("\n\n--- Final Result ---")
    print(final_result) 