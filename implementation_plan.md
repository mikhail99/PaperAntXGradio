# Implementation Plan: Team-Based Research Proposal Agent (V3)

This document outlines the next iteration of the `proposal_agent`, building on lessons learned from the current working implementation. The goal is to maintain the robust human-in-the-loop functionality we've achieved while improving modularity, configurability, and maintainability.

## Current State Assessment

### ✅ What's Working Well
- **Human-in-the-loop functionality** is robust and properly integrated
- **State management** with reducers and checkpointing works correctly
- **Service layer abstraction** cleanly separates graph execution from UI
- **Parrot services** provide excellent testing and debugging capabilities
- **Modular service initialization** with lazy loading patterns
- **Comprehensive logging** aids in debugging and monitoring

### 🔧 Areas for Improvement
- **Monolithic node factory** - `create_llm_node()` has grown too large with special-case logic
- **Hard-coded graph structure** - team composition and workflow are embedded in Python
- **Mixed concerns** - business logic, prompt handling, and graph structure are intertwined
- **Limited configurability** - changing team composition requires code changes
- **Inconsistent error handling** - some nodes handle errors differently
- **Prompt management** - prompts are embedded in JSON but could be more modular

## Guiding Principles for V3

1. **Preserve Working Patterns**: Keep the successful HIL, state management, and service patterns
2. **Incremental Refactoring**: Maintain backward compatibility during transitions
3. **Configuration over Code**: Move team composition and workflows to configuration
4. **Separation of Concerns**: Clearly separate graph structure, business logic, and presentation
5. **Testability**: Maintain the excellent parrot service pattern for testing
6. **Observability**: Enhance logging and monitoring capabilities

## Step 1: Enhanced Configuration Architecture

### `team_workflows.json` (New)
Define the high-level workflow patterns that can be composed together:

```json
{
  "research_proposal_workflow": {
    "description": "Complete research proposal generation with HIL checkpoints",
    "stages": [
      {
        "id": "query_generation",
        "team": "query_generation_team",
        "hil_checkpoint": "human_query_review_node",
        "hil_prompt_template": "query_review_prompt"
      },
      {
        "id": "literature_review", 
        "team": "literature_review_team",
        "hil_checkpoint": "human_insight_review_node",
        "hil_prompt_template": "insight_review_prompt"
      },
      {
        "id": "proposal_creation",
        "team": "proposal_creation_team", 
        "hil_checkpoint": null
      },
      {
        "id": "proposal_review",
        "team": "proposal_review_team",
        "hil_checkpoint": "human_review_node",
        "hil_prompt_template": "final_review_prompt",
        "approval_logic": "is_proposal_approved",
        "max_revisions": 3
      }
    ]
  }
}
```

### `agent_teams.json` (Enhanced)
```json
{
  "teams": {
    "query_generation_team": {
      "description": "Generates and refines search queries",
      "parallel_members": ["query_generator_base"],
      "aggregator": "deduplicate_queries_node",
      "state_inputs": ["topic", "human_feedback"],
      "state_outputs": ["search_queries"]
    },
    "literature_review_team": {
      "description": "Reviews literature and synthesizes findings",
      "parallel_members": ["literature_reviewer_local"],
      "aggregator": "synthesize_literature_review",
      "state_inputs": ["search_queries", "collection_name", "human_feedback"],
      "state_outputs": ["literature_summaries", "knowledge_gap"]
    },
    "proposal_creation_team": {
      "description": "Creates initial proposal draft",
      "parallel_members": ["formulate_plan"],
      "aggregator": null,
      "state_inputs": ["knowledge_gap", "human_feedback"],
      "state_outputs": ["proposal_draft"]
    },
    "proposal_review_team": {
      "description": "Reviews proposal for feasibility and novelty",
      "parallel_members": ["review_novelty", "review_feasibility"],
      "aggregator": "synthesize_review",
      "state_inputs": ["proposal_draft", "knowledge_gap"],
      "state_outputs": ["review_team_feedback", "final_review"]
    }
  }
}
```

### `agent_nodes.json` (Refined)
```json
{
  "nodes": {
    "query_generator_base": {
      "type": "llm_node",
      "name": "Base Query Generator",
      "prompt_template": "generate_query_base",
      "output_schema": "QueryList",
      "input_processor": "query_generator_input_processor",
      "output_processor": "query_generator_output_processor"
    },
    "literature_reviewer_local": {
      "type": "tool_node", 
      "name": "Local Literature Reviewer",
      "tool_function": "paperqa_service.query_documents",
      "input_processor": "literature_reviewer_input_processor",
      "output_processor": "literature_reviewer_output_processor"
    },
    "formulate_plan": {
      "type": "llm_node",
      "name": "Proposal Formulator",
      "prompt_template": "formulate_plan",
      "output_schema": "string",
      "input_processor": "formulate_plan_input_processor",
      "truncation_limits": {
        "knowledge_gap": 1500,
        "proposal_draft": 1500,
        "review_team_feedback": 1500
      }
    }
  }
}
```

### `hil_prompts.json` (New)
```json
{
  "query_review_prompt": {
    "template": "✅ **Queries Generated.** The agent plans to search for the following:\n{query_list}\n\nPlease provide a new query to use instead, or type 'continue' to approve.",
    "variables": ["query_list"]
  },
  "insight_review_prompt": {
    "template": "✅ **Literature Synthesized.** The agent identified this knowledge gap:\n\n*'{knowledge_gap}'*\n\nPlease provide a refined knowledge gap, or type 'continue' to approve.",
    "variables": ["knowledge_gap"]
  },
  "final_review_prompt": {
    "template": "✅ **AI Review Complete.** Please provide your feedback, or type 'continue' to approve.",
    "variables": []
  }
}
```

## Step 2: Modular Node Architecture

### Node Type System
Replace the monolithic `create_llm_node()` with a type-based system:

```python
# core/proposal_agent/node_types.py
class NodeFactory:
    @staticmethod
    def create_llm_node(config: Dict) -> Callable:
        """Creates a node that uses LLM with structured output"""
        
    @staticmethod  
    def create_tool_node(config: Dict) -> Callable:
        """Creates a node that calls external tools/services"""
        
    @staticmethod
    def create_aggregator_node(config: Dict) -> Callable:
        """Creates a node that aggregates results from parallel nodes"""
        
    @staticmethod
    def create_hil_node(config: Dict) -> Callable:
        """Creates a human-in-the-loop checkpoint node"""
```

### Input/Output Processors
Extract the special-case logic into reusable processors:

```python
# core/proposal_agent/processors.py
def query_generator_input_processor(state: ProposalAgentState) -> Dict:
    """Handles the 'continue' logic and topic/feedback processing"""
    
def literature_reviewer_input_processor(state: ProposalAgentState) -> Dict:
    """Determines which query to run and handles human overrides"""
    
def formulate_plan_input_processor(state: ProposalAgentState) -> Dict:
    """Handles truncation and combines multiple state inputs"""
```

## Step 3: Enhanced Graph Builder

### Workflow-Driven Construction
```python
# core/proposal_agent/graph_builder.py
class WorkflowGraphBuilder:
    def __init__(self, workflow_config: Dict, teams_config: Dict, nodes_config: Dict):
        self.workflow = workflow_config
        self.teams = teams_config  
        self.nodes = nodes_config
        
    def build_graph(self) -> StateGraph:
        """Builds the graph from workflow configuration"""
        builder = StateGraph(ProposalAgentState)
        
        for stage in self.workflow["stages"]:
            self._add_stage(builder, stage)
            
        return self._compile_with_checkpoints(builder)
        
    def _add_stage(self, builder: StateGraph, stage_config: Dict):
        """Adds a complete stage (team + HIL) to the graph"""
        team_config = self.teams[stage_config["team"]]
        
        # Add team members in parallel
        for member in team_config["parallel_members"]:
            node_func = self._create_node(member)
            builder.add_node(member, node_func)
            
        # Add aggregator if specified
        if team_config["aggregator"]:
            agg_func = self._create_node(team_config["aggregator"])
            builder.add_node(team_config["aggregator"], agg_func)
            
        # Add HIL checkpoint if specified
        if stage_config["hil_checkpoint"]:
            hil_func = self._create_hil_node(stage_config)
            builder.add_node(stage_config["hil_checkpoint"], hil_func)
```

## Step 4: Enhanced Service Layer

### Configuration-Aware Service
```python
# core/proposal_agent_service.py (Enhanced)
class ProposalAgentService:
    def __init__(self, workflow_name: str = "research_proposal_workflow"):
        self.workflow_config = self._load_workflow_config(workflow_name)
        self.graph = self._build_configured_graph()
        
    def get_hil_prompt(self, paused_on: str, state: Dict) -> str:
        """Generates contextual HIL prompts based on configuration"""
        
    def get_workflow_progress(self, state: Dict) -> Dict:
        """Returns current progress through the workflow stages"""
```

## Step 5: Improved Testing and Development

### Enhanced Parrot Services
```python
# core/proposal_agent/parrot_services.py (Enhanced)
class ConfigurableParrotService:
    def __init__(self, scenario: str = "default"):
        self.scenario_config = self._load_scenario(scenario)
        
    def get_parrot_response(self, node_name: str, input_data: Dict) -> Any:
        """Returns scenario-specific responses for testing"""
```

### Test Scenarios
```json
// test_scenarios.json
{
  "happy_path": {
    "description": "All approvals, no revisions needed",
    "node_responses": {
      "synthesize_review": {"is_approved": true}
    }
  },
  "revision_cycle": {
    "description": "Tests the revision loop",
    "node_responses": {
      "synthesize_review": {"is_approved": false}
    }
  }
}
```

## Step 6: Migration Strategy

### Phase 1: Extract Configurations
1. Move existing prompts to `prompts.json` ✅ (Already done)
2. Create `team_workflows.json` with current workflow
3. Create `agent_teams.json` matching current structure
4. Create `agent_nodes.json` with current node definitions

### Phase 2: Modularize Node Creation
1. Create `NodeFactory` and processor functions
2. Replace `create_llm_node()` with type-based creation
3. Extract special-case logic to processors
4. Maintain backward compatibility

### Phase 3: Configuration-Driven Graph Building
1. Create `WorkflowGraphBuilder`
2. Replace current graph building with configuration-driven approach
3. Enhance service layer with workflow awareness
4. Add comprehensive testing

### Phase 4: Enhanced Features
1. Add workflow progress tracking
2. Implement dynamic HIL prompt generation
3. Add scenario-based testing
4. Improve observability and monitoring

## Benefits of This Approach

1. **Maintainability**: Clear separation of concerns, modular architecture
2. **Configurability**: Easy to modify workflows without code changes  
3. **Testability**: Enhanced parrot services with scenario support
4. **Scalability**: Easy to add new node types, workflows, and teams
5. **Observability**: Better tracking of workflow progress and state
6. **Backward Compatibility**: Incremental migration preserves working functionality

This plan builds on the solid foundation we've established while addressing the architectural debt that has accumulated. The result will be a more maintainable, configurable, and extensible system that preserves all the hard-won stability of the current implementation. 