# Implementation Plan: Team-Based Research Proposal Agent (V3)

This document outlines the next iteration of the `proposal_agent`, building on lessons learned from the current working implementation and modern LangGraph best practices. The goal is to maintain the robust human-in-the-loop functionality we've achieved while improving modularity, configurability, and adopting LangGraph's recommended `interrupt()` pattern.

## Current Design: Single-Query Processing Pattern

### 🎯 **How Query Processing Actually Works**

The current implementation uses a **single-query processing pattern** that efficiently searches literature while avoiding repetition:

1. **Query Generation**: `query_generator_base` generates **ONE focused query** per cycle
2. **Repetition Avoidance**: The LLM considers previous queries (`search_queries` history) to ensure diversity
3. **Literature Search**: `literature_reviewer_local` processes **ONLY the most recent query**
4. **Summary Accumulation**: Results from all searches accumulate in `literature_summaries`

### 📋 **Example Flow:**
```
Cycle 1: Topic "AI safety" → Generate "neural network safety verification" → Search → Summary A
Cycle 2: Topic "AI safety" + Previous ["neural network safety verification"] → Generate "AI safety in robotics" → Search → Summary B  
Cycle 3: User provides "autonomous vehicle safety" → Search → Summary C
Final: literature_summaries = [Summary A, Summary B, Summary C]
```

### ✅ **Benefits of This Design:**
- **Efficient**: One focused search per cycle (faster than parallel multi-query processing)
- **Diverse**: Query generator actively avoids repetition by considering query history
- **Controlled**: Human can override any generated query at checkpoints
- **Cumulative**: Literature knowledge builds up across multiple targeted searches

### ⚠️ **Key Design Decision:**
While `search_queries` maintains a list for history/context, **only the last query is actually searched**. This design prioritizes focused, non-repetitive literature coverage over exhaustive parallel searching.

## Current State Assessment

### ✅ What's Working Well
- **Human-in-the-loop functionality** is robust and properly integrated
- **State management** with reducers and checkpointing works correctly
- **Service layer abstraction** cleanly separates graph execution from UI
- **Parrot services** provide excellent testing and debugging capabilities
- **Modular service initialization** with lazy loading patterns
- **Comprehensive logging** aids in debugging and monitoring

### 🔧 Areas for Improvement
- **Outdated HIL pattern** - Using `interrupt_after` instead of modern `interrupt()` function
- **Monolithic node factory** - `create_llm_node()` has grown too large with special-case logic
- **Hard-coded graph structure** - team composition and workflow are embedded in Python
- **Mixed concerns** - business logic, prompt handling, and graph structure are intertwined
- **Limited configurability** - changing team composition requires code changes
- **Inconsistent error handling** - some nodes handle errors differently
- **Manual pause state management** - Current approach requires explicit `paused_on` field handling

## Guiding Principles for V3

1. **Adopt Modern LangGraph Patterns**: Use `interrupt()` function for cleaner HIL implementation
2. **Preserve Working Patterns**: Keep successful state management and service patterns
3. **Incremental Refactoring**: Maintain backward compatibility during transitions
4. **Configuration over Code**: Move team composition and workflows to configuration
5. **Separation of Concerns**: Clearly separate graph structure, business logic, and presentation
6. **Testability**: Maintain the excellent parrot service pattern for testing
7. **Observability**: Enhance logging and monitoring capabilities

## Step 1: Modern HIL Architecture with `interrupt()`

### Key Changes from Current Implementation
Based on the [LangGraph documentation](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/add-human-in-the-loop/), we should:

1. **Replace `interrupt_after` with `interrupt()` calls** within nodes
2. **Use `Command(resume=...)` for resumption** instead of manual state updates
3. **Eliminate manual `paused_on` field management** - LangGraph handles this automatically
4. **Simplify service layer** - no more complex pause detection logic needed

### New HIL Node Pattern
```python
# core/proposal_agent/hil_nodes.py
from langgraph.types import interrupt, Command

def human_query_review_node(state: ProposalAgentState):
    """Human review of generated queries using modern interrupt() pattern."""
    queries = state.get('search_queries', [])
    query_list = "\n".join(f"- `{q}`" for q in queries)
    
    # Modern interrupt pattern - much cleaner than our current approach
    user_input = interrupt({
        "type": "query_review",
        "message": f"✅ **Queries Generated.** The agent plans to search for:\n{query_list}\n\nProvide a new query or type 'continue' to approve.",
        "queries": queries,
        "context": {
            "stage": "query_generation",
            "topic": state.get("topic", "")
        }
    })
    
    # Handle user input
    if user_input.strip().lower() == 'continue':
        return {}  # No state changes needed
    else:
        # User provided a custom query
        return {"search_queries": [user_input]}

def human_insight_review_node(state: ProposalAgentState):
    """Human review of literature synthesis using interrupt() pattern."""
    knowledge_gap = state.get('knowledge_gap', {}).get('knowledge_gap', 'No gap identified')
    
    user_input = interrupt({
        "type": "insight_review", 
        "message": f"✅ **Literature Synthesized.** Knowledge gap identified:\n\n*'{knowledge_gap}'*\n\nProvide a refined gap or type 'continue' to approve.",
        "knowledge_gap": knowledge_gap,
        "context": {
            "stage": "literature_review",
            "summaries_count": len(state.get('literature_summaries', []))
        }
    })
    
    if user_input.strip().lower() == 'continue':
        return {}
    else:
        # Update the knowledge gap with user input
        current_kg = state.get('knowledge_gap', {})
        current_kg['knowledge_gap'] = user_input
        return {"knowledge_gap": current_kg}

def human_review_node(state: ProposalAgentState):
    """Final human review using interrupt() pattern."""
    proposal = state.get('proposal_draft', 'No proposal generated')
    reviews = state.get('review_team_feedback', {})
    
    user_input = interrupt({
        "type": "final_review",
        "message": "✅ **AI Review Complete.** Please provide feedback or type 'continue' to approve.",
        "proposal_draft": proposal,
        "ai_reviews": reviews,
        "context": {
            "stage": "final_review",
            "revision_cycle": state.get('proposal_revision_cycles', 0)
        }
    })
    
    if user_input.strip().lower() == 'continue':
        return {}
    else:
        # Store human feedback for next revision
        return {"human_feedback": user_input}
```

### Simplified Service Layer
```python
# core/proposal_agent_service.py (Modernized)
from langgraph.types import Command

class ProposalAgentService:
    def __init__(self, workflow_name: str = "research_proposal_workflow"):
        self.workflow_config = self._load_workflow_config(workflow_name)
        self.graph = self._build_configured_graph()
        # No more HIL_NODES or complex pause detection needed!
        
    async def start_agent(self, config: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Starts a new agent run."""
        thread_id = str(uuid.uuid4())
        
        initial_state = ProposalAgentState(
            topic=config.get("topic", ""),
            collection_name=config.get("collection_name", ""),
            # ... other fields
        )
        
        async for result in self._stream_graph(thread_id, initial_state):
            yield result

    async def continue_agent(self, thread_id: str, user_input: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Continues execution using modern Command pattern."""
        # Much simpler - just use Command(resume=...)
        async for result in self._stream_graph(thread_id, Command(resume=user_input)):
            yield result
            
    async def _stream_graph(self, thread_id: str, graph_input: Any):
        """Simplified streaming without manual pause detection."""
        config = {"configurable": {"thread_id": thread_id}}
        
        async for step in self.graph.astream(graph_input, config=config):
            step_name = list(step.keys())[0]
            step_output = step.get(step_name, {})
            
            # Check for interrupt - LangGraph handles this automatically now
            if "__interrupt__" in step:
                interrupt_data = step["__interrupt__"][0]
                yield {
                    "step": "human_input_required",
                    "interrupt_data": interrupt_data.value,  # Our structured data
                    "thread_id": thread_id
                }
            else:
                yield {
                    "step": step_name,
                    "state": step_output,
                    "thread_id": thread_id
                }
```

### Enhanced UI Integration
```python
# ui/ui_research_plan.py (Updated for interrupt() pattern)
async def handle_chat_interaction(collection_name, user_input, chat_history, thread_id, agent_state):
    """Enhanced handler that works with interrupt() pattern."""
    
    # ... existing setup code ...
    
    async for step_data in agent_stream:
        if step_data.get("step") == "human_input_required":
            # Extract structured interrupt data
            interrupt_info = step_data.get("interrupt_data", {})
            interrupt_type = interrupt_info.get("type", "unknown")
            message = interrupt_info.get("message", "Please provide input")
            context = interrupt_info.get("context", {})
            
            # Generate contextual UI based on interrupt type
            if interrupt_type == "query_review":
                # Show query list with approval UI
                pass
            elif interrupt_type == "insight_review":
                # Show knowledge gap with editing UI  
                pass
            elif interrupt_type == "final_review":
                # Show final review UI
                pass
                
            chat_history[-1] = (user_input, message)
            break
            
        # ... handle other step types ...
```

## Step 2: Enhanced Configuration Architecture

### `team_workflows.json` (Updated)
```json
{
  "research_proposal_workflow": {
    "description": "Modern workflow using interrupt() pattern",
    "stages": [
      {
        "id": "query_generation",
        "team": "query_generation_team",
        "hil_node": "human_query_review_node",
        "hil_type": "query_review"
      },
      {
        "id": "literature_review", 
        "team": "literature_review_team",
        "hil_node": "human_insight_review_node", 
        "hil_type": "insight_review"
      },
      {
        "id": "proposal_creation",
        "team": "proposal_creation_team",
        "hil_node": null
      },
      {
        "id": "proposal_review",
        "team": "proposal_review_team",
        "hil_node": "human_review_node",
        "hil_type": "final_review",
        "approval_logic": "is_proposal_approved",
        "max_revisions": 3
      }
    ]
  }
}
```

### `hil_configurations.json` (New)
```json
{
  "interrupt_types": {
    "query_review": {
      "ui_component": "QueryReviewUI",
      "validation": "validate_query_input",
      "timeout_seconds": 300,
      "allow_skip": true
    },
    "insight_review": {
      "ui_component": "InsightReviewUI", 
      "validation": "validate_insight_input",
      "timeout_seconds": 600,
      "allow_skip": false
    },
    "final_review": {
      "ui_component": "FinalReviewUI",
      "validation": "validate_final_input", 
      "timeout_seconds": 900,
      "allow_skip": false
    }
  }
}
```

## Step 3: Improved Graph Construction

### Modern Graph Builder
```python
# core/proposal_agent/graph_builder.py (Updated)
class ModernWorkflowGraphBuilder:
    def build_graph(self) -> StateGraph:
        """Builds graph using modern interrupt() pattern."""
        builder = StateGraph(ProposalAgentState)
        
        # Add nodes with interrupt() calls - no more interrupt_after needed
        for stage in self.workflow["stages"]:
            self._add_stage_with_modern_hil(builder, stage)
            
        # Much simpler compilation - LangGraph handles interrupts automatically
        return builder.compile(checkpointer=MemorySaver())
        
    def _add_stage_with_modern_hil(self, builder: StateGraph, stage_config: Dict):
        """Adds stage using interrupt() pattern instead of interrupt_after."""
        # Add team members
        team_config = self.teams[stage_config["team"]]
        for member in team_config["parallel_members"]:
            builder.add_node(member, self._create_node(member))
            
        # Add HIL node with interrupt() calls
        if stage_config["hil_node"]:
            hil_func = self._create_modern_hil_node(stage_config)
            builder.add_node(stage_config["hil_node"], hil_func)
            
        # Connect with normal edges - no special interrupt handling needed
        self._connect_stage_nodes(builder, stage_config)
```

## Step 4: Enhanced Testing with Modern Patterns

### Interrupt-Aware Parrot Services
```python
# core/proposal_agent/parrot_services.py (Enhanced)
class ModernParrotService:
    def __init__(self, scenario: str = "default"):
        self.scenario = self._load_scenario(scenario)
        
    def handle_interrupt(self, interrupt_data: Dict) -> str:
        """Handles interrupt() calls in test scenarios."""
        interrupt_type = interrupt_data.get("type", "unknown")
        scenario_responses = self.scenario.get("interrupt_responses", {})
        
        return scenario_responses.get(interrupt_type, "continue")
```

### Test Scenarios for Interrupts
```json
{
  "happy_path": {
    "description": "All approvals, no modifications",
    "interrupt_responses": {
      "query_review": "continue",
      "insight_review": "continue", 
      "final_review": "continue"
    },
    "node_responses": {
      "synthesize_review": {"is_approved": true}
    }
  },
  "user_modifications": {
    "description": "User modifies queries and insights",
    "interrupt_responses": {
      "query_review": "machine learning for mathematical reasoning",
      "insight_review": "Need better understanding of neural symbolic integration",
      "final_review": "continue"
    }
  }
}
```

## Step 5: Migration Strategy

### Phase 1: Modernize HIL Pattern
1. ✅ Replace `interrupt_after` with `interrupt()` calls in HIL nodes
2. ✅ Update service layer to use `Command(resume=...)` pattern  
3. ✅ Remove manual `paused_on` state management
4. ✅ Simplify graph compilation (no more interrupt_after list needed)

### Phase 2: Enhanced Configuration
1. Create `hil_configurations.json` for interrupt type definitions
2. Update workflow configurations to use new HIL pattern
3. Add validation and timeout configurations for interrupts

### Phase 3: Improved UI Integration  
1. Update UI to handle structured interrupt data
2. Create type-specific UI components for different interrupt types
3. Add better user feedback and validation

### Phase 4: Advanced Features
1. Add interrupt timeouts and fallback behaviors
2. Implement interrupt validation and retry logic
3. Add interrupt history and audit trails
4. Create interrupt analytics and monitoring

## Benefits of Modern Approach

1. **Cleaner Code**: `interrupt()` is more ergonomic than `interrupt_after` + manual state management
2. **Better Error Handling**: LangGraph automatically handles interrupt state persistence
3. **Improved Testing**: Structured interrupt data makes testing more predictable
4. **Enhanced UX**: Rich interrupt data enables better UI experiences
5. **Reduced Complexity**: Eliminates manual pause detection and resume logic
6. **Future-Proof**: Aligns with LangGraph's recommended patterns and best practices

## Key Considerations

### Side Effects Management
Per the [LangGraph documentation](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/add-human-in-the-loop/#using-with-code-with-side-effects), we must be careful about code placement:

```python
# ✅ CORRECT: Side effects after interrupt
def modern_literature_node(state: ProposalAgentState):
    # Get user approval first
    user_input = interrupt({"type": "query_approval", "queries": queries})
    
    # THEN do expensive operations (they won't re-run on resume)
    if user_input != "continue":
        queries = [user_input]
    
    results = await paperqa_service.query_documents(collection, queries[0])
    return {"literature_summaries": results}

# ❌ INCORRECT: Side effects before interrupt (will re-run)
def bad_literature_node(state: ProposalAgentState):
    results = await paperqa_service.query_documents(collection, query)  # This will re-run!
    user_input = interrupt({"type": "review_results", "results": results})
    return {"literature_summaries": results}
```

This plan modernizes our implementation to use LangGraph's recommended patterns while preserving all the stability and functionality we've built. The result will be cleaner, more maintainable code that follows current best practices. 

# EXAMPLE in literature_reviewer_local:
# INPUT: state = {"search_queries": ["AI safety"], "human_feedback": "continue"}
# OR:    state = {"search_queries": ["AI safety"], "human_feedback": "machine learning ethics"}
# LOGIC: Use human feedback if provided, otherwise use approved query 

# EXAMPLE in literature_reviewer_local (SINGLE QUERY PROCESSING):
# INPUT: state = {"search_queries": ["neural networks", "AI safety"], "human_feedback": "continue"}
# LOGIC: Uses ONLY "AI safety" (the most recent query), ignoring "neural networks"
# OR:    state = {"search_queries": ["AI safety"], "human_feedback": "machine learning ethics"}
# LOGIC: Uses "machine learning ethics" (human override), ignoring all previous queries
# PURPOSE: Focused search with one query per literature review cycle

# EXAMPLE in query_generator_base:
# INPUT: state = {"topic": "AI safety", "human_feedback": "continue"}
# RESULT: Uses original topic "AI safety"
# INPUT: state = {"topic": "AI safety", "human_feedback": "neural networks"}  
# RESULT: Uses "neural networks" instead of original topic 

# EXAMPLE in query_generator_base (REPETITION AVOIDANCE):
# INPUT: state = {"topic": "AI safety", "search_queries": ["neural networks", "machine learning"], "human_feedback": "continue"}
# RESULT: Generates new query like "AI safety verification" (avoids repeating previous queries)
# INPUT: state = {"topic": "AI safety", "human_feedback": "robotics safety"}  
# RESULT: Uses "robotics safety" (human override), ignoring topic and previous queries
# PURPOSE: Ensures diverse literature coverage without repetitive searches

# EXAMPLE in deduplicate_queries_node:
# INPUT: state = {"search_queries": ["AI safety", "machine learning", "AI safety", "neural networks"]}
# OUTPUT: {"search_queries": ["AI safety", "machine learning", "neural networks"]}
# LOGIC: Remove duplicates while preserving order

# EXAMPLE in deduplicate_queries_node (QUERY HISTORY MAINTENANCE):
# INPUT: state = {"search_queries": ["AI safety", "machine learning", "AI safety", "neural networks"]}
# OUTPUT: {"search_queries": ["AI safety", "machine learning", "neural networks"]}
# LOGIC: Remove duplicates while preserving order for query history context
# PURPOSE: Maintains clean query history that informs future query generation (avoiding repetition)
# NOTE: Only the LAST query will be used for literature search, others provide context

# EXAMPLE in is_proposal_approved:
# SCENARIO 1: {"final_review": {"is_approved": True}} → "approved" → END
# SCENARIO 2: {"is_approved": False}, cycles < 3 → "revise" → human review
# SCENARIO 3: {"is_approved": False}, cycles >= 3 → "max_revisions_reached" → END

# EXAMPLE: Overall workflow pattern (SINGLE QUERY PROCESSING)
# START → query_generator_base (generates 1 focused query, avoids previous queries)
#   ↓ 
# deduplicate_queries_node (removes duplicates, maintains query history for context)
#   ↓
# human_query_review_node (PAUSE - user can approve generated query or provide custom one)
#   ↓ (user types "continue" or custom query like "robotics safety")
# literature_reviewer_local (processes ONLY the most recent/relevant query)
#   ↓ 
# synthesize_literature_review (combines current + previous literature summaries)
#   ↓
# human_insight_review_node (PAUSE - user can approve knowledge gap or refine it)
#   ↓ (user types "continue" or refined gap like "focus on autonomous vehicles")
# formulate_plan → [review_novelty + review_feasibility] → synthesize_review
#   ↓ (AI decides: approved, revise, or max_revisions_reached)
# IF approved/max_revisions: clear_pause_state_node → END
# IF revise: human_review_node (PAUSE) → formulate_plan (loop back)
#
# KEY DESIGN PRINCIPLES:
# - Each literature search processes ONE focused query for efficiency
# - Query generator considers previous queries to ensure diversity and avoid repetition
# - Literature summaries accumulate across searches to build comprehensive knowledge base
# - Human can override any generated query with custom input at checkpoints