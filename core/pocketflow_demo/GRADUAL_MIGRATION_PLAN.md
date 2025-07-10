# PocketFlow Demo → Research Proposal Agent Migration Plan

## Overview

This document outlines a **gradual migration strategy** to transform the working `@pocketflow_demo` (hotel/weather assistant) into a Research Proposal Agent that follows the workflow described in `@WORKFLOW_EXAMPLES.md`.

**Core Philosophy:** Evolve, don't rebuild. We will preserve the working HITL patterns and gradually replace the business logic.

---

## Current Working Architecture (Baseline)

### Flow Structure
```python
# Current hub-and-spoke model in flow.py
decide_action = DecideAction()
check_weather = CheckWeather()
book_hotel = BookHotel()
follow_up = FollowUp()
result_notification = ResultNotification()

decide_action - "check-weather" >> check_weather
check_weather >> decide_action
decide_action - "book-hotel" >> book_hotel
book_hotel >> decide_action
decide_action - "follow-up" >> follow_up
decide_action - "result-notification" >> result_notification

return Flow(start=decide_action)
```

### Current Action Space
- `check-weather`: Get weather for a city
- `book-hotel`: Book hotel with dates
- `follow-up`: Ask clarifying questions
- `result-notification`: Deliver final results

### Working HITL Pattern
```python
# From FollowUp and ResultNotification nodes
def prep(self, shared):
    flow_log: Queue = shared["flow_queue"]
    flow_log.put(None)  # Pause the flow
    # ... prepare message

def exec(self, prep_res):
    question, queue = prep_res
    queue.put(question)    # Send message to UI
    queue.put(None)        # Signal completion
    return question

def post(self, shared, prep_res, exec_res):
    # ... cleanup and return "done"
```

---

## Phase 1: Morph the Agent's Brain (Router Evolution)

### Objective
Replace the hotel/weather logic with research workflow logic while keeping the same robust error handling and HITL patterns.

### Step 1.1: Rename Core Components
```python
# OLD: DecideAction → NEW: ResearchAgentRouter
class ResearchAgentRouter(Node):  # Renamed from DecideAction
    # Keep the same prep/exec/post structure
```

### Step 1.2: Update Action Space
Replace the current 4 actions with initial research actions:

```python
# OLD ACTION SPACE:
# [1] check-weather
# [2] book-hotel  
# [3] follow-up
# [4] result-notification

# NEW ACTION SPACE (Phase 1):
# [1] generate-queries    # Start research workflow
# [2] follow-up          # Keep for clarification (unchanged)
```

### Step 1.3: Preserve Error Handling Pattern
Keep the robust error handling from `DecideAction.post()`:

```python
def post(self, shared, prep_res, exec_res):
    # Keep the same try/except pattern that prevents silent failures
    if exec_res["action"] == "generate-queries":
        try:
            topic = exec_res["topic"]
            session["generate_queries_params"] = {"topic": topic}
            flow_log.put(f"➡️ Starting research for: {topic}")
        except KeyError as e:
            # Same defensive pattern as the hotel booking
            question = "I can help you develop a research proposal! What topic would you like to explore? 📚"
            session["follow_up_params"] = {"question": question}
            next_action = "follow-up"
```

### Step 1.4: Update Flow Connections
```python
# OLD (hub-and-spoke):
# decide_action - "check-weather" >> check_weather
# check_weather >> decide_action

# NEW (Phase 1 - simplified):
research_router = ResearchAgentRouter()
generate_queries = GenerateQueries()  # New placeholder node
follow_up = FollowUp()  # Keep unchanged

research_router - "generate-queries" >> generate_queries  
research_router - "follow-up" >> follow_up
```

**Expected Result:** User can say "I want to research LLMs in education" and the agent will route to `generate-queries` action instead of trying to book hotels.

---

## Phase 2: Implement Core Research Pipeline

### Objective
Build the main research workflow nodes while preserving the working HITL patterns.

### Step 2.1: Create `GenerateQueries` Node
Model this after the working `FollowUp` node pattern:

```python
class GenerateQueries(Node):
    def prep(self, shared):
        # Same pause pattern as FollowUp
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(None)  # Pause flow
        
        session = load_conversation(shared["conversation_id"])
        topic = session["generate_queries_params"]["topic"]
        return topic, shared["queue"]
    
    def exec(self, prep_res):
        topic, queue = prep_res
        
        # Call LLM to generate query (simplified to 1 query)
        query = call_llm_for_query(topic)  # Implementation detail
        
        # Use same UI communication pattern as FollowUp
        message = f"📚 Generated research query for '{topic}':\n\n  1. {query}\n\nApprove this query?"
        
        queue.put(message)
        queue.put(None)
        return query
    
    def post(self, shared, prep_res, exec_res):
        # Save query to session (new state management)
        session = load_conversation(shared["conversation_id"])
        session["query"] = exec_res
        session["last_action"] = "generate_queries"
        save_conversation(shared["conversation_id"], session)
        return "done"  # Same completion pattern
```

### Step 2.2: Create `LiteratureReview` Node
```python
class LiteratureReview(Node):
    # Follow the same prep/exec/post pattern
    # In exec(): Use PaperQAService to query documents
    # Preserve the queue communication pattern for showing progress
    
    def prep(self, shared):
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(None)  # Pause flow
        
        session = load_conversation(shared["conversation_id"])
        query = session["query"]
        topic = session["generate_queries_params"]["topic"]
        return query, topic, shared["queue"]
    
    def exec(self, prep_res):
        query, topic, queue = prep_res
        
        # Use PaperQAService to query the single research query
        result = paperqa_service.query(query)
        literature_finding = {
            "query": query,
            "result": result,
            "sources": getattr(result, 'sources', [])
        }
        
        # Show results
        message = f"📚 Literature review complete for '{topic}'!\n\n"
        message += f"Query: {query}\n"
        message += f"Found {len(literature_finding['sources'])} sources\n\n"
        message += "Proceed to gap analysis?"
        
        queue.put(message)
        queue.put(None)
        return literature_finding
```

### Step 2.3: Create `SynthesizeGap` Node
```python
class SynthesizeGap(Node):
    # Follow the same prep/exec/post pattern  
    # In exec(): Use LLM to synthesize knowledge gap
    # Use same HITL pattern to show gap analysis to user
    
    def prep(self, shared):
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(None)  # Pause flow
        
        session = load_conversation(shared["conversation_id"])
        literature_finding = session["literature_finding"]
        topic = session["generate_queries_params"]["topic"]
        return literature_finding, topic, shared["queue"]
    
    def exec(self, prep_res):
        literature_finding, topic, queue = prep_res
        
        # Prepare literature summary for LLM
        literature_summary = f"Query: {literature_finding['query']}\n"
        literature_summary += f"Findings: {str(literature_finding['result'])[:300]}...\n"
        literature_summary += f"Sources: {len(literature_finding.get('sources', []))} papers\n"
        
        # Call LLM to synthesize knowledge gap
        gap_analysis = call_llm_for_gap_analysis(topic, literature_summary)
        
        # Present gap analysis to user
        message = f"🎯 Knowledge Gap Analysis for '{topic}':\n\n{gap_analysis}\n\n"
        message += "Does this gap analysis look good?"
        
        queue.put(message)
        queue.put(None)
        return {
            "topic": topic,
            "gap_analysis": gap_analysis,
            "literature_summary": literature_summary
        }
```

### Step 2.4: Transition to Linear Flow
```python
# Replace hub-and-spoke with linear pipeline
def create_flow():
    router = ResearchAgentRouter()
    generate_queries = GenerateQueries()
    literature_review = LiteratureReview() 
    synthesize_gap = SynthesizeGap()
    follow_up = FollowUp()  # Keep for error cases
    
    # Linear progression
    router - "generate-queries" >> generate_queries
    generate_queries >> literature_review
    literature_review >> synthesize_gap
    
    # Error handling
    router - "follow-up" >> follow_up
    
    return Flow(start=router)
```

**Expected Result:** Complete pipeline from topic → queries → literature review → knowledge gap, with HITL approval at each step.

---

## Phase 3: Add Proposal Writing & Revision Loop

### Objective
Implement the complex proposal writing, review, and revision cycle while maintaining the robust patterns.

### Step 3.1: Create `WriteProposal` Node
```python
class WriteProposal(Node):
    def prep(self, shared):
        # Check if this is a revision (has prior feedback)
        session = load_conversation(shared["conversation_id"])
        knowledge_gap = session.get("knowledge_gap", {})
        prior_feedback = session.get("revision_feedback", "")
        return knowledge_gap, prior_feedback, shared["queue"]
    
    def exec(self, prep_res):
        knowledge_gap, prior_feedback, queue = prep_res
        
        # Generate proposal using LLM
        proposal = call_llm_for_proposal(knowledge_gap, prior_feedback)
        
        # Show proposal to user (same UI pattern)
        queue.put(f"📝 Generated research proposal:\n\n{proposal[:500]}...")
        queue.put(None)
        return proposal
```

### Step 3.2: Create `ReviewProposal` Node  
```python
class ReviewProposal(Node):
    # Use same HITL pattern to show AI review
    # Generate critique using LLM
    # Present review + options to user
```

### Step 3.3: Create `UserApprovalRouter` Node
```python
class UserApprovalRouter(Node):
    def prep(self, shared):
        # Pause for user decision (same pattern as FollowUp)
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(None)
        
        session = load_conversation(shared["conversation_id"])
        proposal = session["proposal_draft"]
        ai_review = session["ai_review"]
        
        # Present options to user
        message = f"Review complete! Options:\n1. ✅ Approve\n2. 📝 Request revision\n3. 🔄 Regenerate"
        return message, shared["queue"]
    
    def exec(self, prep_res):
        message, queue = prep_res
        queue.put(message)
        queue.put(None)
        
        # This will be handled by the continue_workflow mechanism
        # Return routing decision based on user input
        
    def post(self, shared, prep_res, exec_res):
        # Parse user input and return routing decision
        user_input = shared.get("user_input", "").lower()
        
        if "approve" in user_input:
            return "approved"
        elif "revision" in user_input:
            return "revise" 
        else:
            return "regenerate"
```

### Step 3.4: Update Flow with Conditional Routing
```python
def create_flow():
    # ... existing nodes
    write_proposal = WriteProposal()
    review_proposal = ReviewProposal()
    user_approval = UserApprovalRouter()
    
    # Linear flow with conditional routing
    synthesize_gap >> write_proposal
    write_proposal >> review_proposal  
    review_proposal >> user_approval
    
    # Conditional edges for revision loop
    user_approval - "revise" >> write_proposal    # Back to writing with feedback
    user_approval - "regenerate" >> write_proposal # Regenerate without prior feedback
    user_approval - "approved" >> END             # Complete workflow
```

**Expected Result:** Full research proposal workflow with working revision loops, following the patterns from `@WORKFLOW_EXAMPLES.md`.

---

## Migration Principles

### 1. Preserve Working Patterns
- Keep the `prep/exec/post` structure that works
- Maintain the `flow_log.put(None)` pause mechanism
- Preserve the `queue.put(message); queue.put(None)` UI pattern
- Keep the defensive error handling with try/catch

### 2. Gradual Business Logic Replacement
- Replace prompts and LLM calls incrementally
- Keep session management pattern but change state schema
- Evolve the action space gradually (4 actions → 2 actions → 6 actions)

### 3. Test at Each Phase
- Phase 1: Test basic routing works
- Phase 2: Test linear pipeline with HITL
- Phase 3: Test revision loops

### 4. Preserve Session Management
```python
# Keep the conversation.py pattern:
session = load_conversation(conversation_id)
session["new_field"] = new_value  
save_conversation(conversation_id, session)
```

### 5. Maintain UI Compatibility
- Keep the same `chat_fn` signature in `ui_pocketflow_demo.py`
- Preserve the Flow Log + Chat Message pattern
- Maintain the queue-based communication

---

## Implementation Order

1. **Phase 1.1**: Update `ResearchAgentRouter` prompt only
2. **Phase 1.2**: Test basic routing ("research LLMs" → "generate-queries")
3. **Phase 2.1**: Implement `GenerateQueries` node
4. **Phase 2.2**: Test query generation + HITL approval
5. **Phase 2.3**: Implement `LiteratureReview` node  
6. **Phase 2.4**: Implement `SynthesizeGap` node
7. **Phase 3.1**: Implement `WriteProposal` node
8. **Phase 3.2**: Implement `ReviewProposal` node
9. **Phase 3.3**: Implement revision loop routing
10. **Phase 3.4**: End-to-end testing with revision cycles

This approach ensures we always have a working system and can rollback at any point if issues arise. 