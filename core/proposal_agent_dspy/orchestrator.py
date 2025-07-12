"""
Simple HITL orchestrator for research workflow.
No complex threading - just async/await for UI integration.
"""

import asyncio
from typing import Dict, Any, Callable, Optional
from queue import Queue
import json

from .modules import (
    QueryGenerator, 
    LiteratureReviewer, 
    GapAnalyzer, 
    ProposalWriter, 
    ProposalReviewer,
    mock_literature_search
)
from .config import configure_dspy_for_research


class ResearchOrchestrator:
    """Simple orchestrator that handles HITL workflow"""
    
    def __init__(self, chat_queue: Queue, flow_queue: Queue):
        configure_dspy_for_research()
        
        self.chat_queue = chat_queue
        self.flow_queue = flow_queue
        
        # Initialize DSPy modules
        self.query_gen = QueryGenerator()
        self.lit_reviewer = LiteratureReviewer()
        self.gap_analyzer = GapAnalyzer()
        self.proposal_writer = ProposalWriter()
        self.proposal_reviewer = ProposalReviewer()
        
        # Simple state storage
        self.state = {}
        self.human_feedback_event = None
        self.human_response = None
    
    def log_flow(self, message: str):
        """Log to flow queue"""
        self.flow_queue.put(message)
    
    def send_message(self, message: str):
        """Send message to chat"""
        self.chat_queue.put(message)
    
    async def wait_for_human(self, prompt: str) -> str:
        """Wait for human feedback"""
        # Send prompt to UI
        self.send_message(prompt)
        
        # Set up waiting mechanism (you'll need to trigger this from UI)
        self.human_feedback_event = asyncio.Event()
        self.human_response = None
        
        # Wait for human response (UI will set this)
        await self.human_feedback_event.wait()
        
        response = self.human_response
        self.human_response = None
        return response
    
    def set_human_response(self, response: str):
        """Called by UI when human provides feedback"""
        self.human_response = response
        if self.human_feedback_event:
            self.human_feedback_event.set()
    
    async def run_research_workflow(self, topic: str):
        """Main research workflow with HITL checkpoints"""
        try:
            self.log_flow("🚀 Starting research workflow")
            self.state["topic"] = topic
            
            # Step 1: Generate queries
            self.log_flow("🔍 Generating research queries...")
            queries = self.query_gen(topic=topic)
            self.state["queries"] = queries
            
            # HITL Checkpoint 1: Query approval
            query_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(queries)])
            prompt = f"""🔍 **Generated Research Queries:**

{query_text}

**Please review these queries:**
👍 Type **"approved"** to proceed with literature search
👎 Type **"rejected"** to regenerate queries
💭 Or provide specific feedback for improvements"""
            
            feedback = await self.wait_for_human(prompt)
            
            if "rejected" in feedback.lower():
                self.send_message("🔄 Regenerating queries based on your feedback...")
                # Could implement query regeneration with feedback here
                return await self.run_research_workflow(topic)  # Restart
            
            # Step 2: Literature search and review
            self.log_flow("📚 Conducting literature review...")
            search_results = mock_literature_search(queries)  # Replace with real PaperQA
            literature_summary = self.lit_reviewer(
                topic=topic, 
                queries=queries, 
                search_results=search_results
            )
            self.state["literature_summary"] = literature_summary
            
            # Step 3: Gap analysis
            self.log_flow("🧩 Analyzing research gaps...")
            research_gaps = self.gap_analyzer(
                topic=topic, 
                literature_summary=literature_summary
            )
            self.state["research_gaps"] = research_gaps
            
            # Step 4: Generate proposal
            self.log_flow("📝 Writing research proposal...")
            proposal = self.proposal_writer(
                topic=topic,
                literature_summary=literature_summary,
                research_gaps=research_gaps
            )
            self.state["proposal"] = proposal
            
            # HITL Checkpoint 2: Proposal approval
            prompt = f"""📋 **Research Proposal Generated:**

{proposal[:500]}...

**Please review the proposal:**
👍 Type **"approved"** to finalize
👎 Type **"rejected"** to revise
📖 Type **"show full"** to see complete proposal"""
            
            feedback = await self.wait_for_human(prompt)
            
            if "show full" in feedback.lower():
                self.send_message(f"**Complete Proposal:**\n\n{proposal}")
                feedback = await self.wait_for_human("Now, do you approve this proposal? (approved/rejected)")
            
            if "rejected" in feedback.lower():
                self.send_message("🔄 Revising proposal...")
                # Could implement revision logic here
                return
            
            # Step 5: Final review and scoring
            self.log_flow("⭐ Generating final review...")
            review_feedback, score = self.proposal_reviewer(proposal)
            
            # Final result
            final_message = f"""✅ **Research Proposal Complete!**

**Quality Score:** {score:.2f}/1.0

**Expert Review:**
{review_feedback}

**Your research proposal is ready for submission! 🎉**"""
            
            self.send_message(final_message)
            self.log_flow("✅ Research workflow completed successfully")
            
        except Exception as e:
            self.log_flow(f"❌ Error in workflow: {str(e)}")
            self.send_message(f"❌ An error occurred: {str(e)}")
        
        finally:
            # Signal completion
            self.chat_queue.put(None)
            self.flow_queue.put(None)


# Simple factory function
def create_research_orchestrator(chat_queue: Queue, flow_queue: Queue) -> ResearchOrchestrator:
    """Create a new research orchestrator instance"""
    return ResearchOrchestrator(chat_queue, flow_queue)
