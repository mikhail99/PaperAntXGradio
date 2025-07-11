from core.pocketflow_demo.nodes.actions import Action
from core.pocketflow_demo.nodes.base import NodeBase
from core.pocketflow_demo.types import SharedState, ResearchContext
from core.pocketflow_demo.utils.conversation import load_conversation, save_conversation
from typing import Union, Tuple, Dict, Optional
from queue import Queue
from pocketflow import Node

class FlowEntry(NodeBase):
    """Smart entry point that detects where to resume the flow"""
    
    @staticmethod
    def required_params():
        return []
    
    @staticmethod
    def action_type():
        return None  # Special entry node
    
    def prep(self, shared: SharedState) -> str:
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        
        # Check if we're resuming after feedback
        waiting_for = session.get("waiting_for_feedback")
        
        print(f"🎯 FlowEntry: waiting_for_feedback = {waiting_for}")
        
        if waiting_for == "queries":
            print("📍 Resuming at query review")
            return "resume_query_review"
        elif waiting_for == "report":
            print("📍 Resuming at report review") 
            return "resume_report_review"
        else:
            print("📍 Starting new research flow")
            return "start_new_flow"
    
    def exec(self, prep_res: str) -> str:
        return prep_res
    
    def post(self, shared: SharedState, prep_res: str, exec_res: str) -> str:
        return exec_res  # Route based on the detected state

class PauseForFeedback(NodeBase):
    """Special node that properly terminates the flow when waiting for human feedback"""
    
    @staticmethod
    def required_params():
        return []
    
    @staticmethod
    def action_type():
        return None  # Special pause node
    
    def prep(self, shared: SharedState) -> str:
        # This node just terminates the flow cleanly
        return "Pausing flow for human feedback"
    
    def exec(self, prep_res: str) -> str:
        return prep_res
    
    def post(self, shared: SharedState, prep_res: str, exec_res: str) -> str:
        # Return something that terminates the flow without warnings
        # The waiting_for_feedback state is already set by the review node
        return "terminated"

class GenerateQueries(NodeBase):
    @staticmethod
    def required_params():
        return []  # No longer using params - getting from context
    
    @staticmethod
    def action_type():
        return Action.do_generate_queries

    def prep(self, shared: SharedState) -> str:
        # Extract topic from current query (start of research journey)
        topic = shared["query"].strip()
        
        # Store topic as first step in research journey
        self.update_research_journey(shared, "topic_extraction", topic)
        
        return topic

    def exec(self, prep_res: str) -> str:
        topic = prep_res
        queries = f"Generated search queries for {topic}:\n1. '{topic} recent advances 2023-2024'\n2. '{topic} applications in education'\n3. '{topic} limitations and challenges'"
        return queries

    def post(self, shared: SharedState, prep_res: str, exec_res: str) -> str:
        # Update research journey with query generation results
        self.update_research_journey(shared, "query_generation", exec_res)
        self.log_to_flow(shared, f"⬅️ Generated queries: {exec_res}")
        return "default"

class LiteratureReview(NodeBase):
    @staticmethod
    def required_params():
        return []  # Getting from context
    
    @staticmethod
    def action_type():
        return Action.do_literature_review

    def prep(self, shared: SharedState) -> str:
        # Get queries from research context
        context = self.get_research_context(shared)
        queries = context.get("queries")
        
        if not queries:
            # Fallback if no queries in journey
            queries = "No queries found in research journey"
            
        return queries

    def exec(self, prep_res: str) -> str:
        queries = prep_res
        literature = f"Literature review completed for queries:\n{queries}\n\n📚 Found 15 relevant papers\n📊 Key themes: neural networks, transformers, education applications\n📈 Growing trend in personalized learning systems"
        return literature

    def post(self, shared: SharedState, prep_res: str, exec_res: str) -> str:
        # Update research journey with literature review results
        self.update_research_journey(shared, "literature_review", exec_res)
        self.log_to_flow(shared, f"⬅️ Literature review: {exec_res}")
        return "default"

class SynthesizeGap(NodeBase):
    @staticmethod
    def required_params():
        return []  # Getting from context
    
    @staticmethod
    def action_type():
        return Action.do_literature_review_gap

    def prep(self, shared: SharedState) -> Dict[str, str]:
        # Get topic, queries, and literature from research context
        context = self.get_research_context(shared)
        
        return {
            "topic": context.get("topic") or "Unknown topic",
            "queries": context.get("queries") or "No queries",
            "literature": context.get("literature") or "No literature review"
        }

    def exec(self, prep_res: Dict[str, str]) -> str:
        data = prep_res
        gaps = f"Research gaps identified for '{data['topic']}':\n\n🔍 Gap 1: Limited studies on real-time feedback systems\n🔍 Gap 2: Lack of multilingual LLM education research\n🔍 Gap 3: Insufficient evaluation of long-term learning outcomes\n\nBased on literature: {data['literature'][:100]}..."
        return gaps

    def post(self, shared: SharedState, prep_res: Dict[str, str], exec_res: str) -> str:
        # Update research journey with gap analysis results
        self.update_research_journey(shared, "gap_analysis", exec_res)
        self.log_to_flow(shared, f"⬅️ Gap analysis: {exec_res}")
        return "default"

class ReportGeneration(NodeBase):
    @staticmethod
    def required_params():
        return []  # Getting from context
    
    @staticmethod
    def action_type():
        return Action.do_write_proposal

    def prep(self, shared: SharedState) -> ResearchContext:
        # Get everything accumulated so far
        context = self.get_research_context(shared)
        return context

    def exec(self, prep_res: ResearchContext) -> str:
        context = prep_res
        
        # Safely get values with default strings to avoid None indexing issues
        topic = context.get('topic') or 'Unknown'
        queries = context.get('queries') or 'N/A'
        literature = context.get('literature') or 'N/A'
        gaps = context.get('gaps') or 'N/A'
        
        proposal = f"""📋 **Research Proposal Draft:**

**Title:** Advanced LLM Applications in Personalized Education
**Topic:** {topic}

**Objective:** Develop real-time feedback systems for personalized learning

**Background:** 
Based on research queries: {queries[:100]}...
Literature findings: {literature[:100]}...

**Research Gaps Identified:**
{gaps[:200]}...

**Methodology:** 
- Design multilingual LLM framework
- Implement adaptive learning algorithms  
- Conduct longitudinal study with 500 students

**Expected Impact:** Improve learning outcomes by 25% through personalized AI tutoring"""
        
        return proposal

    def post(self, shared: SharedState, prep_res: ResearchContext, exec_res: str) -> str:
        # Update research journey with proposal generation results
        self.update_research_journey(shared, "proposal_generation", exec_res)
        self.log_to_flow(shared, f"⬅️ Proposal generated: {exec_res}")
        return "default"

class FollowUp(NodeBase):
    @staticmethod
    def required_params():
        return []  # Getting from context
    
    @staticmethod
    def action_type():
        return Action.do_follow_up

    def prep(self, shared: SharedState) -> Tuple[str, Queue[str]]:
        # End flow and prepare follow-up question
        self.log_to_flow(shared, None)  # Stop flow thoughts
        
        question = "I need more information to continue. Could you please clarify your request?"
        return question, shared["queue"]

    def exec(self, prep_res: Tuple[str, Queue[str]]) -> str:
        question, queue = prep_res
        queue.put(question)
        # Don't put None in chat queue - that's handled by the UI layer
        return question

    def post(self, shared: SharedState, prep_res: Tuple[str, Queue[str]], exec_res: str) -> str:
        # Don't update research journey for follow-up questions
        return "done"

class ResultNotification(NodeBase):
    @staticmethod
    def required_params():
        return []  # Getting from context
    
    @staticmethod
    def action_type():
        return Action.do_result_notification

    def prep(self, shared: SharedState) -> Tuple[str, Queue[str]]:
        # End flow and prepare final result
        self.log_to_flow(shared, None)  # Stop flow thoughts
        
        # Get the final proposal from research context
        context = self.get_research_context(shared)
        proposal = context.get("proposal") or "No proposal generated"
        
        return proposal, shared["queue"]

    def exec(self, prep_res: Tuple[str, Queue[str]]) -> str:
        proposal, queue = prep_res
        
        final_message = f"""✅ **Research Proposal Completed!**

{proposal}

🎉 Your research proposal is ready! You can now:
- Submit to funding agencies
- Share with collaborators  
- Begin preliminary research

Thank you for using the Research Assistant! 🚀"""
        
        queue.put(final_message)
        # Don't put None in chat queue - that's handled by the UI layer
        return final_message

    def post(self, shared: SharedState, prep_res: Tuple[str, Queue[str]], exec_res: str) -> str:
        # Clear session state to mark completion
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        session["last_action"] = None
        session["waiting_for_feedback"] = None
        save_conversation(conversation_id, session)
        return "done"


