"""
Minimal PocketFlow Research Nodes

Direct copy of cookbook pattern - no abstractions, just business logic.
"""

import json
import threading
from queue import Queue
from pocketflow import Node
from typing import List

from .dspy_modules import QueryGenerator, KnowledgeSynthesizer, ProposalWriter, ProposalReviewer
from core.business_intelligence.business_bridge import get_paperqa_service


class GenerateQueries(Node):
    """Generate search queries using DSPy (like cookbook DecideAction)"""
    
    def prep(self, shared):
        topic = shared["topic"]
        return topic
    
    def exec(self, prep_res):
        topic = prep_res
        generator = QueryGenerator()
        prediction = generator(topic=topic, existing_queries=[])
        return prediction.queries
    
    def post(self, shared, prep_res, exec_res):
        shared["queries"] = exec_res
        flow_log: Queue = shared["flow_queue"]
        chat_queue: Queue = shared["queue"]
        
        flow_log.put(f"🔍 Generated {len(exec_res)} queries")
        for i, query in enumerate(exec_res, 1):
            flow_log.put(f"  {i}. {query}")
        
        # Send structured message for UI
        chat_queue.put(json.dumps({
            "type": "query_review",
            "message": "Please review the generated queries",
            "queries": exec_res
        }))
        
        # Create blocking event for HITL
        if "hitl_event" not in shared:
            shared["hitl_event"] = threading.Event()
        
        hitl_event = shared["hitl_event"]
        hitl_event.clear()  # Reset event
        
        flow_log.put("PAUSED")  # Signal to orchestrator we're paused
        
        print("GenerateQueries: Waiting for user input...")
        hitl_event.wait()  # BLOCK here until user input received
        print(f"GenerateQueries: Received input: {shared.get('user_input', 'N/A')}")
        
        return "default"


class QueryDocuments(Node):
    """Query documents using PaperQA (like cookbook CheckWeather)"""
    
    def prep(self, shared):
        queries = shared["queries"]
        collection = shared["collection"]
        return queries, collection
    
    def exec(self, prep_res):
        queries, collection = prep_res
        paperqa = get_paperqa_service()
        
        summaries = []
        for query in queries:
            try:
                result = paperqa.query_documents(collection, query)
                if result and not result.get("error"):
                    summaries.append(result.get("answer_text", "No summary"))
                else:
                    summaries.append(f"Error: {query}")
            except Exception as e:
                summaries.append(f"Exception: {str(e)}")
        
        return summaries
    
    def post(self, shared, prep_res, exec_res):
        shared["summaries"] = exec_res
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(f"📚 Queried {len(exec_res)} documents")
        return "default"


class SynthesizeGap(Node):
    """Synthesize knowledge gap using DSPy (like cookbook BookHotel)"""
    
    def prep(self, shared):
        topic = shared["topic"]
        summaries = "\n---\n".join(shared["summaries"])
        return topic, summaries
    
    def exec(self, prep_res):
        topic, summaries = prep_res
        synthesizer = KnowledgeSynthesizer()
        prediction = synthesizer(topic=topic, literature_summaries=summaries)
        return prediction.knowledge_gap.model_dump()
    
    def post(self, shared, prep_res, exec_res):
        shared["gap"] = exec_res
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(f"🧠 Synthesized knowledge gap")
        return "default"


class WriteProposal(Node):
    """Write proposal using DSPy (like cookbook FollowUp with HITL)"""
    
    def prep(self, shared):
        gap = shared["gap"]
        return gap
    
    def exec(self, prep_res):
        gap = prep_res
        
        # Send structured message for UI
        chat_queue: Queue = shared["queue"]
        gap_summary = gap.get("knowledge_gap", "Unknown gap")[:100]
        
        chat_queue.put(json.dumps({
            "type": "proposal_approval",
            "message": f"Ready to write proposal for gap: {gap_summary}... Approve?",
            "gap": gap
        }))
        
        # Create blocking event for HITL
        if "hitl_event" not in shared:
            shared["hitl_event"] = threading.Event()
        
        hitl_event = shared["hitl_event"]
        hitl_event.clear()  # Reset event
        
        flow_log: Queue = shared["flow_queue"]
        flow_log.put("PAUSED")  # Signal to orchestrator we're paused
        
        print("WriteProposal: Waiting for user input...")
        hitl_event.wait()  # BLOCK here until user input received
        print(f"WriteProposal: Received input: {shared.get('user_input', 'N/A')}")
        
        # Generate proposal
        writer = ProposalWriter()
        prediction = writer(knowledge_gap_summary=json.dumps(gap), prior_feedback="")
        return prediction.proposal
    
    def post(self, shared, prep_res, exec_res):
        shared["proposal"] = exec_res
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(f"📝 Generated proposal ({len(exec_res)} chars)")
        return "default"


class ReviewProposal(Node):
    """Review proposal using DSPy (like cookbook ResultNotification)"""
    
    def prep(self, shared):
        proposal = shared["proposal"]
        return proposal
    
    def exec(self, prep_res):
        proposal = prep_res
        
        # Generate review
        reviewer = ProposalReviewer()
        prediction = reviewer(proposal_draft=proposal, review_aspect="novelty and contribution")
        
        # Show result to user
        score = prediction.critique.score
        chat_queue: Queue = shared["queue"]
        
        chat_queue.put(json.dumps({
            "type": "final_review",
            "message": f"✅ Proposal reviewed! Score: {score:.2f}",
            "review": prediction.critique.model_dump(),
            "proposal_length": len(proposal)
        }))
        
        # Create blocking event for HITL
        if "hitl_event" not in shared:
            shared["hitl_event"] = threading.Event()
        
        hitl_event = shared["hitl_event"]
        hitl_event.clear()  # Reset event
        
        flow_log: Queue = shared["flow_queue"]
        flow_log.put("PAUSED")  # Signal to orchestrator we're paused
        
        print("ReviewProposal: Waiting for user input...")
        hitl_event.wait()  # BLOCK here until user input received
        print(f"ReviewProposal: Received input: {shared.get('user_input', 'N/A')}")
        
        return prediction.critique.model_dump()
    
    def post(self, shared, prep_res, exec_res):
        shared["review"] = exec_res
        shared["final_proposal"] = shared["proposal"]
        return "done" 