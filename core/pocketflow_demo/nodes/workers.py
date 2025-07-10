from core.pocketflow_demo.nodes.actions import Action
from core.pocketflow_demo.nodes.hitl import ReviewQueries, ReviewReport
from core.pocketflow_demo.utils.conversation import load_conversation, save_conversation
from queue import Queue
from pocketflow import Node

class GenerateQueries(Node):
    @staticmethod
    def required_params():
        return ["topic"]
    
    @staticmethod
    def action_type():
        return Action.do_generate_queries

    def prep(self, shared):
        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        topic = session["params"][Action.do_generate_queries]["topic"]
        return topic

    def exec(self, prep_res):
        topic = prep_res
        return f"Generated search queries for {topic}:\n1. '{topic} recent advances 2023-2024'\n2. '{topic} applications in education'\n3. '{topic} limitations and challenges'"

    def post(self, shared, prep_res, exec_res):
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(f"⬅️ Generated queries: {exec_res}")

        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        session["action_result"] = exec_res
        save_conversation(conversation_id, session)
        return "default"



class LiteratureReview(Node):
    @staticmethod
    def required_params():
        return ["search_query"]
    
    @staticmethod
    def action_type():
        return Action.do_literature_review

    def prep(self, shared):
        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        search_query = session["params"][Action.do_literature_review]["search_query"]
        return search_query

    def exec(self, prep_res):
        search_query = prep_res
        return f"Literature review completed for '{search_query}':\n\n📚 Found 15 relevant papers\n📊 Key themes: neural networks, transformers, education applications\n📈 Growing trend in personalized learning systems"

    def post(self, shared, prep_res, exec_res):
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(f"⬅️ Literature review: {exec_res}")

        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        session["action_result"] = exec_res
        save_conversation(conversation_id, session)
        return "default"

class SynthesizeGap(Node):
    @staticmethod
    def required_params():
        return ["summary"]
    
    @staticmethod
    def action_type():
        return Action.do_literature_review_gap

    def prep(self, shared):
        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        summary = session["params"][Action.do_literature_review_gap]["summary"]
        return summary

    def exec(self, prep_res):
        summary = prep_res
        return f"Research gaps identified:\n\n🔍 Gap 1: Limited studies on real-time feedback systems\n🔍 Gap 2: Lack of multilingual LLM education research\n🔍 Gap 3: Insufficient evaluation of long-term learning outcomes"

    def post(self, shared, prep_res, exec_res):
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(f"⬅️ Gap analysis: {exec_res}")

        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        session["action_result"] = exec_res
        save_conversation(conversation_id, session)
        return "default"

class ReportGeneration(Node):
    @staticmethod
    def required_params():
        return ["gaps"]
    
    @staticmethod
    def action_type():
        return Action.do_write_proposal

    def prep(self, shared):
        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        gaps = session["params"][Action.do_write_proposal]["gaps"]
        return gaps

    def exec(self, prep_res):
        gaps = prep_res
        return f"""📋 **Research Proposal Draft:**

**Title:** Advanced LLM Applications in Personalized Education

**Objective:** Develop real-time feedback systems for personalized learning

**Methodology:** 
- Design multilingual LLM framework
- Implement adaptive learning algorithms  
- Conduct longitudinal study with 500 students

**Expected Impact:** Improve learning outcomes by 25% through personalized AI tutoring"""

    def post(self, shared, prep_res, exec_res):
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(f"⬅️ Proposal generated: {exec_res}")

        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        session["action_result"] = exec_res
        save_conversation(conversation_id, session)
        return "default"


    
class FollowUp(Node):
    @staticmethod
    def required_params():
        return ["question"]
    
    @staticmethod
    def action_type():
        return Action.do_follow_up

    def prep(self, shared):
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(None)

        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        question = session["params"][Action.do_follow_up]["question"]
        return question, shared["queue"]

    def exec(self, prep_res):
        question, queue = prep_res
        queue.put(question)
        queue.put(None)
        return question

    def post(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        session["action_result"] = exec_res
        return "done"

class ResultNotification(Node):
    @staticmethod
    def required_params():
        return ["result"]
    
    @staticmethod
    def action_type():
        return Action.do_result_notification

    def prep(self, shared):
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(None)

        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        result = session["params"][Action.do_result_notification]["result"]
        return result, shared["queue"]

    def exec(self, prep_res):
        result, queue = prep_res
        final_message = f"""✅ **Research Proposal Completed!**

{result}

🎉 Your research proposal is ready! You can now:
- Submit to funding agencies
- Share with collaborators  
- Begin preliminary research

Thank you for using the Research Assistant! 🚀"""
        queue.put(final_message)
        queue.put(None)
        return final_message

    def post(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        session["action_result"] = None
        session["last_action"] = None
        save_conversation(conversation_id, session)
        return "done"


