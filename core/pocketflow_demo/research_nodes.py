from datetime import datetime
from queue import Queue

import yaml
from pocketflow import Node

from core.pocketflow_demo.utils.call_llm import call_llm
#from core.pocketflow_demo.utils.call_mock_api import call_book_hotel_api, call_check_weather_api
from core.pocketflow_demo.utils.conversation import load_conversation, save_conversation
from core.pocketflow_demo.utils.format_chat_history import format_chat_history
from core.pocketflow_demo.research_state import Action

def compute_next_action(history: list[dict], query: str, last_action: str, last_action_result: str) -> str:

    decision = {
        "thinking": "I think so",
        "action": Action.do_literature_review,
        "reason": "I think so",
        "question": "...",
        "topic": "llm for math",
        "hotel": "...", 
    }

    return decision


def prepare_for_next_action(session, decision: dict , flow_log):
    next_action = decision["action"]
    if next_action == Action.do_generate_queries:
            try:
                topic = decision["topic"]
                session["params"][Action.do_generate_queries] = {
                    "topic": topic,
                }
                flow_log.put(f"➡️ Agent decided to {next_action} for: {topic}")
            except KeyError:
                print(f"⚠️ Missing parameter for  {next_action}. Overriding to {Action.do_follow_up}.")
                question = "I can check the weather for you! Which city would you like to know about? 🌤️"
                session["params"][Action.do_follow_up] = {"question": question}
                flow_log.put(f"➡️ Agent needs more info: {question}")
                next_action = Action.do_follow_up

    elif next_action == Action.do_literature_review:
        try:
            search_query = decision["search_query"]
            session["params"][Action.do_literature_review] = {
                "search_query": search_query,
            }
            flow_log.put(f"➡️ Agent decided to {next_action} for: {search_query}")
        except KeyError as e:
            print(f"⚠️ Missing parameter for 'book-hotel': {e}. Overriding to 'follow-up'.")
            question = "I can help with booking a hotel! Could you please provide the hotel name, check-in date, and check-out date? 🏨"
            session["follow_up_params"] = {"question": question}
            flow_log.put(f"➡️ Agent needs more info: {question}")
            next_action = Action.do_follow_up

    elif decision["action"] == "follow-up":
        session["follow_up_params"] = {"question": exec_res["question"]}
        flow_log.put(f"➡️ Agent decided to follow up: {exec_res['question']}")
    elif decision["action"] == "result-notification":
        session["result_notification_params"] = {"result": exec_res["result"]}
        flow_log.put(f"➡️ Agent decided to notify the result: {exec_res['result']}")


class DecideAction(Node):
    def prep(self, shared):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        return session, shared["history"], shared["query"]

    def exec(self, prep_res):
        session, history, query = prep_res
        last_action = session.get("last_action", None) 
        last_action_result = session.get("action_result", None)
        decision = compute_next_action(history, query, last_action, last_action_result)
        return decision

    def post(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        """Save the decision and determine the next step in the flow."""
        session["last_action"] = exec_res["action"]
        flow_log: Queue = shared["flow_queue"]
        next_action = exec_res["action"]  # Default next action

        for line in exec_res["thinking"].split("\n"):
            line = line.replace("-", "").strip()
            if line:
                flow_log.put(f"🤔 {line}")

        prepare_for_next_action(session, exec_res, flow_log)
        
        save_conversation(conversation_id, session)
        # Return the action to determine the next node in the flow
        return next_action


class GenerateLiteratureSearchQuery(Node):
    def prep(self, shared):
        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        topic = session["params"][Action.do_generate_queries]["topic"]
        return topic

    def exec(self, prep_res):
        topic = prep_res
        return f"search for {topic}"

    def post(self, shared, prep_res, exec_res):
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(f"⬅️ Search query: {exec_res}")

        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        session["action_result"] = exec_res
        save_conversation(conversation_id, session)
        return "default"


class LiteratureSearch(Node):
    def prep(self, shared):
        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        search = session["params"][Action.do_literature_review]["search"]
        return search

    def exec(self, prep_res):
        search = prep_res
        return f"Literature search result for {search}"

    def post(self, shared, prep_res, exec_res):
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(f"⬅️ Literature search result: {exec_res}")

        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        session["action_result"] = exec_res
        save_conversation(conversation_id, session)
        return "default"

class LiteratureGapAnalysis(Node):
    def prep(self, shared):
        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        summary = session["params"][Action.do_literature_review_gap]["summary"]
        return summary

    def exec(self, prep_res):
        summary = prep_res
        return f"Literature gap result for {summary}"

    def post(self, shared, prep_res, exec_res):
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(f"⬅️ Literature gap result: {exec_res}")

        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        session["action_result"] = exec_res
        save_conversation(conversation_id, session)
        return "default"

class ReportGeneration(Node):
    def prep(self, shared):
        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        gaps = session["params"][Action.do_write_proposal]["gaps"]
        return gaps

    def exec(self, prep_res):
        gaps = prep_res
        return f"Project report for {gaps}"

    def post(self, shared, prep_res, exec_res):
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(f"⬅️ Project report: {exec_res}")

        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        session["action_result"] = exec_res
        save_conversation(conversation_id, session)
        return "default"
    
class FollowUp(Node):
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
    def prep(self, shared):
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(None)

        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        result = session["params"][Action.do_result_notification]["result"]
        return result, shared["queue"]

    def exec(self, prep_res):
        result, queue = prep_res
        queue.put(result)
        queue.put(None)
        return result

    def post(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        session["action_result"] = None
        session["last_action"] = None
        save_conversation(conversation_id, session)
        return "done"
