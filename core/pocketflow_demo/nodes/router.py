from core.pocketflow_demo.nodes.actions import Action
from core.pocketflow_demo.utils.conversation import load_conversation, save_conversation
from queue import Queue
from pocketflow import Node
from core.pocketflow_demo.nodes.workers import GenerateQueries, LiteratureReview, SynthesizeGap, ReportGeneration, FollowUp, ResultNotification
from core.pocketflow_demo.nodes.hitl import ReviewQueries, ReviewReport

# Action to Node Registry - includes both action nodes and review nodes
ACTION_TO_NODE = {
    Action.do_generate_queries: GenerateQueries,
    Action.do_literature_review: LiteratureReview,
    Action.do_literature_review_gap: SynthesizeGap,
    Action.do_write_proposal: ReportGeneration,
    Action.do_follow_up: FollowUp,
    Action.do_result_notification: ResultNotification,
    # Review nodes
    Action.review_queries: ReviewQueries,
    Action.review_report: ReviewReport,
}


def compute_next_action(history: list[dict], query: str, last_action: str, last_action_result: str) -> str:
    print(f"🤖 Computing next action. Last action: '{last_action}'")
    print(f"📨 Current query: '{query}'")
    
    # Determine next action based on current state
    if last_action is None:
        print("🏁 Starting new flow")
        next_action = Action.do_generate_queries
        
    elif last_action == Action.do_generate_queries:
        print("📝 After query generation → review")
        next_action = Action.review_queries
        
    elif last_action == Action.review_queries:
        print("🔍 At query review step, checking for feedback...")
        # Check for feedback in current query
        feedback = check_feedback_in_message(query)
        
        if feedback == "approved":
            print("✅ User approved queries → literature review")
            next_action = Action.do_literature_review
        elif feedback == "rejected":
            print("❌ User rejected queries → retry generation")
            next_action = Action.do_generate_queries
        else:
            print(f"⚠️ No clear feedback detected, defaulting to follow_up")
            next_action = Action.do_follow_up
            
    elif last_action == Action.do_literature_review:
        print("📚 After literature review → gap analysis")
        next_action = Action.do_literature_review_gap
        
    elif last_action == Action.do_literature_review_gap:
        print("🔍 After gap analysis → proposal generation")
        next_action = Action.do_write_proposal
        
    elif last_action == Action.do_write_proposal:
        print("📋 After proposal generation → review")
        next_action = Action.review_report
        
    elif last_action == Action.review_report:
        print("🔍 At report review step, checking for feedback...")
        # Check for feedback in current query
        feedback = check_feedback_in_message(query)
        
        if feedback == "approved":
            print("✅ User approved report → final result")
            next_action = Action.do_result_notification
        elif feedback == "rejected":
            print("❌ User rejected report → retry generation")
            next_action = Action.do_write_proposal
        else:
            print(f"⚠️ No clear feedback detected, defaulting to follow_up")
            next_action = Action.do_follow_up
            
    else:
        print(f"⚠️ Unhandled last_action: '{last_action}', defaulting to follow_up")
        next_action = Action.do_follow_up
    
    print(f"🎯 Next action determined: {next_action}")
    
    decision = {
        "thinking": f"Moving from {last_action} to {next_action}",
        "action": next_action,
        "reason": "Following research pipeline",
        "question": "...",
        "topic": query,
        "search_query": "llm for math research",
        "summary": "literature summary here",
        "gaps": "research gaps identified",
        "result": "final research proposal generated",
    }
    return decision

def check_feedback_in_message(message: str) -> str:
    """
    Check if a message contains feedback keywords.
    Returns: 'approved', 'rejected', or None
    """
    if not message:
        return None
        
    message_lower = message.lower().strip()
    print(f"🔍 Checking message for feedback: '{message_lower}'")
    
    # Check for approval keywords (more flexible)
    approval_keywords = ["approve", "approved", "proceed", "continue", "yes", "ok", "okay", "good", "looks good"]
    rejection_keywords = ["reject", "rejected", "retry", "redo", "no", "not good", "bad", "change"]
    
    for keyword in approval_keywords:
        if keyword in message_lower:
            print(f"✅ Detected APPROVAL keyword: '{keyword}'")
            return "approved"
    
    for keyword in rejection_keywords:
        if keyword in message_lower:
            print(f"❌ Detected REJECTION keyword: '{keyword}'")
            return "rejected"
    
    print("❓ No feedback keywords detected")
    return None

def prepare_for_next_action(session, decision: dict, flow_log):
    """Generic parameter preparation using node self-declaration"""
    next_action = decision["action"]
    
    # Initialize session params if it doesn't exist
    if "params" not in session:
        session["params"] = {}
    
    # Handle review actions (they use Action enum values now)
    if next_action in [Action.review_queries, Action.review_report]:
        # No parameters needed for review actions
        flow_log.put(f"➡️ Starting review for: {next_action}")
        return next_action
    
    # Find the node class for this action
    if next_action not in ACTION_TO_NODE:
        # Fallback to follow-up for unknown actions
        return handle_missing_action(session, flow_log, next_action)
    
    node_class = ACTION_TO_NODE[next_action]
    required_params = node_class.required_params()
    
    # Extract and validate parameters
    params = {}
    missing_params = []
    
    for param in required_params:
        if param in decision:
            params[param] = decision[param]
        else:
            missing_params.append(param)
    
    if missing_params:
        # Fallback to follow-up with helpful message
        return handle_missing_params(session, flow_log, next_action, missing_params)
    
    # Store parameters and log success
    session["params"][next_action] = params
    flow_log.put(f"➡️ Agent decided to {next_action} with params: {params}")
    return next_action

def handle_missing_action(session, flow_log, action):
    """Handle unknown action by falling back to follow-up"""
    # Initialize session params if it doesn't exist
    if "params" not in session:
        session["params"] = {}
    
    question = f"I'm not sure how to handle the action '{action}'. Could you please rephrase your request?"
    session["params"][Action.do_follow_up] = {"question": question}
    flow_log.put(f"⚠️ Unknown action: {action}. Falling back to follow-up.")
    return Action.do_follow_up

def handle_missing_params(session, flow_log, action, missing_params):
    """Handle missing parameters by falling back to follow-up"""
    # Initialize session params if it doesn't exist
    if "params" not in session:
        session["params"] = {}
    
    question = f"I need more information to {action}. Missing: {', '.join(missing_params)}. Could you provide these details?"
    session["params"][Action.do_follow_up] = {"question": question}
    flow_log.put(f"⚠️ Missing parameters for {action}: {missing_params}. Falling back to follow-up.")
    return Action.do_follow_up

class ResearchAgentRouter(Node):
    @staticmethod
    def required_params():
        return []  # Router doesn't need input parameters
    
    @staticmethod
    def action_type():
        return None  # Router is not an action, it decides actions

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
        session["last_action"] = exec_res["action"]
        flow_log: Queue = shared["flow_queue"]

        for line in exec_res["thinking"].split("\n"):
            line = line.replace("-", "").strip()
            if line:
                flow_log.put(f"🤔 {line}")

        next_action = prepare_for_next_action(session, exec_res, flow_log)
        save_conversation(conversation_id, session)
        return next_action
