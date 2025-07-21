import dspy
import gradio as gr #Refactor
from gradio import ChatMessage #Refactor
import pandas as pd

TABLE_SAVE_FILE ="queries.csv"

class QueyTools:
    """Tools for interacting with the Mem0 memory system."""

    def __init__(self, query_generator : dspy.Module, context : str):
        #self.memory : Dict[str, bool|None] = {}
        self.table = pd.DataFrame(columns=["query", "approved"])
        #self.query_index : int = 0
        self.query_generator = query_generator
        self.context = context

        self.flow_log = []

    def generate_query(self, user_request : str ) -> str:
        """Generate a query from a user query request."""

        self.flow_log.append(ChatMessage(
            role="assistant",
            content=f"Generating query for: {user_request}",
            metadata={"title": "🛠️ Tool Use: Query Generator"}
        ))
        res = self.query_generator(user_request=user_request, context=self.context)
        result = self._add_query(res.query, pd.NA)
        return result
    
    def add_user_query(self, query : str) -> str:
        """Add a user query to the memory."""
        print(f"add_user_query: {query}")
        self._add_query(query,True)
 
        return f"Query added: {query}, index: {len(self.table)-1}"


    def _add_query(self, query : str, approved : bool) -> str:
        """Generate a query for a given topic."""
        print(f"add_query: {query}")
        #if query in self.memory:
        #    return f"Query already exists: {query}"
        self.table.loc[len(self.table)] = [query, approved]
        #self.memory[query]= None
        return f"Query added: {query}, index: {len(self.table)-1} needs to be reviewed"

  
    def add_query_review(self, query_index : int, approved : bool) -> str:
        """Review a query and approve or reject it."""
        print(f"add_query_review: {query_index}, {approved}")
        self.table.loc[query_index, "approved"] = approved
        return f"Query reviewed: {query_index} {approved}"

    def list_all_queries(self) -> gr.ChatMessage:
        """Returns all queries."""
        self.flow_log.append(ChatMessage(
            role="assistant",
            content="Listing all queries",
            metadata={"title": "🛠️ Tool Use: List All Queries"}
        ))
        print(f"search_approve_queries")
        html_table = self.table.to_html(index=False, escape=True)
        return ChatMessage(
            role="assistant",
            content=html_table,
        )
    
    def list_approved_queries(self) -> gr.HTML:
        """Returns approved queries."""
        print(f"search_approve_queries")
        approved_df = self.table[self.table["approved"] == True][["query"]]
        html_table = approved_df.to_html(index=False, escape=True)
        return gr.HTML(html_table)
    
    def list_rejected_queries(self) -> str:
        """Returns rejected queries."""
        print(f"search_rejected_queries")
        return self.table[self.table["approved"] == False]["query"].to_string()
    
    def list_pending_queries(self) -> str:
        """Returns pending queries."""
        print(f"search_pending_queries")
        return self.table[self.table["approved"].isna()]["query"].to_string()
    
    def save(self) -> str:
        """Save the table to a file."""

        self.flow_log.append(ChatMessage(
            role="assistant",
            content=f"Saving table to queries.csv",
            metadata={"title": "💾 Save Queries"}
        ))
        print(f"save: {self.table}")
        self.table.to_csv(TABLE_SAVE_FILE, index=False)
        return f"Table saved to {TABLE_SAVE_FILE}"
    

class QueryHelper(dspy.Signature):
    """
    You're a research assistant helping to generate and bookkeep research questions.
    Your task unclude:
    - Generate a research question from user request.
    - List already created questions satisfying. 
    """
    user_request: str = dspy.InputField()
    past_user_requests: str = dspy.InputField()
    question_topic: str = dspy.InputField()
    response: str = dspy.OutputField()

class QueryAgent(dspy.Module):
    """ReAct agent for query analysis using Yahoo Finance data."""

    def __init__(self):
        super().__init__()

        self.query_generator = dspy.Predict("user_request: str , context: str -> query: str") 
        self.query_tools = QueyTools(self.query_generator, context="LLM agents")  
        #self.table = pd.DataFrame(columns=["query", "approved"])
        self.table = pd.DataFrame(columns=["query"])  
        self.tools = [self.query_tools.generate_query , self.query_tools.list_all_queries]
        '''
        self.tools = [
            self.query_tools.generate_query,
            self.query_tools.add_user_query,
            self.query_tools.add_query_review,
            self.query_tools.list_approved_queries,
            self.query_tools.list_rejected_queries,
            self.query_tools.list_pending_queries,
            self.query_tools.list_all_queries,
            self.query_tools.save
        ]
        '''

        # Initialize ReAct
        dspy.configure(lm=dspy.LM('ollama_chat/qwen3:4b', api_base='http://localhost:11434', api_key=''))
        self.react = dspy.ReAct(
            signature=QueryHelper,
            tools=self.tools,
            max_iters=1
        )

    def forward(self, user_request: str, past_user_requests: str, question_topic: str):
        result = self.react(user_request=user_request, past_user_requests=past_user_requests,question_topic=question_topic)
        return result.response 
