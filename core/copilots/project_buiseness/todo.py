import yfinance as yf
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from dspy.adapters.types.tool import Tool

# Convert LangChain Yahoo Finance tool to DSPy
yahoo_finance_tool = YahooFinanceNewsTool()
finance_news_tool = Tool.from_langchain(yahoo_finance_tool)

print(finance_news_tool.args)
print(yahoo_finance_tool.invoke("NVDA"))

class FinancialAnalysisAgent(dspy.Module):
    """ReAct agent for financial analysis using Yahoo Finance data."""

    def __init__(self):
        super().__init__()

        # Combine all tools
        self.tools = [
            finance_news_tool,  # LangChain Yahoo Finance News
        ]

        # Initialize ReAct
        self.react = dspy.ReAct(
            signature="financial_query -> analysis_response",
            tools=self.tools,
            max_iters=6
        )

    def forward(self, financial_query: str):
        # HACK
        dspy.configure(lm=dspy.LM('ollama_chat/qwen3:4b', api_base='http://localhost:11434', api_key=''))
        print(f"Financial query: {financial_query}")
        try:
            answer = self.react(financial_query=financial_query)
            print(f"Answer: {answer}")
            return answer
        except Exception as e:
            print(f"Error: {e}")
            return f"Error: {e}"
        
class BusinessStrategyAgent(dspy.Module):
    """ReAct agent for financial analysis using Yahoo Finance data."""

    def __init__(self):
        super().__init__()

        # Combine all tools
        self.tools = [
            finance_news_tool,  # LangChain Yahoo Finance News
        ]

        # Initialize ReAct
        self.react = dspy.ReAct(
            signature="financial_query -> analysis_response",
            tools=self.tools,
            max_iters=6
        )

    def forward(self, financial_query: str):
        output = asyncio.run(self.react.acall(financial_query=financial_query))
        return output

