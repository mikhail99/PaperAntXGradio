import gradio as gr
from state.state import get_shared_state
from ui.ui_collections import create_collections_tab
from ui.ui_articles import create_articles_tab
from ui.ui_proposal_agent_debugger import create_proposal_debugger_tab
from ui.ui_copilot import create_copilot_tab
from ui.ui_paperqa import create_paperqa_tab
from ui.ui_research_plan import create_research_plan_tab
from ui.ui_library import create_library_tab
from ui.custom_css import CUSTOM_CSS
from core.article_manager import ArticleManager
from core.collections_manager import CollectionsManager
from core.copilot_service import CopilotService
from core.llm_service import LLMService
from core.mcp_server_manager import MCPServerManager
from core.proposal_agent_ui_service import ProposalAgentUIService


def main():
    with gr.Blocks(css=CUSTOM_CSS) as demo:
        gr.Markdown("# PaperAnt X")
        state = get_shared_state()

        llm_service = LLMService()
        collections_manager = CollectionsManager()
        article_manager = ArticleManager(collections_manager)
        mcp_server_manager = MCPServerManager()
        copilot_service = CopilotService(collections_manager, article_manager, llm_service, mcp_server_manager)
        proposal_agent_ui_service = ProposalAgentUIService(collections_manager)

        with gr.Tabs():
            create_articles_tab(state)
            create_paperqa_tab(state)
            create_research_plan_tab(state)
            create_proposal_debugger_tab(proposal_agent_ui_service)
            create_copilot_tab(state, copilot_service)
            create_collections_tab(state)
            create_library_tab(state)
    demo.launch()

if __name__ == '__main__':
    main() 