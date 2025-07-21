# Configure DSPy for parrot mode at app startup FIRST to avoid threading conflicts
import dspy
from core.copilots.project_buiseness.business_bridge import get_llm
#dspy.configure(lm=get_llm())

import gradio as gr
from ui.state.state import get_shared_state
from ui.old.ui_collections import create_collections_tab
#from ui.ui_articles import create_articles_tab
#from ui.ui_proposal_agent_debugger import create_proposal_debugger_tab
#from ui.ui_copilot import create_copilot_tab
#from ui.ui_paperqa import create_paperqa_tab
#from ui.ui_research_plan import create_research_plan_tab
#from ui.ui_library import create_library_tab
#from ui.ui_manager_review import create_manager_review_tab
#from ui.ui_pocketflow_demo import create_pocketflow_demo_tab
#from ui.ui_research_demo import create_research_demo_tab
from ui.custom_css import CUSTOM_CSS
#from core.article_manager import ArticleManager
#from core.collections_manager import CollectionsManager
#from core.copilot_service import CopilotService
#from core.llm_service import LLMService
#from core.mcp_server_manager import MCPServerManager
#from core.proposal_agent_pf_dspy.main import create_research_service as create_service
#from core.copilots.copilot_business_service import CopilotBusinessService 
#from ui.ui_copilot_business import create_copilot_tab as create_copilot_business_tab
#from ui.old.ui_test import create_ui_test_tab
from ui.ui_copilot_project_proposal import create_copilot_tab as create_copilot_project_proposal_tab
from core.copilots.project_proposal.copilot_project_proposal_service import CopilotProjectProposalService
#from ui.ui_copilot_libraryQA import create_copilot_tab as create_copilot_library_qa_tab
#from core.copilots.copilot_papersQA import CopilotPaperQAService
from ui.ui_copilot_project_portfolio import create_copilot_tab as create_copilot_project_portfolio_tab
from core.copilots.copilot_project_portfolio import CopilotProjectPortfolioService
def main():
    with gr.Blocks(css=CUSTOM_CSS, fill_width=True) as demo:
        gr.Markdown("# PaperAnt X")
        state = get_shared_state()

        #llm_service = LLMService()
        #collections_manager = CollectionsManager()
        #article_manager = ArticleManager(collections_manager)
        #mcp_server_manager = MCPServerManager()
        #copilot_service = CopilotService(collections_manager, article_manager, llm_service, mcp_server_manager)
        #copilot_business_service = CopilotBusinessService()
        copilot_project_proposal_service = CopilotProjectProposalService()
        #copilot_paper_qa_service = CopilotPaperQAService()
        copilot_project_portfolio_service = CopilotProjectPortfolioService()
        #proposal_agent_service = create_service(use_parrot=True)

        # Create the hidden trigger textboxes at the top level, outside the Tabs.
        # This ensures they are present in the DOM from the start, avoiding the race condition.
        proposal_trigger = gr.Textbox(label="proposal_trigger", visible=True, elem_id="copilot_selected_agent_trigger_proposal", elem_classes="hidden-trigger")
        portfolio_trigger = gr.Textbox(label="portfolio_trigger", visible=True, elem_id="copilot_selected_agent_trigger_portfolio", elem_classes="hidden-trigger")

        with gr.Tabs():
            #create_manager_review_tab()
            #create_articles_tab(state)
            #create_paperqa_tab(state)
            #create_research_plan_tab(proposal_agent_service, collections_manager)
            ##create_proposal_debugger_tab(proposal_agent_service, collections_manager)
            ##create_pocketflow_demo_tab()
            #create_research_demo_tab()
            #create_copilot_tab(state, copilot_service)
            #create_copilot_library_qa_tab(state, copilot_paper_qa_service)
            create_copilot_project_proposal_tab(state, copilot_project_proposal_service, trigger=proposal_trigger)
            #create_copilot_business_tab(state, copilot_business_service)
            create_copilot_project_portfolio_tab(state, copilot_project_portfolio_service, trigger=portfolio_trigger)
            #create_collections_tab(state)
            #create_library_tab(state)
    demo.launch()

if __name__ == '__main__':
    main() 