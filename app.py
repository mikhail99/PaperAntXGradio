import gradio as gr
from state.state import get_shared_state
from ui.ui_collections import create_collections_tab
from ui.ui_articles import create_articles_tab
from ui.ui_copilot import create_copilot_tab
from ui.ui_paperqa import create_paperqa_tab
from ui.ui_mindmap import create_mindmap_tab
from ui.custom_css import CUSTOM_CSS

def main():
    with gr.Blocks(css=CUSTOM_CSS) as demo:
        gr.Markdown("# PaperAnt X")
        state = get_shared_state()
        with gr.Tabs():
            create_collections_tab(state)
            create_articles_tab(state)
            create_copilot_tab(state)
            create_paperqa_tab(state)
            create_mindmap_tab(state)
    demo.launch()

if __name__ == '__main__':
    main() 