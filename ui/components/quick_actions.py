import gradio as gr
from typing import List, Dict, Callable, Optional

def create_quick_actions_toolbar(
    actions: List[Dict[str, str]], 
    title: str = "🚀 Quick Actions",
    show_title: bool = True,
    actions_per_row: int = 4
) -> tuple:
    """
    Creates a beautiful quick actions toolbar with gradient buttons.
    
    Args:
        actions: List of action dictionaries with 'label', 'icon', 'variant', 'color_class'
        title: Title for the toolbar section
        show_title: Whether to show the title
        actions_per_row: Number of action buttons per row
        
    Returns:
        tuple: (action_buttons_list, css_html_component)
        
    Example:
        actions = [
            {"label": "Generate Research Questions", "icon": "💡", "variant": "primary", "color_class": "research-btn"},
            {"label": "Create Proposal Outline", "icon": "📋", "variant": "secondary", "color_class": "outline-btn"},
        ]
        buttons, css = create_quick_actions_toolbar(actions)
    """
    
    # Generate CSS for beautiful styling
    css_styles = """
    <style>
    .quick-actions-toolbar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .toolbar-title {
        color: white !important;
        margin: 0 !important;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .quick-actions-row {
        gap: 0.5rem !important;
        margin-top: 0.5rem;
    }
    .quick-action-btn {
        transition: all 0.3s ease !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }
    .quick-action-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }
    .research-btn {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24) !important;
        border: none !important;
    }
    .outline-btn {
        background: linear-gradient(135deg, #4834d4, #686de0) !important;
        border: none !important;
    }
    .search-btn {
        background: linear-gradient(135deg, #00d2d3, #54a0ff) !important;
        border: none !important;
    }
    .method-btn {
        background: linear-gradient(135deg, #5f27cd, #a55eea) !important;
        border: none !important;
    }
    .timeline-btn {
        background: linear-gradient(135deg, #00d2d3, #01a3a4) !important;
        border: none !important;
    }
    .budget-btn {
        background: linear-gradient(135deg, #feca57, #ff9ff3) !important;
        border: none !important;
    }
    .risk-btn {
        background: linear-gradient(135deg, #ff9f43, #ff6348) !important;
        border: none !important;
    }
    .review-btn {
        background: linear-gradient(135deg, #26de81, #20bf6b) !important;
        border: none !important;
    }
    .data-btn {
        background: linear-gradient(135deg, #341f97, #5f27cd) !important;
        border: none !important;
    }
    .analysis-btn {
        background: linear-gradient(135deg, #2d98da, #006ba6) !important;
        border: none !important;
    }
    .export-btn {
        background: linear-gradient(135deg, #8854d0, #3742fa) !important;
        border: none !important;
    }
    .help-btn {
        background: linear-gradient(135deg, #a55eea, #fd79a8) !important;
        border: none !important;
    }
    </style>
    """
    
    # Create the toolbar components
    components = []
    
    # Title section
    if show_title:
        with gr.Row(elem_classes=["quick-actions-toolbar"], equal_height=True):
            title_component = gr.Markdown(f"### {title}", elem_classes=["toolbar-title"])
            components.append(title_component)
    
    # Create action buttons in rows
    action_buttons = []
    for i in range(0, len(actions), actions_per_row):
        row_actions = actions[i:i + actions_per_row]
        
        with gr.Row(elem_classes=["quick-actions-row"], equal_height=True):
            row_buttons = []
            for action in row_actions:
                button = gr.Button(
                    f"{action.get('icon', '🔧')} {action['label']}",
                    variant=action.get('variant', 'secondary'),
                    size="sm",
                    elem_classes=["quick-action-btn", action.get('color_class', 'help-btn')]
                )
                row_buttons.append(button)
                action_buttons.append(button)
            components.extend(row_buttons)
    
    # CSS component
    css_component = gr.HTML(css_styles)
    components.append(css_component)
    
    return action_buttons, css_component

def get_research_proposal_actions() -> List[Dict[str, str]]:
    """Predefined actions for research proposal workflows."""
    return [
        {"label": "Generate Research Questions", "icon": "💡", "variant": "primary", "color_class": "research-btn"},
        {"label": "Create Proposal Outline", "icon": "📋", "variant": "secondary", "color_class": "outline-btn"},
        {"label": "Find Related Work", "icon": "🔍", "variant": "secondary", "color_class": "search-btn"},
        {"label": "Suggest Methodology", "icon": "🔬", "variant": "secondary", "color_class": "method-btn"},
        {"label": "Create Project Timeline", "icon": "📅", "variant": "secondary", "color_class": "timeline-btn"},
        {"label": "Budget Estimation", "icon": "💰", "variant": "secondary", "color_class": "budget-btn"},
        {"label": "Identify Risks", "icon": "⚠️", "variant": "secondary", "color_class": "risk-btn"},
        {"label": "Review & Improve", "icon": "✅", "variant": "secondary", "color_class": "review-btn"},
    ]

def get_business_analysis_actions() -> List[Dict[str, str]]:
    """Predefined actions for business analysis workflows."""
    return [
        {"label": "Market Analysis", "icon": "📊", "variant": "primary", "color_class": "analysis-btn"},
        {"label": "Financial Forecast", "icon": "💹", "variant": "secondary", "color_class": "budget-btn"},
        {"label": "Competitor Research", "icon": "🔍", "variant": "secondary", "color_class": "search-btn"},
        {"label": "SWOT Analysis", "icon": "⚖️", "variant": "secondary", "color_class": "method-btn"},
    ]

def get_data_science_actions() -> List[Dict[str, str]]:
    """Predefined actions for data science workflows."""
    return [
        {"label": "Data Exploration", "icon": "🔍", "variant": "primary", "color_class": "search-btn"},
        {"label": "Feature Engineering", "icon": "🛠️", "variant": "secondary", "color_class": "method-btn"},
        {"label": "Model Training", "icon": "🤖", "variant": "secondary", "color_class": "analysis-btn"},
        {"label": "Results Export", "icon": "📤", "variant": "secondary", "color_class": "export-btn"},
    ]

def create_action_handler(action_prompts: Dict[str, str], state_accessor: Optional[Callable] = None):
    """
    Creates a handler function for quick actions that generates context-aware prompts.
    
    Args:
        action_prompts: Dictionary mapping action labels to prompt templates
        state_accessor: Optional function to access shared state for context
        
    Returns:
        Callable: Handler function that takes action_type and returns a prompt
    """
    def handle_action(action_type: str) -> str:
        context = ""
        if state_accessor:
            try:
                state = state_accessor()
                if state.get("selected_collection_name"):
                    context += f"Working with collection: {state['selected_collection_name']}. "
                if state.get("selected_article_id"):
                    context += f"Focusing on article: {state['selected_article_id']}. "
            except Exception as e:
                print(f"Warning: Could not access state for context: {e}")
        
        prompt_template = action_prompts.get(action_type, "How can I help you?")
        return f"{context}{prompt_template}"
    
    return handle_action 