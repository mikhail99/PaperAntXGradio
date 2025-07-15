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

def get_context_aware_actions(agent_name: str, base_actions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Adapts action labels and prompts based on the selected agent.
    
    Args:
        agent_name: Name of the currently selected agent
        base_actions: Base list of actions to adapt
        
    Returns:
        List of adapted actions with context-aware labels
    """
    if not agent_name:
        return base_actions
    
    # Agent-specific adaptations
    agent_adaptations = {
        "Generate Research Questions": {
            "Generate Literature Review": {"label": "Find Key Research Papers", "icon": "📚"},
            "Generate Project Ideas": {"label": "Brainstorm Project Concepts", "icon": "💡"},
            "Generate Project Proposal": {"label": "Research Proposal Questions", "icon": "❓"},
            "Generate Project Review": {"label": "Review Criteria Questions", "icon": "✅"},
        },
        "Create Proposal Outline": {
            "Generate Literature Review": {"label": "Structure Literature Analysis", "icon": "📋"},
            "Generate Project Ideas": {"label": "Project Idea Framework", "icon": "🏗️"},
            "Generate Project Proposal": {"label": "Proposal Structure", "icon": "📝"},
            "Generate Project Review": {"label": "Review Framework", "icon": "🔍"},
        },
        "Find Related Work": {
            "Generate Literature Review": {"label": "Systematic Literature Search", "icon": "🔍"},
            "Generate Project Ideas": {"label": "Find Similar Projects", "icon": "🔗"},
            "Generate Project Proposal": {"label": "Literature Gap Analysis", "icon": "📊"},
            "Generate Project Review": {"label": "Find Review Templates", "icon": "📑"},
        },
        "Suggest Methodology": {
            "Generate Literature Review": {"label": "Review Methodology", "icon": "🔬"},
            "Generate Project Ideas": {"label": "Ideation Methods", "icon": "🧠"},
            "Generate Project Proposal": {"label": "Research Methodology", "icon": "⚗️"},
            "Generate Project Review": {"label": "Evaluation Methods", "icon": "📏"},
        }
    }
    
    adapted_actions = []
    for action in base_actions:
        adapted_action = action.copy()
        
        # Check if we have adaptations for this action
        action_label = action['label']
        if action_label in agent_adaptations and agent_name in agent_adaptations[action_label]:
            adaptation = agent_adaptations[action_label][agent_name]
            adapted_action['label'] = adaptation['label']
            adapted_action['icon'] = adaptation['icon']
        
        adapted_actions.append(adapted_action)
    
    return adapted_actions

def generate_agent_specific_prompt(action_label: str, agent_name: str, state_context: Dict = None) -> str:
    """
    Generates context-aware prompts based on action and selected agent.
    
    Args:
        action_label: The action button that was clicked
        agent_name: Currently selected agent
        state_context: Optional state context (collection, article, etc.)
        
    Returns:
        Context-aware prompt string
    """
    context = ""
    if state_context:
        if state_context.get("selected_collection_name"):
            context += f"Working with collection '{state_context['selected_collection_name']}'. "
        if state_context.get("selected_article_id"):
            context += f"Focusing on article '{state_context['selected_article_id']}'. "
    
    # Base prompt templates
    base_prompts = {
        "Generate Research Questions": "Generate 5-7 compelling research questions",
        "Find Key Research Papers": "Help me find the most important research papers",
        "Brainstorm Project Concepts": "Help me brainstorm innovative project ideas",
        "Research Proposal Questions": "Generate research questions for a proposal",
        
        "Create Proposal Outline": "Create a detailed outline",
        "Structure Literature Analysis": "Create a structure for literature analysis",
        "Project Idea Framework": "Create a framework for developing project ideas",
        "Proposal Structure": "Create a research proposal structure",
        
        "Find Related Work": "Help me find related work and literature",
        "Systematic Literature Search": "Conduct a systematic literature search",
        "Find Similar Projects": "Find similar projects and initiatives",
        "Literature Gap Analysis": "Analyze gaps in current literature",
        
        "Suggest Methodology": "Suggest appropriate methodologies",
        "Review Methodology": "Suggest methodologies for literature review",
        "Ideation Methods": "Suggest creative ideation methodologies",
        "Research Methodology": "Suggest research methodologies",
        
        "Create Project Timeline": "Create a realistic project timeline",
        "Budget Estimation": "Help estimate project budget and resources",
        "Identify Risks": "Identify potential risks and mitigation strategies",
        "Review & Improve": "Review and suggest improvements"
    }
    
    # Agent-specific context additions
    agent_context = {
        "Generate Literature Review": " for a comprehensive literature review",
        "Generate Project Ideas": " for innovative project development",
        "Generate Project Proposal": " for a research proposal",
        "Generate Project Review": " for project evaluation and review"
    }
    
    base_prompt = base_prompts.get(action_label, f"Help me with {action_label.lower()}")
    agent_suffix = agent_context.get(agent_name, "")
    
    return f"{context}{base_prompt}{agent_suffix}. Please provide detailed, actionable guidance."

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

def create_quick_action_handlers(action_buttons: List[gr.Button], selected_agent_state: gr.State, state: gr.State = None):
    """
    Creates event handlers for quick action buttons that generate context-aware prompts.
    
    Args:
        action_buttons: List of action buttons from create_quick_actions_toolbar
        selected_agent_state: Gradio state containing the selected agent name
        state: Optional shared state for additional context
        
    Returns:
        Dictionary mapping button indices to their generated prompts
    """
    quick_action_outputs = []
    
    def create_action_click_handler(button_label: str):
        def handle_click():
            # Get current agent name
            current_agent = selected_agent_state.value if hasattr(selected_agent_state, 'value') else None
            
            # Get state context
            state_context = {}
            if state and hasattr(state, 'value'):
                state_context = state.value or {}
            
            # Generate context-aware prompt
            prompt = generate_agent_specific_prompt(button_label, current_agent, state_context)
            return prompt
        
        return handle_click
    
    # Create handlers for each button
    for i, button in enumerate(action_buttons):
        # Extract label from button (remove icon)
        button_label = button.value.split(' ', 1)[-1] if hasattr(button, 'value') else f"Action {i+1}"
        handler = create_action_click_handler(button_label)
        quick_action_outputs.append(handler)
    
    return quick_action_outputs

def connect_quick_actions_to_chat(action_buttons: List[gr.Button], base_actions: List[Dict[str, str]], selected_agent_state: gr.State, state: gr.State = None):
    """
    Connects quick action buttons to generate prompts that can be used with chat interface.
    
    Args:
        action_buttons: List of quick action buttons
        base_actions: Original action definitions
        selected_agent_state: State containing selected agent name
        state: Optional shared state
        
    Returns:
        List of outputs that can be connected to chat inputs
    """
    action_outputs = []
    
    for i, (button, action) in enumerate(zip(action_buttons, base_actions)):
        def create_click_handler(action_label: str):
            def on_click(agent_name_state_value, shared_state_value):
                # Handle potential state access issues
                try:
                    agent_name = agent_name_state_value
                    state_context = shared_state_value or {}
                    
                    # Generate the appropriate prompt
                    prompt = generate_agent_specific_prompt(action_label, agent_name, state_context)
                    print(f"Quick action triggered: {action_label} -> {prompt}")
                    return prompt
                    
                except Exception as e:
                    print(f"Error in quick action handler: {e}")
                    return f"Help me with {action_label.lower()}"
            
            return on_click
        
        action_label = action['label']
        handler = create_click_handler(action_label)
        
        # Create a textbox to capture the output
        output_textbox = gr.Textbox(visible=False, label=f"action_output_{i}")
        
        # Connect button click to handler
        button.click(
            fn=handler,
            inputs=[selected_agent_state, state] if state else [selected_agent_state],
            outputs=[output_textbox]
        )
        
        action_outputs.append(output_textbox)
    
    return action_outputs 

def create_dynamic_quick_actions_toolbar(
    base_actions: List[Dict[str, str]], 
    title: str = "🚀 Quick Actions",
    show_title: bool = True,
    actions_per_row: int = 4
) -> tuple:
    """
    Creates a dynamic quick actions toolbar that updates its button labels based on agent selection.
    
    Args:
        base_actions: Base list of actions to adapt
        title: Title for the toolbar section
        show_title: Whether to show the title
        actions_per_row: Number of action buttons per row
        
    Returns:
        tuple: (action_buttons_list, css_html_component, update_function)
    """
    
    # Create the toolbar infrastructure
    action_buttons, css_component = create_quick_actions_toolbar(
        actions=base_actions,
        title=title,
        show_title=show_title,
        actions_per_row=actions_per_row
    )
    
    def update_buttons_for_agent(agent_name: str):
        """
        Returns updated button configurations for the specified agent.
        
        Args:
            agent_name: Name of the selected agent
            
        Returns:
            List of gr.Button.update() objects with new labels
        """
        if not agent_name:
            return [gr.Button.update() for _ in action_buttons]
        
        # Get context-aware actions for this agent
        adapted_actions = get_context_aware_actions(agent_name, base_actions)
        
        # Create button updates
        button_updates = []
        for i, (button, adapted_action) in enumerate(zip(action_buttons, adapted_actions)):
            new_label = f"{adapted_action.get('icon', '🔧')} {adapted_action['label']}"
            button_updates.append(gr.Button.update(value=new_label))
        
        return button_updates
    
    return action_buttons, css_component, update_buttons_for_agent 