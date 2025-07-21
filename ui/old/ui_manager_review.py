import gradio as gr
from core.proposal_agent_pf_dspy.storage import ProposalStorage
import json
import traceback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def create_patent_visualizations(patent_data):
    """Create Plotly visualizations for patent landscape data."""
    if not patent_data or patent_data.get("total_patents", 0) == 0:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No patent data available for visualization<br>Please submit a proposal for analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#666"),
            align="center"
        )
        fig.update_layout(
            height=400,
            template="plotly_white"
        )
        return fig
    
    # Create subplots with a more robust layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Patents by Company", "Filing Trends", "Technology Classifications", "Key Metrics"),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "indicator"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Patents by Company (Bar chart)
    companies = list(patent_data.get("companies", {}).keys())[:8]  # Top 8
    company_counts = [patent_data["companies"][c] for c in companies]
    
    fig.add_trace(
        go.Bar(
            x=companies, 
            y=company_counts, 
            name="Patents",
            marker_color="rgb(55, 83, 109)",
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Filing Trends (Line chart)
    years = list(patent_data.get("year_trends", {}).keys())
    year_counts = list(patent_data.get("year_trends", {}).values())
    
    fig.add_trace(
        go.Scatter(
            x=years, 
            y=year_counts, 
            mode='lines+markers',
            name="Filings",
            line=dict(color="rgb(26, 118, 255)", width=2),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Technology Classifications
    classifications = list(patent_data.get("top_classifications", {}).keys())[:5]
    class_counts = [patent_data["top_classifications"][c] for c in classifications]
    
    fig.add_trace(
        go.Bar(
            x=classifications,
            y=class_counts,
            name="Classifications",
            marker_color="rgb(158, 202, 225)",
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. Key Metrics (Indicator)
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=patent_data.get("total_patents", 0),
            title={"text": "Total Patents<br>in Analysis"},
            delta={"reference": patent_data.get("recent_activity", 0), "relative": True},
            number={"font": {"size": 40}},
        ),
        row=2, col=2
    )
    
    # Update layout with better styling
    fig.update_layout(
        height=800,  # Increased height
        showlegend=False,
        title={
            "text": "Patent Landscape Analysis",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 24}
        },
        template="plotly_white",
        margin=dict(t=120, l=50, r=50, b=50)
    )
    
    # Make x-axis labels more readable
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
    
    # Add gridlines and improve readability
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128, 128, 128, 0.2)")
    
    return fig

def create_manager_review_tab():
    with gr.TabItem("💼 Manager Review"):
        gr.Markdown("## Project Proposal Review with Patent Intelligence")
        gr.Markdown("Load a saved proposal to get comprehensive business analysis including patent landscape insights.")

        # Initialize variables
        storage = None
        business_bridge_service = None
        proposal_choices = []
        service_error = None

        # Try to instantiate services
        try:
            storage = ProposalStorage()
        except Exception as e:
            service_error = f"Storage service error: {e}"
            print(f"Storage service error: {e}")
            traceback.print_exc()

        try:
            from core.copilots.project_buiseness.business_bridge import create_business_bridge_service
            business_bridge_service = create_business_bridge_service()
        except Exception as e:
            service_error = f"Business bridge service error: {e}"
            print(f"Business bridge service error: {e}")
            traceback.print_exc()

        # Try to load proposals
        if storage:
            try:
                proposals = storage.list_saved_proposals()
                proposal_choices = [(f"{p['topic']} ({p.get('collection_name', 'N/A')})", p['file_path']) for p in proposals]
            except Exception as e:
                print(f"Could not load proposals: {e}")
                traceback.print_exc()

        # Show error message if services failed
        if service_error:
            gr.Markdown(f"⚠️ **Service Error:** {service_error}")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Load Saved Proposal")
                proposal_dropdown = gr.Dropdown(
                    choices=proposal_choices,
                    label="Select a Saved Proposal to Load",
                    value=None,
                )
                
                gr.Markdown("### 2. Proposal Content")
                proposal_input = gr.Textbox(
                    lines=15,
                    label="Project Proposal Content",
                    placeholder="Select a proposal from the dropdown or paste the content here..."
                )
                review_button = gr.Button("🚀 Review Proposal with Patent Analysis", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 📝 Business Review")
                review_output = gr.Markdown("*(Waiting for review...)*")
                
                gr.Markdown("### 📊 Patent Landscape")
                # Use gr.Plot directly instead of HTML
                patent_viz = gr.Plot(label="Patent Analysis", show_label=True)
                
                with gr.Accordion("🔍 Patent Search Details", open=False):
                    patent_queries_display = gr.Markdown("*(No queries generated yet)*")
                    patent_raw_data = gr.JSON(label="Raw Patent Data", visible=False)

        def load_proposal_text(file_path):
            if not file_path or not storage:
                return ""
            try:
                data = storage.load_proposal(file_path)
                
                # Nicely format the loaded JSON into a string for the agent
                parts = []
                parts.append(f"# Project Proposal: {data.get('topic', 'N/A')}")
                parts.append(f"**Collection:** {data.get('collection_name', 'N/A')}")
                
                parts.append("\n## Literature Summaries")
                for i, summary in enumerate(data.get('literature_summaries', [])):
                    parts.append(f"### Summary {i+1}\n{summary}")
                
                parts.append("\n## Research Plan")
                plans = data.get('research_plan', [])
                if isinstance(plans, list):
                    for i, plan in enumerate(plans):
                        parts.append(f"### Plan Option {i+1}\n{plan}")
                elif isinstance(plans, str):
                    parts.append(plans)

                parts.append("\n## Novelty Assessment")
                for i, assessment in enumerate(data.get('novelty_assessment', [])):
                    parts.append(f"### Assessment {i+1}")
                    parts.append(f"**Is Novel:** {assessment.get('is_novel')}")
                    parts.append(f"**Justification:** {assessment.get('justification')}")

                return "\n\n".join(parts)
            except Exception as e:
                print(f"Failed to load or parse proposal from {file_path}: {e}")
                traceback.print_exc()
                return f"Error loading file: {e}"

        def run_review(proposal_text):
            if not proposal_text:
                return "*(Please load or enter a proposal to review)*", None, "*(No queries generated yet)*", {}
            
            if not business_bridge_service:
                return "**Error:** Business bridge service not available. Please check the configuration.", None, "*(Service unavailable)*", {}
            
            try:
                result = business_bridge_service.forward(project_proposal=proposal_text)
                
                # Format patent queries for display
                queries_text = "**Generated Patent Search Queries:**\n" + "\n".join([f"• {q}" for q in result.patent_queries])
                
                # Create patent visualizations using Plotly
                fig = create_patent_visualizations(result.patent_data)
                
                return result.review, fig, queries_text, result.patent_data
                
            except Exception as e:
                print(f"Review error: {e}")
                traceback.print_exc()
                return f"**Error:** {e}", None, "*(Error occurred)*", {}

        proposal_dropdown.change(
            load_proposal_text,
            inputs=[proposal_dropdown],
            outputs=[proposal_input]
        )

        review_button.click(
            run_review,
            inputs=[proposal_input],
            outputs=[review_output, patent_viz, patent_queries_display, patent_raw_data]
        ) 