"""
Modern Graph Builder using LangGraph's interrupt() pattern.
This completely replaces the old interrupt_after approach.
"""

import json
from pathlib import Path
from typing import Dict, Any

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from core.proposal_agent.state import ProposalAgentState
from core.proposal_agent.hil_nodes import (
    human_query_review_node,
    human_insight_review_node, 
    human_review_node,
    workflow_complete_node
)
from core.proposal_agent.graph import (
    create_llm_node,
    deduplicate_queries_node,
    is_proposal_approved
)


class ModernWorkflowGraphBuilder:
    """Builds graphs using modern interrupt() pattern."""
    
    def __init__(self, workflow_name: str = "research_proposal_workflow"):
        self.config_path = Path(__file__).parent
        self.workflow_config = self._load_workflow_config(workflow_name)
        self.teams_config = self._load_teams_config()
        self.nodes_config = self._load_nodes_config()
        self.hil_config = self._load_hil_config()
        
    def _load_workflow_config(self, workflow_name: str) -> Dict[str, Any]:
        """Load workflow configuration."""
        with open(self.config_path / "team_workflows.json", "r") as f:
            workflows = json.load(f)
        return workflows[workflow_name]
        
    def _load_teams_config(self) -> Dict[str, Any]:
        """Load teams configuration (using existing agent_config for now)."""
        with open(self.config_path / "agent_config.json", "r") as f:
            config = json.load(f)
        return config["teams"]
        
    def _load_nodes_config(self) -> Dict[str, Any]:
        """Load nodes configuration (using existing agent_config for now)."""
        with open(self.config_path / "agent_config.json", "r") as f:
            config = json.load(f)
        return config["nodes"]
        
    def _load_hil_config(self) -> Dict[str, Any]:
        """Load HIL configuration."""
        with open(self.config_path / "hil_configurations.json", "r") as f:
            return json.load(f)
    
    def build_graph(self) -> StateGraph:
        """Builds graph using modern interrupt() pattern."""
        builder = StateGraph(ProposalAgentState)
        
        # Add all standard nodes from config
        for node_name, node_config in self.nodes_config.items():
            node_func = create_llm_node(node_name, node_config)
            builder.add_node(node_name, node_func)
            
        # Add utility nodes
        builder.add_node("deduplicate_queries_node", deduplicate_queries_node)
        builder.add_node("workflow_complete_node", workflow_complete_node)
        
        # Add modern HIL nodes using interrupt() pattern
        builder.add_node("human_query_review_node", human_query_review_node)
        builder.add_node("human_insight_review_node", human_insight_review_node)
        builder.add_node("human_review_node", human_review_node)
        
        # Build workflow stages
        self._connect_workflow_stages(builder)
        
        # Clean compilation - LangGraph handles interrupts automatically
        return builder.compile(checkpointer=MemorySaver())
        
    def _connect_workflow_stages(self, builder: StateGraph):
        """Connect workflow stages with modern HIL pattern."""
        
        # Stage 1: Query Generation
        query_stage = next(s for s in self.workflow_config["stages"] if s["id"] == "query_generation")
        team_config = self.teams_config[query_stage["team"]]
        
        # Entry Point -> Query Generation Team
        for member in team_config["members"]:
            builder.add_edge(START, member)
        
        # Query Team Members -> Aggregator
        for member in team_config["members"]:
            builder.add_edge(member, team_config["aggregator"])
            
        # FIXED: Don't add self-edge for utility nodes like deduplicate_queries_node
        # Only add aggregator -> next step if aggregator is different from what we're connecting to
        aggregator = team_config["aggregator"]
        if aggregator == "deduplicate_queries_node":
            # For deduplicate_queries_node, it goes directly to HIL
            if query_stage["hil_node"]:
                builder.add_edge("deduplicate_queries_node", query_stage["hil_node"])
                last_node = query_stage["hil_node"]
            else:
                last_node = "deduplicate_queries_node"
        else:
            # For other aggregators, add the normal flow
            builder.add_edge(aggregator, "deduplicate_queries_node")
            if query_stage["hil_node"]:
                builder.add_edge("deduplicate_queries_node", query_stage["hil_node"])
                last_node = query_stage["hil_node"]
            else:
                last_node = "deduplicate_queries_node"
        
        # Stage 2: Literature Review
        lit_stage = next(s for s in self.workflow_config["stages"] if s["id"] == "literature_review")
        lit_team_config = self.teams_config[lit_stage["team"]]
        
        # HIL -> Literature Review Team
        for member in lit_team_config["members"]:
            builder.add_edge(last_node, member)
            
        # Literature Team Members -> Aggregator
        for member in lit_team_config["members"]:
            builder.add_edge(member, lit_team_config["aggregator"])
            
        # Literature Aggregator -> HIL checkpoint
        if lit_stage["hil_node"]:
            builder.add_edge(lit_team_config["aggregator"], lit_stage["hil_node"])
            last_node = lit_stage["hil_node"]
        else:
            last_node = lit_team_config["aggregator"]
            
        # Stage 3: Proposal Creation
        proposal_stage = next(s for s in self.workflow_config["stages"] if s["id"] == "proposal_creation")
        proposal_team_config = self.teams_config[proposal_stage["team"]]
        
        # HIL -> Proposal Creation
        for member in proposal_team_config["members"]:
            builder.add_edge(last_node, member)
            
        # Use the first member as the proposal creator (formulate_plan)
        proposal_creator = proposal_team_config["members"][0]
        
        # Stage 4: Proposal Review
        review_stage = next(s for s in self.workflow_config["stages"] if s["id"] == "proposal_review")
        review_team_config = self.teams_config[review_stage["team"]]
        
        # Proposal Creator -> Review Team
        for member in review_team_config["members"]:
            builder.add_edge(proposal_creator, member)
            
        # Review Team Members -> Aggregator
        for member in review_team_config["members"]:
            builder.add_edge(member, review_team_config["aggregator"])
            
        # FIXED: Always go to human review first, then apply approval logic
        if review_stage["hil_node"]:
            # Review Aggregator -> Human Review HIL (always)
            builder.add_edge(review_team_config["aggregator"], review_stage["hil_node"])
            
            # Human Review HIL -> Approval Decision
            builder.add_conditional_edges(
                review_stage["hil_node"],
                is_proposal_approved,
                {
                    "approved": "workflow_complete_node",
                    "max_revisions_reached": "workflow_complete_node",
                    "revise": proposal_creator
                }
            )
        else:
            # Fallback: if no HIL, go directly to approval logic (shouldn't happen with current config)
            builder.add_conditional_edges(
                review_team_config["aggregator"],
                is_proposal_approved,
                {
                    "approved": "workflow_complete_node",
                    "max_revisions_reached": "workflow_complete_node",
                    "revise": proposal_creator
                }
            )
        
        # Workflow complete -> END
        builder.add_edge("workflow_complete_node", END)


def build_modern_graph(workflow_name: str = "research_proposal_workflow") -> StateGraph:
    """Factory function to build a modern graph with interrupt() pattern."""
    builder = ModernWorkflowGraphBuilder(workflow_name)
    return builder.build_graph()


# For testing
if __name__ == "__main__":
    graph = build_modern_graph()
    print("Modern graph built successfully!")
    print(f"Graph nodes: {list(graph.nodes.keys())}") 