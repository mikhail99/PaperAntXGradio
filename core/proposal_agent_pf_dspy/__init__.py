"""
PocketFlow-based Proposal Generation Agent (Minimal Implementation)

Simplified research workflow using PocketFlow cookbook patterns.
No complexity, no abstractions - just clean business logic.
"""

# Core state management
from .state import ProposalWorkflowState, KnowledgeGap, Critique, create_shared_state

# All implemented nodes (minimal implementation)  
from .nodes import (
    GenerateQueries,
    QueryDocuments, 
    SynthesizeGap,
    WriteProposal,
    ReviewProposal
)

# DSPy modules (engine-agnostic)
from .dspy_modules import (
    QueryGenerator,
    KnowledgeSynthesizer, 
    ProposalWriter,
    ProposalReviewer
)

# Services
from .parrot import MockPaperQAService, MockLM

# Flow definition (minimal implementation)
from .flow import create_research_flow

# Orchestrator (minimal implementation)  
from .main import create_research_service as create_pocketflow_service

__all__ = [
    # State
    "ProposalWorkflowState",
    "KnowledgeGap", 
    "Critique",
    "create_shared_state",
    
    # Nodes
    "GenerateQueries",
    "QueryDocuments",
    "SynthesizeGap",
    "WriteProposal",
    "ReviewProposal",
    
    # DSPy Modules
    "QueryGenerator",
    "KnowledgeSynthesizer",
    "ProposalWriter", 
    "ProposalReviewer",
    
    # Services
    "MockPaperQAService",
    "MockLM",
    
    # Flow & Orchestrator (minimal implementation)
    "create_research_flow",
    "create_pocketflow_service"  # Updated to point to minimal service
]

# Package metadata
__version__ = "0.4.0"
__author__ = "PocketFlow Simplification Team"
__description__ = "Minimal PocketFlow-based research proposal generation following cookbook patterns"
