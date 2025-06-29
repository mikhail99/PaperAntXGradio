"""
Enhanced Parrot Services for testing with modern interrupt() pattern.
This provides scenario-based responses for different interrupt types.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.outputs import LLMResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.runnables import Runnable


class ModernParrotChatOllama(BaseLanguageModel, Runnable):
    """Enhanced Parrot LLM that handles interrupt scenarios."""
    
    def __init__(self, scenario: str = "happy_path"):
        super().__init__()
        self.scenario = self._load_scenario(scenario)
        self.call_count = 0
        
    def _load_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Load test scenario configuration."""
        config_path = Path(__file__).parent / "test_scenarios.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                scenarios = json.load(f)
            return scenarios.get(scenario_name, scenarios.get("happy_path", {}))
        return self._get_default_scenario()
        
    def _get_default_scenario(self) -> Dict[str, Any]:
        """Default scenario for testing."""
        return {
            "description": "Default happy path scenario",
            "node_responses": {
                "query_generator_base": {"queries": ["machine learning safety"]},
                "synthesize_literature_review": {
                    "knowledge_gap": "Limited research on ML safety verification",
                    "synthesized_summary": "Current ML safety research focuses on..."
                },
                "formulate_plan": {
                    "content": "Research Plan: Develop verification methods for ML safety..."
                },
                "review_novelty": {
                    "score": 0.8,
                    "justification": "Novel approach to ML safety verification"
                },
                "review_feasibility": {
                    "score": 0.7,
                    "justification": "Technically feasible with existing tools"
                },
                "synthesize_review": {
                    "is_approved": True,
                    "final_summary": "Proposal approved with high scores"
                }
            }
        }
    
    def _call(self, messages: list, stop: Optional[list] = None, 
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        """Mock LLM call that returns scenario-based responses."""
        self.call_count += 1
        
        # Try to determine which node is calling based on message content
        message_text = str(messages[-1].content) if messages else ""
        
        # Look for node-specific patterns in the message
        for node_name, response in self.scenario.get("node_responses", {}).items():
            if any(keyword in message_text.lower() for keyword in self._get_node_keywords(node_name)):
                return json.dumps(response)
                
        # Default response
        return json.dumps({"content": "Default parrot response"})
        
    def _get_node_keywords(self, node_name: str) -> list:
        """Get keywords that identify specific nodes."""
        keywords = {
            "query_generator_base": ["search queries", "query", "topic"],
            "synthesize_literature_review": ["literature", "summaries", "knowledge gap"],
            "formulate_plan": ["research plan", "proposal", "methodology"],
            "review_novelty": ["novelty", "original", "contribution"],
            "review_feasibility": ["feasibility", "technical", "methodology"],
            "synthesize_review": ["feedback", "review", "final decision"]
        }
        return keywords.get(node_name, [])
    
    def with_structured_output(self, schema):
        """Return self for structured output compatibility."""
        return self
        
    def invoke(self, input_data, config=None):
        """Handle direct invoke calls."""
        if isinstance(input_data, (list, tuple)):
            messages = input_data
        elif hasattr(input_data, 'messages'):
            messages = input_data.messages
        else:
            messages = [HumanMessage(content=str(input_data))]
            
        response = self._call(messages)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"content": response}
    
    @property 
    def _llm_type(self) -> str:
        return "modern_parrot_chat_ollama"


class ModernParrotPaperQAService:
    """Enhanced Parrot PaperQA service with scenario-based responses."""
    
    def __init__(self, scenario: str = "happy_path"):
        self.scenario_name = scenario
        self.scenario = self._load_scenario(scenario)
        self.query_count = 0
        
    def _load_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Load test scenario configuration."""
        config_path = Path(__file__).parent / "test_scenarios.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                scenarios = json.load(f)
            return scenarios.get(scenario_name, scenarios.get("happy_path", {}))
        return self._get_default_scenario()
        
    def _get_default_scenario(self) -> Dict[str, Any]:
        """Default scenario for testing."""
        return {
            "description": "Default literature review responses",
            "literature_responses": [
                "Research shows that machine learning safety is an emerging field with focus on verification methods.",
                "Recent studies indicate gaps in formal verification approaches for neural networks.",
                "Literature suggests need for better safety guarantees in AI systems."
            ]
        }
    
    async def query_documents(self, collection_name: str, query: str) -> Dict[str, Any]:
        """Mock document query with scenario-based responses."""
        self.query_count += 1
        
        responses = self.scenario.get("literature_responses", [
            f"Mock literature response for query: {query}"
        ])
        
        # Cycle through responses if multiple queries
        response_index = (self.query_count - 1) % len(responses)
        response_text = responses[response_index]
        
        return {
            "answer_text": response_text,
            "sources": [
                {"title": f"Mock Paper {self.query_count}", "url": "http://example.com"},
            ],
            "query": query,
            "collection": collection_name
        }


def get_modern_parrot_services(scenario: str = "happy_path"):
    """Factory function to create modern parrot services."""
    json_llm = ModernParrotChatOllama(scenario)
    text_llm = ModernParrotChatOllama(scenario)
    paperqa_service = ModernParrotPaperQAService(scenario)
    
    return json_llm, text_llm, paperqa_service


# Create test scenarios configuration file
def create_test_scenarios_file():
    """Create a test scenarios configuration file."""
    scenarios = {
        "happy_path": {
            "description": "All approvals, no modifications",
            "node_responses": {
                "query_generator_base": {"queries": ["machine learning safety verification"]},
                "synthesize_literature_review": {
                    "knowledge_gap": "Limited formal verification methods for ML safety",
                    "synthesized_summary": "Current research focuses on statistical approaches...",
                    "justification": "Gap identified through systematic review",
                    "is_novel": True
                },
                "formulate_plan": {
                    "content": "Research Plan: Develop formal verification framework for ML safety with focus on neural network verification and safety guarantees."
                },
                "review_novelty": {
                    "score": 0.8,
                    "justification": "Novel integration of formal methods with ML safety"
                },
                "review_feasibility": {
                    "score": 0.75,
                    "justification": "Technically feasible with existing verification tools"
                },
                "synthesize_review": {
                    "is_approved": True,
                    "final_summary": "Proposal approved with strong scores on novelty and feasibility"
                }
            },
            "literature_responses": [
                "Formal verification of neural networks has gained attention, with research focusing on robustness properties and safety constraints.",
                "Machine learning safety literature emphasizes the need for verification methods that can provide guarantees about system behavior.",
                "Recent work in ML safety verification shows promise but lacks comprehensive frameworks for practical deployment."
            ]
        },
        "revision_cycle": {
            "description": "Tests the revision loop with initial rejection",
            "node_responses": {
                "query_generator_base": {"queries": ["AI safety in autonomous systems"]},
                "synthesize_literature_review": {
                    "knowledge_gap": "Need better safety frameworks for autonomous AI",
                    "synthesized_summary": "Autonomous AI safety research is fragmented...",
                    "justification": "Multiple gaps identified in current approaches",
                    "is_novel": True
                },
                "formulate_plan": {
                    "content": "Research Plan: Create comprehensive safety framework for autonomous AI systems."
                },
                "review_novelty": {
                    "score": 0.6,
                    "justification": "Somewhat novel but similar work exists"
                },
                "review_feasibility": {
                    "score": 0.5,
                    "justification": "Challenging scope, may need to be narrowed"
                },
                "synthesize_review": {
                    "is_approved": False,
                    "final_summary": "Proposal needs revision - scope too broad, feasibility concerns"
                }
            },
            "literature_responses": [
                "Autonomous AI safety is a broad field with work spanning robotics, decision-making, and human-AI interaction."
            ]
        },
        "user_modifications": {
            "description": "User provides custom queries and feedback",
            "node_responses": {
                "query_generator_base": {"queries": ["custom user query from interrupt"]},
                "synthesize_literature_review": {
                    "knowledge_gap": "User-refined knowledge gap",
                    "synthesized_summary": "Literature review based on user input...",
                    "justification": "User provided specific direction",
                    "is_novel": True
                },
                "formulate_plan": {
                    "content": "Research Plan: User-guided research direction with specific focus areas."
                },
                "review_novelty": {"score": 0.7, "justification": "User input improved novelty"},
                "review_feasibility": {"score": 0.8, "justification": "User guidance improved feasibility"},
                "synthesize_review": {
                    "is_approved": True,
                    "final_summary": "Approved with user improvements"
                }
            },
            "literature_responses": [
                "Literature response incorporating user-provided direction and custom query focus."
            ]
        }
    }
    
    config_path = Path(__file__).parent / "test_scenarios.json"
    with open(config_path, "w") as f:
        json.dump(scenarios, f, indent=2)
    
    print(f"Test scenarios created at: {config_path}")


if __name__ == "__main__":
    # Create test scenarios file
    create_test_scenarios_file()
    
    # Test parrot services
    json_llm, text_llm, paperqa = get_modern_parrot_services("happy_path")
    print("Modern parrot services created successfully!")
    
    # Test a simple call
    response = json_llm.invoke("Generate search queries for machine learning safety")
    print(f"Test response: {response}") 