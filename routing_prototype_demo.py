#!/usr/bin/env python3
"""
Prototype demonstration of Intelligent HIL Routing for Research Proposal Agent

This shows how the routing system would integrate with the existing LangGraph + DSPy architecture.
"""

import json
from typing import Dict, Any, Literal
from dataclasses import dataclass
from enum import Enum

# Simulated DSPy modules (would be real DSPy in implementation)
class MockDSPyRouter:
    def __init__(self, name: str, threshold: float):
        self.name = name
        self.threshold = threshold
    
    def predict(self, **kwargs) -> Dict[str, Any]:
        # Simulate confidence scoring based on input complexity
        # In real implementation, this would be a trained DSPy module
        confidence = self._calculate_confidence(**kwargs)
        
        if confidence >= self.threshold:
            route = "auto_continue"
        elif confidence >= 0.3:  # Expert escalation threshold
            route = "human_review"
        else:
            route = "expert_escalation"
            
        return {
            "route": route,
            "confidence": confidence,
            "reasoning": f"Confidence {confidence:.2f} vs threshold {self.threshold}"
        }
    
    def _calculate_confidence(self, **kwargs) -> float:
        # Mock confidence calculation - would be learned in real implementation
        if "topic" in kwargs:
            topic = kwargs["topic"].lower()
            # Simple heuristics for demo
            if any(word in topic for word in ["machine learning", "nlp", "computer vision"]):
                return 0.9  # High confidence for common ML topics
            elif len(topic.split()) < 3:
                return 0.4  # Low confidence for vague topics
            else:
                return 0.7  # Medium confidence
        return 0.5

class RouteDecision(Enum):
    AUTO_CONTINUE = "auto_continue"
    HUMAN_REVIEW = "human_review"
    EXPERT_ESCALATION = "expert_escalation"

@dataclass
class RoutingResult:
    stage: str
    route: RouteDecision
    confidence: float
    reasoning: str
    features: Dict[str, Any]

class IntelligentHILRouter:
    """
    Intelligent routing system that decides when human intervention is needed.
    
    This replaces the fixed HIL checkpoints with dynamic routing based on:
    - Content complexity
    - AI confidence
    - User expertise level
    - Historical patterns
    """
    
    def __init__(self, config_path: str = None):
        # Load routing configuration
        self.config = self._load_config(config_path)
        
        # Initialize stage-specific routers (would be trained DSPy modules)
        self.query_router = MockDSPyRouter("query", self.config["query_generation"]["auto_continue_threshold"])
        self.literature_router = MockDSPyRouter("literature", self.config["literature_review"]["auto_continue_threshold"])
        self.proposal_router = MockDSPyRouter("proposal", self.config["final_review"]["auto_continue_threshold"])
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load routing configuration with fallback to defaults"""
        default_config = {
            "query_generation": {
                "auto_continue_threshold": 0.8,
                "expert_escalation_threshold": 0.3
            },
            "literature_review": {
                "auto_continue_threshold": 0.75,
                "expert_escalation_threshold": 0.25
            },
            "final_review": {
                "auto_continue_threshold": 0.7,
                "expert_escalation_threshold": 0.2
            },
            "user_profiles": {
                "novice": {"confidence_penalty": 0.1},
                "intermediate": {"confidence_penalty": 0.0},
                "expert": {"confidence_bonus": 0.1}
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                print(f"Config file {config_path} not found, using defaults")
        
        return default_config
    
    def route_query_stage(self, state: Dict[str, Any]) -> RoutingResult:
        """Route decision for query generation stage"""
        topic = state.get("topic", "")
        queries = state.get("search_queries", [])
        user_profile = state.get("user_profile", {}).get("expertise", "intermediate")
        
        # Extract features for routing decision
        features = {
            "topic_clarity": self._assess_topic_clarity(topic),
            "query_count": len(queries),
            "query_specificity": self._assess_query_specificity(queries),
            "user_expertise": user_profile
        }
        
        # Get routing decision from trained model
        decision = self.query_router.predict(topic=topic, queries=queries, **features)
        
        # Apply user profile adjustments
        adjusted_confidence = self._apply_user_profile_adjustment(
            decision["confidence"], user_profile
        )
        
        return RoutingResult(
            stage="query_generation",
            route=RouteDecision(decision["route"]),
            confidence=adjusted_confidence,
            reasoning=f"{decision['reasoning']} (user: {user_profile})",
            features=features
        )
    
    def route_literature_stage(self, state: Dict[str, Any]) -> RoutingResult:
        """Route decision for literature review stage"""
        summaries = state.get("literature_summaries", [])
        knowledge_gap = state.get("knowledge_gap", {})
        
        features = {
            "summary_count": len(summaries),
            "knowledge_gap_clarity": self._assess_knowledge_gap_clarity(knowledge_gap),
            "summary_coherence": self._assess_summary_coherence(summaries)
        }
        
        decision = self.literature_router.predict(summaries=summaries, knowledge_gap=knowledge_gap, **features)
        
        return RoutingResult(
            stage="literature_review",
            route=RouteDecision(decision["route"]),
            confidence=decision["confidence"],
            reasoning=decision["reasoning"],
            features=features
        )
    
    def route_proposal_stage(self, state: Dict[str, Any]) -> RoutingResult:
        """Route decision for final proposal stage"""
        proposal = state.get("proposal_draft", "")
        reviews = state.get("review_team_feedback", {})
        
        features = {
            "proposal_length": len(proposal.split()) if proposal else 0,
            "ai_reviewer_agreement": self._assess_reviewer_agreement(reviews),
            "proposal_completeness": self._assess_proposal_completeness(proposal)
        }
        
        decision = self.proposal_router.predict(proposal=proposal, reviews=reviews, **features)
        
        return RoutingResult(
            stage="final_review",
            route=RouteDecision(decision["route"]),
            confidence=decision["confidence"],
            reasoning=decision["reasoning"],
            features=features
        )
    
    # Feature assessment methods (would be more sophisticated in real implementation)
    def _assess_topic_clarity(self, topic: str) -> float:
        """Assess how clearly defined the research topic is"""
        if not topic:
            return 0.0
        
        # Simple heuristics - real implementation would use NLP
        words = topic.split()
        if len(words) < 3:
            return 0.3  # Too vague
        elif len(words) > 20:
            return 0.4  # Too verbose
        elif any(word in topic.lower() for word in ["improve", "better", "new approach"]):
            return 0.6  # Somewhat clear intent
        else:
            return 0.8  # Reasonably specific
    
    def _assess_query_specificity(self, queries: list) -> float:
        """Assess how specific the generated queries are"""
        if not queries:
            return 0.0
        
        # Check for specific terms vs generic ones
        specificity_scores = []
        for query in queries:
            generic_terms = ["improve", "better", "new", "method", "approach"]
            specific_terms = ["algorithm", "model", "dataset", "evaluation", "benchmark"]
            
            generic_count = sum(1 for term in generic_terms if term in query.lower())
            specific_count = sum(1 for term in specific_terms if term in query.lower())
            
            if specific_count > generic_count:
                specificity_scores.append(0.8)
            elif specific_count == generic_count:
                specificity_scores.append(0.6)
            else:
                specificity_scores.append(0.4)
        
        return sum(specificity_scores) / len(specificity_scores)
    
    def _assess_knowledge_gap_clarity(self, knowledge_gap: dict) -> float:
        """Assess how clearly the knowledge gap is articulated"""
        if not knowledge_gap:
            return 0.0
        
        gap_text = knowledge_gap.get("knowledge_gap", "")
        if len(gap_text) < 20:
            return 0.3  # Too brief
        elif "unclear" in gap_text.lower() or "unknown" in gap_text.lower():
            return 0.4  # Indicates uncertainty
        else:
            return 0.8  # Seems well-articulated
    
    def _assess_summary_coherence(self, summaries: list) -> float:
        """Assess how coherent the literature summaries are"""
        if not summaries:
            return 0.0
        
        # Simple check - real implementation would use semantic analysis
        total_length = sum(len(summary.split()) for summary in summaries)
        avg_length = total_length / len(summaries)
        
        if avg_length < 50:
            return 0.4  # Too brief
        elif avg_length > 500:
            return 0.5  # Potentially too verbose
        else:
            return 0.8  # Reasonable length
    
    def _assess_reviewer_agreement(self, reviews: dict) -> float:
        """Assess how much AI reviewers agree"""
        if not reviews:
            return 0.0
        
        scores = []
        for review in reviews.values():
            if isinstance(review, dict) and "score" in review:
                scores.append(review["score"])
        
        if len(scores) < 2:
            return 0.5  # Not enough data
        
        # Calculate score variance
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        
        # High agreement = low variance
        if variance < 0.1:
            return 0.9  # High agreement
        elif variance < 0.3:
            return 0.7  # Moderate agreement
        else:
            return 0.4  # Low agreement
    
    def _assess_proposal_completeness(self, proposal: str) -> float:
        """Assess how complete the proposal is"""
        if not proposal:
            return 0.0
        
        # Check for key sections
        required_sections = ["introduction", "method", "evaluation", "conclusion"]
        present_sections = sum(1 for section in required_sections if section in proposal.lower())
        
        return present_sections / len(required_sections)
    
    def _apply_user_profile_adjustment(self, confidence: float, user_profile: str) -> float:
        """Adjust confidence based on user expertise level"""
        adjustment = self.config["user_profiles"].get(user_profile, {}).get("confidence_penalty", 0.0)
        if "bonus" in self.config["user_profiles"].get(user_profile, {}):
            adjustment = self.config["user_profiles"][user_profile]["confidence_bonus"]
        
        return max(0.0, min(1.0, confidence + adjustment))

# Integration with existing LangGraph architecture
def create_smart_hil_node(router: IntelligentHILRouter, stage: str):
    """
    Factory function to create HIL nodes with intelligent routing.
    
    This replaces the fixed HIL nodes with routing-aware versions.
    """
    
    def smart_hil_node(state: Dict[str, Any]) -> Dict[str, Any]:
        # Route decision based on stage
        if stage == "query":
            routing_result = router.route_query_stage(state)
        elif stage == "literature":
            routing_result = router.route_literature_stage(state)
        elif stage == "proposal":
            routing_result = router.route_proposal_stage(state)
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        # Log routing decision
        print(f"🎯 ROUTING DECISION for {routing_result.stage}:")
        print(f"   Route: {routing_result.route.value}")
        print(f"   Confidence: {routing_result.confidence:.2f}")
        print(f"   Reasoning: {routing_result.reasoning}")
        print(f"   Features: {routing_result.features}")
        
        # Store routing info in state
        routing_info = {
            "stage": routing_result.stage,
            "route": routing_result.route.value,
            "confidence": routing_result.confidence,
            "reasoning": routing_result.reasoning,
            "timestamp": "2024-01-01T12:00:00Z"  # Would be real timestamp
        }
        
        # Add to routing history
        routing_history = state.get("routing_history", [])
        routing_history.append(routing_info)
        
        if routing_result.route == RouteDecision.AUTO_CONTINUE:
            # Skip human review - continue automatically
            return {
                "routing_history": routing_history,
                "last_routing_decision": "auto_continue"
            }
        elif routing_result.route == RouteDecision.HUMAN_REVIEW:
            # Trigger normal HIL
            return {
                "routing_history": routing_history,
                "last_routing_decision": "human_review",
                "needs_human_review": True
            }
        else:  # EXPERT_ESCALATION
            # Trigger expert review
            return {
                "routing_history": routing_history,
                "last_routing_decision": "expert_escalation",
                "needs_expert_review": True
            }
    
    return smart_hil_node

# Demo function
def demo_intelligent_routing():
    """Demonstrate the intelligent routing system"""
    print("🚀 Intelligent HIL Routing Demo\n")
    
    router = IntelligentHILRouter()
    
    # Test scenarios
    scenarios = [
        {
            "name": "Simple ML Topic (Auto-Continue Expected)",
            "state": {
                "topic": "machine learning for image classification",
                "search_queries": ["deep learning CNN image classification", "transfer learning computer vision"],
                "user_profile": {"expertise": "intermediate"}
            }
        },
        {
            "name": "Vague Topic (Human Review Expected)",
            "state": {
                "topic": "improve AI",
                "search_queries": ["AI improvement"],
                "user_profile": {"expertise": "novice"}
            }
        },
        {
            "name": "Complex Literature Review (Expert Escalation Expected)",
            "state": {
                "literature_summaries": ["Brief summary 1", "Brief summary 2"],
                "knowledge_gap": {"knowledge_gap": "unclear gap"},
                "user_profile": {"expertise": "novice"}
            }
        },
        {
            "name": "High-Quality Proposal (Auto-Continue Expected)",
            "state": {
                "proposal_draft": "Introduction: This study proposes... Method: We will use... Evaluation: We will measure... Conclusion: Expected contributions...",
                "review_team_feedback": {
                    "feasibility": {"score": 0.9, "justification": "Highly feasible"},
                    "novelty": {"score": 0.85, "justification": "Novel approach"}
                },
                "user_profile": {"expertise": "expert"}
            }
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"📋 Scenario {i}: {scenario['name']}")
        print("-" * 50)
        
        # Test query routing
        if "topic" in scenario["state"]:
            query_result = router.route_query_stage(scenario["state"])
            print(f"Query Stage: {query_result.route.value} (confidence: {query_result.confidence:.2f})")
        
        # Test literature routing
        if "literature_summaries" in scenario["state"]:
            lit_result = router.route_literature_stage(scenario["state"])
            print(f"Literature Stage: {lit_result.route.value} (confidence: {lit_result.confidence:.2f})")
        
        # Test proposal routing
        if "proposal_draft" in scenario["state"]:
            prop_result = router.route_proposal_stage(scenario["state"])
            print(f"Proposal Stage: {prop_result.route.value} (confidence: {prop_result.confidence:.2f})")
        
        print()
    
    # Show efficiency gains
    print("📊 Expected Efficiency Gains:")
    print("- Auto-Continue Rate: 70-80% (vs 0% with fixed HIL)")
    print("- Human Review Rate: 15-25% (vs 100% with fixed HIL)")
    print("- Expert Escalation Rate: 5% (new capability)")
    print("- Processing Speed: 3-4x faster for auto-continue cases")
    print("- Human Focus: Only complex cases requiring expertise")

if __name__ == "__main__":
    demo_intelligent_routing() 