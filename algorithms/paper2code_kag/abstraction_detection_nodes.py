"""
Abstraction Planning Nodes for Paper2ImplementationDoc
Iteration 3: Identify, categorize, and save abstractions using hybrid rule-based + LLM approach.

Flow: identify_abstractions >> categorize_abstractions >> save_abstractions
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional, TypedDict
from pocketflow import Node

# Import our abstraction detector
from utils.abstraction_detector import AbstractionDetector, DetectedAbstraction, AbstractionType

logger = logging.getLogger(__name__)

# Type definitions for shared state
class AbstractionInfo(TypedDict):
    """Information about a detected abstraction."""
    name: str
    type: str  # AbstractionType.value
    description: str
    confidence: float
    detection_method: str
    section_source: str
    keywords: List[str]
    context: str
    metadata: Dict[str, Any]

class CategorizedAbstraction(TypedDict):
    """Categorized abstraction with additional analysis."""
    abstraction: AbstractionInfo
    category: str  # Primary category
    subcategory: str  # More specific classification
    importance_score: float  # 0.0 to 1.0
    relationships: List[str]  # Related abstraction names
    implementation_complexity: str  # "low", "medium", "high"

class AbstractionPlanningSharedState(TypedDict):
    """Shared state for abstraction planning flow."""
    # From previous iterations
    selected_sections: List[Dict[str, Any]]
    planning_summary: Dict[str, Any]
    
    # Abstraction planning results
    raw_abstractions: List[AbstractionInfo]
    categorized_abstractions: List[CategorizedAbstraction]
    abstraction_summary: Dict[str, Any]
    
    # Metadata
    abstraction_detection_method: str
    total_abstractions_found: int

class IdentifyAbstractionsNode(Node):
    """
    Node to identify abstractions from selected sections using hybrid rule-based + LLM approach.
    """
    
    def __init__(self, use_mock_llm: bool = True, target_types: Optional[List[str]] = None, max_retries: int = 2, wait: int = 5):
        super().__init__(max_retries=max_retries, wait=wait)
        self.use_mock_llm = use_mock_llm
        self.target_types = [AbstractionType(t) for t in target_types] if target_types else None
        self.detector = AbstractionDetector(use_mock_llm=use_mock_llm)
    
    def prep(self, shared: AbstractionPlanningSharedState) -> List[Dict[str, str]]:
        """Prepare sections for abstraction detection."""
        sections = shared.get("selected_sections", [])
        if not sections:
            raise ValueError("No selected sections found. Run section planning first.")
        
        # Extract section content and titles
        section_data = []
        for section in sections:
            section_data.append({
                "title": section.get("title", "Unknown"),
                "content": section.get("content", ""),
                "section_type": section.get("section_type", "section")
            })
        
        logger.info(f"Prepared {len(section_data)} sections for abstraction detection")
        return section_data
    
    def exec(self, section_data: List[Dict[str, str]]) -> List[DetectedAbstraction]:
        """Execute abstraction detection on all sections."""
        all_abstractions = []
        
        for section in section_data:
            title = section["title"]
            content = section["content"]
            
            if not content.strip():
                logger.warning(f"Skipping empty section: {title}")
                continue
            
            logger.info(f"Detecting abstractions in section: {title}")
            
            # Use hybrid detection for best results
            abstractions = self.detector.detect_abstractions_hybrid(
                text=content,
                section_title=title,
                target_types=self.target_types
            )
            
            logger.info(f"Found {len(abstractions)} abstractions in {title}")
            all_abstractions.extend(abstractions)
        
        # Sort by confidence (highest first)
        all_abstractions.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Total abstractions detected: {len(all_abstractions)}")
        return all_abstractions
    
    def exec_fallback(self, section_data: List[Dict[str, str]], exc: Exception) -> List[DetectedAbstraction]:
        """Fallback to rule-based only if hybrid detection fails."""
        logger.warning(f"Hybrid detection failed: {str(exc)}. Falling back to rule-based detection.")
        
        all_abstractions = []
        for section in section_data:
            title = section["title"]
            content = section["content"]
            
            if content.strip():
                abstractions = self.detector.detect_abstractions_rule_based(content, title)
                all_abstractions.extend(abstractions)
        
        return all_abstractions
    
    def post(self, shared: AbstractionPlanningSharedState, prep_res: List[Dict[str, str]], exec_res: List[DetectedAbstraction]) -> str:
        """Save detected abstractions to shared state."""
        # Convert DetectedAbstraction objects to AbstractionInfo dicts
        raw_abstractions = []
        
        for abstraction in exec_res:
            abs_info: AbstractionInfo = {
                "name": abstraction.name,
                "type": abstraction.type.value,
                "description": abstraction.description,
                "confidence": abstraction.confidence,
                "detection_method": abstraction.detection_method,
                "section_source": abstraction.section_source,
                "keywords": abstraction.keywords,
                "context": abstraction.context,
                "metadata": {
                    "line_number": abstraction.line_number
                }
            }
            raw_abstractions.append(abs_info)
        
        shared["raw_abstractions"] = raw_abstractions
        shared["total_abstractions_found"] = len(raw_abstractions)
        shared["abstraction_detection_method"] = "hybrid" if exec_res and exec_res[0].detection_method == "hybrid" else "rule-based"
        
        logger.info(f"Stored {len(raw_abstractions)} raw abstractions in shared state")
        return "default"

class CategorizeAbstractionsNode(Node):
    """
    Node to categorize and analyze detected abstractions.
    """
    
    def __init__(self, max_retries: int = 2, wait: int = 5):
        super().__init__(max_retries=max_retries, wait=wait)
    
    def prep(self, shared: AbstractionPlanningSharedState) -> List[AbstractionInfo]:
        """Prepare raw abstractions for categorization."""
        raw_abstractions = shared.get("raw_abstractions", [])
        if not raw_abstractions:
            raise ValueError("No raw abstractions found. Run identification first.")
        
        logger.info(f"Preparing {len(raw_abstractions)} abstractions for categorization")
        return raw_abstractions
    
    def exec(self, raw_abstractions: List[AbstractionInfo]) -> List[CategorizedAbstraction]:
        """Categorize abstractions with additional analysis."""
        categorized_abstractions = []
        
        for abs_info in raw_abstractions:
            # Determine category and subcategory
            category, subcategory = self._determine_category(abs_info)
            
            # Calculate importance score
            importance_score = self._calculate_importance(abs_info)
            
            # Find relationships with other abstractions
            relationships = self._find_relationships(abs_info, raw_abstractions)
            
            # Assess implementation complexity
            complexity = self._assess_complexity(abs_info)
            
            categorized_abs: CategorizedAbstraction = {
                "abstraction": abs_info,
                "category": category,
                "subcategory": subcategory,
                "importance_score": importance_score,
                "relationships": relationships,
                "implementation_complexity": complexity
            }
            
            categorized_abstractions.append(categorized_abs)
        
        # Sort by importance score (highest first)
        categorized_abstractions.sort(key=lambda x: x["importance_score"], reverse=True)
        
        logger.info(f"Categorized {len(categorized_abstractions)} abstractions")
        return categorized_abstractions
    
    def _determine_category(self, abs_info: AbstractionInfo) -> tuple[str, str]:
        """Determine primary category and subcategory."""
        abs_type = abs_info["type"]
        keywords = [kw.lower() for kw in abs_info["keywords"]]
        
        # Category mapping
        if abs_type == "algorithm":
            if any(kw in keywords for kw in ["neural", "deep", "learning", "ml"]):
                return "Machine Learning", "Neural Algorithm"
            elif any(kw in keywords for kw in ["optimization", "search", "sort"]):
                return "Computational", "Optimization Algorithm"
            else:
                return "Computational", "General Algorithm"
        
        elif abs_type == "architecture":
            if any(kw in keywords for kw in ["neural", "network", "cnn", "transformer"]):
                return "Neural Architecture", "Network Design"
            else:
                return "System Architecture", "Framework Design"
        
        elif abs_type == "method":
            if any(kw in keywords for kw in ["preprocessing", "feature"]):
                return "Data Processing", "Feature Engineering"
            else:
                return "Methodology", "General Method"
        
        elif abs_type == "workflow":
            return "Process", "Workflow Design"
        
        elif abs_type == "technique":
            return "Implementation", "Technical Approach"
        
        elif abs_type == "dataset":
            return "Data", "Dataset Resource"
        
        else:
            return "General", "Uncategorized"
    
    def _calculate_importance(self, abs_info: AbstractionInfo) -> float:
        """Calculate importance score based on confidence and keywords."""
        base_score = abs_info["confidence"]
        
        # Boost score for high-value keywords
        high_value_keywords = ["algorithm", "neural network", "transformer", "architecture", "method"]
        keyword_boost = sum(1 for kw in abs_info["keywords"] if any(hv in kw.lower() for hv in high_value_keywords))
        
        # Boost for detection method
        method_boost = 0.1 if abs_info["detection_method"] == "hybrid" else 0.0
        
        # Cap at 1.0
        importance_score = min(1.0, base_score + (keyword_boost * 0.1) + method_boost)
        
        return round(importance_score, 3)
    
    def _find_relationships(self, current_abs: AbstractionInfo, all_abstractions: List[AbstractionInfo]) -> List[str]:
        """Find relationships with other abstractions."""
        relationships = []
        current_keywords = set(kw.lower() for kw in current_abs["keywords"])
        
        for other_abs in all_abstractions:
            if other_abs["name"] == current_abs["name"]:
                continue
            
            other_keywords = set(kw.lower() for kw in other_abs["keywords"])
            
            # Check for keyword overlap
            overlap = current_keywords & other_keywords
            if len(overlap) >= 1:  # At least 1 keyword in common
                relationships.append(other_abs["name"])
        
        return relationships[:3]  # Limit to top 3 relationships
    
    def _assess_complexity(self, abs_info: AbstractionInfo) -> str:
        """Assess implementation complexity."""
        keywords = [kw.lower() for kw in abs_info["keywords"]]
        abs_type = abs_info["type"]
        
        # High complexity indicators
        high_complexity_keywords = ["transformer", "neural network", "deep learning", "optimization"]
        
        # Medium complexity indicators
        medium_complexity_keywords = ["cnn", "attention", "preprocessing", "algorithm"]
        
        if any(kw in keywords for kw in high_complexity_keywords) or abs_type == "architecture":
            return "high"
        elif any(kw in keywords for kw in medium_complexity_keywords) or abs_type == "algorithm":
            return "medium"
        else:
            return "low"
    
    def post(self, shared: AbstractionPlanningSharedState, prep_res: List[AbstractionInfo], exec_res: List[CategorizedAbstraction]) -> str:
        """Save categorized abstractions to shared state."""
        shared["categorized_abstractions"] = exec_res
        
        # Generate summary statistics
        type_counts = {}
        category_counts = {}
        complexity_counts = {"low": 0, "medium": 0, "high": 0}
        
        for cat_abs in exec_res:
            abs_type = cat_abs["abstraction"]["type"]
            category = cat_abs["category"]
            complexity = cat_abs["implementation_complexity"]
            
            type_counts[abs_type] = type_counts.get(abs_type, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1
            complexity_counts[complexity] += 1
        
        shared["abstraction_summary"] = {
            "total_categorized": len(exec_res),
            "type_distribution": type_counts,
            "category_distribution": category_counts,
            "complexity_distribution": complexity_counts,
            "average_importance": sum(cat_abs["importance_score"] for cat_abs in exec_res) / len(exec_res) if exec_res else 0.0
        }
        
        logger.info(f"Categorized {len(exec_res)} abstractions with summary statistics")
        return "default"

class SaveAbstractionsNode(Node):
    """
    Node to save abstraction planning results to JSON file.
    """
    
    def __init__(self, output_dir: str = "output", max_retries: int = 2, wait: int = 5):
        super().__init__(max_retries=max_retries, wait=wait)
        self.output_dir = output_dir
    
    def prep(self, shared: AbstractionPlanningSharedState) -> Dict[str, Any]:
        """Prepare abstraction planning data for saving."""
        categorized_abstractions = shared.get("categorized_abstractions", [])
        if not categorized_abstractions:
            raise ValueError("No categorized abstractions found. Run categorization first.")
        
        # Prepare complete planning data
        planning_data = {
            "abstraction_planning_results": {
                "raw_abstractions": shared.get("raw_abstractions", []),
                "categorized_abstractions": categorized_abstractions,
                "abstraction_summary": shared.get("abstraction_summary", {}),
                "detection_metadata": {
                    "detection_method": shared.get("abstraction_detection_method", "unknown"),
                    "total_found": shared.get("total_abstractions_found", 0),
                    "sections_analyzed": len(shared.get("selected_sections", []))
                }
            },
            "previous_planning": {
                "selected_sections": shared.get("selected_sections", []),
                "planning_summary": shared.get("planning_summary", {})
            }
        }
        
        logger.info(f"Prepared abstraction planning data with {len(categorized_abstractions)} categorized abstractions")
        return planning_data
    
    def exec(self, planning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save abstraction planning data to JSON file."""
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save to JSON file
        output_file = os.path.join(self.output_dir, "abstraction_plan.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(planning_data, f, indent=2, ensure_ascii=False)
        
        file_size = os.path.getsize(output_file)
        
        result = {
            "output_file": output_file,
            "file_size_bytes": file_size,
            "abstractions_saved": len(planning_data["abstraction_planning_results"]["categorized_abstractions"]),
            "success": True
        }
        
        logger.info(f"Saved abstraction planning results to {output_file} ({file_size} bytes)")
        return result
    
    def post(self, shared: AbstractionPlanningSharedState, prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Update shared state with save results."""
        if not exec_res.get("success", False):
            raise RuntimeError(f"Failed to save abstraction planning results")
        
        # Add save metadata to shared state
        shared["abstraction_plan_saved"] = True
        shared["abstraction_plan_file"] = exec_res["output_file"]
        shared["abstraction_plan_file_size"] = exec_res["file_size_bytes"]
        
        logger.info(f"Successfully saved {exec_res['abstractions_saved']} categorized abstractions")
        return "default"

def test_abstraction_planning_nodes():
    """Test the abstraction planning nodes."""
    print("🧪 Testing Abstraction Planning Nodes")
    
    # Create mock shared state based on Iteration 2 results
    shared: AbstractionPlanningSharedState = {
        "selected_sections": [
            {
                "title": "Abstract",
                "content": "This paper presents a novel neural network approach for data processing. Our method uses deep learning techniques with transformer architecture.",
                "section_type": "abstract"
            },
            {
                "title": "Methodology",
                "content": "Step 1: Data preprocessing using matrix operations\nStep 2: Feature extraction with CNN layers\nStep 3: Attention mechanism for sequence processing\nThe algorithm complexity is O(n log n) for the preprocessing phase.",
                "section_type": "methodology"
            }
        ],
        "planning_summary": {"sections_selected": 2},
        "raw_abstractions": [],
        "categorized_abstractions": [],
        "abstraction_summary": {},
        "abstraction_detection_method": "",
        "total_abstractions_found": 0
    }
    
    # Test nodes
    identify_node = IdentifyAbstractionsNode(use_mock_llm=True)
    categorize_node = CategorizeAbstractionsNode()
    save_node = SaveAbstractionsNode(output_dir="test_output")
    
    print("\n🔍 Testing IdentifyAbstractionsNode...")
    action1 = identify_node.run(shared)
    print(f"✅ Identified {shared['total_abstractions_found']} abstractions")
    
    print("\n📂 Testing CategorizeAbstractionsNode...")
    action2 = categorize_node.run(shared)
    print(f"✅ Categorized {len(shared['categorized_abstractions'])} abstractions")
    
    print("\n💾 Testing SaveAbstractionsNode...")
    action3 = save_node.run(shared)
    print(f"✅ Saved planning results to {shared.get('abstraction_plan_file', 'unknown')}")
    
    print(f"\n📊 Summary:")
    summary = shared.get("abstraction_summary", {})
    print(f"  • Total abstractions: {summary.get('total_categorized', 0)}")
    print(f"  • Type distribution: {summary.get('type_distribution', {})}")
    print(f"  • Average importance: {summary.get('average_importance', 0):.2f}")

if __name__ == "__main__":
    test_abstraction_planning_nodes() 