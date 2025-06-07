"""
Abstraction Detector for Paper2ImplementationDoc
Hybrid rule-based + LLM detection of algorithms, methods, datasets, and workflows.
Uses parameterizable abstraction types with structured output.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

class AbstractionType(Enum):
    """Types of abstractions that can be detected."""
    ALGORITHM = "algorithm"
    METHOD = "method"
    DATASET = "dataset"
    WORKFLOW = "workflow"
    TECHNIQUE = "technique"
    ARCHITECTURE = "architecture"

@dataclass
class DetectedAbstraction:
    """Represents a detected abstraction with metadata."""
    name: str
    type: AbstractionType
    description: str
    confidence: float  # 0.0 to 1.0
    detection_method: str  # "rule-based" or "llm" or "hybrid"
    section_source: str
    keywords: List[str]
    context: str  # surrounding text
    line_number: Optional[int] = None

class AbstractionDetector:
    """
    Hybrid rule-based + LLM abstraction detector.
    Supports parameterizable abstraction types.
    """
    
    def __init__(self, use_mock_llm: bool = True):
        self.use_mock_llm = use_mock_llm
        self._setup_rule_patterns()
    
    def _setup_rule_patterns(self):
        """Setup regex patterns for rule-based detection."""
        self.patterns = {
            AbstractionType.ALGORITHM: [
                r'\b(?:algorithm|algorithmic|step\s*\d+|procedure)\b',
                r'\b(?:complexity|O\([^)]+\)|time complexity|space complexity)\b',
                r'\b(?:sorting|searching|optimization|clustering)\b',
                r'\b(?:neural network|deep learning|machine learning|CNN|RNN|transformer)\b'
            ],
            AbstractionType.METHOD: [
                r'\b(?:method|methodology|approach|technique|strategy)\b',
                r'\b(?:framework|model|system|architecture)\b',
                r'\b(?:preprocessing|feature extraction|training|inference)\b'
            ],
            AbstractionType.DATASET: [
                r'\b(?:dataset|data set|corpus|benchmark|collection)\b',
                r'\b(?:training data|test data|validation set|samples)\b',
                r'\b(?:ImageNet|MNIST|CIFAR|CoNLL|Wikipedia)\b'
            ],
            AbstractionType.WORKFLOW: [
                r'\b(?:workflow|pipeline|process|flow|sequence)\b',
                r'\b(?:step\s*\d+|phase\s*\d+|stage\s*\d+)\b',
                r'\b(?:preprocessing.*training|training.*evaluation)\b'
            ],
            AbstractionType.TECHNIQUE: [
                r'\b(?:technique|method|approach|strategy)\b',
                r'\b(?:attention mechanism|batch normalization|dropout)\b',
                r'\b(?:data augmentation|transfer learning|fine-tuning)\b'
            ],
            AbstractionType.ARCHITECTURE: [
                r'\b(?:architecture|network|model structure)\b',
                r'\b(?:layers?|nodes?|units?|connections?)\b',
                r'\b(?:encoder|decoder|embedding|pooling)\b'
            ]
        }
    
    def detect_abstractions_rule_based(self, text: str, section_title: str) -> List[DetectedAbstraction]:
        """
        Rule-based abstraction detection using regex patterns.
        """
        abstractions = []
        lines = text.split('\n')
        
        for abs_type, patterns in self.patterns.items():
            type_score = 0
            matched_keywords = []
            contexts = []
            
            for line_num, line in enumerate(lines):
                line_lower = line.lower()
                line_matches = 0
                
                for pattern in patterns:
                    matches = re.findall(pattern, line_lower, re.IGNORECASE)
                    if matches:
                        line_matches += len(matches)
                        matched_keywords.extend(matches)
                        if line.strip():  # Only add non-empty contexts
                            contexts.append(line.strip())
                
                type_score += line_matches
            
            # Create abstraction if we found evidence
            if type_score > 0:
                confidence = min(0.8, type_score * 0.2)  # Rule-based max confidence: 0.8
                
                # Generate description based on context
                description = self._generate_rule_description(abs_type, matched_keywords, contexts)
                
                abstraction = DetectedAbstraction(
                    name=f"{abs_type.value}_from_{section_title}",
                    type=abs_type,
                    description=description,
                    confidence=confidence,
                    detection_method="rule-based",
                    section_source=section_title,
                    keywords=list(set(matched_keywords)),
                    context="; ".join(contexts[:3])  # Top 3 contexts
                )
                abstractions.append(abstraction)
        
        return abstractions
    
    def _generate_rule_description(self, abs_type: AbstractionType, keywords: List[str], contexts: List[str]) -> str:
        """Generate description for rule-based detection."""
        if not keywords and not contexts:
            return f"Detected {abs_type.value} with no specific details"
        
        desc_parts = [f"Detected {abs_type.value}"]
        
        if keywords:
            unique_keywords = list(set(keywords))[:5]  # Top 5 unique keywords
            desc_parts.append(f"involving: {', '.join(unique_keywords)}")
        
        if contexts:
            desc_parts.append(f"Context: {contexts[0][:100]}...")
        
        return ". ".join(desc_parts)
    
    def detect_abstractions_llm(self, text: str, section_title: str, target_types: Optional[List[AbstractionType]] = None) -> List[DetectedAbstraction]:
        """
        LLM-based abstraction detection with mock fallback.
        """
        if self.use_mock_llm:
            return self._mock_llm_detection(text, section_title, target_types)
        
        # Real LLM implementation would go here
        try:
            from .llm_interface import call_llm
            
            # Prepare target types
            if target_types:
                types_str = ", ".join([t.value for t in target_types])
            else:
                types_str = ", ".join([t.value for t in AbstractionType])
            
            prompt = f"""
Analyze the following text section and identify abstractions.

Section: {section_title}
Text: {text}

Target abstraction types: {types_str}

For each abstraction found, provide:
1. Name (concise identifier)
2. Type (one of: {types_str})
3. Description (detailed explanation)
4. Confidence (0.0 to 1.0)
5. Keywords (relevant terms)

Output in JSON format:
{{
    "abstractions": [
        {{
            "name": "example_algorithm",
            "type": "algorithm",
            "description": "Description of the algorithm",
            "confidence": 0.9,
            "keywords": ["keyword1", "keyword2"]
        }}
    ]
}}
"""
            
            result = call_llm(prompt)
            if result["success"]:
                return self._parse_llm_response(result["response"], section_title)
            else:
                logger.warning(f"LLM call failed, using mock: {result.get('error', 'Unknown error')}")
                return self._mock_llm_detection(text, section_title, target_types)
                
        except Exception as e:
            logger.warning(f"LLM detection failed, using mock: {str(e)}")
            return self._mock_llm_detection(text, section_title, target_types)
    
    def _mock_llm_detection(self, text: str, section_title: str, target_types: Optional[List[AbstractionType]] = None) -> List[DetectedAbstraction]:
        """
        Mock LLM detection with realistic responses based on text analysis.
        """
        abstractions = []
        text_lower = text.lower()
        
        # Mock responses based on content analysis
        mock_patterns = {
            'neural network': (AbstractionType.ARCHITECTURE, "Neural network architecture for data processing", 0.9),
            'deep learning': (AbstractionType.METHOD, "Deep learning methodology for feature extraction", 0.85),
            'transformer': (AbstractionType.ARCHITECTURE, "Transformer architecture with attention mechanism", 0.95),
            'algorithm': (AbstractionType.ALGORITHM, "Core algorithm for computational processing", 0.8),
            'cnn': (AbstractionType.ARCHITECTURE, "Convolutional Neural Network architecture", 0.9),
            'attention': (AbstractionType.TECHNIQUE, "Attention mechanism for sequence processing", 0.85),
            'preprocessing': (AbstractionType.WORKFLOW, "Data preprocessing workflow", 0.7),
            'pytorch': (AbstractionType.TECHNIQUE, "PyTorch implementation framework", 0.8),
            'tensorflow': (AbstractionType.TECHNIQUE, "TensorFlow optimization framework", 0.8),
            'matrix operations': (AbstractionType.METHOD, "Matrix operations for data transformation", 0.75)
        }
        
        # Filter by target types if specified
        valid_types = set(target_types) if target_types else set(AbstractionType)
        
        for keyword, (abs_type, description, confidence) in mock_patterns.items():
            if abs_type in valid_types and keyword in text_lower:
                abstraction = DetectedAbstraction(
                    name=f"{keyword.replace(' ', '_')}_{section_title.lower()}",
                    type=abs_type,
                    description=description,
                    confidence=confidence,
                    detection_method="llm",
                    section_source=section_title,
                    keywords=[keyword],
                    context=self._extract_context_around_keyword(text, keyword)
                )
                abstractions.append(abstraction)
        
        return abstractions
    
    def _extract_context_around_keyword(self, text: str, keyword: str, window: int = 50) -> str:
        """Extract context around a keyword."""
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        index = text_lower.find(keyword_lower)
        if index == -1:
            return text[:100] + "..." if len(text) > 100 else text
        
        start = max(0, index - window)
        end = min(len(text), index + len(keyword) + window)
        context = text[start:end].strip()
        
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
            
        return context
    
    def _parse_llm_response(self, response: str, section_title: str) -> List[DetectedAbstraction]:
        """Parse LLM JSON response into DetectedAbstraction objects."""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                abstractions = []
                for item in data.get("abstractions", []):
                    abs_type = AbstractionType(item["type"])
                    abstraction = DetectedAbstraction(
                        name=item["name"],
                        type=abs_type,
                        description=item["description"],
                        confidence=item["confidence"],
                        detection_method="llm",
                        section_source=section_title,
                        keywords=item.get("keywords", []),
                        context=item.get("context", "")
                    )
                    abstractions.append(abstraction)
                
                return abstractions
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {str(e)}")
        
        return []
    
    def detect_abstractions_hybrid(self, text: str, section_title: str, target_types: Optional[List[AbstractionType]] = None) -> List[DetectedAbstraction]:
        """
        Hybrid detection combining rule-based and LLM approaches.
        """
        # Get rule-based detections
        rule_abstractions = self.detect_abstractions_rule_based(text, section_title)
        
        # Get LLM detections
        llm_abstractions = self.detect_abstractions_llm(text, section_title, target_types)
        
        # Combine and deduplicate
        all_abstractions = rule_abstractions + llm_abstractions
        
        # Simple deduplication by type and similar names
        deduplicated = []
        seen_types = set()
        
        # Prioritize LLM detections (higher confidence), then rule-based
        for abstraction in sorted(all_abstractions, key=lambda x: (-x.confidence, x.detection_method == "llm")):
            # Simple deduplication by type per section
            type_section_key = (abstraction.type, abstraction.section_source)
            if type_section_key not in seen_types:
                seen_types.add(type_section_key)
                # Mark as hybrid if we have both types
                if any(a.type == abstraction.type and a.detection_method != abstraction.detection_method 
                       for a in all_abstractions):
                    abstraction.detection_method = "hybrid"
                deduplicated.append(abstraction)
        
        return deduplicated

def test_abstraction_detector():
    """Test the abstraction detector."""
    print("🧪 Testing Abstraction Detector")
    
    detector = AbstractionDetector(use_mock_llm=True)
    
    # Test text with various abstractions
    test_text = """
    Step 1: Data preprocessing using matrix operations
    Step 2: Feature extraction with CNN layers
    Step 3: Attention mechanism for sequence processing
    The algorithm complexity is O(n log n) for the preprocessing phase.
    Our neural network approach uses transformer architecture.
    """
    
    section_title = "Methodology"
    
    print(f"\n📋 Testing on section: {section_title}")
    print(f"📝 Text: {test_text[:100]}...")
    
    # Test rule-based detection
    print("\n🔍 Rule-based detection:")
    rule_abstractions = detector.detect_abstractions_rule_based(test_text, section_title)
    for abs in rule_abstractions:
        print(f"  ✓ {abs.type.value}: {abs.name} (confidence: {abs.confidence:.2f})")
    
    # Test LLM detection
    print("\n🤖 LLM detection:")
    llm_abstractions = detector.detect_abstractions_llm(test_text, section_title)
    for abs in llm_abstractions:
        print(f"  ✓ {abs.type.value}: {abs.name} (confidence: {abs.confidence:.2f})")
    
    # Test hybrid detection
    print("\n🔀 Hybrid detection:")
    hybrid_abstractions = detector.detect_abstractions_hybrid(test_text, section_title)
    for abs in hybrid_abstractions:
        print(f"  ✓ {abs.type.value}: {abs.name} (confidence: {abs.confidence:.2f}, method: {abs.detection_method})")
    
    print(f"\n✅ Found {len(hybrid_abstractions)} unique abstractions using hybrid approach")

if __name__ == "__main__":
    test_abstraction_detector() 