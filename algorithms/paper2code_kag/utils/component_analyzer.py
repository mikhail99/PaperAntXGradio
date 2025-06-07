"""
Component Analysis utilities for academic papers
Identifies algorithms, mathematical formulations, methodology steps, and system architecture
"""

import re
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class Algorithm:
    """Represents an identified algorithm in the paper."""
    name: str
    description: str
    steps: List[str]
    complexity: str
    confidence: float
    location: str  # which section it was found in

@dataclass
class MathFormulation:
    """Represents a mathematical formulation or equation."""
    formula: str
    description: str
    variables: List[str]
    context: str
    confidence: float

@dataclass
class MethodologyStep:
    """Represents a step in the methodology."""
    step_number: int
    title: str
    description: str
    sub_steps: List[str]
    inputs: List[str]
    outputs: List[str]
    confidence: float

@dataclass
class SystemComponent:
    """Represents a system architecture component."""
    name: str
    type: str  # module, layer, network, etc.
    description: str
    connections: List[str]
    parameters: Dict[str, str]
    confidence: float

class ComponentAnalyzer:
    """Analyzes paper content to identify key implementation components."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Algorithm detection patterns
        self.algorithm_patterns = [
            r'(?i)algorithm\s+(\d+)[:.]?\s*([^.\n]+)',
            r'(?i)procedure\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)',
            r'(?i)function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)',
            r'(?i)method\s+([A-Za-z_][A-Za-z0-9_]*)',
            r'(?i)([\w\s]+)\s+algorithm',
            r'(?i)our\s+(approach|method|technique|algorithm)',
        ]
        
        # Mathematical formula patterns
        self.math_patterns = [
            r'\$([^$]+)\$',  # LaTeX inline math
            r'\\\[([^\]]+)\\\]',  # LaTeX display math
            r'equation\s*\(?(\d+)\)?[:.]?\s*([^.\n]+)',
            r'formula\s*\(?(\d+)\)?[:.]?\s*([^.\n]+)',
            r'([A-Za-z]+)\s*=\s*([^,\n]+)',  # Simple assignments
        ]
        
        # Methodology step indicators
        self.methodology_indicators = [
            r'(?i)step\s+(\d+)[:.]?\s*([^.\n]+)',
            r'(?i)stage\s+(\d+)[:.]?\s*([^.\n]+)',
            r'(?i)phase\s+(\d+)[:.]?\s*([^.\n]+)',
            r'(?i)first[,\s]+([^.\n]+)',
            r'(?i)second[,\s]+([^.\n]+)',
            r'(?i)third[,\s]+([^.\n]+)',
            r'(?i)finally[,\s]+([^.\n]+)',
            r'(?i)(\d+)\.\s+([^.\n]+)',
        ]
        
        # System component patterns
        self.component_patterns = [
            r'(?i)(\w+)\s+(module|component|layer|network|unit)',
            r'(?i)(\w+)\s+(architecture|framework|model)',
            r'(?i)(encoder|decoder|transformer|cnn|rnn|lstm|gru)',
            r'(?i)(neural\s+network|deep\s+learning|machine\s+learning)',
            r'(?i)(input|output|hidden)\s+(layer|unit)',
        ]
    
    def analyze_components(self, sections: List[Dict], text: str) -> Dict[str, Any]:
        """
        Analyze paper content to identify key implementation components.
        
        Args:
            sections: List of detected sections
            text: Full paper text
            
        Returns:
            Dict containing identified components
        """
        if self.verbose:
            logger.debug("Starting component analysis...")
        
        # Analyze different types of components
        algorithms = self._identify_algorithms(sections, text)
        math_formulations = self._identify_math_formulations(sections, text)
        methodology_steps = self._identify_methodology_steps(sections, text)
        system_components = self._identify_system_components(sections, text)
        
        # Extract additional insights
        technical_terms = self._extract_technical_terms(text)
        data_structures = self._identify_data_structures(text)
        dependencies = self._identify_dependencies(text)
        
        result = {
            "algorithms": [self._algorithm_to_dict(alg) for alg in algorithms],
            "math_formulations": [self._math_to_dict(math) for math in math_formulations],
            "methodology_steps": [self._method_to_dict(step) for step in methodology_steps],
            "system_components": [self._component_to_dict(comp) for comp in system_components],
            "technical_terms": technical_terms,
            "data_structures": data_structures,
            "dependencies": dependencies,
            "analysis_stats": {
                "algorithms_found": len(algorithms),
                "formulations_found": len(math_formulations),
                "methodology_steps_found": len(methodology_steps),
                "system_components_found": len(system_components),
                "average_confidence": self._calculate_average_confidence(algorithms, math_formulations, methodology_steps, system_components)
            }
        }
        
        if self.verbose:
            logger.debug(f"Component analysis complete: {result['analysis_stats']}")
        
        return result
    
    def _identify_algorithms(self, sections: List[Dict], text: str) -> List[Algorithm]:
        """Identify algorithms described in the paper."""
        algorithms = []
        
        # Focus on methodology sections
        methodology_text = ""
        for section in sections:
            if section.get("type") in ["methodology", "methods", "approach"]:
                methodology_text += section.get("content", "") + "\n"
        
        if not methodology_text:
            methodology_text = text  # Fallback to full text
        
        # Apply algorithm detection patterns
        for pattern in self.algorithm_patterns:
            matches = re.finditer(pattern, methodology_text, re.MULTILINE)
            for match in matches:
                algorithm_name = match.group(1) if match.groups() else "Unnamed Algorithm"
                
                # Extract context around the match
                start = max(0, match.start() - 200)
                end = min(len(methodology_text), match.end() + 200)
                context = methodology_text[start:end]
                
                # Extract steps from context
                steps = self._extract_algorithm_steps(context)
                
                # Estimate complexity
                complexity = self._estimate_algorithm_complexity(context)
                
                algorithm = Algorithm(
                    name=algorithm_name.strip(),
                    description=context[:100] + "..." if len(context) > 100 else context,
                    steps=steps,
                    complexity=complexity,
                    confidence=self._calculate_algorithm_confidence(context),
                    location="methodology"
                )
                
                algorithms.append(algorithm)
        
        # Remove duplicates and low-confidence algorithms
        algorithms = self._deduplicate_algorithms(algorithms)
        
        return algorithms[:5]  # Limit to top 5
    
    def _identify_math_formulations(self, sections: List[Dict], text: str) -> List[MathFormulation]:
        """Identify mathematical formulations in the paper."""
        formulations = []
        
        for pattern in self.math_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                formula = match.group(1) if match.groups() else match.group(0)
                
                # Extract context
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]
                
                # Extract variables
                variables = self._extract_variables(formula)
                
                formulation = MathFormulation(
                    formula=formula.strip(),
                    description=self._extract_formula_description(context),
                    variables=variables,
                    context=context,
                    confidence=self._calculate_math_confidence(formula, context)
                )
                
                formulations.append(formulation)
        
        return formulations[:10]  # Limit to top 10
    
    def _identify_methodology_steps(self, sections: List[Dict], text: str) -> List[MethodologyStep]:
        """Identify methodology steps from the paper."""
        steps = []
        
        # Focus on methodology sections
        methodology_text = ""
        for section in sections:
            if section.get("type") in ["methodology", "methods", "approach"]:
                methodology_text += section.get("content", "") + "\n"
        
        if not methodology_text:
            return steps
        
        step_counter = 0
        for pattern in self.methodology_indicators:
            matches = re.finditer(pattern, methodology_text, re.MULTILINE)
            for match in matches:
                step_counter += 1
                
                if match.groups() and len(match.groups()) >= 2:
                    step_num = match.group(1)
                    title = match.group(2)
                else:
                    step_num = str(step_counter)
                    title = match.group(1) if match.groups() else match.group(0)
                
                # Extract context
                start = max(0, match.start() - 50)
                end = min(len(methodology_text), match.end() + 200)
                context = methodology_text[start:end]
                
                # Extract sub-steps
                sub_steps = self._extract_sub_steps(context)
                
                # Extract inputs/outputs
                inputs, outputs = self._extract_io_elements(context)
                
                step = MethodologyStep(
                    step_number=int(step_num) if step_num.isdigit() else step_counter,
                    title=title.strip(),
                    description=context,
                    sub_steps=sub_steps,
                    inputs=inputs,
                    outputs=outputs,
                    confidence=self._calculate_step_confidence(context)
                )
                
                steps.append(step)
        
        # Sort by step number and remove duplicates
        steps = sorted(steps, key=lambda x: x.step_number)
        return steps[:8]  # Limit to top 8
    
    def _identify_system_components(self, sections: List[Dict], text: str) -> List[SystemComponent]:
        """Identify system architecture components."""
        components = []
        
        for pattern in self.component_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                comp_name = match.group(1) if match.groups() else match.group(0)
                comp_type = match.group(2) if len(match.groups()) >= 2 else "component"
                
                # Extract context
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 150)
                context = text[start:end]
                
                # Extract connections and parameters
                connections = self._extract_connections(context)
                parameters = self._extract_parameters(context)
                
                component = SystemComponent(
                    name=comp_name.strip(),
                    type=comp_type.strip(),
                    description=context[:100] + "..." if len(context) > 100 else context,
                    connections=connections,
                    parameters=parameters,
                    confidence=self._calculate_component_confidence(context)
                )
                
                components.append(component)
        
        return components[:6]  # Limit to top 6
    
    def _extract_algorithm_steps(self, context: str) -> List[str]:
        """Extract algorithm steps from context."""
        steps = []
        # Look for numbered lists or bullet points
        step_patterns = [
            r'(?i)\d+\.\s+([^.\n]+)',
            r'(?i)[-*]\s+([^.\n]+)',
            r'(?i)step\s*\d+[:.]?\s*([^.\n]+)'
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, context)
            steps.extend(matches[:3])  # Limit per pattern
        
        return steps[:5]  # Max 5 steps
    
    def _estimate_algorithm_complexity(self, context: str) -> str:
        """Estimate algorithmic complexity from context."""
        complexity_patterns = [
            (r'O\([^)]+\)', lambda m: m.group(0)),
            (r'(?i)linear\s+time', lambda m: "O(n)"),
            (r'(?i)quadratic\s+time', lambda m: "O(n²)"),
            (r'(?i)exponential\s+time', lambda m: "O(2^n)"),
            (r'(?i)logarithmic\s+time', lambda m: "O(log n)"),
            (r'(?i)polynomial\s+time', lambda m: "O(n^k)"),
        ]
        
        for pattern, extractor in complexity_patterns:
            match = re.search(pattern, context)
            if match:
                return extractor(match)
        
        return "O(n)"  # Default assumption
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms and concepts."""
        technical_patterns = [
            r'(?i)(neural\s+network|deep\s+learning|machine\s+learning)',
            r'(?i)(convolutional|recurrent|transformer|attention)',
            r'(?i)(gradient\s+descent|backpropagation|optimization)',
            r'(?i)(reinforcement\s+learning|supervised\s+learning|unsupervised\s+learning)',
            r'(?i)(classification|regression|clustering)',
            r'(?i)(feature\s+extraction|dimensionality\s+reduction)',
        ]
        
        terms = set()
        for pattern in technical_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    terms.update(match)
                else:
                    terms.add(match)
        
        return list(terms)[:10]
    
    def _identify_data_structures(self, text: str) -> List[str]:
        """Identify data structures mentioned in the paper."""
        ds_patterns = [
            r'(?i)(matrix|vector|tensor|array)',
            r'(?i)(graph|tree|list|queue|stack)',
            r'(?i)(dictionary|hash\s+table|map)',
            r'(?i)(dataset|database|corpus)',
        ]
        
        structures = set()
        for pattern in ds_patterns:
            matches = re.findall(pattern, text)
            structures.update(matches)
        
        return list(structures)[:8]
    
    def _identify_dependencies(self, text: str) -> List[str]:
        """Identify potential software dependencies."""
        dep_patterns = [
            r'(?i)(tensorflow|pytorch|keras|scikit-learn)',
            r'(?i)(numpy|pandas|scipy|matplotlib)',
            r'(?i)(opencv|pillow|nltk|spacy)',
            r'(?i)(python|java|c\+\+|matlab|r)',
        ]
        
        dependencies = set()
        for pattern in dep_patterns:
            matches = re.findall(pattern, text)
            dependencies.update(matches)
        
        return list(dependencies)[:6]
    
    def _extract_variables(self, formula: str) -> List[str]:
        """Extract variables from mathematical formula."""
        # Simple variable extraction (single letters)
        variables = re.findall(r'\b[a-zA-Z]\b', formula)
        return list(set(variables))[:5]
    
    def _extract_formula_description(self, context: str) -> str:
        """Extract description of a formula from context."""
        # Take text before the formula as description
        lines = context.split('\n')
        return lines[0][:100] if lines else context[:100]
    
    def _extract_sub_steps(self, context: str) -> List[str]:
        """Extract sub-steps from methodology context."""
        sub_patterns = [
            r'(?i)[a-z]\)\s+([^.\n]+)',
            r'(?i)[-*]\s+([^.\n]+)',
        ]
        
        sub_steps = []
        for pattern in sub_patterns:
            matches = re.findall(pattern, context)
            sub_steps.extend(matches[:2])
        
        return sub_steps[:3]
    
    def _extract_io_elements(self, context: str) -> Tuple[List[str], List[str]]:
        """Extract input and output elements from context."""
        input_patterns = [r'(?i)input[:s]?\s+([^.\n,]+)', r'(?i)given\s+([^.\n,]+)']
        output_patterns = [r'(?i)output[:s]?\s+([^.\n,]+)', r'(?i)return[:s]?\s+([^.\n,]+)']
        
        inputs = []
        outputs = []
        
        for pattern in input_patterns:
            matches = re.findall(pattern, context)
            inputs.extend(matches[:2])
        
        for pattern in output_patterns:
            matches = re.findall(pattern, context)
            outputs.extend(matches[:2])
        
        return inputs[:3], outputs[:3]
    
    def _extract_connections(self, context: str) -> List[str]:
        """Extract component connections from context."""
        connection_patterns = [
            r'(?i)connect(?:ed)?\s+to\s+([^.\n,]+)',
            r'(?i)feed(?:s)?\s+into\s+([^.\n,]+)',
            r'(?i)followed\s+by\s+([^.\n,]+)',
        ]
        
        connections = []
        for pattern in connection_patterns:
            matches = re.findall(pattern, context)
            connections.extend(matches[:2])
        
        return connections[:3]
    
    def _extract_parameters(self, context: str) -> Dict[str, str]:
        """Extract parameters from context."""
        param_patterns = [
            r'(?i)(\w+)\s*=\s*([^,\n]+)',
            r'(?i)with\s+(\w+)\s+of\s+([^,\n]+)',
        ]
        
        parameters = {}
        for pattern in param_patterns:
            matches = re.findall(pattern, context)
            for key, value in matches[:3]:
                parameters[key.strip()] = value.strip()
        
        return parameters
    
    def _calculate_algorithm_confidence(self, context: str) -> float:
        """Calculate confidence score for algorithm detection."""
        confidence = 0.5
        
        # Boost for algorithm keywords
        if re.search(r'(?i)(algorithm|procedure|function|method)', context):
            confidence += 0.2
        
        # Boost for steps
        if re.search(r'(?i)(step|stage|phase)', context):
            confidence += 0.1
        
        # Boost for complexity mentions
        if re.search(r'O\([^)]+\)', context):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_math_confidence(self, formula: str, context: str) -> float:
        """Calculate confidence for math formulation."""
        confidence = 0.4
        
        if re.search(r'[=+\-*/^]', formula):
            confidence += 0.2
        
        if re.search(r'(?i)(equation|formula)', context):
            confidence += 0.2
        
        if len(formula) > 3:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_step_confidence(self, context: str) -> float:
        """Calculate confidence for methodology step."""
        confidence = 0.6
        
        if re.search(r'(?i)(step|stage|phase)', context):
            confidence += 0.2
        
        if re.search(r'\d+\.', context):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_component_confidence(self, context: str) -> float:
        """Calculate confidence for system component."""
        confidence = 0.5
        
        if re.search(r'(?i)(module|component|layer|network)', context):
            confidence += 0.2
        
        if re.search(r'(?i)(architecture|framework)', context):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _deduplicate_algorithms(self, algorithms: List[Algorithm]) -> List[Algorithm]:
        """Remove duplicate algorithms based on name similarity."""
        unique = []
        seen_names = set()
        
        for alg in algorithms:
            name_key = alg.name.lower().strip()
            if name_key not in seen_names and alg.confidence > 0.5:
                seen_names.add(name_key)
                unique.append(alg)
        
        return sorted(unique, key=lambda x: x.confidence, reverse=True)
    
    def _calculate_average_confidence(self, *component_lists) -> float:
        """Calculate average confidence across all components."""
        all_confidences = []
        
        for comp_list in component_lists:
            for comp in comp_list:
                all_confidences.append(comp.confidence)
        
        return sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    
    # Conversion methods to dictionaries
    def _algorithm_to_dict(self, alg: Algorithm) -> Dict:
        return {
            "name": alg.name,
            "description": alg.description,
            "steps": alg.steps,
            "complexity": alg.complexity,
            "confidence": alg.confidence,
            "location": alg.location
        }
    
    def _math_to_dict(self, math: MathFormulation) -> Dict:
        return {
            "formula": math.formula,
            "description": math.description,
            "variables": math.variables,
            "context": math.context[:200],  # Truncate context
            "confidence": math.confidence
        }
    
    def _method_to_dict(self, step: MethodologyStep) -> Dict:
        return {
            "step_number": step.step_number,
            "title": step.title,
            "description": step.description[:200],  # Truncate description
            "sub_steps": step.sub_steps,
            "inputs": step.inputs,
            "outputs": step.outputs,
            "confidence": step.confidence
        }
    
    def _component_to_dict(self, comp: SystemComponent) -> Dict:
        return {
            "name": comp.name,
            "type": comp.type,
            "description": comp.description,
            "connections": comp.connections,
            "parameters": comp.parameters,
            "confidence": comp.confidence
        } 