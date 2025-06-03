"""
Paper2ImplementationDoc Nodes - Processing components
Following PocketFlow-Tutorial-Codebase-Knowledge pattern
Now with real PDF processing and section detection (Iteration 2)
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
from datetime import datetime

# Import PocketFlow base node
from pocketflow import Node

# Import our utilities
from utils.pdf_processor import PDFProcessor
from utils.section_detector import SectionDetector
from utils.component_analyzer import ComponentAnalyzer

# Assuming utils.call_llm might be used by some nodes later
# from .utils.call_llm import call_llm 

class PDFInputNode(Node):
    """Node for handling PDF input (file or ArXiv download) - now with real validation"""
    
    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__(**kwargs) # Pass any PocketFlow Node args like max_retries, wait
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        if self.verbose: 
            self.logger.setLevel(logging.DEBUG)
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor(verbose=verbose)

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare PDF input source information."""
        input_source = shared["input_source"]
        input_type = shared["input_type"]
        
        if self.verbose:
            self.logger.debug(f"Preparing PDF input: {input_source} (type: {input_type})")
        
        return {
            "input_source": input_source,
            "input_type": input_type
        }

    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Execute PDF input validation and preparation."""
        input_source = prep_res["input_source"]
        input_type = prep_res["input_type"]
        
        self.logger.info(f"📥 Processing {input_type} input: {input_source}")
        
        if input_type == "pdf":
            # Real PDF validation
            validation_result = self.pdf_processor.validate_pdf(input_source)
            
            if not validation_result["valid"]:
                raise FileNotFoundError(f"PDF validation failed: {validation_result['details']}")
            
            result = {
                "validated_source": input_source,
                "file_type": input_type,
                "file_size": validation_result["file_size"],
                "file_size_mb": validation_result["file_size_mb"],
                "page_count": validation_result["page_count"],
                "validation_status": "success",
                "metadata": {
                    "processor": "PDFInputNode",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        elif input_type == "arxiv":
            # Placeholder: ArXiv download logic (will be implemented in future iteration)
            # For now, treat as if downloaded to a local file
            self.logger.warning(f"ArXiv download not yet implemented - using placeholder")
            downloaded_path = f"./downloads/{input_source}.pdf"
            
            result = {
                "validated_source": downloaded_path,
                "file_type": "pdf",  # ArXiv papers are PDFs
                "arxiv_id": input_source,
                "file_size": "unknown",
                "validation_status": "placeholder",
                "metadata": {
                    "processor": "PDFInputNode",
                    "timestamp": datetime.now().isoformat(),
                    "warning": "ArXiv download not implemented - using placeholder"
                }
            }
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
            
        return result

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Store validated input information in shared store."""
        shared["validated_input"] = exec_res
        shared["metadata"]["processing_steps"].append({
            "step": "pdf_input",
            "status": exec_res["validation_status"],
            "timestamp": exec_res["metadata"]["timestamp"]
        })
        
        # Add warning if ArXiv placeholder was used
        if "warning" in exec_res["metadata"]:
            shared["metadata"]["warnings"].append(exec_res["metadata"]["warning"])
        
        if self.verbose:
            self.logger.debug(f"✅ PDF input validation completed: {exec_res['validation_status']}")
        
        return "default"  # Continue to next node


class TextExtractionNode(Node):
    """Node for extracting text content from PDF - now with real PyMuPDF extraction"""
    
    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        if self.verbose: 
            self.logger.setLevel(logging.DEBUG)
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor(verbose=verbose)

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare for text extraction."""
        validated_input = shared["validated_input"]
        return validated_input

    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Execute text extraction from PDF using PyMuPDF."""
        source = prep_res["validated_source"]
        file_type = prep_res["file_type"]
        
        self.logger.info("🔍 Extracting text from PDF...")
        
        if file_type == "pdf" and Path(source).exists():
            # Real PDF text extraction
            try:
                extraction_result = self.pdf_processor.extract_text_from_pdf(source)
                
                result = {
                    "raw_text": extraction_result["raw_text"],
                    "cleaned_text": extraction_result["cleaned_text"],
                    "page_texts": extraction_result["page_texts"],
                    "page_count": extraction_result["page_count"],
                    "word_count": extraction_result["word_count"],
                    "char_count": extraction_result["char_count"],
                    "extraction_method": extraction_result["extraction_method"],
                    "extraction_quality": extraction_result["extraction_quality"],
                    "metadata": {
                        "processor": "TextExtractionNode",
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
            except Exception as e:
                self.logger.error(f"PDF extraction failed: {str(e)}")
                # Fallback to placeholder if extraction fails
                result = {
                    "raw_text": f"[PDF extraction failed for {source}]",
                    "cleaned_text": f"PDF extraction failed: {str(e)}",
                    "page_texts": [],
                    "page_count": 0,
                    "word_count": 0,
                    "char_count": 0,
                    "extraction_method": "fallback",
                    "extraction_quality": "failed",
                    "metadata": {
                        "processor": "TextExtractionNode",
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e)
                    }
                }
        else:
            # Fallback for non-existent files (e.g., ArXiv placeholder)
            self.logger.warning(f"File not found or not PDF: {source}")
            result = {
                "raw_text": f"Sample research paper text for {source}...",
                "cleaned_text": f"Sample cleaned research paper text for {source}...",
                "page_texts": [{"page_number": 1, "text": "Sample page text", "char_count": 50}],
                "page_count": 1,
                "word_count": 100,
                "char_count": 500,
                "extraction_method": "placeholder",
                "extraction_quality": "medium",
                "metadata": {
                    "processor": "TextExtractionNode",
                    "timestamp": datetime.now().isoformat(),
                    "warning": f"Using placeholder text for {source}"
                }
            }
        
        return result

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Store extracted text in shared store."""
        shared["extracted_text"] = exec_res
        shared["metadata"]["processing_steps"].append({
            "step": "text_extraction",
            "word_count": exec_res["word_count"],
            "quality": exec_res["extraction_quality"],
            "method": exec_res["extraction_method"],
            "timestamp": exec_res["metadata"]["timestamp"]
        })
        
        # Add warning if placeholder or error occurred
        if "warning" in exec_res["metadata"]:
            shared["metadata"]["warnings"].append(exec_res["metadata"]["warning"])
        if "error" in exec_res["metadata"]:
            shared["metadata"]["warnings"].append(f"Text extraction error: {exec_res['metadata']['error']}")
        
        if self.verbose:
            self.logger.debug(f"✅ Text extraction completed: {exec_res['word_count']} words")
        
        return "default"


class StructureAnalysisNode(Node):
    """Node for analyzing paper structure and identifying sections - now with advanced component analysis (Iteration 3)"""
    
    def __init__(self, max_sections: int = 10, enable_component_analysis: bool = True, verbose: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.max_sections = max_sections
        self.enable_component_analysis = enable_component_analysis
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        if self.verbose: 
            self.logger.setLevel(logging.DEBUG)
        
        # Initialize section detector
        self.section_detector = SectionDetector(verbose=verbose)
        
        # Initialize component analyzer (new for Iteration 3)
        if self.enable_component_analysis:
            self.component_analyzer = ComponentAnalyzer(verbose=verbose)

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare text and section data for enhanced structure analysis."""
        extracted_text = shared["extracted_text"]
        return {
            "text": extracted_text["cleaned_text"],
            "raw_text": extracted_text["raw_text"],
            "extraction_quality": extracted_text["extraction_quality"]
        }

    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced structure analysis with component identification."""
        text = prep_res["text"]
        raw_text = prep_res["raw_text"]
        extraction_quality = prep_res["extraction_quality"]
        
        self.logger.info("📋 Analyzing paper structure with advanced component detection...")
        
        if not text or text.startswith("[PDF extraction failed"):
            # Handle failed extraction
            self.logger.warning("Cannot analyze structure - text extraction failed")
            result = {
                "sections": [
                    {"title": "Unknown Section", "content": "Structure analysis failed", "type": "unknown"}
                ],
                "document_type": "unknown",
                "complexity_score": 0.0,
                "identified_algorithms": [],
                "structure_analysis": {
                    "structure_quality": "failed",
                    "section_count": 0
                },
                "component_analysis": self._get_empty_component_analysis(),
                "metadata": {
                    "processor": "StructureAnalysisNode",
                    "max_sections_limit": self.max_sections,
                    "component_analysis_enabled": self.enable_component_analysis,
                    "timestamp": datetime.now().isoformat(),
                    "error": "Text extraction failed"
                }
            }
        else:
            # Real section detection
            try:
                detected_sections = self.section_detector.detect_sections(text, self.max_sections)
                structure_analysis = self.section_detector.analyze_paper_structure(detected_sections)
                
                # Convert Section objects to dictionaries
                sections = []
                for section in detected_sections:
                    sections.append({
                        "title": section.title,
                        "content": section.content[:1000] + "..." if len(section.content) > 1000 else section.content,  # Truncate long content
                        "type": section.section_type,
                        "confidence": section.confidence,
                        "start_position": section.start_position,
                        "end_position": section.end_position
                    })
                
                # Enhanced component analysis (Iteration 3)
                component_analysis = self._get_empty_component_analysis()
                if self.enable_component_analysis:
                    try:
                        component_analysis = self.component_analyzer.analyze_components(sections, text)
                        self.logger.info(f"🔬 Component analysis: {component_analysis['analysis_stats']['algorithms_found']} algorithms, "
                                       f"{component_analysis['analysis_stats']['methodology_steps_found']} steps, "
                                       f"{component_analysis['analysis_stats']['system_components_found']} components")
                    except Exception as e:
                        self.logger.warning(f"Component analysis failed: {str(e)}")
                        component_analysis["analysis_stats"]["error"] = str(e)
                
                # Legacy algorithm identification (for backward compatibility)
                legacy_algorithms = self._identify_algorithms_legacy(text, detected_sections)
                
                result = {
                    "sections": sections,
                    "document_type": structure_analysis["document_type"],
                    "complexity_score": self._calculate_enhanced_complexity_score(structure_analysis, text, component_analysis),
                    "identified_algorithms": legacy_algorithms,  # Keep for backward compatibility
                    "structure_analysis": structure_analysis,
                    "component_analysis": component_analysis,  # New for Iteration 3
                    "metadata": {
                        "processor": "StructureAnalysisNode",
                        "max_sections_limit": self.max_sections,
                        "sections_found": len(sections),
                        "component_analysis_enabled": self.enable_component_analysis,
                        "extraction_quality": extraction_quality,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
            except Exception as e:
                self.logger.error(f"Structure analysis failed: {str(e)}")
                # Fallback to simple analysis
                result = {
                    "sections": [
                        {"title": "Document", "content": text[:1000] + "...", "type": "document"}
                    ],
                    "document_type": "document",
                    "complexity_score": 5.0,
                    "identified_algorithms": ["Unknown Algorithm"],
                    "structure_analysis": {
                        "structure_quality": "error",
                        "section_count": 1
                    },
                    "component_analysis": self._get_empty_component_analysis(),
                    "metadata": {
                        "processor": "StructureAnalysisNode",
                        "max_sections_limit": self.max_sections,
                        "component_analysis_enabled": self.enable_component_analysis,
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e)
                    }
                }
        
        return result
    
    def _get_empty_component_analysis(self) -> Dict[str, Any]:
        """Get empty component analysis structure for error cases."""
        return {
            "algorithms": [],
            "math_formulations": [],
            "methodology_steps": [],
            "system_components": [],
            "technical_terms": [],
            "data_structures": [],
            "dependencies": [],
            "analysis_stats": {
                "algorithms_found": 0,
                "formulations_found": 0,
                "methodology_steps_found": 0,
                "system_components_found": 0,
                "average_confidence": 0.0
            }
        }
    
    def _identify_algorithms_legacy(self, text: str, sections: List) -> List[str]:
        """Legacy algorithm identification for backward compatibility."""
        algorithms = []
        
        # Look for algorithm-related keywords
        algorithm_patterns = [
            r'algorithm\s+\d+',
            r'procedure\s+\w+',
            r'function\s+\w+',
            r'method\s+\w+',
            r'approach\s+\w+',
            r'technique\s+\w+'
        ]
        
        import re
        for pattern in algorithm_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:3]:  # Limit to 3 per pattern
                algorithms.append(match.title())
        
        # If no algorithms found, use generic names
        if not algorithms:
            algorithms = ["Main Algorithm", "Processing Method"]
        
        return algorithms[:5]  # Limit total algorithms
    
    def _calculate_enhanced_complexity_score(self, structure_analysis: Dict, text: str, component_analysis: Dict) -> float:
        """Calculate enhanced complexity score using component analysis."""
        # Base score from structure
        base_score = structure_analysis.get("structure_score", 0.5) * 5  # Scale to 0-5
        
        # Adjust based on document length
        word_count = len(text.split())
        if word_count > 5000:
            base_score += 2
        elif word_count > 2000:
            base_score += 1
        
        # Adjust based on structure quality
        quality = structure_analysis.get("structure_quality", "poor")
        if quality == "excellent":
            base_score += 1
        elif quality == "good":
            base_score += 0.5
        
        # New: Adjust based on component analysis
        if component_analysis and "analysis_stats" in component_analysis:
            stats = component_analysis["analysis_stats"]
            
            # More algorithms = higher complexity
            base_score += min(stats.get("algorithms_found", 0) * 0.5, 2.0)
            
            # More math formulations = higher complexity
            base_score += min(stats.get("formulations_found", 0) * 0.3, 1.5)
            
            # More methodology steps = higher complexity
            base_score += min(stats.get("methodology_steps_found", 0) * 0.2, 1.0)
            
            # More system components = higher complexity
            base_score += min(stats.get("system_components_found", 0) * 0.3, 1.5)
        
        return min(base_score, 10.0)  # Cap at 10

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Store enhanced structure analysis results."""
        shared["paper_structure"] = exec_res
        shared["metadata"]["processing_steps"].append({
            "step": "structure_analysis",
            "sections_found": len(exec_res["sections"]),
            "complexity_score": exec_res["complexity_score"],
            "document_type": exec_res["document_type"],
            "component_analysis_enabled": self.enable_component_analysis,
            "algorithms_found": exec_res["component_analysis"]["analysis_stats"]["algorithms_found"],
            "methodology_steps_found": exec_res["component_analysis"]["analysis_stats"]["methodology_steps_found"],
            "timestamp": exec_res["metadata"]["timestamp"]
        })
        
        # Add error/warning if any
        if "error" in exec_res["metadata"]:
            shared["metadata"]["warnings"].append(f"Structure analysis error: {exec_res['metadata']['error']}")
        
        if self.verbose:
            component_stats = exec_res["component_analysis"]["analysis_stats"]
            self.logger.debug(f"✅ Enhanced structure analysis completed: {len(exec_res['sections'])} sections, "
                            f"{component_stats['algorithms_found']} algorithms, "
                            f"{component_stats['methodology_steps_found']} methodology steps")
        
        return "default"


class ImplementationAnalysisNode(Node):
    """Node for analyzing implementation details and generating implementation guide - enhanced with component analysis"""
    
    def __init__(self, analysis_depth: str = "detailed", verbose: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.analysis_depth = analysis_depth
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        if self.verbose: 
            self.logger.setLevel(logging.DEBUG)

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare paper structure and component analysis for implementation analysis."""
        paper_structure = shared["paper_structure"]
        config = shared["config"]
        
        return {
            "sections": paper_structure["sections"],
            "algorithms": paper_structure["identified_algorithms"],
            "component_analysis": paper_structure.get("component_analysis", {}),  # New for Iteration 3
            "analysis_depth": self.analysis_depth,
            "document_type": paper_structure["document_type"],
            "complexity_score": paper_structure["complexity_score"],
            "config": config
        }

    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced implementation analysis using component data."""
        sections = prep_res["sections"]
        algorithms = prep_res["algorithms"]
        component_analysis = prep_res["component_analysis"]
        depth = prep_res["analysis_depth"]
        complexity_score = prep_res["complexity_score"]
        
        self.logger.info(f"🔬 Performing {depth} implementation analysis with component insights...")
        
        # Enhanced analysis based on actual sections and components
        complexity_factor = 1.5 if depth == "detailed" else 1.0
        
        # Analyze methodology section for implementation details
        methodology_content = ""
        for section in sections:
            if section["type"] in ["methodology", "methods", "approach"]:
                methodology_content += section["content"] + "\n"
        
        # Extract implementation insights from methodology and component analysis
        impl_insights = self._extract_enhanced_implementation_insights(methodology_content, component_analysis)
        
        # Use component analysis data for better algorithm analysis
        enhanced_algorithms = self._enhance_algorithms_with_components(algorithms, component_analysis)
        
        result = {
            "implementation_guide": {
                "overview": f"Enhanced implementation guide for {len(enhanced_algorithms)} algorithms (depth: {depth})",
                "architecture": {
                    "components": impl_insights.get("components", ["Data Loader", "Processor", "Output Manager"]),
                    "interfaces": impl_insights.get("interfaces", ["API Interface", "Config Interface"]),
                    "data_flow": impl_insights.get("data_flow", "Input → Process → Validate → Output"),
                    "system_components": impl_insights.get("system_components", [])  # New from component analysis
                },
                "algorithms": enhanced_algorithms,
                "mathematical_formulations": impl_insights.get("math_formulations", []),  # New
                "methodology_steps": impl_insights.get("methodology_steps", [])  # New
            },
            "implementation_complexity": f"{depth}_complexity_enhanced",
            "estimated_effort": f"{len(enhanced_algorithms) * int(complexity_factor * complexity_score * 2)} hours",
            "implementation_insights": impl_insights,
            "component_insights": self._extract_component_insights(component_analysis),  # New
            "metadata": {
                "processor": "ImplementationAnalysisNode",
                "analysis_depth": depth,
                "methodology_analyzed": len(methodology_content) > 0,
                "component_analysis_used": bool(component_analysis),
                "algorithms_enhanced": len(enhanced_algorithms),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return result
    
    def _extract_enhanced_implementation_insights(self, methodology_text: str, component_analysis: Dict) -> Dict[str, List[str]]:
        """Extract enhanced implementation-relevant information using component analysis."""
        insights = {
            "components": [],
            "interfaces": [],
            "data_flow": "",
            "data_structures": [],
            "dependencies": [],
            "system_components": [],
            "math_formulations": [],
            "methodology_steps": []
        }
        
        # Use component analysis data if available
        if component_analysis:
            # Extract from component analysis
            insights["system_components"] = [comp["name"] for comp in component_analysis.get("system_components", [])]
            insights["data_structures"] = component_analysis.get("data_structures", [])
            insights["dependencies"] = component_analysis.get("dependencies", [])
            insights["math_formulations"] = [form["formula"] for form in component_analysis.get("math_formulations", [])]
            insights["methodology_steps"] = [step["title"] for step in component_analysis.get("methodology_steps", [])]
        
        # Fallback to text analysis if component analysis is empty or failed
        if not methodology_text:
            return insights
        
        import re
        
        # Look for common components (enhanced with component analysis)
        component_patterns = [
            r'(\w+)\s+(?:module|component|layer|network)',
            r'(?:using|with)\s+(\w+)\s+(?:architecture|framework)',
            r'(\w+)\s+(?:algorithm|method|approach)'
        ]
        
        for pattern in component_patterns:
            matches = re.findall(pattern, methodology_text, re.IGNORECASE)
            for match in matches[:3]:
                if len(match) > 2:
                    insights["components"].append(match.title())
        
        # Merge with component analysis results
        if insights["system_components"]:
            insights["components"].extend(insights["system_components"][:3])
        
        # Look for data structures (enhanced)
        if not insights["data_structures"]:
            data_structure_keywords = ['matrix', 'vector', 'graph', 'tree', 'array', 'list', 'dictionary', 'tensor']
            for keyword in data_structure_keywords:
                if keyword in methodology_text.lower():
                    insights["data_structures"].append(keyword.title())
        
        # Look for dependencies/frameworks (enhanced)
        if not insights["dependencies"]:
            framework_keywords = ['tensorflow', 'pytorch', 'numpy', 'scipy', 'sklearn', 'pandas', 'opencv']
            for keyword in framework_keywords:
                if keyword in methodology_text.lower():
                    insights["dependencies"].append(keyword)
        
        # Remove duplicates and limit
        insights["components"] = list(set(insights["components"]))[:8]
        insights["data_structures"] = list(set(insights["data_structures"]))[:8]
        insights["dependencies"] = list(set(insights["dependencies"]))[:8]
        
        # Set defaults if empty
        if not insights["components"]:
            insights["components"] = ["Input Processor", "Main Algorithm", "Output Generator"]
        if not insights["data_structures"]:
            insights["data_structures"] = ["Array", "Dictionary", "Graph"]
        if not insights["dependencies"]:
            insights["dependencies"] = ["numpy", "scipy"]
        
        return insights
    
    def _enhance_algorithms_with_components(self, legacy_algorithms: List[str], component_analysis: Dict) -> List[Dict]:
        """Enhance legacy algorithms with component analysis data."""
        enhanced_algorithms = []
        
        # Use component analysis algorithms if available
        if component_analysis and component_analysis.get("algorithms"):
            for alg_data in component_analysis["algorithms"]:
                enhanced_algorithms.append({
                    "name": alg_data["name"],
                    "complexity": alg_data["complexity"],
                    "implementation_steps": alg_data.get("steps", []),
                    "data_structures": component_analysis.get("data_structures", ["Array", "Dictionary"]),
                    "dependencies": component_analysis.get("dependencies", ["numpy", "scipy"]),
                    "confidence": alg_data.get("confidence", 0.5),
                    "description": alg_data.get("description", ""),
                    "location": alg_data.get("location", "unknown")
                })
        
        # Fallback to legacy algorithms if component analysis failed
        if not enhanced_algorithms:
            for alg_name in legacy_algorithms:
                enhanced_algorithms.append({
                    "name": alg_name,
                    "complexity": "O(n)",
                    "implementation_steps": [
                        f"Step 1: Initialize {alg_name}",
                        f"Step 2: Process input for {alg_name}",
                        f"Step 3: Generate output for {alg_name}"
                    ],
                    "data_structures": ["Array", "Dictionary"],
                    "dependencies": ["numpy", "scipy"],
                    "confidence": 0.5,
                    "description": f"Legacy algorithm: {alg_name}",
                    "location": "unknown"
                })
        
        return enhanced_algorithms
    
    def _extract_component_insights(self, component_analysis: Dict) -> Dict[str, Any]:
        """Extract insights specifically from component analysis."""
        if not component_analysis:
            return {"status": "not_available"}
        
        stats = component_analysis.get("analysis_stats", {})
        
        return {
            "status": "available",
            "summary": {
                "algorithms_identified": stats.get("algorithms_found", 0),
                "math_formulations": stats.get("formulations_found", 0),
                "methodology_steps": stats.get("methodology_steps_found", 0),
                "system_components": stats.get("system_components_found", 0),
                "average_confidence": stats.get("average_confidence", 0.0)
            },
            "technical_terms": component_analysis.get("technical_terms", []),
            "implementation_complexity": self._assess_implementation_complexity(component_analysis),
            "recommended_approach": self._recommend_implementation_approach(component_analysis)
        }
    
    def _assess_implementation_complexity(self, component_analysis: Dict) -> str:
        """Assess implementation complexity based on component analysis."""
        if not component_analysis:
            return "medium"
        
        stats = component_analysis.get("analysis_stats", {})
        total_components = (
            stats.get("algorithms_found", 0) +
            stats.get("methodology_steps_found", 0) +
            stats.get("system_components_found", 0)
        )
        
        if total_components >= 15:
            return "very_high"
        elif total_components >= 10:
            return "high"
        elif total_components >= 5:
            return "medium"
        else:
            return "low"
    
    def _recommend_implementation_approach(self, component_analysis: Dict) -> str:
        """Recommend implementation approach based on component analysis."""
        if not component_analysis:
            return "standard_approach"
        
        technical_terms = component_analysis.get("technical_terms", [])
        algorithms = component_analysis.get("algorithms", [])
        
        # Check for ML/AI terms
        ml_terms = ["neural network", "deep learning", "machine learning", "transformer", "attention"]
        has_ml = any(term.lower() in " ".join(technical_terms).lower() for term in ml_terms)
        
        if has_ml:
            return "ml_framework_approach"
        elif len(algorithms) > 3:
            return "modular_approach"
        else:
            return "standard_approach"

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Store enhanced implementation analysis results."""
        shared["implementation_analysis"] = exec_res
        shared["metadata"]["processing_steps"].append({
            "step": "implementation_analysis",
            "depth": self.analysis_depth,
            "algorithms_analyzed": len(exec_res["implementation_guide"]["algorithms"]),
            "estimated_effort": exec_res["estimated_effort"],
            "component_analysis_used": exec_res["metadata"]["component_analysis_used"],
            "implementation_complexity": exec_res["component_insights"].get("implementation_complexity", "unknown"),
            "timestamp": exec_res["metadata"]["timestamp"]
        })
        
        if self.verbose:
            component_status = exec_res["component_insights"]["status"]
            self.logger.debug(f"✅ Enhanced implementation analysis completed: {exec_res['implementation_complexity']} "
                            f"(component analysis: {component_status})")
        
        return "default"


class DocumentationGenerationNode(Node):
    """Node for generating final implementation documentation"""
    
    def __init__(self, output_format: str = "markdown", include_diagrams: bool = False, verbose: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.output_format = output_format
        self.include_diagrams = include_diagrams
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        if self.verbose: 
            self.logger.setLevel(logging.DEBUG)

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare all analysis results for documentation generation."""
        return {
            "paper_structure": shared["paper_structure"],
            "implementation_analysis": shared["implementation_analysis"],
            "output_dir": shared["output_dir"],
            "config": shared["config"]
        }

    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Execute documentation generation."""
        analysis = prep_res["implementation_analysis"]
        structure = prep_res["paper_structure"]
        output_dir = prep_res["output_dir"]
        
        self.logger.info(f"📝 Generating {self.output_format} documentation...")
        
        # Enhanced documentation generation
        doc_content = self._generate_enhanced_documentation(analysis, structure)
        
        # Generate file paths
        base_name = "implementation_guide"
        extension = {"markdown": ".md", "html": ".html", "latex": ".tex"}[self.output_format]
        
        output_files = [f"{base_name}{extension}"]
        
        if self.include_diagrams:
            diagram_content = self._generate_diagrams(analysis)
            output_files.append(f"{base_name}_diagram.md")
        
        result = {
            "documentation": {
                "content": doc_content,
                "format": self.output_format,
                "includes_diagrams": self.include_diagrams
            },
            "output_files": output_files,
            "file_paths": [f"{output_dir}/{file}" for file in output_files],
            "quality_metrics": self._calculate_quality_metrics(analysis, structure),
            "metadata": {
                "processor": "DocumentationGenerationNode",
                "output_format": self.output_format,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return result
    
    def _generate_enhanced_documentation(self, analysis: Dict, structure: Dict) -> str:
        """Generate enhanced documentation based on real analysis and component data."""
        guide = analysis["implementation_guide"]
        component_analysis = structure.get("component_analysis", {})
        
        doc = f"""# Implementation Guide

## Overview
{guide['overview']}

**Document Type**: {structure.get('document_type', 'Unknown')}
**Complexity Score**: {structure.get('complexity_score', 'N/A')}/10
**Estimated Effort**: {analysis.get('estimated_effort', 'Unknown')}

## Paper Structure Analysis
**Sections Identified**: {len(structure.get('sections', []))}
**Structure Quality**: {structure.get('structure_analysis', {}).get('structure_quality', 'Unknown')}

"""
        
        # Add section overview
        sections = structure.get('sections', [])
        if sections:
            doc += "**Detected Sections**:\n"
            for section in sections:
                confidence = section.get('confidence', 0)
                doc += f"- {section['title']} ({section['type']}) - Confidence: {confidence:.1f}\n"
            doc += "\n"
        
        # Add component analysis summary (NEW for Iteration 3)
        if component_analysis and component_analysis.get("analysis_stats"):
            stats = component_analysis["analysis_stats"]
            doc += f"""## Component Analysis Summary

**Algorithms Identified**: {stats.get('algorithms_found', 0)}
**Mathematical Formulations**: {stats.get('formulations_found', 0)}
**Methodology Steps**: {stats.get('methodology_steps_found', 0)}
**System Components**: {stats.get('system_components_found', 0)}
**Average Confidence**: {stats.get('average_confidence', 0.0):.2f}

"""
            
            # Add technical terms if available
            technical_terms = component_analysis.get("technical_terms", [])
            if technical_terms:
                doc += f"**Technical Terms Identified**: {', '.join(technical_terms[:5])}\n\n"
        
        doc += f"""## System Architecture

### Components
{', '.join(guide['architecture']['components'])}

### Data Flow
{guide['architecture']['data_flow']}

### Interfaces
{', '.join(guide['architecture']['interfaces'])}

"""
        
        # Add system components from component analysis
        system_components = guide['architecture'].get('system_components', [])
        if system_components:
            doc += f"""### Identified System Components
{', '.join(system_components)}

"""
        
        doc += "## Implementation Details\n\n"
        
        # Add algorithms with enhanced details from component analysis
        for i, alg in enumerate(guide['algorithms'], 1):
            doc += f"""### {i}. {alg['name']}

**Complexity**: {alg['complexity']}
**Dependencies**: {', '.join(alg['dependencies'])}
**Data Structures**: {', '.join(alg['data_structures'])}
"""
            
            # Add confidence and description if available (from component analysis)
            if 'confidence' in alg:
                doc += f"**Confidence**: {alg['confidence']:.2f}\n"
            if 'description' in alg and alg['description']:
                doc += f"**Description**: {alg['description'][:100]}...\n"
            
            doc += "\n**Implementation Steps**:\n"
            for step in alg['implementation_steps']:
                doc += f"- {step}\n"
            doc += "\n"
        
        # Add mathematical formulations if available (NEW)
        math_formulations = guide.get('mathematical_formulations', [])
        if math_formulations:
            doc += f"""## Mathematical Formulations

"""
            for i, formula in enumerate(math_formulations[:5], 1):
                doc += f"{i}. `{formula}`\n"
            doc += "\n"
        
        # Add methodology steps if available (NEW)
        methodology_steps = guide.get('methodology_steps', [])
        if methodology_steps:
            doc += f"""## Methodology Steps

"""
            for i, step in enumerate(methodology_steps[:8], 1):
                doc += f"{i}. {step}\n"
            doc += "\n"
        
        # Add implementation insights if available
        insights = analysis.get('implementation_insights', {})
        if insights:
            doc += f"""## Technical Considerations

### Key Components
{', '.join(insights.get('components', []))}

### Required Data Structures
{', '.join(insights.get('data_structures', []))}

### Recommended Dependencies
{', '.join(insights.get('dependencies', []))}

"""
        
        # Add component insights (NEW)
        component_insights = analysis.get('component_insights', {})
        if component_insights.get('status') == 'available':
            summary = component_insights.get('summary', {})
            doc += f"""## Component Analysis Insights

**Implementation Complexity**: {component_insights.get('implementation_complexity', 'Unknown')}
**Recommended Approach**: {component_insights.get('recommended_approach', 'Standard')}

### Analysis Summary
- Algorithms: {summary.get('algorithms_identified', 0)}
- Math Formulations: {summary.get('math_formulations', 0)}
- Methodology Steps: {summary.get('methodology_steps', 0)}
- System Components: {summary.get('system_components', 0)}

"""
        
        doc += """## Implementation Roadmap

1. **Phase 1**: Set up development environment and dependencies
2. **Phase 2**: Implement core data structures and interfaces
3. **Phase 3**: Develop main algorithms
4. **Phase 4**: Testing and optimization
5. **Phase 5**: Integration and deployment

---
*Generated by Paper2ImplementationDoc (Iteration 3) - Now with advanced component analysis*
"""
        
        return doc
    
    def _generate_diagrams(self, analysis: Dict) -> str:
        """Generate Mermaid diagrams based on analysis."""
        guide = analysis["implementation_guide"]
        
        diagram = """# Implementation Diagrams

## System Architecture

```mermaid
flowchart TD
    A[Input Data] --> B[Data Processor]
    B --> C{Algorithm Selection}
"""
        
        # Add algorithm nodes
        algorithms = guide.get('algorithms', [])
        for i, alg in enumerate(algorithms, 1):
            diagram += f"    C --> D{i}[{alg['name']}]\n"
        
        for i in range(1, len(algorithms) + 1):
            diagram += f"    D{i} --> E[Output Generator]\n"
        
        diagram += "    E --> F[Final Results]\n```\n\n"
        
        # Add component diagram
        components = guide['architecture'].get('components', [])
        if len(components) > 1:
            diagram += """## Component Interaction

```mermaid
graph LR
"""
            for i, comp in enumerate(components):
                if i < len(components) - 1:
                    next_comp = components[i + 1]
                    diagram += f"    {comp.replace(' ', '')} --> {next_comp.replace(' ', '')}\n"
            diagram += "```\n"
        
        return diagram
    
    def _calculate_quality_metrics(self, analysis: Dict, structure: Dict) -> Dict[str, float]:
        """Calculate quality metrics based on analysis results."""
        # Base metrics
        completeness = 75.0
        clarity = 80.0
        implementability = 70.0
        
        # Adjust based on structure analysis
        structure_quality = structure.get('structure_analysis', {}).get('structure_quality', 'poor')
        if structure_quality == 'excellent':
            completeness += 15
            clarity += 10
        elif structure_quality == 'good':
            completeness += 10
            clarity += 5
        elif structure_quality == 'fair':
            completeness += 5
        
        # Adjust based on sections found
        sections_found = len(structure.get('sections', []))
        if sections_found >= 5:
            completeness += 10
        elif sections_found >= 3:
            completeness += 5
        
        # Adjust based on methodology analysis
        if analysis.get('metadata', {}).get('methodology_analyzed', False):
            implementability += 15
            clarity += 10
        
        # Cap at 100
        return {
            "completeness": min(completeness, 100.0),
            "clarity": min(clarity, 100.0),
            "implementability": min(implementability, 100.0)
        }

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Save generated documentation and finalize results."""
        output_dir = Path(prep_res["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main documentation
        doc_content = exec_res["documentation"]["content"]
        main_file = output_dir / exec_res["output_files"][0]
        
        with open(main_file, "w", encoding="utf-8") as f:
            f.write(doc_content)
        
        # Save diagram if requested
        if self.include_diagrams and len(exec_res["output_files"]) > 1:
            diagram_file = output_dir / exec_res["output_files"][1]
            diagram_content = self._generate_diagrams(prep_res["implementation_analysis"])
            with open(diagram_file, "w", encoding="utf-8") as f:
                f.write(diagram_content)
        
        # Store results in shared
        shared["final_documentation"] = exec_res
        shared["output_files"] = exec_res["file_paths"]
        shared["quality_score"] = sum(exec_res["quality_metrics"].values()) / len(exec_res["quality_metrics"])
        
        shared["metadata"]["processing_steps"].append({
            "step": "documentation_generation",
            "format": self.output_format,
            "files_generated": len(exec_res["output_files"]),
            "quality_score": shared["quality_score"],
            "timestamp": exec_res["metadata"]["timestamp"]
        })
        
        self.logger.info(f"📄 Generated {len(exec_res['output_files'])} file(s) in {output_dir}")
        if self.verbose:
            self.logger.debug(f"✅ Documentation generation completed: {shared['quality_score']:.1f}% quality")
        
        return "default"  # End of pipeline 