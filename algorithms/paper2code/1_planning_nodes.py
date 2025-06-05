"""
Planning Nodes for Paper2ImplementationDoc
Stage 1: Section Splitting & Planning

This module contains nodes for section splitting, section selection, and planning the extraction workflow.
"""

import logging
import yaml
from typing import Dict, Any, List, TypedDict, NotRequired
from pocketflow import Node
from utils.section_detector import SectionDetector, Section
from utils.llm_interface import call_llm

logger = logging.getLogger(__name__)

# Type Definitions for planning stage shared state

class SectionInfo(TypedDict):
    title: str
    content: str
    section_type: str
    start_position: int
    end_position: int
    confidence: float

class SelectedSection(TypedDict):
    title: str
    content: str
    section_type: str
    selection_reason: str
    priority: int

class PlanningSharedState(TypedDict):
    # Input from previous stage
    pdf_path: str
    raw_text: str
    cleaned_text: str
    pdf_metadata: Dict[str, Any]
    text_stats: Dict[str, Any]
    # Planning stage outputs
    sections: NotRequired[List[SectionInfo]]
    selected_sections: NotRequired[List[SelectedSection]]
    planning_summary: NotRequired[Dict[str, Any]]
    output_dir: NotRequired[str]


class SplitSectionsNode(Node):
    """
    Node to split paper text into sections using regex patterns.
    """
    
    def prep(self, shared: PlanningSharedState) -> str:
        """
        Prepare section splitting by getting raw text (preserves newlines for section detection).
        """
        # Use raw_text instead of cleaned_text since section detection needs newlines
        raw_text = shared.get("raw_text")
        if not raw_text:
            # Fallback to cleaned_text if raw_text not available
            cleaned_text = shared.get("cleaned_text")
            if not cleaned_text:
                raise ValueError("Neither raw_text nor cleaned_text found in shared store")
            logger.warning("raw_text not available, using cleaned_text (may affect section detection)")
            return cleaned_text
        
        return raw_text
    
    def exec(self, input_text: str) -> List[SectionInfo]:
        """
        Split text into sections using SectionDetector.
        """
        detector = SectionDetector(verbose=True)
        sections = detector.detect_sections(input_text, max_sections=20)
        
        # Convert Section objects to SectionInfo TypedDict
        section_infos = []
        for section in sections:
            section_info: SectionInfo = {
                "title": section.title,
                "content": section.content,
                "section_type": section.section_type,
                "start_position": section.start_position,
                "end_position": section.end_position,
                "confidence": section.confidence
            }
            section_infos.append(section_info)
        
        # If no sections found, create a fallback section from the entire text
        if not section_infos:
            logger.warning("No sections detected, creating fallback section from entire text")
            section_info: SectionInfo = {
                "title": "Full Document",
                "content": input_text,
                "section_type": "document",
                "start_position": 0,
                "end_position": len(input_text),
                "confidence": 0.5
            }
            section_infos.append(section_info)
        
        logger.info(f"Detected {len(section_infos)} sections")
        return section_infos
    
    def post(self, shared: PlanningSharedState, prep_res: str, exec_res: List[SectionInfo]) -> str:
        """
        Store detected sections in shared store.
        """
        shared["sections"] = exec_res
        
        # Log section summary
        section_types = [s["section_type"] for s in exec_res]
        type_counts = {}
        for stype in section_types:
            type_counts[stype] = type_counts.get(stype, 0) + 1
        
        logger.info(f"Section types detected: {type_counts}")
        return "default"


class SelectSectionsNode(Node):
    """
    Node to select relevant sections using LLM analysis.
    """
    
    def prep(self, shared: PlanningSharedState) -> Dict[str, Any]:
        """
        Prepare section selection by getting sections and paper metadata.
        """
        sections = shared.get("sections")
        if not sections:
            raise ValueError("sections not found in shared store")
        
        pdf_metadata = shared.get("pdf_metadata", {})
        text_stats = shared.get("text_stats", {})
        
        return {
            "sections": sections,
            "pdf_metadata": pdf_metadata,
            "text_stats": text_stats
        }
    
    def exec(self, prep_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to select and prioritize relevant sections.
        """
        sections = prep_data["sections"]
        
        # Handle single document section (fallback case)
        if len(sections) == 1 and sections[0]["section_type"] == "document":
            logger.info("Single document section detected, selecting it directly")
            selected_section: SelectedSection = {
                "title": "Full Document Analysis",
                "content": sections[0]["content"],
                "section_type": "document",
                "selection_reason": "Single section containing entire document",
                "priority": 1
            }
            return {
                "selected_sections": [selected_section],
                "planning_summary": {
                    "total_sections_analyzed": 1,
                    "selection_method": "single_document_fallback",
                    "selection_criteria": "Entire document as single section"
                },
                "llm_analysis": {
                    "success": False,
                    "fallback_used": True,
                    "reason": "Single document section"
                }
            }
        
        # Create section summary for LLM
        section_summaries = []
        for i, section in enumerate(sections):
            summary = {
                "index": i,
                "title": section["title"],
                "type": section["section_type"],
                "content_preview": section["content"][:200] + "..." if len(section["content"]) > 200 else section["content"],
                "confidence": section["confidence"],
                "length": len(section["content"])
            }
            section_summaries.append(summary)
        
        # Create LLM prompt for section selection
        prompt = f"""You are analyzing an academic paper to identify the most important sections for implementation documentation.

Paper has {len(sections)} sections detected. Your task is to:
1. Select 3-5 most important sections for understanding algorithms, methods, and implementation details
2. Prioritize sections that contain technical content, algorithms, methodology, and implementation details
3. Exclude references, acknowledgments, and purely theoretical discussions

Available sections:
{yaml.dump(section_summaries, default_flow_style=False)}

Please respond in YAML format:

```yaml
selected_sections:
  - index: 2
    title: "Methodology"
    section_type: "methodology"  
    selection_reason: "Contains core algorithm description and implementation approach"
    priority: 1
  - index: 4
    title: "System Architecture"
    section_type: "section"
    selection_reason: "Describes system design and components needed for implementation"
    priority: 2
planning_summary:
  total_sections_analyzed: {len(sections)}
  selection_criteria: "Focus on implementation-relevant content"
  key_focus_areas: ["algorithms", "methodology", "system design"]
```

Now provide your analysis:"""

        # Call LLM
        llm_result = call_llm(prompt, use_mock=True)  # Use mock for now
        
        if not llm_result["success"]:
            # Fallback: select sections heuristically
            logger.warning("LLM call failed, using heuristic selection")
            return self._heuristic_selection(sections)
        
        # Parse LLM response
        try:
            response_text = llm_result["response"]
            
            # Extract YAML content
            if "```yaml" in response_text:
                yaml_content = response_text.split("```yaml")[1].split("```")[0].strip()
            else:
                yaml_content = response_text
            
            parsed_result = yaml.safe_load(yaml_content)
            
            # Validate structure
            if not isinstance(parsed_result, dict) or "selected_sections" not in parsed_result:
                raise ValueError("Invalid LLM response structure")
            
            # Map selected sections back to full content
            selected_sections = []
            for selection in parsed_result["selected_sections"]:
                section_idx = selection["index"]
                if 0 <= section_idx < len(sections):
                    original_section = sections[section_idx]
                    selected_section: SelectedSection = {
                        "title": original_section["title"],
                        "content": original_section["content"],
                        "section_type": original_section["section_type"],
                        "selection_reason": selection.get("selection_reason", "Selected by LLM"),
                        "priority": selection.get("priority", 5)
                    }
                    selected_sections.append(selected_section)
            
            return {
                "selected_sections": selected_sections,
                "planning_summary": parsed_result.get("planning_summary", {}),
                "llm_analysis": {
                    "success": True,
                    "model": llm_result.get("model", "unknown"),
                    "response_length": llm_result.get("response_length", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            return self._heuristic_selection(sections)
    
    def _heuristic_selection(self, sections: List[SectionInfo]) -> Dict[str, Any]:
        """Fallback heuristic selection when LLM fails."""
        # Prioritize sections by type and content
        priority_types = {
            "methodology": 1,
            "abstract": 2,
            "introduction": 3,
            "results": 4,
            "section": 5,
            "conclusion": 6,
            "document": 7  # Fallback document type
        }
        
        scored_sections = []
        for section in sections:
            score = priority_types.get(section["section_type"], 10)
            
            # Boost score for sections with technical keywords
            content_lower = section["content"].lower()
            technical_keywords = ["algorithm", "method", "implementation", "architecture", "system", "model", "approach"]
            keyword_count = sum(1 for keyword in technical_keywords if keyword in content_lower)
            score -= keyword_count * 0.5  # Lower score = higher priority
            
            scored_sections.append((score, section))
        
        # Sort by score and select top 4
        scored_sections.sort(key=lambda x: x[0])
        top_sections = scored_sections[:4]
        
        selected_sections = []
        for i, (score, section) in enumerate(top_sections):
            selected_section: SelectedSection = {
                "title": section["title"],
                "content": section["content"],
                "section_type": section["section_type"],
                "selection_reason": f"Selected by heuristic (score: {score:.1f})",
                "priority": i + 1
            }
            selected_sections.append(selected_section)
        
        return {
            "selected_sections": selected_sections,
            "planning_summary": {
                "total_sections_analyzed": len(sections),
                "selection_method": "heuristic_fallback",
                "selection_criteria": "Type priority + technical keyword density"
            },
            "llm_analysis": {
                "success": False,
                "fallback_used": True
            }
        }
    
    def post(self, shared: PlanningSharedState, prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """
        Store selected sections and planning summary.
        """
        shared["selected_sections"] = exec_res["selected_sections"]
        shared["planning_summary"] = exec_res["planning_summary"]
        
        # Log selection results
        selected_count = len(exec_res["selected_sections"])
        method = "LLM" if exec_res["llm_analysis"]["success"] else "heuristic"
        
        logger.info(f"Selected {selected_count} sections using {method} method")
        for i, section in enumerate(exec_res["selected_sections"]):
            logger.info(f"  {i+1}. {section['section_type']}: {section['title']}")
        
        return "default"


class SavePlanningResultsNode(Node):
    """
    Node to save planning results to output directory.
    """
    
    def prep(self, shared: PlanningSharedState) -> Dict[str, Any]:
        """
        Prepare saving by collecting planning results.
        """
        required_keys = ["selected_sections", "planning_summary"]
        missing_keys = [key for key in required_keys if key not in shared]
        
        if missing_keys:
            raise ValueError(f"Missing required planning data: {missing_keys}")
        
        output_dir = shared.get("output_dir", "output")
        
        return {
            "output_dir": output_dir,
            "planning_data": {
                "selected_sections": shared["selected_sections"],
                "planning_summary": shared["planning_summary"],
                "total_sections_detected": len(shared.get("sections", [])),
                "pdf_metadata": shared.get("pdf_metadata", {}),
                "text_stats": shared.get("text_stats", {})
            }
        }
    
    def exec(self, prep_data: Dict[str, Any]) -> str:
        """
        Save planning results to JSON file.
        """
        import json
        import os
        from pathlib import Path
        
        output_dir = prep_data["output_dir"]
        planning_data = prep_data["planning_data"]
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save planning results
        output_file = os.path.join(output_dir, "planning_results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(planning_data, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def post(self, shared: PlanningSharedState, prep_res: Dict[str, Any], exec_res: str) -> str:
        """
        Store output file path in shared store.
        """
        shared["planning_output_file"] = exec_res
        
        logger.info(f"Planning results saved to: {exec_res}")
        return "default" 