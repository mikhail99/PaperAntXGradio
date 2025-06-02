"""
Section Detection utilities for academic papers
Uses pattern matching and heuristics to identify paper sections
"""

import re
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Section:
    """Represents a detected section in a paper."""
    title: str
    content: str
    section_type: str
    start_position: int
    end_position: int
    confidence: float

class SectionDetector:
    """Detects and extracts sections from academic paper text."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Define section patterns (order matters - more specific first)
        self.section_patterns = [
            # Abstract patterns
            (r'(?i)^(?:abstract|summary)\s*$', 'abstract', 0.95),
            (r'(?i)^(?:abstract|summary)[:.]?\s', 'abstract', 0.90),
            
            # Introduction patterns
            (r'(?i)^(?:1\.?\s*)?introduction\s*$', 'introduction', 0.95),
            (r'(?i)^(?:1\.?\s*)?(?:introduction|background)[:.]?\s', 'introduction', 0.85),
            
            # Methods/Methodology patterns
            (r'(?i)^(?:\d+\.?\s*)?(?:methodology|methods|approach|model)\s*$', 'methodology', 0.95),
            (r'(?i)^(?:\d+\.?\s*)?(?:methodology|methods|approach|model|technique)s?[:.]?\s', 'methodology', 0.85),
            
            # Results patterns
            (r'(?i)^(?:\d+\.?\s*)?(?:results|findings|experiments?|evaluation)\s*$', 'results', 0.95),
            (r'(?i)^(?:\d+\.?\s*)?(?:results|findings|experiments?|evaluation)[:.]?\s', 'results', 0.85),
            
            # Discussion patterns
            (r'(?i)^(?:\d+\.?\s*)?discussion\s*$', 'discussion', 0.95),
            (r'(?i)^(?:\d+\.?\s*)?discussion[:.]?\s', 'discussion', 0.85),
            
            # Conclusion patterns
            (r'(?i)^(?:\d+\.?\s*)?(?:conclusions?|summary|final remarks)\s*$', 'conclusion', 0.95),
            (r'(?i)^(?:\d+\.?\s*)?(?:conclusions?|summary|final remarks)[:.]?\s', 'conclusion', 0.85),
            
            # References patterns
            (r'(?i)^(?:references|bibliography|citations?)\s*$', 'references', 0.95),
            (r'(?i)^(?:references|bibliography|citations?)[:.]?\s', 'references', 0.90),
            
            # Related work patterns
            (r'(?i)^(?:\d+\.?\s*)?(?:related work|prior work|literature review)\s*$', 'related_work', 0.95),
            (r'(?i)^(?:\d+\.?\s*)?(?:related work|prior work|literature review)[:.]?\s', 'related_work', 0.85),
            
            # Generic numbered sections
            (r'^(?:\d+\.?\s+)([A-Z][^.\n]*?)(?:\s*$)', 'section', 0.70),
        ]
    
    def detect_sections(self, text: str, max_sections: int = 15) -> List[Section]:
        """
        Detect sections in academic paper text.
        
        Args:
            text: The paper text to analyze
            max_sections: Maximum number of sections to detect
            
        Returns:
            List of detected sections
        """
        if not text:
            return []
        
        # Split text into potential section headers (lines that might be titles)
        lines = text.split('\n')
        sections = []
        
        # Find section boundaries
        section_boundaries = self._find_section_boundaries(lines)
        
        if self.verbose:
            logger.debug(f"Found {len(section_boundaries)} potential section boundaries")
        
        # Extract sections based on boundaries
        for i, (start_line, section_type, title, confidence) in enumerate(section_boundaries[:max_sections]):
            # Determine end position
            if i + 1 < len(section_boundaries):
                end_line = section_boundaries[i + 1][0]
            else:
                end_line = len(lines)
            
            # Extract content
            content_lines = lines[start_line + 1:end_line]
            content = '\n'.join(content_lines).strip()
            
            # Calculate positions in original text
            start_pos = sum(len(line) + 1 for line in lines[:start_line])
            end_pos = sum(len(line) + 1 for line in lines[:end_line])
            
            section = Section(
                title=title,
                content=content,
                section_type=section_type,
                start_position=start_pos,
                end_position=end_pos,
                confidence=confidence
            )
            
            sections.append(section)
        
        if self.verbose:
            logger.debug(f"Detected {len(sections)} sections")
            for section in sections:
                logger.debug(f"  {section.section_type}: {section.title[:50]}...")
        
        return sections
    
    def _find_section_boundaries(self, lines: List[str]) -> List[Tuple[int, str, str, float]]:
        """Find potential section header lines and classify them."""
        boundaries = []
        
        for line_idx, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip empty lines or very short lines
            if len(line_stripped) < 2:
                continue
            
            # Check against all patterns
            for pattern, section_type, base_confidence in self.section_patterns:
                match = re.match(pattern, line_stripped)
                if match:
                    # Extract title
                    if match.groups():
                        title = match.group(1).strip()
                    else:
                        title = line_stripped
                    
                    # Adjust confidence based on context
                    confidence = self._calculate_confidence(
                        line_stripped, line_idx, lines, base_confidence
                    )
                    
                    boundaries.append((line_idx, section_type, title, confidence))
                    break  # Use first matching pattern
        
        # Sort by line position and filter overlapping/low-confidence matches
        boundaries = sorted(boundaries, key=lambda x: x[0])
        boundaries = self._filter_boundaries(boundaries)
        
        return boundaries
    
    def _calculate_confidence(self, line: str, line_idx: int, lines: List[str], base_confidence: float) -> float:
        """Calculate confidence score for a potential section header."""
        confidence = base_confidence
        
        # Boost confidence for typical academic section formatting
        if re.match(r'^\d+\.?\s+[A-Z]', line):  # Numbered sections
            confidence += 0.05
        
        if line.isupper() and len(line.split()) <= 5:  # ALL CAPS short titles
            confidence += 0.05
        
        if re.match(r'^[A-Z][^.]*$', line) and len(line.split()) <= 6:  # Title case
            confidence += 0.03
        
        # Reduce confidence for very long lines (probably not headers)
        if len(line.split()) > 10:
            confidence -= 0.10
        
        # Check surrounding context
        if line_idx > 0:
            prev_line = lines[line_idx - 1].strip()
            if not prev_line:  # Blank line before (good for headers)
                confidence += 0.05
        
        if line_idx < len(lines) - 1:
            next_line = lines[line_idx + 1].strip()
            if not next_line:  # Blank line after (good for headers)
                confidence += 0.05
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _filter_boundaries(self, boundaries: List[Tuple[int, str, str, float]]) -> List[Tuple[int, str, str, float]]:
        """Filter out overlapping or low-confidence boundaries."""
        if not boundaries:
            return []
        
        # Remove duplicates and low-confidence matches
        filtered = []
        seen_types = set()
        
        for boundary in boundaries:
            line_idx, section_type, title, confidence = boundary
            
            # Skip very low confidence matches
            if confidence < 0.5:
                continue
            
            # For certain important sections, allow duplicates with higher confidence
            if section_type in seen_types and section_type not in ['section']:
                # Keep if significantly higher confidence
                existing = next((b for b in filtered if b[1] == section_type), None)
                if existing and confidence > existing[3] + 0.1:
                    # Replace existing with higher confidence match
                    filtered = [b for b in filtered if b[1] != section_type]
                    filtered.append(boundary)
                    continue
                else:
                    continue  # Skip lower confidence duplicate
            
            seen_types.add(section_type)
            filtered.append(boundary)
        
        return sorted(filtered, key=lambda x: x[0])
    
    def analyze_paper_structure(self, sections: List[Section]) -> Dict[str, Any]:
        """Analyze the overall structure of the detected sections."""
        if not sections:
            return {"document_type": "unknown", "structure_quality": "poor"}
        
        section_types = [s.section_type for s in sections]
        section_count = len(sections)
        
        # Determine document type
        if 'abstract' in section_types and 'methodology' in section_types:
            doc_type = "research_paper"
        elif 'introduction' in section_types and 'conclusion' in section_types:
            doc_type = "academic_paper"
        else:
            doc_type = "document"
        
        # Assess structure quality
        expected_sections = ['abstract', 'introduction', 'methodology', 'results', 'conclusion']
        found_expected = sum(1 for s in expected_sections if s in section_types)
        structure_score = found_expected / len(expected_sections)
        
        if structure_score > 0.8:
            quality = "excellent"
        elif structure_score > 0.6:
            quality = "good"
        elif structure_score > 0.4:
            quality = "fair"
        else:
            quality = "poor"
        
        # Calculate average confidence
        avg_confidence = sum(s.confidence for s in sections) / len(sections) if sections else 0
        
        return {
            "document_type": doc_type,
            "structure_quality": quality,
            "structure_score": structure_score,
            "section_count": section_count,
            "section_types": section_types,
            "average_confidence": avg_confidence,
            "has_abstract": 'abstract' in section_types,
            "has_methodology": 'methodology' in section_types,
            "has_results": 'results' in section_types,
            "has_conclusion": 'conclusion' in section_types
        } 