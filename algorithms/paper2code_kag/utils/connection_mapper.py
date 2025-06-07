"""
Connection Mapper for analyzing dependencies and relationships between abstractions.
Supports rule-based pattern matching and LLM-based intelligent analysis.
"""

import re
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Set, Tuple, Optional
from enum import Enum

class ConnectionType(Enum):
    """Types of connections between abstractions."""
    DEPENDENCY = "dependency"           # A requires B
    WORKFLOW = "workflow"              # A feeds into B
    COMPOSITION = "composition"        # A is part of B
    ALTERNATIVE = "alternative"        # A or B (mutually exclusive)
    SEMANTIC = "semantic"              # A relates to B conceptually
    IMPLEMENTATION = "implementation"   # A implements B

@dataclass
class Connection:
    """Represents a connection between two abstractions."""
    source_id: str                    # ID of source abstraction
    target_id: str                    # ID of target abstraction
    connection_type: ConnectionType   # Type of connection
    confidence: float                 # Confidence score (0.0-1.0)
    description: str                  # Human-readable description
    evidence: List[str]              # Supporting evidence/keywords
    detection_method: str            # "rule_based", "llm", "hybrid"
    bidirectional: bool = False      # Whether connection works both ways

@dataclass 
class DependencyChain:
    """Represents a chain of dependencies."""
    chain_id: str
    abstractions: List[str]          # List of abstraction IDs in order
    chain_type: str                  # "linear", "branching", "circular"
    description: str

class ConnectionMapper:
    """Analyzes dependencies and relationships between abstractions."""
    
    def __init__(self, use_mock_llm: bool = True):
        self.use_mock_llm = use_mock_llm
        self.logger = logging.getLogger(__name__)
        
        # Rule-based patterns for different connection types
        self.dependency_patterns = [
            (r"requires?\s+(\w+)", ConnectionType.DEPENDENCY),
            (r"depends?\s+on\s+(\w+)", ConnectionType.DEPENDENCY),
            (r"needs?\s+(\w+)", ConnectionType.DEPENDENCY),
            (r"built\s+on\s+(\w+)", ConnectionType.DEPENDENCY),
            (r"based\s+on\s+(\w+)", ConnectionType.DEPENDENCY),
        ]
        
        self.workflow_patterns = [
            (r"step\s+\d+.*?(\w+).*?step\s+\d+.*?(\w+)", ConnectionType.WORKFLOW),
            (r"first.*?(\w+).*?then.*?(\w+)", ConnectionType.WORKFLOW),
            (r"(\w+)\s+follows?\s+(\w+)", ConnectionType.WORKFLOW),
            (r"after\s+(\w+).*?(\w+)", ConnectionType.WORKFLOW),
            (r"(\w+)\s+feeds?\s+into\s+(\w+)", ConnectionType.WORKFLOW),
        ]
        
        self.composition_patterns = [
            (r"(\w+)\s+contains?\s+(\w+)", ConnectionType.COMPOSITION),
            (r"(\w+)\s+includes?\s+(\w+)", ConnectionType.COMPOSITION),
            (r"(\w+)\s+consists?\s+of\s+(\w+)", ConnectionType.COMPOSITION),
            (r"part\s+of\s+(\w+)", ConnectionType.COMPOSITION),
        ]

    def detect_connections_rule_based(self, abstractions: List[Dict[str, Any]], 
                                    context_text: str) -> List[Connection]:
        """Detect connections using rule-based pattern matching."""
        connections = []
        
        # Create abstraction lookup
        abs_lookup = {abs_info["name"].lower(): abs_info for abs_info in abstractions}
        abs_names = list(abs_lookup.keys())
        
        # Try each pattern type
        pattern_groups = [
            (self.dependency_patterns, "dependency"),
            (self.workflow_patterns, "workflow"), 
            (self.composition_patterns, "composition")
        ]
        
        for patterns, pattern_type in pattern_groups:
            for pattern, conn_type in patterns:
                matches = re.finditer(pattern, context_text.lower(), re.IGNORECASE)
                
                for match in matches:
                    # Extract potential abstraction names from match
                    groups = match.groups()
                    if len(groups) >= 2:
                        source_name = groups[0].strip()
                        target_name = groups[1].strip()
                    elif len(groups) == 1:
                        # Single group - find closest abstraction
                        matched_name = groups[0].strip()
                        source_name = matched_name
                        target_name = self._find_closest_abstraction(matched_name, abs_names)
                    else:
                        continue
                    
                    # Check if both are valid abstractions
                    if source_name in abs_lookup and target_name in abs_lookup:
                        source_abs = abs_lookup[source_name]
                        target_abs = abs_lookup[target_name]
                        
                        connection = Connection(
                            source_id=source_abs["id"] if "id" in source_abs else source_abs["name"],
                            target_id=target_abs["id"] if "id" in target_abs else target_abs["name"],
                            connection_type=conn_type,
                            confidence=0.7,  # Rule-based confidence
                            description=f"{source_abs['name']} {conn_type.value} {target_abs['name']}",
                            evidence=[match.group(0)],
                            detection_method="rule_based"
                        )
                        connections.append(connection)
        
        return connections

    def detect_connections_llm(self, abstractions: List[Dict[str, Any]], 
                             context_text: str) -> List[Connection]:
        """Detect connections using mock LLM analysis."""
        if not self.use_mock_llm:
            return []
            
        # Mock LLM - generate realistic connections based on abstraction types and names
        connections = []
        
        for i, abs1 in enumerate(abstractions):
            for abs2 in abstractions[i+1:]:
                connection = self._mock_llm_analyze_connection(abs1, abs2, context_text)
                if connection:
                    connections.append(connection)
        
        return connections

    def _mock_llm_analyze_connection(self, abs1: Dict[str, Any], abs2: Dict[str, Any], 
                                   context: str) -> Optional[Connection]:
        """Mock LLM analysis of connection between two abstractions."""
        name1 = abs1["name"].lower()
        name2 = abs2["name"].lower()
        type1 = abs1.get("type", "")
        type2 = abs2.get("type", "")
        
        # Mock intelligent connection detection based on patterns
        connection_rules = [
            # Algorithm dependencies
            (lambda: "algorithm" in type1 and "method" in type2, 
             ConnectionType.DEPENDENCY, 0.8, f"{abs1['name']} algorithm uses {abs2['name']} method"),
            
            # Architecture compositions
            (lambda: "architecture" in type1 and ("algorithm" in type2 or "method" in type2),
             ConnectionType.COMPOSITION, 0.7, f"{abs1['name']} architecture contains {abs2['name']}"),
            
            # Workflow sequences
            (lambda: any(word in name1 for word in ["preprocessing", "step", "phase"]) and 
                    any(word in name2 for word in ["processing", "analysis", "extraction"]),
             ConnectionType.WORKFLOW, 0.6, f"{abs1['name']} flows into {abs2['name']}"),
            
            # Neural network dependencies
            (lambda: "neural" in name1 and ("layer" in name2 or "attention" in name2),
             ConnectionType.DEPENDENCY, 0.8, f"{abs1['name']} requires {abs2['name']}"),
            
            # Implementation relationships
            (lambda: "framework" in name1 or "library" in name1,
             ConnectionType.IMPLEMENTATION, 0.7, f"{abs2['name']} implemented using {abs1['name']}"),
        ]
        
        for condition, conn_type, confidence, description in connection_rules:
            try:
                if condition():
                    return Connection(
                        source_id=abs1.get("id", abs1["name"]),
                        target_id=abs2.get("id", abs2["name"]),
                        connection_type=conn_type,
                        confidence=confidence,
                        description=description,
                        evidence=[f"LLM analysis: {name1} -> {name2}"],
                        detection_method="llm"
                    )
            except:
                continue
        
        return None

    def detect_connections_hybrid(self, abstractions: List[Dict[str, Any]], 
                                context_text: str) -> List[Connection]:
        """Detect connections using hybrid rule-based + LLM approach."""
        rule_connections = self.detect_connections_rule_based(abstractions, context_text)
        llm_connections = self.detect_connections_llm(abstractions, context_text)
        
        # Combine and deduplicate
        all_connections = rule_connections + llm_connections
        
        # Deduplicate based on source-target pairs
        seen_pairs = set()
        deduplicated = []
        
        for conn in all_connections:
            pair_key = f"{conn.source_id}->{conn.target_id}-{conn.connection_type.value}"
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                deduplicated.append(conn)
            else:
                # If duplicate, keep the one with higher confidence
                existing_idx = next(i for i, c in enumerate(deduplicated) 
                                  if f"{c.source_id}->{c.target_id}-{c.connection_type.value}" == pair_key)
                if conn.confidence > deduplicated[existing_idx].confidence:
                    deduplicated[existing_idx] = conn
        
        # Mark hybrid connections
        for conn in deduplicated:
            if any(c.source_id == conn.source_id and c.target_id == conn.target_id 
                  for c in rule_connections) and \
               any(c.source_id == conn.source_id and c.target_id == conn.target_id 
                  for c in llm_connections):
                conn.detection_method = "hybrid"
                conn.confidence = min(conn.confidence + 0.1, 1.0)  # Boost hybrid confidence
        
        return deduplicated

    def analyze_dependency_chains(self, connections: List[Connection]) -> List[DependencyChain]:
        """Analyze dependency chains from connections."""
        dependency_connections = [c for c in connections 
                                if c.connection_type == ConnectionType.DEPENDENCY]
        
        chains = []
        
        # Build dependency graph
        graph = {}
        for conn in dependency_connections:
            if conn.source_id not in graph:
                graph[conn.source_id] = []
            graph[conn.source_id].append(conn.target_id)
        
        # Find linear chains
        visited = set()
        chain_id = 0
        
        for start_node in graph:
            if start_node not in visited:
                chain = self._trace_dependency_chain(graph, start_node, visited)
                if len(chain) > 1:
                    chains.append(DependencyChain(
                        chain_id=f"dep_chain_{chain_id}",
                        abstractions=chain,
                        chain_type="linear" if len(set(chain)) == len(chain) else "circular",
                        description=f"Dependency chain: {' -> '.join(chain)}"
                    ))
                    chain_id += 1
        
        return chains

    def _trace_dependency_chain(self, graph: Dict[str, List[str]], 
                              start: str, visited: Set[str]) -> List[str]:
        """Trace a dependency chain from a starting node."""
        chain = [start]
        current = start
        local_visited = {start}
        
        while current in graph and graph[current]:
            # Take first unvisited target
            next_nodes = [n for n in graph[current] if n not in local_visited]
            if not next_nodes:
                break
            
            next_node = next_nodes[0]
            chain.append(next_node)
            local_visited.add(next_node)
            current = next_node
        
        visited.update(local_visited)
        return chain

    def _find_closest_abstraction(self, target: str, abs_names: List[str]) -> str:
        """Find the closest matching abstraction name."""
        target_lower = target.lower()
        
        # Exact match
        if target_lower in abs_names:
            return target_lower
        
        # Partial match
        for name in abs_names:
            if target_lower in name or name in target_lower:
                return name
        
        # Return first as fallback
        return abs_names[0] if abs_names else target_lower

    def create_connection_summary(self, connections: List[Connection], 
                                chains: List[DependencyChain]) -> Dict[str, Any]:
        """Create summary statistics for connections."""
        type_counts = {}
        method_counts = {}
        
        for conn in connections:
            conn_type = conn.connection_type.value
            type_counts[conn_type] = type_counts.get(conn_type, 0) + 1
            
            method = conn.detection_method
            method_counts[method] = method_counts.get(method, 0) + 1
        
        avg_confidence = sum(c.confidence for c in connections) / len(connections) if connections else 0
        
        return {
            "total_connections": len(connections),
            "total_dependency_chains": len(chains),
            "connection_type_distribution": type_counts,
            "detection_method_distribution": method_counts,
            "average_confidence": round(avg_confidence, 3),
            "has_circular_dependencies": any(c.chain_type == "circular" for c in chains),
            "longest_chain_length": max(len(c.abstractions) for c in chains) if chains else 0
        } 