"""
Connection Planning Nodes for analyzing dependencies and relationships between abstractions.
Part of Iteration 4: Extended Planning Stage 3 (Connection Planning).
"""

import json
import os
import logging
from typing import Dict, Any, List, TypedDict
from dataclasses import asdict

# Import PocketFlow base class
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pocketflow import Node

# Import connection analysis utilities
from utils.connection_mapper import ConnectionMapper, Connection, DependencyChain, ConnectionType

# Type definitions for better structure
class ConnectionInfo(TypedDict):
    """Connection information structure."""
    source_id: str
    target_id: str
    connection_type: str
    confidence: float
    description: str
    evidence: List[str]
    detection_method: str
    bidirectional: bool

class DependencyChainInfo(TypedDict):
    """Dependency chain information structure."""
    chain_id: str
    abstractions: List[str]
    chain_type: str
    description: str

class ConnectionPlanningSharedState(TypedDict, total=False):
    """Shared state structure for connection planning."""
    # Input from abstraction planning
    categorized_abstractions: List[Dict[str, Any]]
    raw_abstractions: List[Dict[str, Any]]
    abstraction_summary: Dict[str, Any]
    
    # Connection analysis results
    detected_connections: List[ConnectionInfo]
    dependency_chains: List[DependencyChainInfo]
    connection_summary: Dict[str, Any]
    connection_matrix: Dict[str, List[str]]
    workflow_sequences: List[List[str]]
    
    # Metadata
    total_connections_found: int
    connection_analysis_method: str
    connection_planning_completed: bool
    connection_planning_flow_status: str
    connection_plan_saved: bool
    connection_plan_file: str
    connection_plan_file_size: int

class AnalyzeDependenciesNode(Node):
    """Analyzes dependencies between abstractions."""
    
    def __init__(self, use_mock_llm: bool = True, max_retries: int = 3, wait: int = 1):
        super().__init__(max_retries=max_retries, wait=wait)
        self.use_mock_llm = use_mock_llm
        self.connection_mapper = ConnectionMapper(use_mock_llm=use_mock_llm)
        self.logger = logging.getLogger(__name__)
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare abstractions data for dependency analysis."""
        if "categorized_abstractions" not in shared:
            raise ValueError("Missing categorized_abstractions in shared state")
        
        categorized_abs = shared["categorized_abstractions"]
        if not categorized_abs:
            raise ValueError("No categorized abstractions found")
        
        # Collect all text context from abstractions
        context_text = ""
        for cat_abs in categorized_abs:
            abs_info = cat_abs["abstraction"]
            context_text += f" {abs_info.get('description', '')} {abs_info.get('context', '')}"
        
        self.logger.info(f"Analyzing dependencies for {len(categorized_abs)} abstractions")
        
        return {
            "abstractions": [cat_abs["abstraction"] for cat_abs in categorized_abs],
            "context_text": context_text.strip()
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dependency analysis."""
        abstractions = prep_res["abstractions"]
        context_text = prep_res["context_text"]
        
        # Add IDs to abstractions if missing
        for i, abs_info in enumerate(abstractions):
            if "id" not in abs_info:
                abs_info["id"] = f"abs_{i}_{abs_info['name'].replace(' ', '_').lower()}"
        
        # Detect connections using hybrid approach
        connections = self.connection_mapper.detect_connections_hybrid(abstractions, context_text)
        
        # Analyze dependency chains
        dependency_chains = self.connection_mapper.analyze_dependency_chains(connections)
        
        # Create connection matrix for visualization
        connection_matrix = self._create_connection_matrix(abstractions, connections)
        
        self.logger.info(f"Found {len(connections)} connections and {len(dependency_chains)} dependency chains")
        
        return {
            "connections": connections,
            "dependency_chains": dependency_chains,
            "connection_matrix": connection_matrix,
            "analysis_stats": {
                "total_connections": len(connections),
                "dependency_connections": len([c for c in connections if c.connection_type == ConnectionType.DEPENDENCY]),
                "workflow_connections": len([c for c in connections if c.connection_type == ConnectionType.WORKFLOW]),
                "composition_connections": len([c for c in connections if c.connection_type == ConnectionType.COMPOSITION])
            }
        }
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Save dependency analysis results to shared state."""
        connections = exec_res["connections"]
        dependency_chains = exec_res["dependency_chains"]
        
        # Convert to serializable format
        connection_info_list = []
        for conn in connections:
            connection_info = ConnectionInfo(
                source_id=conn.source_id,
                target_id=conn.target_id,
                connection_type=conn.connection_type.value,
                confidence=conn.confidence,
                description=conn.description,
                evidence=conn.evidence,
                detection_method=conn.detection_method,
                bidirectional=conn.bidirectional
            )
            connection_info_list.append(connection_info)
        
        # Convert dependency chains
        chain_info_list = []
        for chain in dependency_chains:
            chain_info = DependencyChainInfo(
                chain_id=chain.chain_id,
                abstractions=chain.abstractions,
                chain_type=chain.chain_type,
                description=chain.description
            )
            chain_info_list.append(chain_info)
        
        # Update shared state
        shared["detected_connections"] = connection_info_list
        shared["dependency_chains"] = chain_info_list
        shared["connection_matrix"] = exec_res["connection_matrix"]
        shared["total_connections_found"] = len(connections)
        shared["connection_analysis_method"] = "hybrid"
        
        self.logger.info(f"Dependency analysis completed: {len(connections)} connections, {len(dependency_chains)} chains")
        
        return "default"
    
    def _create_connection_matrix(self, abstractions: List[Dict[str, Any]], 
                                connections: List[Connection]) -> Dict[str, List[str]]:
        """Create connection matrix showing relationships."""
        matrix = {}
        
        # Initialize matrix
        for abs_info in abstractions:
            abs_id = abs_info["id"]
            matrix[abs_id] = []
        
        # Populate matrix with connections
        for conn in connections:
            if conn.source_id in matrix:
                target_entry = f"{conn.target_id}:{conn.connection_type.value}"
                matrix[conn.source_id].append(target_entry)
        
        return matrix

class MapConnectionsNode(Node):
    """Maps workflow connections and relationships between abstractions."""
    
    def __init__(self, max_retries: int = 3, wait: int = 1):
        super().__init__(max_retries=max_retries, wait=wait)
        self.logger = logging.getLogger(__name__)
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare connection data for workflow mapping."""
        if "detected_connections" not in shared:
            raise ValueError("Missing detected_connections in shared state")
        
        connections = shared["detected_connections"]
        abstractions = [cat_abs["abstraction"] for cat_abs in shared.get("categorized_abstractions", [])]
        
        return {
            "connections": connections,
            "abstractions": abstractions,
            "dependency_chains": shared.get("dependency_chains", [])
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow connection mapping."""
        connections = prep_res["connections"]
        abstractions = prep_res["abstractions"]
        dependency_chains = prep_res["dependency_chains"]
        
        # Identify workflow sequences
        workflow_sequences = self._identify_workflow_sequences(connections, abstractions)
        
        # Analyze connection patterns
        connection_patterns = self._analyze_connection_patterns(connections)
        
        # Create workflow graph
        workflow_graph = self._create_workflow_graph(connections, abstractions)
        
        # Generate insights
        workflow_insights = self._generate_workflow_insights(workflow_sequences, dependency_chains)
        
        self.logger.info(f"Mapped {len(workflow_sequences)} workflow sequences")
        
        return {
            "workflow_sequences": workflow_sequences,
            "connection_patterns": connection_patterns,
            "workflow_graph": workflow_graph,
            "workflow_insights": workflow_insights
        }
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Save workflow mapping results to shared state."""
        shared["workflow_sequences"] = exec_res["workflow_sequences"]
        shared["connection_patterns"] = exec_res["connection_patterns"]
        shared["workflow_graph"] = exec_res["workflow_graph"]
        shared["workflow_insights"] = exec_res["workflow_insights"]
        
        self.logger.info(f"Workflow mapping completed: {len(exec_res['workflow_sequences'])} sequences")
        
        return "default"
    
    def _identify_workflow_sequences(self, connections: List[ConnectionInfo], 
                                   abstractions: List[Dict[str, Any]]) -> List[List[str]]:
        """Identify workflow sequences from connections."""
        workflow_connections = [c for c in connections if c["connection_type"] == "workflow"]
        
        if not workflow_connections:
            return []
        
        # Build workflow graph
        graph = {}
        for conn in workflow_connections:
            source = conn["source_id"]
            target = conn["target_id"]
            if source not in graph:
                graph[source] = []
            graph[source].append(target)
        
        # Find sequences
        sequences = []
        visited = set()
        
        for start_node in graph:
            if start_node not in visited:
                sequence = self._trace_workflow_sequence(graph, start_node, visited)
                if len(sequence) > 1:
                    sequences.append(sequence)
        
        return sequences
    
    def _trace_workflow_sequence(self, graph: Dict[str, List[str]], 
                               start: str, visited: set) -> List[str]:
        """Trace a workflow sequence from a starting node."""
        sequence = [start]
        current = start
        local_visited = {start}
        
        while current in graph and graph[current]:
            # Take first unvisited target
            next_nodes = [n for n in graph[current] if n not in local_visited]
            if not next_nodes:
                break
            
            next_node = next_nodes[0]
            sequence.append(next_node)
            local_visited.add(next_node)
            current = next_node
        
        visited.update(local_visited)
        return sequence
    
    def _analyze_connection_patterns(self, connections: List[ConnectionInfo]) -> Dict[str, Any]:
        """Analyze patterns in connections."""
        patterns = {
            "most_connected_abstractions": {},
            "connection_type_frequency": {},
            "bidirectional_connections": 0,
            "circular_references": []
        }
        
        # Count connections per abstraction
        for conn in connections:
            source = conn["source_id"]
            target = conn["target_id"]
            conn_type = conn["connection_type"]
            
            # Most connected
            patterns["most_connected_abstractions"][source] = patterns["most_connected_abstractions"].get(source, 0) + 1
            patterns["most_connected_abstractions"][target] = patterns["most_connected_abstractions"].get(target, 0) + 1
            
            # Connection type frequency
            patterns["connection_type_frequency"][conn_type] = patterns["connection_type_frequency"].get(conn_type, 0) + 1
            
            # Bidirectional
            if conn["bidirectional"]:
                patterns["bidirectional_connections"] += 1
        
        return patterns
    
    def _create_workflow_graph(self, connections: List[ConnectionInfo], 
                             abstractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create workflow graph representation."""
        nodes = {}
        edges = []
        
        # Create nodes
        for abs_info in abstractions:
            abs_id = abs_info["id"]
            nodes[abs_id] = {
                "id": abs_id,
                "name": abs_info["name"],
                "type": abs_info.get("type", "unknown"),
                "category": abs_info.get("category", "unknown")
            }
        
        # Create edges
        for conn in connections:
            edge = {
                "source": conn["source_id"],
                "target": conn["target_id"],
                "type": conn["connection_type"],
                "confidence": conn["confidence"],
                "description": conn["description"]
            }
            edges.append(edge)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "graph_type": "directed"
            }
        }
    
    def _generate_workflow_insights(self, workflow_sequences: List[List[str]], 
                                  dependency_chains: List[DependencyChainInfo]) -> Dict[str, Any]:
        """Generate insights about workflow patterns."""
        insights = {
            "total_sequences": len(workflow_sequences),
            "longest_sequence_length": max(len(seq) for seq in workflow_sequences) if workflow_sequences else 0,
            "average_sequence_length": sum(len(seq) for seq in workflow_sequences) / len(workflow_sequences) if workflow_sequences else 0,
            "total_dependency_chains": len(dependency_chains),
            "workflow_complexity": "low"  # Default
        }
        
        # Determine complexity
        if insights["longest_sequence_length"] > 5 or len(dependency_chains) > 3:
            insights["workflow_complexity"] = "high"
        elif insights["longest_sequence_length"] > 3 or len(dependency_chains) > 1:
            insights["workflow_complexity"] = "medium"
        
        return insights

class SaveConnectionsNode(Node):
    """Saves connection plan with metadata to JSON file."""
    
    def __init__(self, output_dir: str = "output", max_retries: int = 3, wait: int = 1):
        super().__init__(max_retries=max_retries, wait=wait)
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare connection data for saving."""
        required_keys = ["detected_connections", "dependency_chains", "connection_matrix"]
        
        for key in required_keys:
            if key not in shared:
                raise ValueError(f"Missing {key} in shared state")
        
        return {
            "connections": shared["detected_connections"],
            "dependency_chains": shared["dependency_chains"],
            "connection_matrix": shared["connection_matrix"],
            "workflow_sequences": shared.get("workflow_sequences", []),
            "workflow_graph": shared.get("workflow_graph", {}),
            "workflow_insights": shared.get("workflow_insights", {}),
            "connection_patterns": shared.get("connection_patterns", {}),
            "total_connections": shared.get("total_connections_found", 0),
            "analysis_method": shared.get("connection_analysis_method", "unknown")
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Execute saving of connection plan."""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create connection summary
        summary = self._create_connection_summary(prep_res)
        
        # Prepare complete connection plan
        connection_plan = {
            "connection_planning_results": {
                "detected_connections": prep_res["connections"],
                "dependency_chains": prep_res["dependency_chains"],
                "workflow_sequences": prep_res["workflow_sequences"],
                "connection_matrix": prep_res["connection_matrix"],
                "workflow_graph": prep_res["workflow_graph"],
                "connection_summary": summary,
                "workflow_insights": prep_res["workflow_insights"],
                "connection_patterns": prep_res["connection_patterns"],
                "analysis_metadata": {
                    "total_connections_analyzed": prep_res["total_connections"],
                    "analysis_method": prep_res["analysis_method"],
                    "analysis_timestamp": self._get_timestamp()
                }
            },
            "previous_planning": {
                "section_planning": "Available in planning_results.json",
                "abstraction_planning": "Available in abstraction_plan.json"
            }
        }
        
        # Save to file
        output_file = os.path.join(self.output_dir, "connection_plan.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(connection_plan, f, indent=2, ensure_ascii=False)
        
        file_size = os.path.getsize(output_file)
        
        self.logger.info(f"Connection plan saved to {output_file} ({file_size} bytes)")
        
        return {
            "output_file": output_file,
            "file_size": file_size,
            "summary": summary
        }
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Update shared state with save results."""
        shared["connection_plan_saved"] = True
        shared["connection_plan_file"] = exec_res["output_file"]
        shared["connection_plan_file_size"] = exec_res["file_size"]
        shared["connection_summary"] = exec_res["summary"]
        shared["connection_planning_completed"] = True
        shared["connection_planning_flow_status"] = "success"
        
        self.logger.info("Connection planning completed successfully")
        
        return "default"
    
    def _create_connection_summary(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of connection analysis."""
        connections = prep_res["connections"]
        
        # Type distribution
        type_counts = {}
        method_counts = {}
        confidence_sum = 0
        
        for conn in connections:
            conn_type = conn["connection_type"]
            type_counts[conn_type] = type_counts.get(conn_type, 0) + 1
            
            method = conn["detection_method"]
            method_counts[method] = method_counts.get(method, 0) + 1
            
            confidence_sum += conn["confidence"]
        
        avg_confidence = confidence_sum / len(connections) if connections else 0
        
        return {
            "total_connections": len(connections),
            "total_dependency_chains": len(prep_res["dependency_chains"]),
            "total_workflow_sequences": len(prep_res["workflow_sequences"]),
            "connection_type_distribution": type_counts,
            "detection_method_distribution": method_counts,
            "average_confidence": round(avg_confidence, 3),
            "workflow_complexity": prep_res["workflow_insights"].get("workflow_complexity", "unknown"),
            "longest_sequence_length": prep_res["workflow_insights"].get("longest_sequence_length", 0)
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat() 