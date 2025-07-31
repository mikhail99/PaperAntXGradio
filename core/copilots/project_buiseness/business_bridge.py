import dspy
import os
import requests
import json
from typing import Dict, List, Any
#from mem0 import Memory
import time
from collections import Counter, defaultdict

# Global flag to control LLM usage
USE_MOCK_LLM = False  # Set to False for real LLM

def get_llm():
    """Get LLM instance based on USE_MOCK_LLM flag."""
    if USE_MOCK_LLM:
        from core.proposal_agent_pf_dspy.parrot import MockLM
        return MockLM()
    else:
        provider = os.getenv("DEFAULT_LLM_PROVIDER", "ollama").lower()
        if provider == "ollama":
            model_name = os.getenv("OLLAMA_MODEL", "qwen3:4b")
            return dspy.LM(f'ollama_chat/{model_name}', api_base='http://localhost:11434', api_key='')
        elif provider == "openai":
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
            return dspy.OpenAI(model=model_name, max_tokens=4000)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

def is_using_mock():
    """Return True if we're using mock services."""
    return USE_MOCK_LLM

def get_paperqa_service():
    """Get PaperQA service instance based on USE_MOCK_LLM flag."""
    if USE_MOCK_LLM:
        from core.proposal_agent_pf_dspy.parrot import MockPaperQAService
        service = MockPaperQAService()
    else:
        from core.paperqa_service import PaperQAService
        service = PaperQAService()
    
    # Wrap with sync interface for nodes
    class SyncPaperQAWrapper:
        def __init__(self, async_service):
            self.async_service = async_service
        
        def query_documents(self, collection: str, query: str) -> dict:
            """Sync wrapper for async query_documents."""
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(
                self.async_service.query_documents(collection, query)
            )
    
    return SyncPaperQAWrapper(service)


class PatentQueryGenerator(dspy.Signature):
    """
    You are a patent research expert. Extract key technical concepts from a project proposal 
    and generate effective patent search queries. Focus on the core technology, applications, 
    and technical methods described.
    """
    project_proposal = dspy.InputField(desc="The technical project proposal to analyze.")
    patent_queries = dspy.OutputField(desc="List of 3-5 targeted patent search queries, separated by semicolons.")

class PatentLandscapeAnalyzer:
    """Analyzes patent data and provides aggregated business intelligence."""
    
    def __init__(self):
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    def search_google_patents(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search patents using Google Custom Search API (free tier available)."""
        # Note: This would require a Google Custom Search API key
        # For now, we'll simulate the response structure
        return self._simulate_patent_results(query, max_results)
    
    def _simulate_patent_results(self, query: str, max_results: int) -> List[Dict]:
        """Simulate patent search results for demonstration."""
        # In production, this would call actual patent APIs
        simulated_results = []
        companies = ["Apple Inc.", "Google LLC", "Microsoft Corp.", "Samsung Electronics", "IBM Corp.", "Intel Corp."]
        
        for i in range(max_results):
            result = {
                "title": f"Patent related to {query} - Method {i+1}",
                "patent_number": f"US{10000000 + i}",
                "assignee": companies[i % len(companies)],
                "filing_date": f"202{i % 4}-0{(i % 12) + 1}-15",
                "publication_date": f"202{(i % 4) + 1}-0{(i % 12) + 1}-15",
                "abstract": f"This patent describes innovative methods for {query.lower()} with improved efficiency and novel applications.",
                "classification": f"G06F {i+1}/00"
            }
            simulated_results.append(result)
        
        return simulated_results
    
    def aggregate_patent_data(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate patent results into business intelligence."""
        if not all_results:
            return {"total_patents": 0, "companies": {}, "trends": {}, "recent_activity": []}
        
        # Count patents by company
        company_counts = Counter([patent.get("assignee", "Unknown") for patent in all_results])
        
        # Analyze filing trends by year
        year_trends = defaultdict(int)
        for patent in all_results:
            filing_date = patent.get("filing_date", "")
            if filing_date:
                year = filing_date.split("-")[0]
                year_trends[year] += 1
        
        # Recent activity (last 2 years)
        recent_patents = [p for p in all_results if p.get("filing_date", "").startswith(("2023", "2024", "2025"))]
        
        # Technology classifications
        classifications = Counter([patent.get("classification", "Unknown") for patent in all_results])
        
        return {
            "total_patents": len(all_results),
            "companies": dict(company_counts.most_common(10)),
            "year_trends": dict(sorted(year_trends.items())),
            "recent_activity": len(recent_patents),
            "top_classifications": dict(classifications.most_common(5)),
            "key_players": list(company_counts.keys())[:5],
            "patent_density": len(all_results) / max(len(set([p.get("assignee") for p in all_results])), 1)
        }

class BusinessTechnicalBridgeQA(dspy.Signature):
    """
    You are an expert business analyst and technical lead.
    Your task is to review a technical project proposal and assess its business viability.
    Analyze the proposal and provide a comprehensive review covering technical feasibility,
    business impact, competitive landscape, and resource planning.
    
    Use the patent landscape data to inform your competitive analysis and IP risk assessment.
    """
    project_proposal = dspy.InputField(desc="The technical project proposal.")
    patent_landscape = dspy.InputField(desc="Patent landscape analysis data.")
    review = dspy.OutputField(desc="A comprehensive business and technical review including patent landscape insights.")

class BusinessTechnicalBridgeAgent(dspy.Module):
    """A ReAct agent for bridging technical proposals with business impact, enhanced with patent intelligence."""

    def __init__(self, memory = None):
        super().__init__()
        self.patent_analyzer = PatentLandscapeAnalyzer()
        self.query_generator = dspy.Predict(PatentQueryGenerator)
        self.predictor = dspy.Predict(BusinessTechnicalBridgeQA)

    def forward(self, project_proposal: str):
        """Processes the project proposal with patent landscape analysis."""
        
        # Step 1: Generate patent search queries
        query_result = self.query_generator(project_proposal=project_proposal)
        queries = [q.strip() for q in query_result.patent_queries.split(";")]
        
        # Step 2: Search patents for each query
        all_patent_results = []
        for query in queries:
            if query:  # Skip empty queries
                results = self.patent_analyzer.search_google_patents(query, max_results=10)
                all_patent_results.extend(results)
        
        # Step 3: Aggregate patent data
        patent_landscape = self.patent_analyzer.aggregate_patent_data(all_patent_results)
        
        # Step 4: Generate comprehensive review with patent insights
        result = self.predictor(
            project_proposal=project_proposal,
            patent_landscape=json.dumps(patent_landscape, indent=2)
        )
        
        # Add patent data to result for UI display
        result.patent_data = patent_landscape
        result.patent_queries = queries
        
        return result

def create_business_bridge_service():
    """Factory function to create the BusinessTechnicalBridgeAgent."""
    # --- DSPy Model Configuration ---

    llm = get_llm()
    dspy.configure(lm=llm)
    print(f"--- Business Bridge DSPy configured with: {llm.__class__.__name__} ---")
    
    # Here we would initialize memory and other dependencies.
    # For now, we create a simple agent.
    agent = BusinessTechnicalBridgeAgent()
    return agent 