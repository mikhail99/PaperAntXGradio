# Patent Intelligence Feature Implementation Guide

## Overview

This guide explains how to implement the Patent Intelligence feature in PaperAntGradio, which provides real-time patent landscape analysis for research proposals. The feature integrates with USPTO PatentsView API to provide business intelligence insights.

**Current Status**: Core files `business_bridge.py` and `ui_manager_review.py` are already implemented with simulated data. This guide covers completing the USPTO API integration and testing.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Current Implementation Status](#current-implementation-status)
4. [Completing the Implementation](#completing-the-implementation)
5. [Testing](#testing)
6. [Deployment Considerations](#deployment-considerations)

## Architecture Overview

```
User Proposal → Patent Query Generation → USPTO API → Data Aggregation → Visualization
                      ↓                        ↓            ↓               ↓
               DSPy LLM Processing    → Real Patent Data → Statistics → Plotly Charts
```

### Key Components

1. **PatentQueryGenerator**: DSPy module that extracts patent-relevant keywords from proposals
2. **PatentLandscapeAnalyzer**: Main service that searches patents and aggregates data
3. **BusinessTechnicalBridgeAgent**: Orchestrates the entire analysis workflow
4. **Visualization Layer**: Plotly-based charts integrated with Gradio

## Prerequisites

### Dependencies

```bash
pip install plotly requests dspy-ai gradio
```

### Environment Variables

```bash
# Optional: If using other LLM providers
DEFAULT_LLM_PROVIDER=ollama  # or openai
OLLAMA_MODEL=gemma3:4b       # or your preferred model
OPENAI_MODEL=gpt-4o          # if using OpenAI
```

## Current Implementation Status

### ✅ Already Implemented

1. **`core/business_intelligence/business_bridge.py`**: Core logic with simulated data
2. **`ui/ui_manager_review.py`**: UI components with Plotly visualization
3. **Basic error handling and visualization framework**

### 🔄 Needs Completion

1. **Real USPTO API integration** in `PatentLandscapeAnalyzer.search_uspto_patents()`
2. **API response processing** in `PatentLandscapeAnalyzer.process_patent_data()`
3. **Error handling for API failures**

## Completing the Implementation

### Step 1: Update PatentLandscapeAnalyzer for Real API Calls

Replace the `_simulate_patent_results` method in `business_bridge.py` with real USPTO API integration:

```python
def search_uspto_patents(self, query: str, max_results: int = 50) -> Dict[str, Any]:
    """Search patents using USPTO PatentsView API."""
    try:
        payload = {
            "q": {"_text_any": {"patent_title": query}},
            "f": [
                "patent_number", "patent_title", "assignee_organization", 
                "application_date", "grant_date", "cpc_subgroup_id"
            ],
            "o": {"per_page": min(max_results, 100)}
        }
        
        response = requests.post(self.base_url, json=payload, timeout=10)
        response.raise_for_status()
        
        time.sleep(self.rate_limit_delay)  # Rate limiting
        
        return {
            "data": response.json(),
            "source": "USPTO PatentsView API",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query
        }
        
    except Exception as e:
        print(f"USPTO API error for query '{query}': {e}")
        return {
            "error": str(e),
            "status": "API_ERROR",
            "message": f"Unable to fetch patent data: {str(e)}",
            "query": query
        }

def process_patent_data(self, api_response: Dict) -> List[Dict]:
    """Process USPTO API response into standardized format."""
    if "error" in api_response:
        return []
        
    patents = api_response.get("data", {}).get("patents", [])
    if not patents:
        return []
        
    processed_results = []
    
    for patent in patents:
        # Extract assignee organization (handle different API response structures)
        assignees = patent.get("assignees", [])
        assignee_name = "Unknown"
        if assignees and len(assignees) > 0:
            assignee_name = assignees[0].get("assignee_organization", "Unknown")
        
        # Extract CPC classification
        cpc_groups = patent.get("cpc_subgroups", [])
        classification = ""
        if cpc_groups and len(cpc_groups) > 0:
            classification = cpc_groups[0].get("cpc_subgroup_id", "")
        
        result = {
            "title": patent.get("patent_title", ""),
            "patent_number": patent.get("patent_number", ""),
            "assignee": assignee_name,
            "filing_date": patent.get("application_date", ""),
            "publication_date": patent.get("grant_date", ""),
            "classification": classification
        }
        processed_results.append(result)
        
    return processed_results
```

### Step 2: Update Error Handling in UI

Modify `create_patent_visualizations` in `ui_manager_review.py` to handle API errors:

```python
def create_patent_visualizations(patent_data):
    """Create Plotly visualizations for patent landscape data."""
    
    # Handle error cases
    if not patent_data or patent_data.get("error"):
        fig = go.Figure()
        error_msg = patent_data.get("message", "No patent data available") if patent_data else "No patent data available"
        fig.add_annotation(
            text=f"Patent Analysis Unavailable<br>{error_msg}",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#d62728"), align="center"
        )
        fig.update_layout(height=400, template="plotly_white")
        return fig
    
    if patent_data.get("total_patents", 0) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No patents found for this query<br>Try different keywords or broader terms",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#666"), align="center"
        )
        fig.update_layout(height=400, template="plotly_white")
        return fig
    
    # Rest of visualization code...
```

### Step 3: Test the Implementation

Create a test script to validate the USPTO API integration.

## Testing

### USPTO API Test Script

Create `test_patent_api.py` to test the implementation:

```python
#!/usr/bin/env python3
"""
Test script for USPTO PatentsView API integration
Tests patent search for "SPAD" keyword and aggregates by company and year
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.business_intelligence.business_bridge import PatentLandscapeAnalyzer
import json
from pprint import pprint

def test_spad_patents():
    """Test patent search for SPAD (Single Photon Avalanche Diode) technology."""
    print("🔍 Testing USPTO PatentsView API with keyword: SPAD")
    print("=" * 60)
    
    # Initialize the analyzer
    analyzer = PatentLandscapeAnalyzer()
    
    # Search for SPAD patents
    print("1. Searching for SPAD patents...")
    api_response = analyzer.search_uspto_patents("SPAD", max_results=50)
    
    if "error" in api_response:
        print(f"❌ API Error: {api_response['message']}")
        return
    
    print(f"✅ API call successful. Source: {api_response.get('source', 'Unknown')}")
    print(f"📅 Timestamp: {api_response.get('timestamp', 'Unknown')}")
    
    # Process the patent data
    print("\n2. Processing patent data...")
    processed_patents = analyzer.process_patent_data(api_response)
    print(f"📊 Processed {len(processed_patents)} patents")
    
    if not processed_patents:
        print("⚠️  No patents found or processing failed")
        return
    
    # Aggregate the data
    print("\n3. Aggregating patent statistics...")
    aggregated_data = analyzer.aggregate_patent_data(processed_patents)
    
    print(f"\n📈 Patent Statistics for 'SPAD':")
    print("=" * 40)
    print(f"Total Patents: {aggregated_data['total_patents']}")
    print(f"Recent Activity (2022+): {aggregated_data['recent_activity']}")
    
    print(f"\n🏢 Top Companies:")
    for i, (company, count) in enumerate(aggregated_data['companies'].items(), 1):
        print(f"  {i}. {company}: {count} patents")
        if i >= 5:  # Show top 5
            break
    
    print(f"\n📅 Patents by Year:")
    for year in sorted(aggregated_data['year_trends'].keys()):
        count = aggregated_data['year_trends'][year]
        print(f"  {year}: {count} patents")
    
    print(f"\n🔬 Top Technology Classifications:")
    for i, (classification, count) in enumerate(aggregated_data['top_classifications'].items(), 1):
        print(f"  {i}. {classification}: {count} patents")
        if i >= 3:  # Show top 3
            break
    
    # Save detailed results
    print(f"\n💾 Saving detailed results...")
    results = {
        "query": "SPAD",
        "api_response_metadata": {
            "source": api_response.get('source'),
            "timestamp": api_response.get('timestamp'),
            "total_processed": len(processed_patents)
        },
        "aggregated_statistics": aggregated_data,
        "sample_patents": processed_patents[:5]  # First 5 patents as samples
    }
    
    with open("spad_patent_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to 'spad_patent_test_results.json'")
    print(f"\n🎯 Test completed successfully!")

def test_api_error_handling():
    """Test error handling with invalid query."""
    print("\n🧪 Testing error handling...")
    print("=" * 30)
    
    analyzer = PatentLandscapeAnalyzer()
    
    # Test with empty query
    response = analyzer.search_uspto_patents("", max_results=10)
    if "error" in response:
        print("✅ Error handling works correctly for empty query")
    else:
        print("⚠️  Expected error for empty query, but got response")
    
    # Test data processing with error response
    processed = analyzer.process_patent_data(response)
    aggregated = analyzer.aggregate_patent_data(processed)
    
    print(f"✅ Error handling complete. Empty aggregation: {aggregated['total_patents'] == 0}")

if __name__ == "__main__":
    print("🚀 USPTO PatentsView API Test Suite")
    print("Testing SPAD (Single Photon Avalanche Diode) patent landscape")
    print("=" * 70)
    
    try:
        test_spad_patents()
        test_api_error_handling()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Test suite completed.")
```

### Running the Test

```bash
# From the project root directory
python test_patent_api.py
```

### Expected Output

The test should produce:
1. API call results showing successful connection to USPTO
2. Aggregated statistics for SPAD patents by company and year
3. A JSON file with detailed results
4. Error handling verification

## Questions to Address

Based on your request, here are some clarifying questions:

1. **API Rate Limits**: Should we implement more aggressive rate limiting for the prototype, or is 1 request/second sufficient?

2. **Data Caching**: Do you want to implement basic caching (e.g., file-based) for the test results to avoid repeated API calls during development?

3. **Error Recovery**: Should the system fall back to simulated data if the API is temporarily unavailable, or always show the error state?

4. **Query Expansion**: For SPAD testing, should we also test related terms like "avalanche photodiode" or "single photon detector"?

5. **Visualization Scope**: For the prototype, should we focus on the 2x2 dashboard (Companies, Years, Classifications, Total) or simplify further to just Companies and Years?

## Next Steps

1. **Install dependencies**: `pip install plotly requests`
2. **Run the test script**: `python test_patent_api.py`
3. **Update business_bridge.py** with the real API implementation
4. **Test the full UI integration** in the Manager Review tab
5. **Iterate based on test results**

This approach gives you a working prototype with real patent data while maintaining the transparent error handling we discussed earlier. 