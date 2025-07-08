#!/usr/bin/env python3
"""
Test script for PatentsView API integration (replacing deprecated USPTO PEDS API).
Tests patent search for "SPAD" keyword and aggregates by company and year.

Note: This script now uses the PatentsView API which requires an API key.
Get your API key from: https://patentsview-support.atlassian.net/servicedesk/customer/portals
"""

import sys
import os
import urllib.parse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.business_intelligence.business_bridge import PatentLandscapeAnalyzer
import json
import requests
import time
from pprint import pprint

class PatentsViewAnalyzer(PatentLandscapeAnalyzer):
    """
    Enhanced PatentLandscapeAnalyzer with PatentsView API integration.
    This replaces the deprecated USPTO PEDS API.
    """
    
    def __init__(self, api_key=None):
        super().__init__()
        # Use the PatentsView API endpoint
        self.base_url = "https://search.patentsview.org/api/v1/patent"
        self.api_key = api_key or os.getenv('PATENTSVIEW_API_KEY')
        self.rate_limit_delay = 1.5  # PatentsView allows 45 requests/minute
        
        if not self.api_key:
            print("⚠️  WARNING: No API key provided. Please set PATENTSVIEW_API_KEY environment variable")
            print("   or pass api_key parameter. Get your key from:")
            print("   https://patentsview-support.atlassian.net/servicedesk/customer/portals")
    
    def search_patents_view(self, query: str, max_results: int = 50):
        """Search patents using the PatentsView API."""
        try:
            # PatentsView API uses JSON query format
            query_params = {
                "q": {
                    "_text_any": {
                        "patent_title": query
                    }
                },
                "f": [
                    "patent_id",
                    "patent_number", 
                    "patent_title",
                    "patent_date",
                    "patent_num_claims",
                    "assignees.assignee_organization",
                    "assignees.assignee_first_name",
                    "assignees.assignee_last_name",
                    "cpcs.cpc_section_id",
                    "cpcs.cpc_class_id",
                    "application_number",
                    "patent_processing_time"
                ],
                "s": [{"patent_date": "desc"}],
                "o": {
                    "size": min(max_results, 1000)  # PatentsView max is 1000
                }
            }
            
            print(f"🔗 Making API request to: {self.base_url}")
            print(f"📝 Query: {query}")
            print(f"📊 Requesting {max_results} results")
            
            headers = {
                'Content-Type': 'application/json',
                'X-Api-Key': self.api_key
            }
            
            # Convert query to proper format
            params = {
                'q': json.dumps(query_params['q']),
                'f': json.dumps(query_params['f']),
                's': json.dumps(query_params['s']),
                'o': json.dumps(query_params['o'])
            }
            
            response = requests.get(
                self.base_url, 
                params=params, 
                headers=headers, 
                timeout=30
            )
            
            print(f"🔍 Request URL: {response.url}")
            print(f"📡 Response Status: {response.status_code}")
            
            if response.status_code == 403:
                print("❌ Authentication failed - please check your API key")
                return {"error": "Authentication failed", "status": "AUTH_ERROR"}
            elif response.status_code == 429:
                print("⏳ Rate limit exceeded - waiting...")
                time.sleep(60)  # Wait 1 minute before retrying
                return {"error": "Rate limit exceeded", "status": "RATE_LIMIT"}
            
            response.raise_for_status()
            
            time.sleep(self.rate_limit_delay)
            
            response_data = response.json()
            
            return {
                "data": response_data,
                "source": "PatentsView API",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "query": query,
                "total_results": response_data.get('total_hits', 0)
            }
            
        except requests.exceptions.RequestException as e:
            print(f"🚫 Request error for query '{query}': {e}")
            return {"error": str(e), "status": "REQUEST_ERROR"}
        except Exception as e:
            print(f"❌ Unexpected error for query '{query}': {e}")
            return {"error": str(e), "status": "UNKNOWN_ERROR"}
    
    def process_patentsview_data(self, api_response):
        """Process PatentsView API response into a standardized format."""
        if "error" in api_response:
            return []
            
        patents = api_response.get("data", {}).get("patents", [])
        if not patents:
            print("⚠️ No patents found in API response.")
            return []
            
        print(f"📊 Processing {len(patents)} patents from PatentsView API")
        processed_results = []
        
        for patent_data in patents:
            try:
                # Extract assignee information
                assignees = patent_data.get("assignees", [])
                assignee_name = "Unknown"
                if assignees:
                    assignee = assignees[0]  # Take first assignee
                    if assignee.get("assignee_organization"):
                        assignee_name = assignee["assignee_organization"]
                    elif assignee.get("assignee_first_name") or assignee.get("assignee_last_name"):
                        first_name = assignee.get("assignee_first_name", "")
                        last_name = assignee.get("assignee_last_name", "")
                        assignee_name = f"{first_name} {last_name}".strip()
                
                # Extract CPC classification
                cpcs = patent_data.get("cpcs", [])
                classification = "Unknown"
                if cpcs:
                    cpc = cpcs[0]
                    section = cpc.get("cpc_section_id", "")
                    class_id = cpc.get("cpc_class_id", "")
                    classification = f"{section}{class_id}"
                
                result = {
                    "title": patent_data.get("patent_title", "Unknown"),
                    "patent_number": patent_data.get("patent_number", "Unknown"),
                    "patent_id": patent_data.get("patent_id", "Unknown"),
                    "assignee": assignee_name,
                    "filing_date": patent_data.get("patent_date", "").split("T")[0] if patent_data.get("patent_date") else "Unknown",
                    "publication_date": patent_data.get("patent_date", "").split("T")[0] if patent_data.get("patent_date") else "Unknown",
                    "classification": classification,
                    "num_claims": patent_data.get("patent_num_claims", 0),
                    "application_number": patent_data.get("application_number", "Unknown"),
                    "processing_time": patent_data.get("patent_processing_time", 0)
                }
                processed_results.append(result)
                
            except Exception as e:
                print(f"⚠️ Error processing patent record: {e}")
                continue
                
        print(f"✅ Successfully processed {len(processed_results)} patents")
        return processed_results

def test_spad_patents_with_patentsview():
    """Test patent search for SPAD technology using PatentsView API."""
    print("🔍 Testing PatentsView API with keyword: SPAD")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv('PATENTSVIEW_API_KEY')
    if not api_key:
        print("❌ ERROR: PatentsView API key not found!")
        print("   Please set the PATENTSVIEW_API_KEY environment variable")
        print("   Get your API key from: https://patentsview-support.atlassian.net/servicedesk/customer/portals")
        return None
    
    analyzer = PatentsViewAnalyzer(api_key=api_key)
    
    print("1. Searching for SPAD patents...")
    api_response = analyzer.search_patents_view("SPAD", max_results=50)
    
    if "error" in api_response:
        print(f"❌ API Error: {api_response['error']}")
        if api_response.get('status') == 'AUTH_ERROR':
            print("   Please check your API key is valid")
        return None
    
    print(f"✅ API call successful! Found {api_response.get('total_results', 0)} total matches")
    
    print("\n2. Processing patent data...")
    processed_patents = analyzer.process_patentsview_data(api_response)
    
    if not processed_patents:
        print("⚠️ No patents found or processing failed.")
        return None
        
    print("\n3. Aggregating patent statistics...")
    aggregated_data = analyzer.aggregate_patent_data(processed_patents)
    
    # Add API-specific metadata
    aggregated_data['api_info'] = {
        'source': 'PatentsView API',
        'query': 'SPAD',
        'total_matches': api_response.get('total_results', 0),
        'processed_patents': len(processed_patents),
        'timestamp': api_response.get('timestamp')
    }
    
    print(f"\n📈 Patent Statistics for 'SPAD' (PatentsView API):")
    print(f"Total Patents Found: {aggregated_data['total_patents']}")
    print(f"Total Matches in Database: {api_response.get('total_results', 0)}")
    
    # Show top assignees
    if 'assignee_stats' in aggregated_data:
        print(f"\n🏢 Top 5 Assignees:")
        for i, (assignee, count) in enumerate(list(aggregated_data['assignee_stats'].items())[:5], 1):
            print(f"   {i}. {assignee}: {count} patents")
    
    output_file = "spad_patent_patentsview_results.json"
    with open(output_file, "w") as f:
        json.dump(aggregated_data, f, indent=2)
    
    print(f"\n✅ Results saved to '{output_file}'")
    print("🎯 PatentsView API test completed successfully!")
    
    return aggregated_data

def setup_instructions():
    """Print setup instructions for the PatentsView API."""
    print("📋 Setup Instructions for PatentsView API")
    print("=" * 50)
    print("1. Get an API key from PatentsView:")
    print("   https://patentsview-support.atlassian.net/servicedesk/customer/portals")
    print("")
    print("2. Set your API key as an environment variable:")
    print("   export PATENTSVIEW_API_KEY='your_api_key_here'")
    print("")
    print("3. Or create a .env file with:")
    print("   PATENTSVIEW_API_KEY=your_api_key_here")
    print("")
    print("4. Run this script again")
    print("")
    print("📊 API Limits:")
    print("   - 45 requests per minute")
    print("   - 1,000 results per request maximum")
    print("   - Free tier available")

if __name__ == "__main__":
    print("🚀 PatentsView API Test Suite")
    print("   (Replacement for deprecated USPTO PEDS API)")
    print("=" * 70)
    
    # Check if API key is available
    if not os.getenv('PATENTSVIEW_API_KEY'):
        setup_instructions()
        sys.exit(1)
    
    try:
        results = test_spad_patents_with_patentsview()
        if results:
            print("\n📊 Summary:")
            print(f"   - API Source: {results['api_info']['source']}")
            print(f"   - Total Database Matches: {results['api_info']['total_matches']}")
            print(f"   - Processed Patents: {results['api_info']['processed_patents']}")
            print(f"   - Unique Assignees: {len(results.get('assignee_stats', {}))}")
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Test suite completed.") 