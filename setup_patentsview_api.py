#!/usr/bin/env python3
"""
Setup script for PatentsView API integration.
This script helps you set up your environment and test the API connection.
"""

import os
import sys
import json
import requests
from getpass import getpass

def check_api_key():
    """Check if API key is already set."""
    api_key = os.getenv('PATENTSVIEW_API_KEY')
    if api_key:
        print(f"✅ API Key found: {api_key[:8]}{'*' * 8}")
        return api_key
    else:
        print("❌ No API key found in environment")
        return None

def set_api_key():
    """Help user set their API key."""
    print("\n📝 Setting up PatentsView API Key")
    print("=" * 40)
    
    api_key = getpass("Enter your PatentsView API key: ")
    
    if not api_key:
        print("❌ No API key provided")
        return None
    
    # Test the API key
    if test_api_key(api_key):
        print("✅ API key is valid!")
        
        # Offer to save to environment
        save_choice = input("\n💾 Save API key to environment? (y/n): ").lower()
        if save_choice == 'y':
            save_api_key_to_env(api_key)
        
        return api_key
    else:
        print("❌ API key is invalid or API is unreachable")
        return None

def test_api_key(api_key):
    """Test if the API key works."""
    try:
        url = "https://search.patentsview.org/api/v1/patent"
        headers = {
            'Content-Type': 'application/json',
            'X-Api-Key': api_key
        }
        
        # Simple test query
        params = {
            'q': json.dumps({"patent_id": "10000000"}),  # Test query
            'f': json.dumps(["patent_id"]),
            'o': json.dumps({"size": 1})
        }
        
        print("🔍 Testing API key...")
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return True
        elif response.status_code == 403:
            print("❌ Authentication failed - invalid API key")
            return False
        else:
            print(f"⚠️  Unexpected response: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing API key: {e}")
        return False

def save_api_key_to_env(api_key):
    """Save API key to environment file."""
    try:
        # Create/update .env file
        with open('.env', 'w') as f:
            f.write(f"PATENTSVIEW_API_KEY={api_key}\n")
        
        print("✅ API key saved to .env file")
        print("   Add this to your shell profile for permanent use:")
        print(f"   echo 'export PATENTSVIEW_API_KEY=\"{api_key}\"' >> ~/.bashrc")
        
    except Exception as e:
        print(f"❌ Error saving API key: {e}")

def run_test_search():
    """Run a test search to verify everything works."""
    try:
        from test_patent_api import PatentsViewAnalyzer
        
        api_key = os.getenv('PATENTSVIEW_API_KEY')
        if not api_key:
            print("❌ No API key available for testing")
            return False
        
        print("\n🔍 Running test search...")
        analyzer = PatentsViewAnalyzer(api_key=api_key)
        
        # Test with a simple query
        response = analyzer.search_patents_view("laser", max_results=5)
        
        if "error" in response:
            print(f"❌ Test search failed: {response['error']}")
            return False
        
        patents = analyzer.process_patentsview_data(response)
        
        if patents:
            print(f"✅ Test search successful! Found {len(patents)} patents")
            print(f"   Total matches in database: {response.get('total_results', 0)}")
            
            # Show first result
            if patents:
                first_patent = patents[0]
                print(f"\n📄 Sample result:")
                print(f"   Title: {first_patent.get('title', 'N/A')}")
                print(f"   Patent #: {first_patent.get('patent_number', 'N/A')}")
                print(f"   Assignee: {first_patent.get('assignee', 'N/A')}")
            
            return True
        else:
            print("⚠️  No patents found in test search")
            return False
            
    except ImportError:
        print("❌ Could not import test_patent_api.py")
        print("   Make sure the file exists and is properly formatted")
        return False
    except Exception as e:
        print(f"❌ Error running test search: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 PatentsView API Setup Script")
    print("=" * 50)
    
    # Step 1: Check existing API key
    api_key = check_api_key()
    
    if not api_key:
        print("\n📋 To get an API key:")
        print("   1. Visit: https://patentsview-support.atlassian.net/servicedesk/customer/portals")
        print("   2. Request an API key")
        print("   3. Come back here and run this script again")
        
        choice = input("\n❓ Do you already have an API key? (y/n): ").lower()
        if choice == 'y':
            api_key = set_api_key()
        else:
            print("\n👉 Please get an API key first, then run this script again")
            return
    
    if not api_key:
        print("\n❌ Setup incomplete - no valid API key")
        return
    
    # Step 2: Test the API connection
    print("\n🧪 Testing API connection...")
    if test_api_key(api_key):
        print("✅ API connection successful!")
    else:
        print("❌ API connection failed")
        return
    
    # Step 3: Run full test
    print("\n🔬 Running full test...")
    if run_test_search():
        print("\n🎉 Setup complete! Your PatentsView API is ready to use.")
        print("\n📚 Next steps:")
        print("   - Run: python test_patent_api.py")
        print("   - Check the migration guide: PATENT_API_MIGRATION_GUIDE.md")
        print("   - Update any existing code that used the old PEDS API")
    else:
        print("\n⚠️  Setup completed but test search failed")
        print("   Your API key works, but there may be issues with the test script")

if __name__ == "__main__":
    main() 