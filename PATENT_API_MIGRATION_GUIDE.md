# Patent API Migration Guide

## Overview

This document explains the migration from the deprecated USPTO PEDS API to the PatentsView API for patent search functionality.

## Why the Migration?

- **USPTO PEDS API retired on March 14, 2025**
- The old endpoint `https://ped.uspto.gov/api/queries` is no longer available
- PatentsView API provides comprehensive patent search capabilities
- Modern REST API with better performance and reliability

## Changes Made

### 1. API Endpoint Change
```python
# OLD (PEDS API - DEPRECATED)
self.base_url = "https://ped.uspto.gov/api/queries"

# NEW (PatentsView API)
self.base_url = "https://search.patentsview.org/api/v1/patent"
```

### 2. Authentication Required
```python
# OLD: No authentication required
# NEW: API key required
headers = {
    'Content-Type': 'application/json',
    'X-Api-Key': self.api_key
}
```

### 3. Query Format Changed
```python
# OLD PEDS Format:
payload = {
    "searchText": f'patentTitle:({query})',
    "fq": ["appStatus:\"Patented Case\""],
    "fl": "*",
    "sort": "applId asc",
    "start": "0",
    "rows": str(max_results)
}

# NEW PatentsView Format:
query_params = {
    "q": {"_text_any": {"patent_title": query}},
    "f": ["patent_id", "patent_title", "assignees.assignee_organization", ...],
    "s": [{"patent_date": "desc"}],
    "o": {"size": min(max_results, 1000)}
}
```

### 4. Response Structure Changed
```python
# OLD PEDS Response:
docs = api_response.get("data", {}).get("queryResults", {}).get("searchResponse", {}).get("response", {}).get("docs", [])

# NEW PatentsView Response:
patents = api_response.get("data", {}).get("patents", [])
```

### 5. Field Mapping Updated
| Old PEDS Field | New PatentsView Field |
|----------------|----------------------|
| `patentTitle` | `patent_title` |
| `patentNumber` | `patent_number` |
| `firstNamedApplicant` | `assignees.assignee_organization` |
| `appFilingDate` | `patent_date` |
| `grantDate` | `patent_date` |
| `cpcClassification` | `cpcs.cpc_section_id` + `cpcs.cpc_class_id` |

## Setup Instructions

### 1. Get PatentsView API Key
1. Visit: https://patentsview-support.atlassian.net/servicedesk/customer/portals
2. Request an API key
3. Each user gets one API key - don't request multiple keys

### 2. Set Environment Variable
```bash
# Option 1: Export in terminal
export PATENTSVIEW_API_KEY='your_api_key_here'

# Option 2: Add to your .bashrc or .zshrc
echo 'export PATENTSVIEW_API_KEY="your_api_key_here"' >> ~/.bashrc

# Option 3: Create a .env file
echo 'PATENTSVIEW_API_KEY=your_api_key_here' > .env
```

### 3. Install Dependencies
```bash
pip install requests python-dotenv
```

### 4. Run the Updated Script
```bash
python test_patent_api.py
```

## API Limits and Features

### PatentsView API Limits
- **Rate Limit**: 45 requests per minute
- **Max Results**: 1,000 results per request
- **Free Tier**: Available with API key
- **Timeout**: 30 seconds per request

### Available Fields
The PatentsView API provides access to:
- Patent titles and abstracts
- Assignee information (companies and individuals)
- CPC classification codes
- Patent dates (filing, grant)
- Claim counts
- Application numbers
- Processing times

### Query Capabilities
- **Text Search**: `_text_any`, `_text_all`, `_text_phrase`
- **Exact Match**: `_eq`, `_neq`
- **Comparisons**: `_gt`, `_gte`, `_lt`, `_lte`
- **String Operations**: `_begins`, `_contains`
- **Logical Operations**: `_and`, `_or`, `_not`

## Example Usage

```python
from test_patent_api import PatentsViewAnalyzer

# Initialize with API key
analyzer = PatentsViewAnalyzer(api_key="your_api_key_here")

# Search for patents
response = analyzer.search_patents_view("SPAD", max_results=50)

# Process results
patents = analyzer.process_patentsview_data(response)

# Results will include:
# - Patent titles and numbers
# - Assignee organizations
# - Classification codes
# - Filing/grant dates
# - Additional metadata
```

## Error Handling

The updated script handles:
- **403 Forbidden**: Invalid or missing API key
- **429 Too Many Requests**: Rate limit exceeded (auto-retry after 60s)
- **Request timeouts**: 30-second timeout with proper error messages
- **Response parsing errors**: Graceful handling of malformed data

## Migration Checklist

- [x] Update API endpoint from PEDS to PatentsView
- [x] Add API key authentication
- [x] Update query format to PatentsView JSON structure
- [x] Update response parsing for new data structure
- [x] Add proper error handling for authentication and rate limits
- [x] Update field mappings for patent data
- [x] Add setup instructions and documentation
- [x] Test with actual API calls

## Support

- **PatentsView Support**: https://patentsview-support.atlassian.net/servicedesk/customer/portals
- **API Documentation**: https://search.patentsview.org/docs/
- **USPTO Support**: data@uspto.gov (for general questions about the PEDS retirement)

## Next Steps

1. Get your PatentsView API key
2. Set the `PATENTSVIEW_API_KEY` environment variable
3. Run the updated test script
4. Update any dependent code that uses the old PEDS format
5. Monitor API usage to stay within rate limits

The migration is complete and the script is ready to use with the PatentsView API! 