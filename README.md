# Patent API Integration - PatentsView Migration

## 🚨 Important Notice

The **USPTO PEDS API was retired on March 14, 2025**. This project has been migrated to use the **PatentsView API** as a replacement.

## 📋 Quick Start

1. **Get an API key** from PatentsView:
   - Visit: https://patentsview-support.atlassian.net/servicedesk/customer/portals
   - Request a free API key

2. **Set up your environment**:
   ```bash
   python setup_patentsview_api.py
   ```

3. **Run the test script**:
   ```bash
   python test_patent_api.py
   ```

## 📁 Files in this Project

- **`test_patent_api.py`** - Main test script using PatentsView API
- **`setup_patentsview_api.py`** - Helper script to set up API key and test connection
- **`PATENT_API_MIGRATION_GUIDE.md`** - Detailed migration documentation
- **`README.md`** - This file

## 🔧 What Changed

| Aspect | Old (PEDS API) | New (PatentsView API) |
|--------|----------------|----------------------|
| **Endpoint** | `https://ped.uspto.gov/api/queries` | `https://search.patentsview.org/api/v1/patent` |
| **Authentication** | None required | API key required |
| **Rate Limit** | Unknown | 45 requests/minute |
| **Max Results** | 50 | 1,000 per request |
| **Query Format** | Custom POST JSON | REST GET with JSON params |
| **Response Format** | Nested `queryResults` | Direct `patents` array |

## 🚀 Key Features

- **Modern API**: Uses the actively maintained PatentsView API
- **Better Error Handling**: Handles authentication, rate limiting, and timeouts
- **Rich Data**: Access to assignees, classifications, claims, and more
- **Flexible Queries**: Support for text search, exact matching, and complex logic
- **Rate Limiting**: Automatic rate limit handling and retry logic

## 📊 Example Usage

```python
from test_patent_api import PatentsViewAnalyzer

# Initialize with your API key
analyzer = PatentsViewAnalyzer(api_key="your_api_key_here")

# Search for patents
response = analyzer.search_patents_view("SPAD", max_results=50)

# Process the results
patents = analyzer.process_patentsview_data(response)

# Each patent includes:
# - title, patent_number, assignee
# - filing_date, classification
# - num_claims, processing_time
```

## 🛠️ Environment Setup

Set your API key in one of these ways:

```bash
# Option 1: Environment variable
export PATENTSVIEW_API_KEY="your_api_key_here"

# Option 2: .env file
echo "PATENTSVIEW_API_KEY=your_api_key_here" > .env

# Option 3: Pass directly to the class
analyzer = PatentsViewAnalyzer(api_key="your_api_key_here")
```

## 📈 API Limits

- **Rate Limit**: 45 requests per minute
- **Results**: Up to 1,000 results per request
- **Timeout**: 30 seconds per request
- **Free Tier**: Available with API key registration

## 🔍 Search Capabilities

The PatentsView API supports:
- **Text Search**: Search titles, abstracts, and descriptions
- **Assignee Search**: Find patents by company or inventor
- **Date Filtering**: Filter by filing or grant dates
- **Classification Search**: Search by CPC classification codes
- **Complex Logic**: Combine criteria with AND/OR/NOT operators

## 🆘 Support

- **PatentsView API**: https://search.patentsview.org/docs/
- **API Support**: https://patentsview-support.atlassian.net/servicedesk/customer/portals
- **Migration Guide**: See `PATENT_API_MIGRATION_GUIDE.md`

## ✅ Migration Status

- [x] ✅ API endpoint updated to PatentsView
- [x] ✅ Authentication implemented
- [x] ✅ Query format updated
- [x] ✅ Response parsing updated
- [x] ✅ Error handling improved
- [x] ✅ Rate limiting implemented
- [x] ✅ Documentation created
- [x] ✅ Setup script provided

**The migration is complete and ready for use!** 