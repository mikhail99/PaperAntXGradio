# Hugging Face Paper Crawler

This module contains scripts for collecting AI research papers from the Hugging Face Papers section and enhancing them with metadata from arXiv.

## Overview

The crawler is a two-step pipeline orchestrated by `main.py`:

1.  **Hugging Face Fetcher**: The `huggingface_fetcher.py` module scrapes the Hugging Face "Daily Papers" section to collect a list of papers, their vote counts, and arXiv IDs. The results are saved to a CSV file.
2.  **ArXiv Enhancer**: The `arxiv_fetcher.py` module reads the CSV file, takes the arXiv IDs, and fetches detailed metadata (abstracts, authors, categories, etc.) from the official arXiv API. The final, enriched data is saved to a JSON file.

The entire process is designed to be incremental, meaning it will skip dates and papers that have already been processed on subsequent runs.

## Setup

Install required dependencies:

```bash
pip install requests beautifulsoup4 pandas pydantic arxiv tqdm
```

## Usage

To run the entire pipeline, simply execute the main script from the project's root directory:

```bash
python -m library_crawler.main
```

This single command will:
1.  Fetch new papers from Hugging Face and add them to `output/huggingface_papers.csv`.
2.  Enhance the newly added papers with data from arXiv and save the complete, updated list to `output/huggingface_papers_enhanced.json`.

### Configuration

You can configure the crawler by editing the constants at the top of `library_crawler/main.py`:

-   `FROM_DATE`: Change the start date for fetching papers (format: "YYYY-MM-DD").
-   `DEBUG_MODE`: Set to `True` to save the raw HTML from Hugging Face for debugging purposes.
-   `OUTPUT_DIR`, `CSV_OUTPUT_FILE`, `JSON_OUTPUT_FILE`: Modify the output paths if needed.

## Important Notes

### Incremental Processing

Both fetchers use incremental processing to avoid re-downloading data:

-   The **Hugging Face fetcher** checks the existing CSV file and skips any dates that have already been successfully scraped.
-   The **arXiv fetcher** checks the existing JSON file and only fetches details for arXiv IDs that have not yet been processed.

### Managing Data Files

**Be careful when moving or deleting the data files in the `output/` directory**:

-   If you delete `output/huggingface_papers.csv`, the crawler will re-download all paper information from Hugging Face from your `FROM_DATE`.
-   If you delete `output/huggingface_papers_enhanced.json`, the arXiv fetcher will re-download all paper details, which can be slow due to API rate limits.

### Testing with Limited Data

If you want to test the arXiv fetcher with a limited number of new papers, you can modify the `TEST_LIMIT` variable in `library_crawler/arxiv_fetcher.py`:

```python
# Set to a number like 50 for testing, or None to process all papers
TEST_LIMIT = 50
```

## File Structure

-   `main.py`: The master script that configures and runs the entire pipeline.
-   `huggingface_fetcher.py`: Module responsible for scraping data from the Hugging Face website.
-   `arxiv_fetcher.py`: Module responsible for fetching detailed metadata from the arXiv API.
-   `models.py`: Pydantic data models used for data validation and structure.
-   `io.py`: Handles all file input/output operations (CSV and JSON).
-   `output/`: Directory containing the generated data files.
    -   `huggingface_papers.csv`: Contains basic paper information from Hugging Face.
    -   `huggingface_papers_enhanced.json`: Contains the final, enhanced paper data with details from arXiv. 