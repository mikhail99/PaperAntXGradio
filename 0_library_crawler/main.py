import logging
from pathlib import Path
from library_crawler.huggingface_fetcher import main as fetch_huggingface_papers
from library_crawler.arxiv_fetcher import main as fetch_arxiv_details

# --- Configuration ---
# Configure logging for the entire application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Define file paths in one central place
OUTPUT_DIR = Path(__file__).parent / "output"
CSV_OUTPUT_FILE = OUTPUT_DIR / "huggingface_papers.csv"
JSON_OUTPUT_FILE = OUTPUT_DIR / "huggingface_papers_enhanced.json"

# Fetching parameters
FROM_DATE = "2024-01-01"  # Start date in YYYY-MM-DD format
DEBUG_MODE = False  # Set to True to save Hugging Face Page HTML for debugging

def main():
    """
    Master script to run the full paper crawling and enhancement pipeline.
    """
    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Step 1: Fetch papers from Hugging Face ---
    logging.info("--- Starting Step 1: Fetching papers from Hugging Face ---")
    fetch_huggingface_papers(
        csv_output_file=CSV_OUTPUT_FILE,
        from_date_str=FROM_DATE,
        debug=DEBUG_MODE
    )
    logging.info("--- Finished Step 1 ---")

    # --- Step 2: Enhance paper data with details from arXiv ---
    logging.info("--- Starting Step 2: Enhancing papers with arXiv data ---")
    fetch_arxiv_details(
        csv_input_path=CSV_OUTPUT_FILE,
        json_output_path=JSON_OUTPUT_FILE
    )
    logging.info("--- Finished Step 2 ---")

    logging.info("Full pipeline completed successfully.")

if __name__ == '__main__':
    main() 