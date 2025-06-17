import pandas as pd
import json
import arxiv
import logging
import time
import re
import os
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Input/output file paths are now passed as arguments
# CSV_PATH = Path(__file__).parent / "huggingface_papers.csv"
# OUTPUT_PATH = Path(__file__).parent / "huggingface_papers_enhanced.json"

# Maximum number of papers to fetch in each batch
BATCH_SIZE = 20
# Delay between batches in seconds to avoid rate limiting
BATCH_DELAY = 5
# Process only this many papers as a test run (set to None to process all)
TEST_LIMIT = None

def extract_arxiv_id_from_url(url: Optional[str]) -> Optional[str]:
    """Extract arXiv ID from a URL or return None if no valid ID found."""
    if not url:
        return None
        
    # Look for patterns like /abs/2301.12345 or /pdf/2301.12345
    match = re.search(r'(?:abs|pdf)/(\d{4}\.\d{5,})', url)
    if match:
        return match.group(1)
    
    # Look for older format IDs 
    match = re.search(r'(?:abs|pdf)/([a-z\-]+(?:\.[A-Z]+)?/\d{7}|\d{7})', url)
    if match:
        return match.group(1)
        
    return None

def normalize_arxiv_id(arxiv_id: str) -> str:
    """Normalize arxiv ID to ensure it's in the format the API expects."""
    if not arxiv_id:
        return ""
        
    # Some IDs might have 'v1', 'v2' etc. at the end - remove those
    id_no_version = re.sub(r'v\d+$', '', arxiv_id)
    
    # Log the original and normalized ID for debugging
    if id_no_version != arxiv_id:
        logging.debug(f"Normalized arXiv ID from '{arxiv_id}' to '{id_no_version}'")
    
    return id_no_version

def load_existing_papers(json_output_path: Path) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Load existing papers from the JSON file and return a set of processed arXiv IDs."""
    existing_papers = []
    processed_ids = set()
    
    if not json_output_path.exists() or json_output_path.stat().st_size == 0:
        logging.info(f"No existing papers file found at {json_output_path} or file is empty")
        return [], set()
        
    try:
        with open(json_output_path, 'r', encoding='utf-8') as f:
            existing_papers = json.load(f)
            
        # Extract all normalized arXiv IDs that have already been processed
        for paper in existing_papers:
            if 'arxiv_id' in paper and paper['arxiv_id']:
                arxiv_id = paper['arxiv_id']
                
                # If it's a URL, extract the ID
                if isinstance(arxiv_id, str) and arxiv_id.startswith('http'):
                    extracted_id = extract_arxiv_id_from_url(arxiv_id)
                    if extracted_id:
                        processed_ids.add(normalize_arxiv_id(extracted_id))
                else:
                    # Otherwise, normalize and add to processed set
                    processed_ids.add(normalize_arxiv_id(str(arxiv_id)))
        
        logging.info(f"Loaded {len(existing_papers)} existing papers with {len(processed_ids)} unique arXiv IDs")
        
        # Log a few examples
        if processed_ids:
            sample_ids = list(processed_ids)[:5]
            logging.info(f"Sample processed IDs: {sample_ids}")
            
        return existing_papers, processed_ids
        
    except Exception as e:
        logging.error(f"Error loading existing papers: {e}")
        return [], set()

def load_arxiv_ids_from_csv(csv_input_path: Path, processed_ids: Set[str], limit: Optional[int] = TEST_LIMIT) -> List[str]:
    """Load arxiv IDs from the CSV file, skipping already processed IDs."""
    logging.info(f"Loading arxiv IDs from {csv_input_path}")
    
    try:
        df = pd.read_csv(csv_input_path, dtype={'arxiv_id': str})
        
        # Show sample rows for debugging
        logging.info("Sample rows from CSV:")
        for _, row in df.head(3).iterrows():
            logging.info(f"  Row: {dict(row)}")
        
        # Extract ArXiv IDs, handling potential URL formats
        valid_ids = []
        skipped_count = 0
        
        for idx, arxiv_id in enumerate(df['arxiv_id'].dropna()):
            normalized_id = None
            
            # For URLs, extract the ID part
            if isinstance(arxiv_id, str) and arxiv_id.startswith('http'):
                extracted_id = extract_arxiv_id_from_url(arxiv_id)
                if extracted_id:
                    normalized_id = normalize_arxiv_id(extracted_id)
            else:
                normalized_id = normalize_arxiv_id(str(arxiv_id))
            
            # Skip if already processed
            if normalized_id and normalized_id in processed_ids:
                skipped_count += 1
                continue
                
            # Add valid IDs that haven't been processed yet
            if normalized_id:
                valid_ids.append(normalized_id)
                
            # Debug the first few IDs
            if idx < 5:
                status = "Skipped (already processed)" if normalized_id in processed_ids else "Added" if normalized_id else "Invalid"
                logging.info(f"Processing ID {idx+1}: '{arxiv_id}' → '{normalized_id}' - {status}")
        
        # Remove duplicates
        unique_ids = list(dict.fromkeys(valid_ids))
        
        # Apply limit if specified
        if limit and len(unique_ids) > limit:
            logging.info(f"Limiting to {limit} IDs for testing")
            unique_ids = unique_ids[:limit]
        
        logging.info(f"Found {len(unique_ids)} new unique arxiv IDs to process (skipped {skipped_count} already processed)")
        
        # Debug: print first few IDs to verify format
        if unique_ids:
            logging.info(f"Sample new IDs: {unique_ids[:min(5, len(unique_ids))]}")
        
        return unique_ids
        
    except Exception as e:
        logging.error(f"Error loading CSV: {e}")
        return []

def fetch_paper_details_batch(arxiv_ids: List[str]) -> Dict[str, Any]:
    """Fetch detailed information for a batch of arxiv IDs in a single API call."""
    results = {}
    
    if not arxiv_ids:
        return results
        
    try:
        # Create a client with appropriate parameters
        client = arxiv.Client(
            page_size=100,  # Maximum page size for efficiency
            delay_seconds=1,
            num_retries=3
        )
        
        # Create a search query for all IDs in the batch
        search = arxiv.Search(
            id_list=arxiv_ids,
            max_results=len(arxiv_ids)
        )
        
        # Process each result
        for result in client.results(search):
            try:
                # Get basic version (with v1, v2, etc.)
                arxiv_id_with_version = result.get_short_id()
                # Also get normalized version (without v1, v2)
                arxiv_id_normalized = normalize_arxiv_id(arxiv_id_with_version)
                
                paper_info = {
                    'title': result.title,
                    'abstract': result.summary,
                    'authors': [author.name for author in result.authors],
                    'published': result.published.strftime('%Y-%m-%d'),
                    'updated': result.updated.strftime('%Y-%m-%d') if result.updated else None,
                    'categories': result.categories,
                    'arxiv_url': result.entry_id,
                    'pdf_url': result.pdf_url,
                    'doi': result.doi,
                    'comment': result.comment,
                    'journal_ref': result.journal_ref
                }
                
                # Store using BOTH the normalized ID and the ID with version
                # This provides two ways to look up the paper later
                results[arxiv_id_normalized] = paper_info
                if arxiv_id_with_version != arxiv_id_normalized:
                    results[arxiv_id_with_version] = paper_info
                    
                logging.info(f"Successfully fetched details for {arxiv_id_with_version} (base: {arxiv_id_normalized})")
                
            except Exception as e:
                logging.error(f"Error extracting paper info: {e}")
                
    except Exception as e:
        logging.error(f"Error fetching batch of papers: {e}")
    
    return results

def fetch_paper_details(arxiv_ids: List[str]) -> Dict[str, Any]:
    """Fetch detailed information for each arxiv ID, processing in batches."""
    all_results = {}
    total_ids = len(arxiv_ids)
    
    # Process in batches
    for i in range(0, total_ids, BATCH_SIZE):
        batch = arxiv_ids[i:i+BATCH_SIZE]
        logging.info(f"Processing batch {i//BATCH_SIZE + 1}/{(total_ids+BATCH_SIZE-1)//BATCH_SIZE} ({len(batch)} papers)")
        
        # Fetch details for the entire batch in one request
        batch_results = fetch_paper_details_batch(batch)
        all_results.update(batch_results)
        
        # Delay between batches to avoid rate limiting
        if i + BATCH_SIZE < total_ids:
            logging.info(f"Waiting {BATCH_DELAY} seconds before next batch...")
            time.sleep(BATCH_DELAY)
    
    logging.info(f"Successfully fetched details for {len(all_results)} papers")
    return all_results

def merge_with_csv_data(csv_input_path: Path, arxiv_details: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Merge the arxiv details with the original CSV data."""
    logging.info("Merging arxiv details with original CSV data")
    
    df = pd.read_csv(csv_input_path, dtype={'arxiv_id': str, 'votes': 'Int64'})
    
    # Create a list to store the merged data
    merged_papers = []
    missing_count = 0
    found_count = 0
    
    for _, row in df.iterrows():
        # Skip rows without arxiv IDs
        if pd.isna(row['arxiv_id']):
            continue
            
        arxiv_id = row['arxiv_id']
        normalized_id = None
        
        # Normalize or extract the ID
        if isinstance(arxiv_id, str) and arxiv_id.startswith('http'):
            extracted_id = extract_arxiv_id_from_url(arxiv_id)
            if extracted_id:
                normalized_id = normalize_arxiv_id(extracted_id)
        else:
            normalized_id = normalize_arxiv_id(str(arxiv_id))
            
        if not normalized_id:
            continue
        
        # Try multiple ways to match the IDs
        arxiv_info = None
        match_key = None
        
        # Try direct match with normalized ID (most common case)
        if normalized_id in arxiv_details:
            arxiv_info = arxiv_details[normalized_id]
            match_key = normalized_id
        # Try potential versioned IDs (v1, v2, etc.)
        elif any(key.startswith(normalized_id + 'v') and key in arxiv_details for key in arxiv_details.keys()):
            for key in arxiv_details.keys():
                if key.startswith(normalized_id + 'v'):
                    arxiv_info = arxiv_details[key]
                    match_key = key
                    break
        # Try extracted ID from URL if it's a URL
        elif isinstance(arxiv_id, str) and arxiv_id.startswith('http'):
            extracted_id = extract_arxiv_id_from_url(arxiv_id)
            if extracted_id and extracted_id in arxiv_details:
                arxiv_info = arxiv_details[extracted_id]
                match_key = extracted_id
        
        if not arxiv_info:
            # If we couldn't fetch details for this ID
            missing_count += 1
            if missing_count <= 10:  # Limit log spam
                logging.warning(f"No arxiv details found for ID: {arxiv_id} (normalized: {normalized_id})")
            continue
            
        # Merge CSV data with arxiv details
        paper = {
            'fetch_date': row['fetch_date'],
            'hf_id': row['hf_id'],
            'hf_link': f"https://huggingface.co/papers/{row['hf_id']}",
            'arxiv_id': arxiv_id,
            'arxiv_link': f"https://arxiv.org/abs/{arxiv_id}" if not isinstance(arxiv_id, str) or not arxiv_id.startswith('http') else arxiv_id,
            'votes': int(row['votes']) if not pd.isna(row['votes']) else 0,
            **arxiv_info
        }
        
        merged_papers.append(paper)
        found_count += 1
        
        # Log some successful matches for debugging
        if found_count <= 5:
            logging.info(f"Successfully matched ID: {arxiv_id} → {match_key}")
    
    if missing_count > 10:
        logging.warning(f"... and {missing_count - 10} more IDs without details")
    
    logging.info(f"Created {len(merged_papers)} merged paper records")
    return merged_papers

def save_to_json(papers: List[Dict[str, Any]], json_output_path: Path) -> None:
    """Save the enhanced paper data as JSON."""
    logging.info(f"Saving enhanced data to {json_output_path}")
    
    if not papers:
        logging.error("No papers to save! Check previous error messages.")
        # Save an empty list rather than nothing, at least it's valid JSON
        papers = []
    
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Successfully saved {len(papers)} papers to {json_output_path}")

def main(csv_input_path: Path, json_output_path: Path):
    """Main function to orchestrate the process."""
    logging.info("Starting arxiv data enhancement process")
    
    # Load existing papers
    existing_papers, processed_ids = load_existing_papers(json_output_path)
    
    # Load new arXiv IDs from CSV, excluding already processed ones
    new_arxiv_ids = load_arxiv_ids_from_csv(csv_input_path, processed_ids)
    
    if not new_arxiv_ids:
        logging.info("No new arXiv IDs to process")
        if existing_papers:
            logging.info(f"Keeping existing {len(existing_papers)} papers")
            save_to_json(existing_papers, json_output_path)
        else:
            save_to_json([], json_output_path)  # Save empty list if no existing or new papers
        return
    
    # Fetch details for new papers only
    arxiv_details = fetch_paper_details(new_arxiv_ids)
    
    if not arxiv_details:
        logging.error("Failed to fetch any new paper details from arXiv")
        if existing_papers:
            logging.info(f"Keeping existing {len(existing_papers)} papers")
            save_to_json(existing_papers, json_output_path)
        else:
            save_to_json([], json_output_path)  # Save empty list if no existing or new papers
        return
    
    # Merge with original CSV data to create new paper records
    new_merged_papers = merge_with_csv_data(csv_input_path, arxiv_details)
    
    # Combine existing and new papers
    all_papers = existing_papers + new_merged_papers
    logging.info(f"Combined {len(existing_papers)} existing and {len(new_merged_papers)} new papers")
    
    # Save all papers (existing + new)
    save_to_json(all_papers, json_output_path)
    
    logging.info("ArXiv data enhancement process completed")

if __name__ == '__main__':
    # This part is for standalone execution, if needed
    # You would need to define CSV_PATH and OUTPUT_PATH here for it to work
    # For now, it's designed to be called from the master main.py
    pass 