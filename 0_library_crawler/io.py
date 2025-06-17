import pandas as pd
import logging
from pathlib import Path
from typing import List, Set, Union, Optional
from datetime import date, datetime
from library_crawler.models import HFPaperInfo

# Define a constant for field names to ensure consistency
CSV_FIELDNAMES = ['fetch_date', 'title', 'hf_id', 'arxiv_id', 'votes']

def parse_date(date_str: Union[str, None]) -> Union[date, None]:
    """Safely parse a date string, handling potential None input."""
    if not date_str or pd.isna(date_str): 
        return None
    try:
        return datetime.strptime(str(date_str), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None

def load_existing_dates(file_path: Path) -> Set[date]:
    """Load already processed dates from the CSV file using Pandas."""
    processed_dates = set()
    if not file_path.exists():
        return processed_dates

    try:
        df = pd.read_csv(file_path, usecols=['fetch_date'])
        valid_dates = df['fetch_date'].apply(parse_date).dropna().unique()
        processed_dates = set(valid_dates)
    except Exception as e:
        # Simplified exception handling with specific error message context
        logging.warning(f"Could not load dates from {file_path}: {type(e).__name__}: {e}")
    
    return processed_dates

def append_to_csv(file_path: Path, papers: List[HFPaperInfo]):
    """Append new paper data to the CSV file using Pandas."""
    if not papers:
        return  # Nothing to append

    try:
        # Convert list of Pydantic objects to list of dicts, then to DataFrame
        papers_dict_list = [paper.to_csv_dict() for paper in papers]
        df_to_append = pd.DataFrame(papers_dict_list)

        # Check if file exists and is not empty to determine header writing
        file_exists = file_path.exists()
        write_header = not file_exists or file_path.stat().st_size == 0

        # Append data using pandas with correct column order
        df_to_append.to_csv(
            file_path,
            mode='a',
            header=write_header,
            index=False,
            columns=CSV_FIELDNAMES
        )
    except Exception as e:
        logging.error(f"Failed to append data to {file_path}: {type(e).__name__}: {e}")

def load_papers_from_csv(file_path: Path) -> List[HFPaperInfo]:
    """Loads all papers from the specified CSV file using Pandas."""
    papers = []
    if not file_path.exists():
        logging.warning(f"Data file not found: {file_path}")
        return papers

    try:
        # Read CSV into DataFrame with appropriate types
        df = pd.read_csv(file_path, dtype={'arxiv_id': str, 'votes': 'Int64'})
        
        # Check required columns
        required_fields = {'fetch_date', 'title', 'hf_id'}
        if not required_fields.issubset(df.columns):
            missing = required_fields - set(df.columns)
            logging.error(f"CSV file missing required columns: {missing}")
            return papers

        # Replace NaN/NA with None and convert to HFPaperInfo objects
        df = df.replace({pd.NA: None, float('nan'): None})
        
        # Convert rows to objects with error handling per row
        for i, row_dict in enumerate(df.to_dict('records')):
            try:
                paper = HFPaperInfo.from_csv_dict(row_dict)
                papers.append(paper)
            except Exception as e:
                logging.error(f"Error in row {i+1}: {type(e).__name__}: {e}")
    except Exception as e:
        logging.error(f"Failed to load papers from {file_path}: {type(e).__name__}: {e}")

    return papers 