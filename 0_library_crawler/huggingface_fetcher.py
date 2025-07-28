import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
import re
from typing import List, Optional
import logging
import os
from pathlib import Path

from .models import HFPaperInfo, BASE_URL
from .io import load_existing_dates, append_to_csv, parse_date

def get_papers_for_date(target_date: date, save_debug_html: bool = False) -> Optional[List[HFPaperInfo]]:
    """
    Fetches paper information from Hugging Face daily papers for a specific date.
    
    Args:
        target_date: A date object representing the desired date.
        save_debug_html: Whether to save the HTML to a file for debugging.
        
    Returns:
        A list of HFPaperInfo objects, or None if the request fails.
    """
    date_str = target_date.strftime("%Y-%m-%d")
    url = f"{BASE_URL}/papers/date/{date_str}"
    logging.info(f"Fetching papers from: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching URL {url}: {e}")
        return None
        
    if save_debug_html:
        debug_dir = "debug"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        debug_file = os.path.join(debug_dir, f"hf_page_{date_str}.html")
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        logging.info(f"Saved HTML to {debug_file} for inspection")
            
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all('article')
    
    if not articles:
        logging.info(f"No papers found for {date_str}")
        return []  # Return empty list instead of None
        
    papers = []
    
    for i, article in enumerate(articles):
        # Extract vote count with multiple methods
        vote_found = False
        votes = 0
        
        # Method 1: Find any span with only a number as content
        for span in article.find_all('span'):
            if span.string and span.string.strip().isdigit():
                votes = int(span.string.strip())
                vote_found = True
                break
                
        # Method 2: Look for a div at the start of the article that might have a number
        if not vote_found:
            divs = article.find_all('div', recursive=False)
            for div in divs:
                for text in div.stripped_strings:
                    if text.isdigit():
                        votes = int(text)
                        vote_found = True
                        break
                if vote_found:
                    break
                    
        # Method 3: Try to find a pattern like "76" or any number before the title
        if not vote_found:
            vote_pattern = re.compile(r'\b\d+\b')
            for element in article.find_all(text=vote_pattern):
                if vote_pattern.search(element):
                    try:
                        votes = int(vote_pattern.search(element).group())
                        vote_found = True
                        break
                    except (ValueError, AttributeError):
                        continue
                        
        # Find title and link
        title_tag = article.find('h3')
        if title_tag and title_tag.a and 'href' in title_tag.a.attrs:
            title = title_tag.a.get_text(strip=True)
            hf_paper_path = title_tag.a['href']
            # Ensure link starts with /
            if not hf_paper_path.startswith('/'):
                logging.warning(f"Unexpected hf_paper_path format: {hf_paper_path}")
                continue 
            hf_link = BASE_URL + hf_paper_path
            
            # Construct arXiv link from the path
            arxiv_link = None
            try:
                arxiv_id_match = re.search(r'(\d{4}\.\d{5,})', hf_paper_path)
                if arxiv_id_match:
                    arxiv_id = arxiv_id_match.group(1)
                    arxiv_link = f"https://arxiv.org/abs/{arxiv_id}"
            except IndexError:
                pass # No arXiv ID found, which is fine
                
            # Create Paper object with Pydantic
            try:
                paper = HFPaperInfo(
                    fetch_date=date_str,
                    title=title,
                    hf_link=hf_link,
                    arxiv_link=arxiv_link,
                    votes=votes
                )
                papers.append(paper)
            except Exception as e:
                logging.error(f"Error creating HFPaperInfo object: {e}")
                continue
        else:
            logging.warning(f"Skipping article {i+1} on {date_str} due to missing title/link.")
            
    # Sort papers by votes in descending order
    papers.sort(key=lambda p: p.votes, reverse=True)
    return papers

def main(csv_output_file: Path, from_date_str: str, debug: bool = False):
    """
    Main function to fetch Hugging Face papers and save them to a CSV file.
    """
    start_date = parse_date(from_date_str)
    if not start_date:
        logging.error(f"Invalid start_date format: {from_date_str}. Use YYYY-MM-DD.")
        return

    end_date = date.today() - timedelta(days=1)
    if start_date > end_date:
        logging.info("Start date is today or later, nothing to fetch.")
        return

    logging.info(f"Output CSV file: {csv_output_file.resolve()}")
    logging.info(f"Checking for papers from {start_date} to {end_date}")

    processed_dates = load_existing_dates(csv_output_file)
    logging.info(f"Found {len(processed_dates)} dates already processed.")

    current_date = start_date
    new_papers_count = 0
    fetched_dates_count = 0

    while current_date <= end_date:
        if current_date in processed_dates:
            logging.debug(f"Skipping already processed date: {current_date}")
        else:
            logging.info(f"Fetching papers for: {current_date}")
            fetched_dates_count += 1
            papers_data = get_papers_for_date(current_date, save_debug_html=debug)

            if papers_data is not None:
                if papers_data:
                    append_to_csv(csv_output_file, papers_data)
                    logging.info(f"-> Saved {len(papers_data)} papers for {current_date}")
                    new_papers_count += len(papers_data)
                else:
                    logging.info(f"-> No papers listed on {current_date}, marked as checked.")
            else:
                logging.error(f"Failed to fetch data for {current_date}. Stopping.")
                break  # Stop processing if a request fails

        current_date += timedelta(days=1)

    logging.info(f"Hugging Face fetcher complete. Checked {fetched_dates_count} new dates, added {new_papers_count} papers.") 