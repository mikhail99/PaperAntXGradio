import os
import json
from bs4 import BeautifulSoup, NavigableString, Tag
import re


def extract_text_content(element):
    """
    Extract clean text content from a BeautifulSoup element, handling special cases.
    """
    if not element:
        return ""
    
    # Handle NavigableString (text nodes)
    if isinstance(element, NavigableString):
        text = str(element)
    else:
        # Remove unwanted elements before extracting text
        for unwanted in element.find_all(['script', 'style', 'button']):
            unwanted.decompose()
        
        # Get text and clean it up
        text = element.get_text()
    
    # Clean up whitespace and special characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
    text = text.strip()
    
    # Remove specific unwanted phrases
    unwanted_phrases = [
        'Report issue for preceding element',
        'Report issue for preceding element\n\n',
    ]
    
    for phrase in unwanted_phrases:
        text = text.replace(phrase, '')
    
    # Clean up any remaining extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def extract_text_with_boundary(start_element, end_element):
    """
    Extracts text content from elements between a start and end element,
    stopping if a heading tag is encountered.
    """
    text = ""
    for elem in start_element.find_next_siblings():
        if elem == end_element:
            break
        
        # If the sibling is a heading, stop.
        if elem.name and re.match(r'^h[1-6]$', elem.name):
            break

        # Check for nested headings before extracting text
        if elem.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            # This element contains a heading, so we need to be careful.
            # We'll recursively process its children.
            temp_text = ""
            for child in elem.children:
                if child.name and re.match(r'^h[1-6]$', child.name):
                    break # Stop at the nested heading
                temp_text += extract_text_content(child) + " "
            text += temp_text
        else:
            # No nested headings, safe to extract all text
            text += extract_text_content(elem) + " "

    return re.sub(r'\s+', ' ', text).strip()


def parse_html_sections(html_content):
    """
    Parse HTML content and extract sections based on heading tags.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    sections = []
    
    # Remove unwanted elements from the entire document first
    for unwanted in soup.find_all(['script', 'style', 'button']):
        unwanted.decompose()
    
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    
    for i, heading in enumerate(headings):
        level = int(heading.name[1])
        title = extract_text_content(heading)
        if not title:
            continue
        
        # The end of the section is the start of the next heading
        next_heading = headings[i + 1] if i + 1 < len(headings) else None
        
        content = extract_text_with_boundary(heading, next_heading)
        
        if content:
            sections.append({
                "title": title,
                "level": level,
                "content": content
            })
    
    return sections


def html_section_splitting(paper_id):
    """
    Process HTML file for a given paper ID and extract sections.
    """
    # Find HTML file in the paper directory
    paper_dir = os.path.join('papers', paper_id)
    if not os.path.exists(paper_dir):
        raise FileNotFoundError(f"Paper directory not found: {paper_dir}")
    
    html_files = [f for f in os.listdir(paper_dir) if f.lower().endswith('.html')]
    if not html_files:
        print(f"No HTML file found in {paper_dir}")
        return None
    
    html_path = os.path.join(paper_dir, html_files[0])
    print(f"Processing HTML file: {html_path}")
    
    # Create output directory
    output_dir = os.path.join('outputs', paper_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and parse HTML
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        sections = parse_html_sections(html_content)
        
        # Save sections to JSON
        output_path = os.path.join(output_dir, 'sections.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sections, f, indent=2, ensure_ascii=False)
        
        print(f"Extracted {len(sections)} sections and saved to {output_path}")
        return sections
        
    except Exception as e:
        print(f"Error processing {html_path}: {str(e)}")
        return None


def batch_html_section_splitting():
    """
    Process all papers with HTML files in the papers directory.
    """
    papers_dir = 'papers'
    if not os.path.exists(papers_dir):
        print(f"Papers directory not found: {papers_dir}")
        return
    
    processed_count = 0
    for paper_id in os.listdir(papers_dir):
        paper_path = os.path.join(papers_dir, paper_id)
        if os.path.isdir(paper_path):
            # Check if this paper has an HTML file
            html_files = [f for f in os.listdir(paper_path) if f.lower().endswith('.html')]
            if html_files:
                output_path = os.path.join('outputs', paper_id, 'sections.json')
                if not os.path.exists(output_path):
                    print(f"\nProcessing paper: {paper_id}")
                    sections = html_section_splitting(paper_id)
                    if sections:
                        processed_count += 1
                else:
                    print(f"Sections already exist for {paper_id}, skipping...")
    
    print(f"\nBatch processing completed. Processed {processed_count} papers.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "all":
            batch_html_section_splitting()
        else:
            paper_id = sys.argv[1]
            html_section_splitting(paper_id)
    else:
        batch_html_section_splitting() 