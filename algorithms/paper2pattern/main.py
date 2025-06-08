import argparse
import json
import os
import dspy

from src.dspy_config import configure_dspy
from src.modules import DictionaryExtractor, StandalonePatternExtractor

def find_section_content(sections, title_keyword):
    """Finds the content of the first section that contains a keyword in its title."""
    for section in sections:
        if title_keyword.lower() in section.get('title', '').lower():
            return section.get('content', '')
    return None

def run_dictionary_extraction(paper_id, sections):
    """Runs the domain dictionary extraction process."""
    print("\n--- Running Domain Dictionary Extraction ---")
    paper_title = sections[0].get('title', 'No Title Found')
    abstract_content = find_section_content(sections, 'abstract')
    target_section_content = find_section_content(sections, 'introduction')

    if not abstract_content:
        print("Warning: Abstract not found. Proceeding without it.")
        abstract_content = ""
        
    if not target_section_content:
        print("Error: Could not find 'Introduction' section.")
        return

    extractor = DictionaryExtractor()
    result = extractor(
        title=paper_title,
        abstract=abstract_content,
        paper_section=target_section_content
    )
    domain_dictionary = result.domain_dictionary
    print("\n--- Extracted Domain Dictionary ---")
    print(domain_dictionary)
    print("-----------------------------------")

    output_dir = os.path.join("outputs", paper_id)
    dictionary_path = os.path.join(output_dir, "dictionary.json")
    with open(dictionary_path, 'w') as f:
        json.dump({"dictionary": domain_dictionary}, f, indent=4)
    print(f"Successfully saved dictionary to {dictionary_path}")

def run_pattern_extraction(paper_id, sections):
    """Runs the full design pattern extraction process."""
    print("\n--- Running Full Pattern Extraction ---")
    
    # Concatenate all section content to form the full paper context
    paper_context = "\\n\\n".join([s.get('content', '') for s in sections if s.get('content')])
    
    if not paper_context:
        print("Error: No content found in sections to form paper context.")
        return

    extractor = StandalonePatternExtractor()
    extracted_pattern = extractor(paper_context=paper_context)
    
    output_dir = os.path.join("outputs", paper_id)
    pattern_path = os.path.join(output_dir, "pattern.json")
    with open(pattern_path, 'w') as f:
        json.dump(extracted_pattern, f, indent=4)
    print(f"Successfully saved pattern to {pattern_path}")


def main(paper_id, process_type):
    """
    Main function to run the extraction process for a single paper.
    """
    print(f"Starting {process_type} extraction for paper: {paper_id}")

    # 1. Configure DSPy
    try:
        configure_dspy()
        print("DSPy configured successfully.")
    except Exception as e:
        print(f"Error configuring DSPy: {e}")
        return

    # 2. Load the paper's sections
    output_dir = os.path.join("outputs", paper_id)
    sections_path = os.path.join(output_dir, "sections.json")

    if not os.path.exists(sections_path):
        print(f"Error: sections.json not found for paper_id '{paper_id}' at {sections_path}")
        return

    try:
        with open(sections_path, 'r') as f:
            sections = json.load(f)
        print(f"Loaded {len(sections)} sections from {sections_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {sections_path}")
        return

    if not sections:
        print("Error: The sections file is empty.")
        return

    # 3. Run the selected process
    if process_type == "dictionary":
        run_dictionary_extraction(paper_id, sections)
    elif process_type == "pattern":
        run_pattern_extraction(paper_id, sections)
    else:
        print(f"Error: Unknown process type '{process_type}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract information from a scientific paper.")
    parser.add_argument("paper_id", type=str, help="The ID of the paper to process.")
    parser.add_argument(
        "process_type", 
        type=str, 
        choices=['dictionary', 'pattern'],
        help="The type of extraction to perform: 'dictionary' or 'pattern'."
    )
    args = parser.parse_args()
    
    main(args.paper_id, args.process_type)
