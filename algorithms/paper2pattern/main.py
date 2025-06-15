import argparse
import json
import os
import dspy

from src.dspy_config import configure_dspy
from src.modules import DictionaryExtractor, BlueprintExtractor, ImportanceAssessor
from src.html_section_splitter import html_section_splitting

def find_section_content(sections, title_keyword):
    """Finds the content of the first section that contains a keyword in its title."""
    for section in sections:
        if title_keyword.lower() in section.get('title', '').lower():
            return section.get('content', '')
    return None

def run_dictionary_extraction(paper_id, sections):
    """Runs the domain dictionary extraction process for all sections."""
    print("\n--- Running Domain Dictionary Extraction for All Sections ---")
    
    output_dir = os.path.join("outputs", paper_id)
    dictionary_path = os.path.join(output_dir, "dictionary.json")
    importance_path = os.path.join(output_dir, "importance.json")

    # Load importance data
    if not os.path.exists(importance_path):
        print(f"Error: Importance data not found at {importance_path}. Please run the 'assess' process first.")
        return
    try:
        with open(importance_path, 'r') as f:
            importance_data = json.load(f)
        importance_map = {item['section_title']: item['is_important'] for item in importance_data}
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"Error: Could not load or parse importance data from {importance_path}. Please run the 'assess' process first.")
        return

    # Load existing dictionary if it exists
    try:
        if os.path.exists(dictionary_path):
            with open(dictionary_path, 'r') as f:
                full_dictionary_by_section = json.load(f)
        else:
            full_dictionary_by_section = {}
    except json.JSONDecodeError:
        print(f"Warning: Could not parse existing dictionary file at {dictionary_path}. Starting fresh.")
        full_dictionary_by_section = {}

    paper_title = sections[0].get('title', 'No Title Found')
    abstract_content = find_section_content(sections, 'abstract')
    if not abstract_content:
        print("Warning: Abstract not found. Proceeding without it.")
        abstract_content = ""

    extractor = DictionaryExtractor()

    for i, section in enumerate(sections):
        section_title = section.get('title', f'Section {i+1}')
        section_content = section.get('content', '')

        if section_title in full_dictionary_by_section:
            print(f"Skipping section '{section_title}' (already processed).")
            continue
        
        # Use pre-computed importance
        if not importance_map.get(section_title, False):
            print(f"Skipping unimportant section: '{section_title}'")
            continue

        if not section_content or len(section_content.split()) < 20:
            print(f"Skipping short section: '{section_title}'")
            continue

        print(f"Processing section: '{section_title}'...")
        result = extractor(
            title=paper_title,
            abstract=abstract_content,
            paper_section=section_content
        )
        domain_dictionary_str = result.domain_dictionary

        if domain_dictionary_str.strip().startswith("```json"):
            json_content = domain_dictionary_str.strip()[7:-3].strip()
        else:
            json_content = domain_dictionary_str

        try:
            parsed_terms = json.loads(json_content)
            if isinstance(parsed_terms, list) and parsed_terms:
                full_dictionary_by_section[section_title] = parsed_terms
                # Save after each successful extraction
                with open(dictionary_path, 'w') as f:
                    json.dump(full_dictionary_by_section, f, indent=4)
                print(f"Saved dictionary for section: '{section_title}'")
        except (json.JSONDecodeError, TypeError):
            print(f"Warning: Could not parse dictionary from section '{section_title}'. Skipping.")
            continue

    print(f"\n--- Completed Dictionary Extraction for {len(full_dictionary_by_section)} Sections ---")
    print(f"Final dictionary saved to {dictionary_path}")


def run_blueprint_extraction(paper_id, sections):
    """Runs the implementation blueprint extraction process for each section."""
    print("\n--- Running Implementation Blueprint Extraction for All Sections ---")
    
    output_dir = os.path.join("outputs", paper_id)
    blueprint_path = os.path.join(output_dir, "blueprint.json")
    importance_path = os.path.join(output_dir, "importance.json")
    dictionary_path = os.path.join(output_dir, "dictionary.json")

    # Load importance data
    if not os.path.exists(importance_path):
        print(f"Error: Importance data not found at {importance_path}. Please run the 'assess' process first.")
        return
    try:
        with open(importance_path, 'r') as f:
            importance_data = json.load(f)
        importance_map = {item['section_title']: item['is_important'] for item in importance_data}
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"Error: Could not load or parse importance data from {importance_path}. Please run the 'assess' process first.")
        return

    # Load domain dictionary data
    if not os.path.exists(dictionary_path):
        print(f"Error: Domain dictionary not found at {dictionary_path}. Please run the 'dictionary' process first.")
        return
    try:
        with open(dictionary_path, 'r') as f:
            full_dictionary_by_section = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"Error: Could not load or parse domain dictionary from {dictionary_path}. Please run 'dictionary' process first.")
        return

    # Load existing blueprints if the file exists
    try:
        if os.path.exists(blueprint_path):
            with open(blueprint_path, 'r') as f:
                full_blueprint = json.load(f)
        else:
            full_blueprint = []
    except json.JSONDecodeError:
        print(f"Warning: Could not parse existing blueprint file at {blueprint_path}. Starting fresh.")
        full_blueprint = []
    
    # Create a set of processed section titles for quick lookup
    processed_titles = {item['section_title'] for item in full_blueprint}

    paper_title = sections[0].get('title', 'No Title Found')
    abstract_content = find_section_content(sections, 'abstract')
    if not abstract_content:
        print("Warning: Abstract not found. Proceeding without it.")
        abstract_content = ""

    extractor = BlueprintExtractor()

    for i, section in enumerate(sections):
        section_title = section.get('title', f'Section {i+1}')
        section_content = section.get('content', '')

        if section_title in processed_titles:
            print(f"Skipping section '{section_title}' (already processed).")
            continue
        
        # Use pre-computed importance
        if not importance_map.get(section_title, False):
            print(f"Skipping unimportant section: '{section_title}'")
            continue

        # Skip very short sections or sections that are just references
        if not section_content or len(section_content.split()) < 50:
            print(f"Skipping short section: '{section_title}'")
            continue
        
        # Get the domain dictionary for this section
        section_dictionary = full_dictionary_by_section.get(section_title, [])
        section_dictionary_str = json.dumps(section_dictionary) if section_dictionary else "{}"

        print(f"Processing section: '{section_title}'...")
        result = extractor(
            title=paper_title,
            abstract=abstract_content,
            paper_section=section_content,
            domain_dictionary=section_dictionary_str
        )
        
        full_blueprint.append({
            "section_title": section_title,
            "blueprint": result.implementation_blueprint
        })
        # Save after each successful extraction
        with open(blueprint_path, 'w') as f:
            json.dump(full_blueprint, f, indent=4)
        print(f"Saved blueprint for section: '{section_title}'")

    print(f"\n--- Completed Blueprint Generation for {len(full_blueprint)} Sections ---")
    print(f"Final blueprint list saved to {blueprint_path}")


def run_importance_assessment(paper_id, sections):
    """Runs the section importance assessment process and saves the results."""
    print("\n--- Running Section Importance Assessment ---")

    output_dir = os.path.join("outputs", paper_id)
    importance_path = os.path.join(output_dir, "importance.json")

    # Load existing importance data if it exists
    try:
        if os.path.exists(importance_path):
            with open(importance_path, 'r') as f:
                importance_data = json.load(f)
        else:
            importance_data = []
    except json.JSONDecodeError:
        print(f"Warning: Could not parse existing importance file at {importance_path}. Starting fresh.")
        importance_data = []

    processed_titles = {item['section_title'] for item in importance_data}
    assessor = ImportanceAssessor()

    for i, section in enumerate(sections):
        section_title = section.get('title', f'Section {i+1}')
        section_content = section.get('content', '')

        if section_title in processed_titles:
            print(f"Skipping section '{section_title}' (already assessed).")
            continue

        if not section_content.strip():
            print(f"Skipping empty section: '{section_title}'")
            continue

        print(f"Assessing section: '{section_title}'...")
        content_preview = " ".join(section_content.split()[:100])
        is_important = assessor(section_title=section_title, content_preview=content_preview)

        importance_data.append({
            "section_title": section_title,
            "is_important": is_important
        })

        with open(importance_path, 'w') as f:
            json.dump(importance_data, f, indent=4)
        
        assessment = "Important" if is_important else "Unimportant"
        print(f"Saved assessment for '{section_title}': {assessment}")

    print(f"\n--- Completed Importance Assessment for {len(importance_data)} Sections ---")
    print(f"Assessment data saved to {importance_path}")


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
    elif process_type == "blueprint":
        run_blueprint_extraction(paper_id, sections)
    elif process_type == "assess":
        run_importance_assessment(paper_id, sections)
    else:
        print(f"Error: Unknown process type '{process_type}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract information from a scientific paper.")
    parser.add_argument("paper_id", type=str, help="The ID of the paper to process.")
    parser.add_argument(
        "process_type", 
        type=str, 
        choices=['dictionary', 'blueprint', 'assess', 'split'],
        help="The type of processing to perform: 'split', 'assess', 'dictionary', or 'blueprint'."
    )
    args = parser.parse_args()
    
    # The 'split' process is special as it doesn't need pre-loaded sections or dspy config
    if args.process_type == 'split':
        print(f"Starting HTML section splitting for paper: {args.paper_id}")
        html_section_splitting(args.paper_id)
    else:
        main(args.paper_id, args.process_type)
