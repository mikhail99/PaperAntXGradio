"""
Validation script for Iteration 2: Section Splitting & Planning

This script tests the planning pipeline:
1. Uses output from Iteration 1 (PDF processing)
2. Runs section splitting and selection
3. Validates the results
4. Generates a summary report
"""

import os
import sys
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_pdf_extraction_output():
    """Find the most recent PDF extraction output file."""
    output_dir = "output"
    if not os.path.exists(output_dir):
        return None
    
    # Look for *_extracted.json files
    extraction_files = [f for f in os.listdir(output_dir) if f.endswith("_extracted.json")]
    
    if not extraction_files:
        return None
    
    # Get the most recent one
    extraction_files.sort(key=lambda f: os.path.getmtime(os.path.join(output_dir, f)), reverse=True)
    return os.path.join(output_dir, extraction_files[0])

def run_planning_test(extraction_file_path):
    """Run the planning pipeline and return results."""
    
    # Import the planning flow
    import importlib.util
    spec = importlib.util.spec_from_file_location("planning_flow", "1_planning_flow.py")
    planning_flow_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planning_flow_module)
    
    # Load extracted data from PDF processing
    with open(extraction_file_path, 'r', encoding='utf-8') as f:
        pdf_data = json.load(f)
    
    # Create shared store with data from PDF processing
    shared = {
        "pdf_path": pdf_data.get("pdf_path", ""),
        "raw_text": pdf_data.get("raw_text", ""),
        "cleaned_text": pdf_data.get("cleaned_text", ""),
        "pdf_metadata": pdf_data.get("pdf_metadata", {}),
        "text_stats": pdf_data.get("text_stats", {}),
        "output_dir": "output"
    }
    
    logger.info(f"Loaded PDF data: {shared['text_stats'].get('word_count', 0)} words")
    
    # Create and run the planning flow
    flow = planning_flow_module.create_planning_flow()
    
    logger.info("Starting planning pipeline...")
    
    try:
        flow.run(shared)
        logger.info("Planning pipeline completed successfully!")
        return shared
    except Exception as e:
        logger.error(f"Error during planning pipeline: {str(e)}")
        raise

def validate_planning_results(shared):
    """Validate that all expected planning results are present and correct."""
    validation_results = {
        "passed": True,
        "checks": [],
        "errors": []
    }
    
    def add_check(name, condition, message):
        check_result = {
            "name": name,
            "passed": condition,
            "message": message
        }
        validation_results["checks"].append(check_result)
        if not condition:
            validation_results["passed"] = False
            validation_results["errors"].append(f"{name}: {message}")
    
    # Check section detection
    sections = shared.get("sections")
    add_check(
        "Section Detection",
        sections and len(sections) > 0,
        "Should detect at least one section"
    )
    
    if sections:
        add_check(
            "Section Structure",
            all(isinstance(s, dict) and "title" in s and "content" in s for s in sections),
            "All sections should have title and content"
        )
        
        add_check(
            "Section Types",
            any(s.get("section_type") in ["abstract", "introduction", "methodology", "results"] for s in sections),
            "Should detect common academic section types"
        )
    
    # Check section selection
    selected_sections = shared.get("selected_sections")
    add_check(
        "Section Selection",
        selected_sections and len(selected_sections) > 0,
        "Should select at least one section"
    )
    
    if selected_sections:
        add_check(
            "Selected Section Structure",
            all(isinstance(s, dict) and "title" in s and "selection_reason" in s for s in selected_sections),
            "All selected sections should have title and selection reason"
        )
        
        add_check(
            "Selection Priorities",
            all("priority" in s and isinstance(s["priority"], int) for s in selected_sections),
            "All selected sections should have numeric priorities"
        )
        
        add_check(
            "Reasonable Selection Count",
            1 <= len(selected_sections) <= 8,
            "Should select between 1-8 sections"
        )
    
    # Check planning summary
    planning_summary = shared.get("planning_summary")
    add_check(
        "Planning Summary",
        planning_summary and isinstance(planning_summary, dict),
        "Should generate planning summary"
    )
    
    # Check output file creation
    planning_output_file = shared.get("planning_output_file")
    add_check(
        "Planning Output File",
        planning_output_file and os.path.exists(planning_output_file),
        "Should create planning output file"
    )
    
    # Validate output file content
    if planning_output_file and os.path.exists(planning_output_file):
        try:
            with open(planning_output_file, 'r') as f:
                output_data = json.load(f)
            add_check(
                "Output File Content",
                "selected_sections" in output_data and "planning_summary" in output_data,
                "Output file should contain selected sections and planning summary"
            )
        except Exception as e:
            add_check(
                "Output File Content",
                False,
                f"Error reading output file: {str(e)}"
            )
    
    return validation_results

def generate_planning_report(shared, validation_results):
    """Generate a summary report of the planning results."""
    
    report = []
    report.append("=" * 60)
    report.append("ITERATION 2 VALIDATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Overall status
    status = "✅ PASSED" if validation_results["passed"] else "❌ FAILED"
    report.append(f"Overall Status: {status}")
    report.append("")
    
    # Input Information
    text_stats = shared.get("text_stats", {})
    report.append("Input Text Information:")
    report.append(f"  - Word Count: {text_stats.get('word_count', 'N/A')}")
    report.append(f"  - Character Count: {text_stats.get('char_count', 'N/A')}")
    report.append(f"  - Quality: {text_stats.get('extraction_quality', 'N/A')}")
    report.append("")
    
    # Section Detection Results
    sections = shared.get("sections", [])
    report.append("Section Detection:")
    report.append(f"  - Total Sections Detected: {len(sections)}")
    
    if sections:
        section_types = {}
        for section in sections:
            stype = section.get("section_type", "unknown")
            section_types[stype] = section_types.get(stype, 0) + 1
        
        report.append("  - Section Types:")
        for stype, count in section_types.items():
            report.append(f"    • {stype}: {count}")
    report.append("")
    
    # Section Selection Results
    selected_sections = shared.get("selected_sections", [])
    report.append("Section Selection:")
    report.append(f"  - Total Sections Selected: {len(selected_sections)}")
    
    if selected_sections:
        report.append("  - Selected Sections:")
        for i, section in enumerate(selected_sections):
            priority = section.get("priority", "N/A")
            section_type = section.get("section_type", "unknown")
            title = section.get("title", "Untitled")[:50]
            report.append(f"    {i+1}. [{priority}] {section_type}: {title}")
    report.append("")
    
    # Planning Summary
    planning_summary = shared.get("planning_summary", {})
    if planning_summary:
        report.append("Planning Summary:")
        report.append(f"  - Selection Method: {planning_summary.get('selection_method', 'N/A')}")
        report.append(f"  - Selection Criteria: {planning_summary.get('selection_criteria', 'N/A')}")
        key_areas = planning_summary.get("key_focus_areas", [])
        if key_areas:
            report.append(f"  - Key Focus Areas: {', '.join(key_areas)}")
    report.append("")
    
    # Output Files
    planning_output_file = shared.get("planning_output_file")
    report.append("Output Files:")
    report.append(f"  - Planning Results: {planning_output_file}")
    report.append("")
    
    # Validation Checks
    report.append("Validation Checks:")
    for check in validation_results["checks"]:
        status_icon = "✅" if check["passed"] else "❌"
        report.append(f"  {status_icon} {check['name']}: {check['message']}")
    report.append("")
    
    # Errors (if any)
    if validation_results["errors"]:
        report.append("Errors:")
        for error in validation_results["errors"]:
            report.append(f"  ❌ {error}")
        report.append("")
    
    # Content Sample
    if selected_sections:
        sample_section = selected_sections[0]
        content = sample_section.get("content", "")
        if content:
            sample_length = min(200, len(content))
            sample = content[:sample_length]
            if len(content) > sample_length:
                sample += "..."
            report.append("Sample Selected Content:")
            report.append(f"  Section: {sample_section.get('title', 'Untitled')}")
            report.append(f"  Content: {sample}")
            report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)

def main():
    """Main validation function."""
    logger.info("Starting Iteration 2 validation...")
    
    try:
        # Step 1: Find PDF extraction output
        extraction_file = find_pdf_extraction_output()
        
        if not extraction_file:
            print("❌ No PDF extraction output found. Please run Iteration 1 first.")
            print("   Expected file: output/*_extracted.json")
            return False
        
        logger.info(f"Using PDF extraction data: {extraction_file}")
        
        # Step 2: Run planning pipeline
        shared = run_planning_test(extraction_file)
        
        # Step 3: Validate results
        validation_results = validate_planning_results(shared)
        
        # Step 4: Generate report
        report = generate_planning_report(shared, validation_results)
        
        # Print report
        print(report)
        
        # Save report to file
        os.makedirs("output", exist_ok=True)
        report_file = "output/iteration2_validation_report.txt"
        with open(report_file, "w") as f:
            f.write(report)
        
        logger.info(f"Validation report saved to: {report_file}")
        
        return validation_results["passed"]
        
    except Exception as e:
        logger.error(f"Validation failed with error: {str(e)}")
        print(f"❌ Validation failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 