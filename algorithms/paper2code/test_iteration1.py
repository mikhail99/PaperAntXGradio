"""
Validation script for Iteration 1: Core PDF Processing

This script tests the entire PDF processing pipeline:
1. Creates a test PDF (if needed)
2. Runs the PDF processing flow
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

def create_test_pdf_if_needed():
    """Create a test PDF if it doesn't exist."""
    test_pdf_path = "test_paper.pdf"
    
    if os.path.exists(test_pdf_path):
        logger.info(f"Test PDF already exists: {test_pdf_path}")
        return test_pdf_path
    
    try:
        # Try to create PDF with reportlab
        from create_test_pdf import create_test_pdf
        pdf_path = create_test_pdf(test_pdf_path)
        logger.info(f"Created test PDF: {pdf_path}")
        return pdf_path
    except ImportError:
        logger.warning("reportlab not available. Using fallback text file method.")
        # Create a simple text file for manual conversion
        text_content = """A Novel Approach to Machine Learning Optimization

Abstract

This paper presents a novel optimization algorithm for machine learning models.
Our approach combines gradient descent with adaptive learning rates to achieve
faster convergence and better performance.

1. Introduction

Machine learning optimization is a fundamental challenge in artificial intelligence.
In this work, we propose AdaptiveGrad, a novel optimization algorithm.

2. Methodology  

Our proposed AdaptiveGrad algorithm consists of three main components:
1. Gradient History Tracking
2. Curvature Estimation  
3. Adaptive Rate Adjustment

3. Results

We evaluated AdaptiveGrad on benchmark datasets and show improvements.

4. Conclusion

We have presented AdaptiveGrad, a novel optimization algorithm."""
        
        with open("test_paper_content.txt", "w") as f:
            f.write(text_content)
        
        logger.info("Created test_paper_content.txt - please convert to PDF manually")
        return None

def run_pdf_processing_test(pdf_path):
    """Run the PDF processing pipeline and return results."""
    
    # Import the flow (handle the numeric filename)
    import importlib.util
    spec = importlib.util.spec_from_file_location("pdf_flow", "0_pdf_process_flow.py")
    pdf_flow_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pdf_flow_module)
    
    # Create shared store
    shared = {
        "pdf_path": pdf_path,
        "output_dir": "output"
    }
    
    # Create and run the flow
    flow = pdf_flow_module.create_pdf_processing_flow()
    
    logger.info(f"Starting PDF processing for: {pdf_path}")
    
    try:
        flow.run(shared)
        logger.info("PDF processing completed successfully!")
        return shared
    except Exception as e:
        logger.error(f"Error during PDF processing: {str(e)}")
        raise

def validate_results(shared):
    """Validate that all expected results are present and correct."""
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
    
    # Check PDF validation results
    pdf_validation = shared.get("pdf_validation")
    add_check(
        "PDF Validation", 
        pdf_validation and pdf_validation.get("valid") == True,
        "PDF should be validated successfully"
    )
    
    # Check PDF metadata
    pdf_metadata = shared.get("pdf_metadata")
    add_check(
        "PDF Metadata",
        pdf_metadata and "page_count" in pdf_metadata,
        "PDF metadata should include page count"
    )
    
    # Check text extraction
    raw_text = shared.get("raw_text")
    add_check(
        "Raw Text Extraction",
        raw_text and len(raw_text.strip()) > 0,
        "Raw text should be extracted and non-empty"
    )
    
    cleaned_text = shared.get("cleaned_text")
    add_check(
        "Cleaned Text",
        cleaned_text and len(cleaned_text.strip()) > 0,
        "Cleaned text should be processed and non-empty"
    )
    
    # Check text statistics
    text_stats = shared.get("text_stats")
    add_check(
        "Text Statistics",
        text_stats and text_stats.get("word_count", 0) > 0,
        "Text statistics should include word count > 0"
    )
    
    # Check output file creation
    output_file = shared.get("extraction_output_file")
    add_check(
        "Output File Creation",
        output_file and os.path.exists(output_file),
        "Output JSON file should be created and exist"
    )
    
    # Validate output file content
    if output_file and os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                output_data = json.load(f)
            add_check(
                "Output File Content",
                "raw_text" in output_data and "text_stats" in output_data,
                "Output file should contain extracted text and statistics"
            )
        except Exception as e:
            add_check(
                "Output File Content",
                False,
                f"Error reading output file: {str(e)}"
            )
    
    return validation_results

def generate_summary_report(shared, validation_results):
    """Generate a summary report of the processing results."""
    
    report = []
    report.append("=" * 60)
    report.append("ITERATION 1 VALIDATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Overall status
    status = "✅ PASSED" if validation_results["passed"] else "❌ FAILED"
    report.append(f"Overall Status: {status}")
    report.append("")
    
    # PDF Information
    pdf_metadata = shared.get("pdf_metadata", {})
    report.append("PDF Information:")
    report.append(f"  - Pages: {pdf_metadata.get('page_count', 'N/A')}")
    report.append(f"  - File Size: {pdf_metadata.get('file_size_mb', 'N/A')} MB")
    report.append("")
    
    # Text Statistics
    text_stats = shared.get("text_stats", {})
    report.append("Text Extraction:")
    report.append(f"  - Word Count: {text_stats.get('word_count', 'N/A')}")
    report.append(f"  - Character Count: {text_stats.get('char_count', 'N/A')}")
    report.append(f"  - Quality: {text_stats.get('extraction_quality', 'N/A')}")
    report.append(f"  - Method: {text_stats.get('extraction_method', 'N/A')}")
    report.append("")
    
    # Output Files
    output_file = shared.get("extraction_output_file")
    report.append("Output Files:")
    report.append(f"  - Extraction Data: {output_file}")
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
    
    # Text Sample
    cleaned_text = shared.get("cleaned_text", "")
    if cleaned_text:
        sample_length = min(200, len(cleaned_text))
        sample = cleaned_text[:sample_length]
        if len(cleaned_text) > sample_length:
            sample += "..."
        report.append("Text Sample:")
        report.append(f"  {sample}")
        report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)

def main():
    """Main validation function."""
    logger.info("Starting Iteration 1 validation...")
    
    try:
        # Step 1: Create test PDF if needed
        pdf_path = create_test_pdf_if_needed()
        
        if not pdf_path or not os.path.exists(pdf_path):
            print("❌ No test PDF available. Please create a PDF file manually and rerun.")
            return False
        
        # Step 2: Run PDF processing
        shared = run_pdf_processing_test(pdf_path)
        
        # Step 3: Validate results
        validation_results = validate_results(shared)
        
        # Step 4: Generate report
        report = generate_summary_report(shared, validation_results)
        
        # Print report
        print(report)
        
        # Save report to file
        os.makedirs("output", exist_ok=True)
        report_file = "output/iteration1_validation_report.txt"
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