#!/usr/bin/env python3
"""
Paper2ImplementationDoc - Generate implementation documentation from academic papers
Following PocketFlow-Tutorial-Codebase-Knowledge pattern
"""

import argparse
import os
import sys
from pathlib import Path
import logging
import time

from flow import create_paper2doc_flow

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )

def main():
    parser = argparse.ArgumentParser(
        description='Generate implementation documentation from academic papers using PocketFlow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --pdf paper.pdf --output ./docs --verbose
  %(prog)s --pdf paper.pdf --analysis-depth simplified --include-diagrams
  %(prog)s --arxiv 2301.07041 --output ./arxiv_docs
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--pdf', type=str, help='Path to PDF file')
    input_group.add_argument('--arxiv', type=str, help='ArXiv paper ID (e.g., 2301.07041)')
    
    # Output options
    parser.add_argument('--output', type=str, default='./output',
                       help='Output directory (default: ./output)')
    
    # Analysis options
    parser.add_argument('--analysis-depth', choices=['simplified', 'detailed'], 
                       default='detailed', help='Analysis depth (default: detailed)')
    parser.add_argument('--output-format', choices=['markdown', 'latex', 'html'], 
                       default='markdown', help='Output format (default: markdown)')
    parser.add_argument('--include-diagrams', action='store_true',
                       help='Include implementation diagrams')
    parser.add_argument('--max-sections', type=int, default=10,
                       help='Maximum sections to analyze (default: 10)')
    
    # System options
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0 (PocketFlow Edition)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Determine input
    if args.pdf:
        input_source = args.pdf
        input_type = "pdf"
    else:
        input_source = args.arxiv
        input_type = "arxiv"
    
    logger.info(f"🚀 Starting Paper2ImplementationDoc...")
    logger.info(f"   Input: {input_source} ({input_type})")
    logger.info(f"   Output: {args.output}")
    
    try:
        # Create the PocketFlow pipeline
        flow = create_paper2doc_flow(
            analysis_depth=args.analysis_depth,
            output_format=args.output_format,
            include_diagrams=args.include_diagrams,
            max_sections=args.max_sections,
            verbose=args.verbose
        )
        
        # Initialize shared data store
        shared = {
            "input_source": input_source,
            "input_type": input_type,
            "output_dir": args.output,
            "config": {
                "analysis_depth": args.analysis_depth,
                "output_format": args.output_format,
                "include_diagrams": args.include_diagrams,
                "max_sections": args.max_sections,
                "verbose": args.verbose
            },
            "metadata": {
                "processing_steps": [],
                "warnings": [],
                "stats": {},
                "flow_start_time": time.time()
            }
        }
        
        # Run the PocketFlow pipeline
        logger.info("🔄 Running PocketFlow pipeline...")
        flow.run(shared)
        
        # Post-processing
        shared["metadata"]["stats"]["total_flow_time"] = time.time() - shared["metadata"]["flow_start_time"]
        
        # Show results
        logger.info("🎉 Processing completed successfully!")
        if "output_files" in shared:
            logger.info(f"📄 Generated files: {shared['output_files']}")
        if "quality_score" in shared:
            logger.info(f"📊 Quality score: {shared['quality_score']}")
        
        # Save metadata
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        metadata_file = output_dir / "flow_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(shared["metadata"], f, indent=2, default=str)
        logger.info(f"💾 Metadata saved to {metadata_file}")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 