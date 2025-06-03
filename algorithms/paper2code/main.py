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
    """
    Main entry point for Paper2ImplementationDoc (Iteration 3)
    Enhanced with advanced component analysis
    """
    parser = argparse.ArgumentParser(description="Paper2ImplementationDoc - Convert academic papers to implementation guides")
    parser.add_argument("input_source", help="PDF file path or ArXiv ID")
    parser.add_argument("--input-type", choices=["pdf", "arxiv"], default="pdf", help="Input type")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--analysis-depth", choices=["simple", "detailed"], default="detailed", help="Analysis depth")
    parser.add_argument("--output-format", choices=["markdown", "html", "latex"], default="markdown", help="Output format")
    parser.add_argument("--include-diagrams", action="store_true", help="Include Mermaid diagrams")
    parser.add_argument("--max-sections", type=int, default=10, help="Maximum sections to detect")
    parser.add_argument("--disable-component-analysis", action="store_true", help="Disable advanced component analysis")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Paper2ImplementationDoc (Iteration 3) - Enhanced Component Analysis")
    
    # Create configuration
    config = {
        "analysis_depth": args.analysis_depth,
        "output_format": args.output_format,
        "include_diagrams": args.include_diagrams,
        "max_sections": args.max_sections,
        "enable_component_analysis": not args.disable_component_analysis,
        "verbose": args.verbose
    }
    
    # Initialize shared store
    shared = {
        "input_source": args.input_source,
        "input_type": args.input_type,
        "output_dir": args.output_dir,
        "config": config,
        "metadata": {
            "start_time": time.time(),
            "processing_steps": [],
            "warnings": [],
            "iteration": "3"
        }
    }
    
    try:
        # Create and run the enhanced flow
        flow = create_paper2doc_flow(config)
        
        logger.info(f"📄 Processing {args.input_type}: {args.input_source}")
        logger.info(f"🔬 Component analysis: {'enabled' if config['enable_component_analysis'] else 'disabled'}")
        
        # Execute the flow
        flow.run(shared)
        
        # Calculate processing time
        processing_time = time.time() - shared["metadata"]["start_time"]
        
        # Display results
        logger.info("✅ Processing completed successfully!")
        logger.info(f"⏱️  Total processing time: {processing_time:.3f} seconds")
        logger.info(f"📊 Quality score: {shared.get('quality_score', 0):.1f}%")
        
        # Display component analysis results if enabled
        if config['enable_component_analysis'] and 'paper_structure' in shared:
            component_stats = shared['paper_structure'].get('component_analysis', {}).get('analysis_stats', {})
            logger.info(f"🔬 Component analysis results:")
            logger.info(f"   - Algorithms found: {component_stats.get('algorithms_found', 0)}")
            logger.info(f"   - Math formulations: {component_stats.get('formulations_found', 0)}")
            logger.info(f"   - Methodology steps: {component_stats.get('methodology_steps_found', 0)}")
            logger.info(f"   - System components: {component_stats.get('system_components_found', 0)}")
            logger.info(f"   - Average confidence: {component_stats.get('average_confidence', 0.0):.2f}")
        
        # Display output files
        if "output_files" in shared:
            logger.info("📁 Generated files:")
            for file_path in shared["output_files"]:
                logger.info(f"   - {file_path}")
        
        # Display warnings if any
        if shared["metadata"]["warnings"]:
            logger.warning("⚠️  Warnings encountered:")
            for warning in shared["metadata"]["warnings"]:
                logger.warning(f"   - {warning}")
        
        # Display processing steps summary
        if args.verbose:
            logger.debug("📋 Processing steps summary:")
            for step in shared["metadata"]["processing_steps"]:
                logger.debug(f"   - {step['step']}: {step.get('status', 'completed')}")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {str(e)}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    main() 