#!/usr/bin/env python3
"""
Main script to run the refactored beam graph filter pipeline
"""

import sys
import os
import argparse
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline.beam_filter_pipeline import BeamFilterPipeline
from pipeline.cli import load_samples

def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description="Beam Graph Filter Pipeline")
    parser.add_argument("--input", type=str, default="raw_test.json", help="Input JSON file")
    parser.add_argument("--output_dir", type=str, default="beam_filter_output", help="Output directory")
    parser.add_argument("--min_relevance", type=float, default=0.15, help="Minimum relevance threshold")
    parser.add_argument("--beam_width", type=int, default=40, help="Beam width")
    parser.add_argument("--max_depth", type=int, default=120, help="Maximum depth")
    parser.add_argument("--max_paths", type=int, default=200, help="Maximum paths")
    parser.add_argument("--max_final_sentences", type=int, default=30, help="Maximum final sentences")
    parser.add_argument("--beam_sentences", type=int, default=50, help="Maximum beam sentences")
    parser.add_argument("--beam_only", action="store_true", help="Use only beam search")
    parser.add_argument("--filter_only", action="store_true", help="Use only filtering")
    parser.add_argument("--use_sbert", action="store_true", help="Enable SBERT")
    parser.add_argument("--use_contradiction", action="store_true", help="Enable contradiction detection")
    parser.add_argument("--use_nli", action="store_true", help="Enable NLI")
    parser.add_argument("--vncorenlp_dir", type=str, default="", help="VnCoreNLP directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples to process")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract input name for output file naming
    input_name = os.path.splitext(os.path.basename(args.input))[0]
    
    # Load samples
    try:
        samples = load_samples(args.input, args.max_samples)
        print(f"Processing {len(samples)} samples from {args.input} with min_relevance_score={args.min_relevance} ...")
    except Exception as e:
        print(f"Error loading samples: {e}")
        return
    
    # Create pipeline
    pipeline = BeamFilterPipeline(
        vncorenlp_dir=args.vncorenlp_dir,
        use_sbert=args.use_sbert,
        use_contradiction_detection=args.use_contradiction,
        use_nli=args.use_nli
    )
    
    # Process batch
    try:
        results = pipeline.process_batch(
            samples=samples,
            output_dir=args.output_dir,
            min_relevance=args.min_relevance,
            beam_width=args.beam_width,
            max_depth=args.max_depth,
            max_paths=args.max_paths,
            max_final_sentences=args.max_final_sentences,
            beam_sentences=args.beam_sentences,
            beam_only=args.beam_only,
            filter_only=args.filter_only,
            input_name=input_name
        )
        
        print(f"\n✅ Processing completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Files created:")
        print(f"  • {results['detailed_file']}")
        print(f"  • {results['simple_file']}")
        print(f"  • {results['stats_file']}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return

if __name__ == "__main__":
    main() 