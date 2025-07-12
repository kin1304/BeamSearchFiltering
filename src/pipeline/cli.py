"""
CLI script for beam graph filter pipeline
"""

import argparse
import json
import os
from typing import List, Dict, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.pipeline.beam_filter_pipeline import BeamFilterPipeline


def load_samples(input_file: str, max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load samples from input file
    
    Args:
        input_file (str): Path to input JSON file
        max_samples (int): Maximum number of samples to load
        
    Returns:
        List[Dict]: List of samples
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    with open(input_file, "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    # Handle both array and single object
    if isinstance(samples, list):
        if len(samples) == 0:
            raise ValueError("Input file contains empty array")
    else:
        # Handle single object
        samples = [samples]
    
    # Apply max_samples slicing if requested
    if max_samples is not None and max_samples < len(samples):
        samples = samples[:max_samples]
        print(f"⚠️  Only processing {len(samples)} samples according to --max_samples={max_samples}")
    
    return samples


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Beam Graph + Advanced Filter pipeline")
    parser.add_argument("--input", type=str, default="raw_test.json", help="Input JSON file")
    parser.add_argument("--output_dir", type=str, default="beam_filter_output", help="Output directory")
    parser.add_argument("--min_relevance", type=float, default=0.15, help="Minimum relevance threshold")
    parser.add_argument("--beam_width", type=int, default=40, help="Beam width (number of paths to keep each step)")
    parser.add_argument("--max_depth", type=int, default=120, help="Maximum depth of beam search")
    parser.add_argument("--max_paths", type=int, default=200, help="Maximum number of paths to return")
    parser.add_argument("--max_final_sentences", type=int, default=30, help="Maximum final sentences to keep")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--beam_sentences", type=int, default=50, help="Maximum sentences from Beam Search before filtering")
    parser.add_argument("--beam_only", action="store_true", help="Use only Beam Search, no Advanced Data Filter")
    parser.add_argument("--filter_only", action="store_true", help="Use only Advanced Data Filter, no Beam Search")
    parser.add_argument("--use_sbert", action="store_true", help="Enable SBERT semantic filtering")
    parser.add_argument("--use_contradiction", action="store_true", help="Enable contradiction detection")
    parser.add_argument("--use_nli", action="store_true", help="Enable NLI stance detection")
    parser.add_argument("--vncorenlp_dir", type=str, default="", help="Path to VnCoreNLP directory")
    
    args = parser.parse_args()
    
    # Convert output_dir to absolute path
    args.output_dir = os.path.abspath(args.output_dir)
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
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return


if __name__ == "__main__":
    main() 