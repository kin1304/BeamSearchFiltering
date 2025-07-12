#!/usr/bin/env python3
"""
CLI for BeamSearch - Simple terminal commands to test beam search functions
"""

import argparse
import sys
import os
from .beam_search import BeamSearchPathFinder
from mint.graph.text_graph import TextGraph
from src.nlp.vncorenlp_wrapper import VnCoreNLPWrapper

def test_init():
    """Test BeamSearchPathFinder initialization"""
    print("🔧 Testing BeamSearchPathFinder initialization...")
    try:
        # Create a simple graph for testing
        graph = TextGraph()
        graph.add_claim_node("Test claim")
        graph.add_sentence_node(0, "Test sentence")
        
        path_finder = BeamSearchPathFinder(graph, beam_width=5, max_depth=10)
        print("✅ BeamSearchPathFinder initialized")
        return path_finder
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_basic_search(context, claim, beam_width=10, max_depth=15, max_paths=5):
    """Test basic beam search"""
    print("🔧 Testing basic beam search...")
    try:
        # Build graph
        graph = TextGraph()
        nlp = VnCoreNLPWrapper()
        
        context_tokens = nlp.annotate_text(context)
        claim_tokens = nlp.annotate_text(claim)
        graph.build_from_vncorenlp_output(context_tokens, claim, claim_tokens)
        
        # Create path finder
        path_finder = BeamSearchPathFinder(graph, beam_width=beam_width, max_depth=max_depth)
        
        # Find paths
        paths = path_finder.find_best_paths(max_paths=max_paths)
        print(f"✅ Found {len(paths)} paths")
        
        for i, path in enumerate(paths[:3]):
            print(f"  Path {i+1}: Score={path.score:.4f}, Length={len(path.nodes)}")
            print(f"    Nodes: {' -> '.join(path.nodes[:5])}...")
        
        return paths
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def test_multi_level_search(context, claim, max_levels=3, beam_width_per_level=3, max_depth=20):
    """Test multi-level beam search"""
    print("🔧 Testing multi-level beam search...")
    try:
        # Build graph
        graph = TextGraph()
        nlp = VnCoreNLPWrapper()
        
        context_tokens = nlp.annotate_text(context)
        claim_tokens = nlp.annotate_text(claim)
        graph.build_from_vncorenlp_output(context_tokens, claim, claim_tokens)
        
        # Create path finder
        path_finder = BeamSearchPathFinder(graph, beam_width=25, max_depth=max_depth)
        
        # Run multi-level search
        multi_results = path_finder.multi_level_beam_search(
            max_levels=max_levels,
            beam_width_per_level=beam_width_per_level,
            min_new_sentences=0
        )
        
        print(f"✅ Multi-level search completed")
        for level, paths in multi_results.items():
            print(f"  Level {level}: {len(paths)} paths")
            if paths:
                avg_score = sum(p.score for p in paths) / len(paths)
                print(f"    Avg score: {avg_score:.4f}")
        
        return multi_results
    except Exception as e:
        print(f"❌ Error: {e}")
        return {}

def test_export_paths(paths, output_dir="output"):
    """Test exporting paths"""
    print("🔧 Testing path export...")
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a simple graph for export
        graph = TextGraph()
        graph.add_claim_node("Test claim")
        path_finder = BeamSearchPathFinder(graph)
        
        # Export JSON
        json_path = path_finder.export_paths_to_file(paths, f"{output_dir}/paths_test.json")
        print(f"✅ JSON exported: {json_path}")
        
        # Export summary
        summary_path = path_finder.export_paths_summary(paths, f"{output_dir}/paths_summary.txt")
        print(f"✅ Summary exported: {summary_path}")
        
        return True
    except Exception as e:
        print(f"❌ Export error: {e}")
        return False

def test_path_analysis(paths):
    """Test path analysis"""
    print("🔧 Testing path analysis...")
    try:
        if not paths:
            print("⚠️ No paths to analyze")
            return
        
        total_paths = len(paths)
        scores = [p.score for p in paths]
        lengths = [len(p.nodes) for p in paths]
        
        sentences_reached = sum(1 for p in paths if any(
            node.startswith('sentence') for node in p.nodes
        ))
        
        print(f"📊 Path Analysis:")
        print(f"  - Total paths: {total_paths}")
        print(f"  - Avg score: {sum(scores) / total_paths:.4f}")
        print(f"  - Avg length: {sum(lengths) / total_paths:.2f}")
        print(f"  - Paths to sentences: {sentences_reached}")
        print(f"  - Sentence reach rate: {sentences_reached / total_paths:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='BeamSearch CLI')
    parser.add_argument('command', choices=['init', 'basic', 'multi', 'export', 'analyze', 'all'], 
                       help='Command to run')
    parser.add_argument('--context', default='SAWACO thông báo tạm ngưng cung cấp nước từ 22 giờ đến 4 giờ. Các khu vực bị ảnh hưởng gồm quận 6, 8, 12.',
                       help='Context text')
    parser.add_argument('--claim', default='SAWACO thông báo tạm ngưng cung cấp nước.',
                       help='Claim text')
    parser.add_argument('--beam-width', type=int, default=10, help='Beam width')
    parser.add_argument('--max-depth', type=int, default=15, help='Max depth')
    parser.add_argument('--max-paths', type=int, default=5, help='Max paths')
    parser.add_argument('--max-levels', type=int, default=3, help='Max levels for multi-level')
    parser.add_argument('--beam-width-per-level', type=int, default=3, help='Beam width per level')
    
    args = parser.parse_args()
    
    if args.command == 'init':
        test_init()
    
    elif args.command == 'basic':
        test_basic_search(args.context, args.claim, args.beam_width, args.max_depth, args.max_paths)
    
    elif args.command == 'multi':
        test_multi_level_search(args.context, args.claim, args.max_levels, args.beam_width_per_level, args.max_depth)
    
    elif args.command == 'export':
        paths = test_basic_search(args.context, args.claim, args.beam_width, args.max_depth, args.max_paths)
        if paths:
            test_export_paths(paths)
    
    elif args.command == 'analyze':
        paths = test_basic_search(args.context, args.claim, args.beam_width, args.max_depth, args.max_paths)
        if paths:
            test_path_analysis(paths)
    
    elif args.command == 'all':
        print("🚀 Running all beam search tests...")
        test_init()
        paths = test_basic_search(args.context, args.claim, args.beam_width, args.max_depth, args.max_paths)
        if paths:
            test_export_paths(paths)
            test_path_analysis(paths)
        test_multi_level_search(args.context, args.claim, args.max_levels, args.beam_width_per_level, args.max_depth)
        print("🎉 All beam search tests completed!")

if __name__ == '__main__':
    main() 