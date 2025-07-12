#!/usr/bin/env python3
"""
CLI for TextGraph - Simple terminal commands to test graph functions
"""

import argparse
import sys
import os
from .text_graph import TextGraph
from src.nlp.vncorenlp_wrapper import VnCoreNLPWrapper

def test_init():
    """Test TextGraph initialization"""
    print("🔧 Testing TextGraph initialization...")
    graph = TextGraph()
    stats = graph.get_statistics()
    print(f"✅ TextGraph initialized: {stats['total_nodes']} nodes")
    return True

def test_build_graph(context, claim):
    """Test building graph from context and claim"""
    print("🔧 Building graph...")
    graph = TextGraph()
    nlp = VnCoreNLPWrapper()
    
    try:
        context_tokens = nlp.annotate_text(context)
        claim_tokens = nlp.annotate_text(claim)
        graph.build_from_vncorenlp_output(context_tokens, claim, claim_tokens)
        
        stats = graph.get_statistics()
        print(f"✅ Graph built: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
        
        detailed = graph.get_detailed_statistics()
        print(f"📊 Shared words: {detailed['shared_words_count']}")
        
        return graph
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_semantic(graph):
    """Test semantic analysis"""
    print("🔧 Testing semantic analysis...")
    try:
        edges_before = graph.get_statistics()['total_edges']
        edges_added = graph.build_semantic_similarity_edges()
        edges_after = graph.get_statistics()['total_edges']
        
        print(f"📊 Semantic edges: {edges_before} -> {edges_after} (+{edges_added})")
        
        semantic_stats = graph.get_semantic_statistics()
        print(f"📊 Semantic stats: {semantic_stats['total_semantic_edges']} edges")
        
        return True
    except Exception as e:
        print(f"❌ Semantic error: {e}")
        return False

def test_beam_search(graph, beam_width=10, max_depth=15, max_paths=5):
    """Test beam search"""
    print("🔧 Testing beam search...")
    try:
        paths = graph.beam_search_paths(beam_width=beam_width, max_depth=max_depth, max_paths=max_paths)
        print(f"✅ Found {len(paths)} paths")
        
        for i, path in enumerate(paths[:3]):
            print(f"  Path {i+1}: Score={path.score:.4f}, Length={len(path.nodes)}")
        
        quality = graph.analyze_paths_quality(paths)
        print(f"📊 Quality: avg_score={quality['avg_score']:.4f}")
        
        return paths
    except Exception as e:
        print(f"❌ Beam search error: {e}")
        return []

def test_export(graph, paths=None):
    """Test exporting data"""
    print("🔧 Testing export...")
    try:
        os.makedirs('output', exist_ok=True)
        
        # Export GEXF
        graph.save_graph('output/graph_test.gexf')
        print("✅ GEXF exported")
        
        # Export JSON
        json_data = graph.export_to_json()
        with open('output/graph_test.json', 'w', encoding='utf-8') as f:
            f.write(json_data)
        print("✅ JSON exported")
        
        # Export beam search results
        if paths:
            json_path, summary_path = graph.export_beam_search_results(paths, 'output', 'beam_search')
            print("✅ Beam search results exported")
        
        return True
    except Exception as e:
        print(f"❌ Export error: {e}")
        return False

def test_statistics(graph):
    """Test detailed statistics"""
    print("🔧 Testing statistics...")
    try:
        detailed = graph.get_detailed_statistics()
        print(f"📊 Total nodes: {detailed['total_nodes']}")
        print(f"📊 Total edges: {detailed['total_edges']}")
        print(f"📊 Word nodes: {detailed['word_nodes']}")
        print(f"📊 Sentence nodes: {detailed['sentence_nodes']}")
        print(f"📊 Shared words: {detailed['shared_words_count']}")
        
        shared_words = graph.get_shared_words()
        print(f"🔍 Shared words: {len(shared_words)}")
        for word_info in shared_words[:3]:
            print(f"  - {word_info['word']} (POS: {word_info['pos']})")
        
        return True
    except Exception as e:
        print(f"❌ Statistics error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='TextGraph CLI')
    parser.add_argument('command', choices=['init', 'build', 'semantic', 'beam', 'export', 'stats', 'all'], 
                       help='Command to run')
    parser.add_argument('--context', default='SAWACO thông báo tạm ngưng cung cấp nước từ 22 giờ đến 4 giờ.',
                       help='Context text')
    parser.add_argument('--claim', default='SAWACO thông báo tạm ngưng cung cấp nước.',
                       help='Claim text')
    parser.add_argument('--beam-width', type=int, default=10, help='Beam width')
    parser.add_argument('--max-depth', type=int, default=15, help='Max depth')
    parser.add_argument('--max-paths', type=int, default=5, help='Max paths')
    
    args = parser.parse_args()
    
    if args.command == 'init':
        test_init()
    
    elif args.command == 'build':
        test_build_graph(args.context, args.claim)
    
    elif args.command == 'semantic':
        graph = test_build_graph(args.context, args.claim)
        if graph:
            test_semantic(graph)
    
    elif args.command == 'beam':
        graph = test_build_graph(args.context, args.claim)
        if graph:
            test_beam_search(graph, args.beam_width, args.max_depth, args.max_paths)
    
    elif args.command == 'export':
        graph = test_build_graph(args.context, args.claim)
        if graph:
            paths = test_beam_search(graph, args.beam_width, args.max_depth, args.max_paths)
            test_export(graph, paths)
    
    elif args.command == 'stats':
        graph = test_build_graph(args.context, args.claim)
        if graph:
            test_statistics(graph)
    
    elif args.command == 'all':
        print("🚀 Running all tests...")
        test_init()
        graph = test_build_graph(args.context, args.claim)
        if graph:
            test_semantic(graph)
            paths = test_beam_search(graph, args.beam_width, args.max_depth, args.max_paths)
            test_export(graph, paths)
            test_statistics(graph)
        print("🎉 All tests completed!")

if __name__ == '__main__':
    main() 