#!/usr/bin/env python3
"""
CLI for Advanced Data Filtering - Simple terminal commands to test filtering functions
"""

import argparse
import sys
import os
import json
from .advanced_data_filtering import AdvancedDataFilter

def test_init():
    """Test AdvancedDataFilter initialization"""
    print("🔧 Testing AdvancedDataFilter initialization...")
    try:
        filter_obj = AdvancedDataFilter()
        print("✅ AdvancedDataFilter initialized successfully")
        return filter_obj
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_quality_filtering(filter_obj, sentences):
    """Test quality filtering"""
    print("🔧 Testing quality filtering...")
    try:
        filtered = filter_obj._stage1_quality_filtering(sentences, min_quality_score=0.3)
        print(f"✅ Quality filtering: {len(filtered)}/{len(sentences)} kept")
        return filtered
    except Exception as e:
        print(f"❌ Quality filtering error: {e}")
        return sentences

def test_semantic_filtering(filter_obj, sentences, claim_text):
    """Test semantic relevance filtering"""
    print("🔧 Testing semantic relevance filtering...")
    try:
        filtered = filter_obj._stage2_semantic_relevance_filtering(
            sentences, claim_text, min_relevance_score=0.25, max_final_sentences=30
        )
        print(f"✅ Semantic filtering: {len(filtered)}/{len(sentences)} kept")
        return filtered
    except Exception as e:
        print(f"❌ Semantic filtering error: {e}")
        return sentences

def test_entity_filtering(filter_obj, sentences, entities, claim_text):
    """Test entity-based filtering"""
    print("🔧 Testing entity-based filtering...")
    try:
        filtered = filter_obj._stage3_entity_based_filtering(
            sentences, entities, claim_text, min_entity_score=0.05, min_entity_keep=5
        )
        print(f"✅ Entity filtering: {len(filtered)}/{len(sentences)} kept")
        return filtered
    except Exception as e:
        print(f"❌ Entity filtering error: {e}")
        return sentences

def test_contradiction_detection(filter_obj, sentences, claim_text):
    """Test contradiction detection"""
    print("🔧 Testing contradiction detection...")
    try:
        filtered = filter_obj._stage4_contradiction_detection(sentences, claim_text, delta=0.1)
        print(f"✅ Contradiction detection: {len(filtered)}/{len(sentences)} kept")
        return filtered
    except Exception as e:
        print(f"❌ Contradiction detection error: {e}")
        return sentences

def test_multi_stage_pipeline(filter_obj, sentences, claim_text, context_text="", entities=None):
    """Test complete multi-stage filtering pipeline"""
    print("🔧 Testing multi-stage filtering pipeline...")
    try:
        results = filter_obj.multi_stage_filtering_pipeline(
            sentences, claim_text, context_text, entities,
            min_quality_score=0.3,
            min_relevance_score=0.25,
            min_entity_score=0.05,
            stance_delta=0.1,
            max_final_sentences=30,
            min_entity_keep=5
        )
        print(f"✅ Multi-stage pipeline completed")
        print(f"📊 Input: {results['input_count']} sentences")
        print(f"📊 Output: {results['final_count']} sentences")
        
        # Print stage results
        for stage, stats in results['stage_results'].items():
            print(f"  {stage}: {stats['output']}/{stats['input']} kept")
        
        return results
    except Exception as e:
        print(f"❌ Multi-stage pipeline error: {e}")
        return None

def create_sample_data():
    """Create sample sentences for testing"""
    sentences = [
        {
            'text': 'SAWACO thông báo tạm ngưng cung cấp nước từ 22 giờ đến 4 giờ.',
            'id': 1,
            'quality_score': 0.8
        },
        {
            'text': 'Các khu vực bị ảnh hưởng gồm quận 6, 8, 12.',
            'id': 2,
            'quality_score': 0.7
        },
        {
            'text': 'Việc cúp nước là để thực hiện công tác bảo trì, bảo dưỡng định kỳ.',
            'id': 3,
            'quality_score': 0.9
        },
        {
            'text': 'SAWACO cho biết đây là phương án để đảm bảo cung cấp nước sạch an toàn.',
            'id': 4,
            'quality_score': 0.8
        },
        {
            'text': 'Thời gian thực hiện dự kiến từ 22 giờ ngày 25-3 đến 4 giờ ngày 26-3.',
            'id': 5,
            'quality_score': 0.7
        }
    ]
    return sentences

def test_individual_functions(filter_obj):
    """Test individual filtering functions"""
    print("🔧 Testing individual filtering functions...")
    
    # Create sample data
    sentences = create_sample_data()
    claim_text = "SAWACO thông báo tạm ngưng cung cấp nước."
    entities = ["SAWACO", "quận 6", "quận 8", "quận 12"]
    
    print(f"📊 Sample data: {len(sentences)} sentences")
    
    # Test quality filtering
    quality_filtered = test_quality_filtering(filter_obj, sentences)
    
    # Test semantic filtering
    semantic_filtered = test_semantic_filtering(filter_obj, quality_filtered, claim_text)
    
    # Test entity filtering
    entity_filtered = test_entity_filtering(filter_obj, semantic_filtered, entities, claim_text)
    
    # Test contradiction detection
    final_filtered = test_contradiction_detection(filter_obj, entity_filtered, claim_text)
    
    print(f"✅ Individual functions test completed")
    print(f"📊 Final result: {len(final_filtered)} sentences")
    
    return final_filtered

def test_export_results(results, output_dir="output"):
    """Test exporting filtering results"""
    print("🔧 Testing export results...")
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Export detailed results
        json_path = f"{output_dir}/filtering_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ Results exported to: {json_path}")
        
        # Export summary
        summary_path = f"{output_dir}/filtering_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Advanced Data Filtering Results\n")
            f.write("=" * 40 + "\n")
            f.write(f"Input sentences: {results['input_count']}\n")
            f.write(f"Final sentences: {results['final_count']}\n")
            f.write(f"Filtered out: {results['input_count'] - results['final_count']}\n\n")
            
            f.write("Stage Results:\n")
            for stage, stats in results['stage_results'].items():
                f.write(f"  {stage}: {stats['output']}/{stats['input']} kept\n")
        
        print(f"✅ Summary exported to: {summary_path}")
        return True
    except Exception as e:
        print(f"❌ Export error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Advanced Data Filtering CLI')
    parser.add_argument('command', choices=['init', 'quality', 'semantic', 'entity', 'contradiction', 'pipeline', 'individual', 'export', 'all'], 
                       help='Command to run')
    parser.add_argument('--claim', default='SAWACO thông báo tạm ngưng cung cấp nước.',
                       help='Claim text')
    parser.add_argument('--context', default='SAWACO thông báo tạm ngưng cung cấp nước từ 22 giờ đến 4 giờ.',
                       help='Context text')
    parser.add_argument('--entities', default='SAWACO,quận 6,quận 8,quận 12',
                       help='Comma-separated entities')
    parser.add_argument('--min-quality', type=float, default=0.3, help='Minimum quality score')
    parser.add_argument('--min-relevance', type=float, default=0.25, help='Minimum relevance score')
    parser.add_argument('--min-entity', type=float, default=0.05, help='Minimum entity score')
    parser.add_argument('--stance-delta', type=float, default=0.1, help='Stance detection delta')
    parser.add_argument('--max-sentences', type=int, default=30, help='Maximum final sentences')
    
    args = parser.parse_args()
    
    if args.command == 'init':
        test_init()
    
    elif args.command == 'quality':
        filter_obj = test_init()
        if filter_obj:
            sentences = create_sample_data()
            test_quality_filtering(filter_obj, sentences)
    
    elif args.command == 'semantic':
        filter_obj = test_init()
        if filter_obj:
            sentences = create_sample_data()
            test_semantic_filtering(filter_obj, sentences, args.claim)
    
    elif args.command == 'entity':
        filter_obj = test_init()
        if filter_obj:
            sentences = create_sample_data()
            entities = args.entities.split(',') if args.entities else []
            test_entity_filtering(filter_obj, sentences, entities, args.claim)
    
    elif args.command == 'contradiction':
        filter_obj = test_init()
        if filter_obj:
            sentences = create_sample_data()
            test_contradiction_detection(filter_obj, sentences, args.claim)
    
    elif args.command == 'pipeline':
        filter_obj = test_init()
        if filter_obj:
            sentences = create_sample_data()
            entities = args.entities.split(',') if args.entities else []
            test_multi_stage_pipeline(filter_obj, sentences, args.claim, args.context, entities)
    
    elif args.command == 'individual':
        filter_obj = test_init()
        if filter_obj:
            test_individual_functions(filter_obj)
    
    elif args.command == 'export':
        filter_obj = test_init()
        if filter_obj:
            sentences = create_sample_data()
            entities = args.entities.split(',') if args.entities else []
            results = test_multi_stage_pipeline(filter_obj, sentences, args.claim, args.context, entities)
            if results:
                test_export_results(results)
    
    elif args.command == 'all':
        print("🚀 Running all filtering tests...")
        filter_obj = test_init()
        if filter_obj:
            test_individual_functions(filter_obj)
            sentences = create_sample_data()
            entities = args.entities.split(',') if args.entities else []
            results = test_multi_stage_pipeline(filter_obj, sentences, args.claim, args.context, entities)
            if results:
                test_export_results(results)
        print("🎉 All filtering tests completed!")

if __name__ == '__main__':
    main() 