#!/usr/bin/env python3
"""
Example usage of the refactored beam graph filter pipeline
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import BeamFilterPipeline


def example_single_sample():
    """Example of processing a single sample"""
    
    # Sample data
    sample = {
        "context": "SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệp. Thời gian thực hiện dự kiến từ 22 giờ ngày 25-3 (thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật).",
        "claim": "SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệp, thời gian thực hiện dự kiến từ 12 giờ ngày 25-3 (thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật)."
    }
    
    # Create pipeline
    pipeline = BeamFilterPipeline(
        use_sbert=False,
        use_contradiction_detection=False,
        use_nli=False
    )
    
    # Process sample
    simple_result, detailed_result, raw_count, beam_count, final_count = pipeline.process_sample(
        sample=sample,
        min_relevance=0.15,
        beam_width=40,
        max_depth=120,
        max_paths=200,
        max_final_sentences=30,
        beam_sentences=50
    )
    
    print("=== SINGLE SAMPLE PROCESSING ===")
    print(f"Raw sentences: {raw_count}")
    print(f"Beam search sentences: {beam_count}")
    print(f"Final sentences: {final_count}")
    print(f"Evidence: {simple_result.get('multi_level_evidence', [])}")
    print("================================")


def example_batch_processing():
    """Example of batch processing"""
    
    # Sample data
    samples = [
        {
            "context": "SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệp. Thời gian thực hiện dự kiến từ 22 giờ ngày 25-3 (thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật).",
            "claim": "SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệp, thời gian thực hiện dự kiến từ 12 giờ ngày 25-3 (thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật)."
        },
        {
            "context": "Các nhà khoa học tại Đại học Stanford đã phát triển AI để nghiên cứu voi châu Phi. Nghiên cứu này được công bố trên tạp chí Nature.",
            "claim": "Các nhà khoa học tại Đại học Stanford đã phát triển AI để nghiên cứu voi châu Phi."
        }
    ]
    
    # Create pipeline
    pipeline = BeamFilterPipeline(
        use_sbert=False,
        use_contradiction_detection=False,
        use_nli=False
    )
    
    # Process batch
    results = pipeline.process_batch(
        samples=samples,
        output_dir="example_output",
        min_relevance=0.15,
        beam_width=40,
        max_depth=120,
        max_paths=200,
        max_final_sentences=30,
        beam_sentences=50
    )
    
    print("=== BATCH PROCESSING ===")
    print(f"Results saved to: {results['simple_file']}")
    print(f"Statistics: {results['statistics']}")
    print("========================")


if __name__ == "__main__":
    print("Running example usage...")
    
    # Example 1: Single sample processing
    example_single_sample()
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Batch processing
    example_batch_processing() 