#!/usr/bin/env python3
"""
🚀 BEAM GRAPH FILTER PIPELINE
=============================

Pipeline mới để:
1. Tiền xử lý context (xoá xuống dòng, chuẩn hoá khoảng trắng, bỏ dấu cách trước dấu câu).
2. Cắt câu bằng regex.
3. Dùng VnCoreNLP để tách từ, POS, dependency.
4. Xây TextGraph, chạy Beam Search để lấy tập câu liên quan.
5. Lọc lại tập câu bằng AdvancedDataFilter(use_sbert=False, use_contradiction_detection=False, use_nli=False).

Usage:
    python beam_graph_filter_pipeline.py --input raw_test.json --output_dir advanced_filtering_output \
           --min_relevance 0.15 --beam_width 20 --max_depth 40

Author: AI Assistant & NguyenNha
Date: 2025-07-12
"""

import os
import sys
import json
import re
import argparse
from typing import List, Dict
from datetime import datetime
import contextlib
import io

# Bảo đảm import được py_vncorenlp (đường dẫn chứa VnCoreNLP-1.2.jar và models)
VNCORENLP_DIR = os.path.join(os.path.dirname(__file__), "vncorenlp")
sys.path.append(VNCORENLP_DIR)
import py_vncorenlp  # type: ignore

from mint.text_graph import TextGraph
from advanced_data_filtering import AdvancedDataFilter

###############################################################################
# 🛠️  TIỆN ÍCH TIỀN XỬ LÝ
###############################################################################

def clean_text(text: str) -> str:
    """Loại bỏ xuống dòng, chuẩn hoá khoảng trắng và xoá khoảng trắng trước dấu câu"""
    if not text:
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    # Bỏ khoảng trắng trước dấu câu (, . ; : ! ? )
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    # Bỏ khoảng trắng sau "("
    text = re.sub(r"\(\s+", "(", text)
    # Bỏ khoảng trắng trước ")"
    text = re.sub(r"\s+\)", ")", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Cắt câu đơn giản bằng regex: sau . ! ? và khoảng trắng"""
    if not text:
        return []
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]

###############################################################################
# 🛠️  EXTRACT SENTENCES TỪ BEAM SEARCH PATHS
###############################################################################

def extract_sentences_from_paths(paths, text_graph: TextGraph, top_n: int | None = 30) -> List[Dict]:
    """Trích xuất sentences duy nhất từ các BeamSearch Path, kèm score cao nhất"""
    if not paths:
        return []

    sentence_best_score = {}
    for path_obj in paths:
        path_score = getattr(path_obj, "score", 0.0)
        path_nodes = getattr(path_obj, "nodes", [])
        for node_id in path_nodes:
            if node_id.startswith("sentence") and node_id in text_graph.graph.nodes:
                sent_text = text_graph.graph.nodes[node_id].get("text", "")
                if not sent_text:
                    continue
                prev = sentence_best_score.get(sent_text)
                if prev is None or path_score > prev:
                    sentence_best_score[sent_text] = path_score

    # Sắp xếp giảm dần theo score
    sorted_sentences = sorted(sentence_best_score.items(), key=lambda x: x[1], reverse=True)
    if top_n is None:
        return [{"sentence": s, "score": sc} for s, sc in sorted_sentences]
    return [{"sentence": s, "score": sc} for s, sc in sorted_sentences[:top_n]]

###############################################################################
# 🚀 XỬ LÝ MỘT SAMPLE
###############################################################################

def process_sample(sample: Dict, model, filter_sys: AdvancedDataFilter, min_relevance: float,
                   beam_width: int, max_depth: int, max_paths: int,
                   max_final_sentences: int = 30, beam_sentences: int = 50):
    """Process một sample, trả về (raw_count, beam_count, final_count)"""
    context_raw = sample.get("context", "")
    claim = sample.get("claim", "")

    # 1️⃣ Tiền xử lý và cắt câu (cho debug / fallback)
    context_clean = clean_text(context_raw)
    raw_sentences = split_sentences(context_clean)

    # 2️⃣ VnCoreNLP annotate
    context_tokens = model.annotate_text(context_clean)
    claim_tokens = model.annotate_text(claim)

    # 3️⃣ Build TextGraph
    tg = TextGraph()
    tg.build_from_vncorenlp_output(context_tokens, claim, claim_tokens)

    # 4️⃣ Beam Search lấy path -> sentences
    paths = tg.beam_search_paths(beam_width=beam_width, max_depth=max_depth, max_paths=max_paths)
    candidate_sentences = extract_sentences_from_paths(paths, tg, top_n=beam_sentences)

    # Fallback nếu beam không ra câu nào
    if not candidate_sentences:
        candidate_sentences = [{"sentence": s} for s in raw_sentences]

    # 5️⃣ AdvancedDataFilter (luôn bật – log bị ẩn để gọn console)
    silent_buf = io.StringIO()
    with contextlib.redirect_stdout(silent_buf):
        results = filter_sys.multi_stage_filtering_pipeline(
            sentences=candidate_sentences,
            claim_text=claim,
            min_relevance_score=min_relevance,
            max_final_sentences=max_final_sentences
        )
    final_sentences = results["filtered_sentences"]

    sample["filtered_evidence"] = [d["sentence"] for d in final_sentences]

    # --- Chuẩn hoá kết quả giống process_multi_hop_multi_beam_search ---
    simple_result = {
        **{k: sample.get(k) for k in ("context", "claim", "evidence", "label") if k in sample},
        "multi_level_evidence": [d["sentence"] for d in final_sentences]
    }
    detailed_result = {
        **{k: sample.get(k) for k in ("context", "claim", "evidence", "label") if k in sample},
        "multi_level_evidence": final_sentences,
        "statistics": {
            "beam": {
                "total_paths": len(paths),
                "unique_sentences": len(candidate_sentences)
            }
        }
    }

    return simple_result, detailed_result, len(raw_sentences), len(candidate_sentences), len(final_sentences)

###############################################################################
# 🏁 MAIN
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Beam Graph + Advanced Filter pipeline")
    parser.add_argument("--input", type=str, default="raw_test.json", help="File JSON input")
    parser.add_argument("--output_dir", type=str, default="beam_filter_output", help="Thư mục lưu output")
    parser.add_argument("--min_relevance", type=float, default=0.15, help="Ngưỡng relevance tối thiểu")
    # Giá trị mặc định mới để “vét” nhiều câu hơn
    parser.add_argument("--beam_width", type=int, default=40,
                         help="Beam width (số path giữ mỗi bước)")
    parser.add_argument("--max_depth", type=int, default=120,
                         help="Độ sâu tối đa của beam search")
    parser.add_argument("--max_paths", type=int, default=200,
                         help="Số paths tối đa trả về")
    parser.add_argument("--max_final_sentences", type=int, default=30, help="Số câu cuối cùng giữ lại")
    parser.add_argument("--max_samples", type=int, default=None, help="Giới hạn số sample xử lý")
    parser.add_argument("--beam_sentences", type=int, default=50,
                    help="Số câu tối đa lấy từ Beam Search trước khi lọc")
    args = parser.parse_args()

    # 👉 luôn dùng đường dẫn tuyệt đối cho output_dir
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        args.output_dir,
        f"{os.path.splitext(os.path.basename(args.input))[0]}_beam_filtered_{args.min_relevance}_{timestamp}.json"
    )

    # 📁 Định nghĩa đường dẫn file output (ghi một mảng JSON duy nhất sau khi chạy xong)
    simple_file   = output_file.replace(".json", "_simple.json")
    detailed_file = output_file.replace(".json", "_detailed.json")
    stats_file    = output_file.replace(".json", "_stats.json")

    # 🔐 bảo đảm thư mục đích tồn tại và reset file cũ
    for fp in (simple_file, detailed_file, stats_file):
        os.makedirs(os.path.dirname(fp), exist_ok=True)
    for fp in (simple_file, detailed_file):
        if os.path.exists(fp):
            os.remove(fp)

    # Load samples
    with open(args.input, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # Áp dụng max_samples nếu có
    if args.max_samples is not None and args.max_samples < len(samples):
        samples = samples[:args.max_samples]
        print(f"⚠️  Chỉ xử lý {len(samples)} sample đầu tiên theo --max_samples={args.max_samples}")

    total_to_process = len(samples)
    print(f"Processing {total_to_process} samples from {args.input} with min_relevance_score={args.min_relevance} ...")

    # Setup VnCoreNLP model
    print("🔧 Loading VnCoreNLP model ...")
    model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "ner", "parse"], save_dir=VNCORENLP_DIR)

    # Advanced filter (không SBERT, không NLI, không contradiction detection)
    filter_sys = AdvancedDataFilter(use_sbert=False, use_contradiction_detection=False, use_nli=False)

    total_raw = total_beam = total_final = 0
    simple_outputs, detailed_outputs = [], []

    for idx, sample in enumerate(samples):
        print(f"\n👉 Sample {idx+1}/{total_to_process}")
        s_res, d_res, r_raw, r_beam, r_final = process_sample(
            sample, model, filter_sys, args.min_relevance,
            args.beam_width, args.max_depth, args.max_paths,
            args.max_final_sentences, beam_sentences=args.beam_sentences)
        simple_outputs.append(s_res)
        detailed_outputs.append(d_res)
        total_raw   += r_raw
        total_beam  += r_beam
        total_final += r_final

        if (idx + 1) % 50 == 0:
            print(f"  -> {idx + 1} samples processed ...")

        # 👉 Bỏ ghi từng dòng JSONL để quay lại ghi một lần cuối – giữ bộ nhớ ở mức chấp nhận được

    # 📝 Ghi danh sách output (định dạng JSON array)
    with open(simple_file, "w", encoding="utf-8") as f:
        json.dump(simple_outputs, f, ensure_ascii=False, indent=2)
    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(detailed_outputs, f, ensure_ascii=False, indent=2)
    # --- Ghi file thống kê tổng ---
    run_stats = {
        "total_context_sentences": total_raw,
        "total_beam_sentences":    total_beam,
        "total_final_sentences":   total_final,
        "num_samples":             total_to_process,
        "beam_parameters": {
            "beam_width": args.beam_width,
            "max_depth":  args.max_depth,
            "max_paths":  args.max_paths,
            "beam_sentences": args.beam_sentences
        }
    }
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(run_stats, f, ensure_ascii=False, indent=2)

    print("\n================= TỔNG KẾT =================")
    print(f"Tổng câu sau tách: {total_raw}")
    print(f"Sau Beam Search:   {total_beam}")
    print(f"Sau Lọc nâng cao:  {total_final}")
    print("===========================================")

    print(f"✅ Done! Output saved to:\n   • {simple_file}\n   • {detailed_file}\n   • {stats_file}")


if __name__ == "__main__":
    main() 