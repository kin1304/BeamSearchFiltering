#!/usr/bin/env python3
"""
🚀 BEAM GRAPH FILTER PIPELINE
=============================

Main steps:
1. Pre-process context (remove line breaks, normalize whitespaces, trim spaces before punctuation).
2. Split context into sentences via regex.
3. Annotate with VnCoreNLP (tokenize, POS, NER, dependency).
4. Build a TextGraph and run Beam Search to collect candidate sentences.
5. Re-filter the sentences with **AdvancedDataFilter** (default: no SBERT/NLI/contradiction detection).

Usage example:

```bash
python beam_graph_filter_pipeline.py \
   --input raw_test.json \
   --output_dir advanced_filtering_output \
   --min_relevance 0.15 --beam_width 20 --max_depth 40
```

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
# 🛠️  PRE-PROCESSING UTILITIES
###############################################################################

def clean_text(text: str) -> str:
    """Remove line breaks, normalize whitespaces and trim spaces before punctuation."""
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
    """Simple sentence splitter using regex: break after . ! ? followed by whitespace."""
    if not text:
        return []
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]

###############################################################################
# 🛠️  EXTRACT SENTENCES FROM BEAM SEARCH PATHS
###############################################################################

def extract_sentences_from_paths(paths, text_graph: TextGraph, top_n: int | None = 30) -> List[Dict]:
    """Extract unique sentences from BeamSearch paths together with their highest score."""
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

    # Sort descending by score
    sorted_sentences = sorted(sentence_best_score.items(), key=lambda x: x[1], reverse=True)
    if top_n is None:
        return [{"sentence": s, "score": sc} for s, sc in sorted_sentences]
    return [{"sentence": s, "score": sc} for s, sc in sorted_sentences[:top_n]]

###############################################################################
# 🚀 PROCESS A SINGLE SAMPLE
###############################################################################

def process_sample(sample: Dict, model, filter_sys: AdvancedDataFilter, min_relevance: float,
                   beam_width: int, max_depth: int, max_paths: int,
                   max_final_sentences: int = 30, beam_sentences: int = 50):
    """Process one sample and return stats (raw_count, beam_count, final_count)."""
    context_raw = sample.get("context", "")
    claim = sample.get("claim", "")

    # 1️⃣ Pre-processing & sentence split (debug / fallback)
    context_clean = clean_text(context_raw)
    raw_sentences = split_sentences(context_clean)

    # 2️⃣ VnCoreNLP annotation
    context_tokens = model.annotate_text(context_clean)
    claim_tokens = model.annotate_text(claim)

    # 3️⃣ Build TextGraph
    tg = TextGraph()
    tg.build_from_vncorenlp_output(context_tokens, claim, claim_tokens)

    # 4️⃣ Beam Search to collect sentence paths
    paths = tg.beam_search_paths(beam_width=beam_width, max_depth=max_depth, max_paths=max_paths)
    candidate_sentences = extract_sentences_from_paths(paths, tg, top_n=beam_sentences)

    # Fallback when beam search returns no sentence
    if not candidate_sentences:
        candidate_sentences = [{"sentence": s} for s in raw_sentences]

    # 4.5️⃣ Build a set of candidate texts to compute leftovers
    cand_text_set = {d["sentence"] for d in candidate_sentences}
    leftover_sentences = [{"sentence": s} for s in raw_sentences if s not in cand_text_set]

    # 5️⃣ Run AdvancedDataFilter on candidate sentences
    silent_buf = io.StringIO()
    with contextlib.redirect_stdout(silent_buf):
        results = filter_sys.multi_stage_filtering_pipeline(
            sentences=candidate_sentences,
            claim_text=claim,
            min_relevance_score=min_relevance,
            max_final_sentences=max_final_sentences
        )
    final_sentences = results["filtered_sentences"]

    # 5.5️⃣ If leftovers exist → filter them and merge results
    if leftover_sentences:
        with contextlib.redirect_stdout(silent_buf):
            left_results = filter_sys.multi_stage_filtering_pipeline(
                sentences=leftover_sentences,
                claim_text=claim,
                min_relevance_score=min_relevance,
                max_final_sentences=max_final_sentences
            )
        extra = left_results["filtered_sentences"]
        exists = {d["sentence"] for d in final_sentences}
        for d in extra:
            if d["sentence"] not in exists:
                final_sentences.append(d)
                exists.add(d["sentence"])

    sample["filtered_evidence"] = [d["sentence"] for d in final_sentences]

    # --- Normalize output to match process_multi_hop_multi_beam_search format ---
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
# 🏁 MAIN ENTRY
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

    # 👉 Always convert output_dir to absolute path
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

    # 🔐 Ensure destination directory exists & remove previous files
    for fp in (simple_file, detailed_file, stats_file):
        os.makedirs(os.path.dirname(fp), exist_ok=True)
    for fp in (simple_file, detailed_file):
        if os.path.exists(fp):
            os.remove(fp)

    # Load samples
    with open(args.input, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # Apply max_samples slicing if requested
    if args.max_samples is not None and args.max_samples < len(samples):
        samples = samples[:args.max_samples]
        print(f"⚠️  Chỉ xử lý {len(samples)} sample đầu tiên theo --max_samples={args.max_samples}")

    total_to_process = len(samples)
    print(f"Processing {total_to_process} samples from {args.input} with min_relevance_score={args.min_relevance} ...")

    # Setup VnCoreNLP model
    print("🔧 Loading VnCoreNLP model ...")
    model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "ner", "parse"], save_dir=VNCORENLP_DIR)

    # Advanced filter (SBERT/NLI/contradiction detection disabled by default)
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

        # 👉 Skip per-line JSONL writing; we dump once at the end to save memory

    # 📝 Dump output lists (JSON array)
    with open(simple_file, "w", encoding="utf-8") as f:
        json.dump(simple_outputs, f, ensure_ascii=False, indent=2)
    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(detailed_outputs, f, ensure_ascii=False, indent=2)
    # --- Dump global statistics ---
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

    print("\n================= SUMMARY =================")
    print(f"Total sentences after split: {total_raw}")
    print(f"After Beam Search:           {total_beam}")
    print(f"After Advanced Filtering:    {total_final}")
    print("===========================================")

    print(f"✅ Done! Output saved to:\n   • {simple_file}\n   • {detailed_file}\n   • {stats_file}")


if __name__ == "__main__":
    main() 