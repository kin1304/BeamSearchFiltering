"""
Beam Graph Filter Pipeline - Main pipeline module
"""

import json
import os
from typing import Dict, List, Tuple
from datetime import datetime

from ..utils.text_preprocessor import clean_text, split_sentences
from ..nlp.vncorenlp_wrapper import VnCoreNLPWrapper
from ..graph.text_graph_simple import TextGraphSimple
from ..filtering.filter_wrapper import FilterWrapper


def extract_sentences_from_paths(paths, text_graph: TextGraphSimple, top_n: int = 30) -> List[Dict]:
    """
    Extract unique sentences from BeamSearch paths together with their highest score.
    
    Args:
        paths: List of beam search paths
        text_graph: TextGraphSimple instance
        top_n: Maximum number of sentences to return
        
    Returns:
        List[Dict]: List of sentences with scores
    """
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
    return [{"sentence": s, "score": sc} for s, sc in sorted_sentences[:top_n]]


class BeamFilterPipeline:
    """
    Main beam filter pipeline class
    """
    
    def __init__(self, vncorenlp_dir: str = "", use_sbert: bool = False, 
                 use_contradiction_detection: bool = False, use_nli: bool = False):
        """
        Initialize beam filter pipeline
        
        Args:
            vncorenlp_dir (str): Path to VnCoreNLP directory
            use_sbert (bool): Enable SBERT semantic filtering
            use_contradiction_detection (bool): Enable contradiction detection
            use_nli (bool): Enable NLI stance detection
        """
        self.nlp_wrapper = VnCoreNLPWrapper(vncorenlp_dir)
        self.filter_wrapper = FilterWrapper(
            use_sbert=use_sbert,
            use_contradiction_detection=use_contradiction_detection,
            use_nli=use_nli
        )
    
    def process_sample(self, sample: Dict, min_relevance: float = 0.15,
                      beam_width: int = 40, max_depth: int = 120, max_paths: int = 200,
                      max_final_sentences: int = 30, beam_sentences: int = 50,
                      beam_only: bool = False, filter_only: bool = False) -> Tuple[Dict, Dict, int, int, int]:
        """
        Process one sample and return results
        
        Args:
            sample (Dict): Sample data with context and claim
            min_relevance (float): Minimum relevance score
            beam_width (int): Beam search width
            max_depth (int): Maximum beam search depth
            max_paths (int): Maximum number of paths
            max_final_sentences (int): Maximum final sentences
            beam_sentences (int): Maximum sentences from beam search
            beam_only (bool): Use only beam search, no filtering
            filter_only (bool): Use only filtering, no beam search
            
        Returns:
            Tuple[Dict, Dict, int, int, int]: (simple_result, detailed_result, raw_count, beam_count, final_count)
        """
        # Extract data
        context_raw = sample.get("context", "")
        claim = sample.get("claim", "")
        
        if not context_raw or not claim:
            print("  ⚠️ Missing context or claim")
            return {}, {}, 0, 0, 0
        
        # 1️⃣ Pre-processing & sentence split
        context_clean = clean_text(context_raw)
        raw_sentences = split_sentences(context_clean)

        # 2️⃣ VnCoreNLP annotation
        context_tokens = self.nlp_wrapper.annotate_text(context_clean)
        claim_tokens = self.nlp_wrapper.annotate_text(claim)

        # 3️⃣ Build TextGraph
        tg = TextGraphSimple()
        tg.build_from_vncorenlp_output(context_tokens, claim, claim_tokens)

        # 4️⃣ Beam Search to collect sentence paths (or skip if filter_only)
        if filter_only:
            print("  ⏭️ Filter-only mode: Skipping Beam Search")
            # Use all sentences as candidates
            candidate_sentences = [{"sentence": s} for s in raw_sentences]
            leftover_sentences = []  # No leftovers in filter-only mode
            paths = []  # Empty paths for filter-only mode
        else:
            paths = tg.beam_search_paths(beam_width=beam_width, max_depth=max_depth, max_paths=max_paths)
            candidate_sentences = extract_sentences_from_paths(paths, tg, top_n=beam_sentences)

            # Fallback when beam search returns no sentence
            if not candidate_sentences:
                candidate_sentences = [{"sentence": s} for s in raw_sentences]

            # Build a set of candidate texts to compute leftovers
            cand_text_set = {d["sentence"] for d in candidate_sentences}
            leftover_sentences = [{"sentence": s} for s in raw_sentences if s not in cand_text_set]

        # 5️⃣ Run AdvancedDataFilter on candidate sentences (or skip if beam_only)
        if beam_only:
            print("  ⏭️ Beam-only mode: Skipping Advanced Data Filter")
            final_sentences = candidate_sentences[:max_final_sentences]  # Just take top N
        else:
            results = self.filter_wrapper.filter_sentences(
                sentences=candidate_sentences,
                claim_text=claim,
                min_relevance_score=min_relevance,
                max_final_sentences=max_final_sentences
            )
            final_sentences = results["filtered_sentences"]

            # If leftovers exist → filter them and merge results
            if leftover_sentences:
                left_results = self.filter_wrapper.filter_sentences(
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

        # Normalize output to match process_multi_hop_multi_beam_search format
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
    
    def process_batch(self, samples: List[Dict], output_dir: str = "beam_filter_output",
                     min_relevance: float = 0.15, beam_width: int = 40, max_depth: int = 120,
                     max_paths: int = 200, max_final_sentences: int = 30, beam_sentences: int = 50,
                     beam_only: bool = False, filter_only: bool = False) -> Dict:
        """
        Process a batch of samples
        
        Args:
            samples (List[Dict]): List of samples to process
            output_dir (str): Output directory
            min_relevance (float): Minimum relevance score
            beam_width (int): Beam search width
            max_depth (int): Maximum beam search depth
            max_paths (int): Maximum number of paths
            max_final_sentences (int): Maximum final sentences
            beam_sentences (int): Maximum sentences from beam search
            beam_only (bool): Use only beam search, no filtering
            filter_only (bool): Use only filtering, no beam search
            
        Returns:
            Dict: Processing results and statistics
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define output files
        simple_file = os.path.join(output_dir, f"simple_{timestamp}.json")
        detailed_file = os.path.join(output_dir, f"detailed_{timestamp}.json")
        stats_file = os.path.join(output_dir, f"stats_{timestamp}.json")
        
        total_to_process = len(samples)
        print(f"Processing {total_to_process} samples with min_relevance_score={min_relevance} ...")

        total_raw = total_beam = total_final = 0
        simple_outputs, detailed_outputs = [], []

        for idx, sample in enumerate(samples):
            print(f"\n👉 Sample {idx+1}/{total_to_process}")
            s_res, d_res, r_raw, r_beam, r_final = self.process_sample(
                sample, min_relevance, beam_width, max_depth, max_paths,
                max_final_sentences, beam_sentences, beam_only, filter_only
            )
            simple_outputs.append(s_res)
            detailed_outputs.append(d_res)
            total_raw += r_raw
            total_beam += r_beam
            total_final += r_final

            if (idx + 1) % 50 == 0:
                print(f"  -> {idx + 1} samples processed ...")

        # Save results
        with open(simple_file, "w", encoding="utf-8") as f:
            json.dump(simple_outputs, f, ensure_ascii=False, indent=2)
        with open(detailed_file, "w", encoding="utf-8") as f:
            json.dump(detailed_outputs, f, ensure_ascii=False, indent=2)
        
        # Save statistics
        run_stats = {
            "total_context_sentences": total_raw,
            "total_beam_sentences": total_beam,
            "total_final_sentences": total_final,
            "num_samples": total_to_process,
            "beam_parameters": {
                "beam_width": beam_width,
                "max_depth": max_depth,
                "max_paths": max_paths,
                "beam_sentences": beam_sentences
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
        
        return {
            "simple_file": simple_file,
            "detailed_file": detailed_file,
            "stats_file": stats_file,
            "statistics": run_stats
        } 