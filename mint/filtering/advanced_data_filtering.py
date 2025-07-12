#!/usr/bin/env python3
"""
🔍 ADVANCED DATA FILTERING SYSTEM
==================================

Advanced data filtering system to improve classification accuracy:

1. Semantic Relevance Filtering
2. Quality-Based Filtering  
3. Contradiction Detection
4. Entity-Based Filtering
5. Length & Structure Filtering
6. Duplicate Detection & Removal
7. Confidence Scoring
8. Multi-Stage Filtering Pipeline

Author: AI Assistant & NguyenNha
Date: 2025-01-03
"""

import re
import json
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter
from datetime import datetime

# Try to import SBERT for semantic filtering
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("⚠️ SBERT not available for semantic filtering")

# Try to import HuggingFace transformers NLI model (XLM-R XNLI – supports Vietnamese)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers.pipelines import pipeline  # type: ignore
    _tokenizer_nli = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
    _model_nli = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")
    NLI_AVAILABLE = True
    print("✅ Loaded XLM-RoBERTa XNLI model for Vietnamese NLI")
except Exception as _e:
    NLI_AVAILABLE = False
    _tokenizer_nli = None
    _model_nli = None
    print(f"⚠️  NLI model not available: {_e}")


class AdvancedDataFilter:
    """
    🔍 Advanced Data Filtering System with multiple filtering strategies
    """
    
    def __init__(self, use_sbert=True, use_contradiction_detection=True, use_nli=True):
        self.use_sbert = use_sbert and SBERT_AVAILABLE
        self.use_contradiction_detection = use_contradiction_detection
        self.use_nli = use_nli and NLI_AVAILABLE
        
        # Initialize SBERT if available
        if self.use_sbert:
            try:
                self.sbert_model = SentenceTransformer("keepitreal/vietnamese-sbert")
                print("✅ SBERT model loaded for semantic filtering")
            except Exception as e:
                print(f"⚠️ SBERT failed, using fallback: {e}")
                try:
                    self.sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                    print("✅ Fallback SBERT model loaded")
                except Exception as e2:
                    print(f"❌ All SBERT models failed: {e2}")
                    self.use_sbert = False
                    self.sbert_model = None
        else:
            self.sbert_model = None
        
        # Vietnamese stop words
        self.stop_words = {
            'và', 'của', 'là', 'có', 'được', 'cho', 'với', 'từ', 'trong', 'ngoài',
            'trên', 'dưới', 'trước', 'sau', 'khi', 'nếu', 'thì', 'mà', 'nhưng',
            'hoặc', 'vì', 'do', 'bởi', 'tại', 'theo', 'qua', 'bằng', 'về', 'đến',
            'này', 'kia', 'đó', 'ấy', 'nọ', 'kìa', 'đây', 'đấy', 'thế', 'vậy',
            'rồi', 'xong', 'xong', 'hết', 'còn', 'vẫn', 'đang', 'sẽ', 'đã', 'chưa',
            'không', 'chẳng', 'chả', 'mới', 'cũng', 'cũng', 'cũng', 'cũng', 'cũng'
        }
        
        # Contradiction indicators
        self.contradiction_indicators = {
            'negation': ['không', 'chẳng', 'chả', 'không phải', 'không phải là'],
            'opposition': ['nhưng', 'tuy nhiên', 'mặc dù', 'dù', 'dù rằng'],
            'contrast': ['trái lại', 'ngược lại', 'thay vào đó', 'thay thế'],
            'disagreement': ['sai', 'không đúng', 'không chính xác', 'không phù hợp']
        }
        
        print(f"🔧 Advanced Data Filter initialized:")
        print(f"   - SBERT Semantic Filtering: {'✅' if self.use_sbert else '❌'}")
        print(f"   - Contradiction Detection: {'✅' if self.use_contradiction_detection else '❌'}")
        print(f"   - NLI Stance Model: {'✅' if self.use_nli else '❌'}")

        # Load NLI model if available
        if self.use_nli and NLI_AVAILABLE:
            try:
                self.nli_pipeline = pipeline(
                    "text-classification",
                    model=_model_nli,
                    tokenizer=_tokenizer_nli,
                    return_all_scores=False,
                    function_to_apply="softmax",
                    truncation=True,
                    max_length=512,
                )
                print("✅ HuggingFace XLM-R XNLI pipeline ready for stance detection")
            except Exception as e:
                print(f"⚠️  NLI pipeline init failed: {e}")
                self.use_nli = False
                self.nli_pipeline = None
        else:
            self.nli_pipeline = None

    def multi_stage_filtering_pipeline(self, sentences: List[Dict], claim_text: str,
                                     context_text: str = "", entities: Optional[List[str]] = None,
                                     min_quality_score: float = 0.3,
                                     min_relevance_score: float = 0.25,
                                     min_entity_score: float = 0.05,
                                     stance_delta: float = 0.1,
                                     subject_keywords: Optional[Set[str]] = None,
                                     max_final_sentences: int = 30,
                                     min_entity_keep: int = 5) -> Dict:
        """
        🚀 Multi-Stage Filtering Pipeline with comprehensive filtering
        """
        pipeline_results = {
            'input_count': len(sentences),
            'stage_results': {},
            'final_count': 0,
            'filtering_statistics': {}
        }
        
        print(f"🔍 Starting Ultra-Optimized Filtering Pipeline...")
        print(f"📊 Input: {len(sentences)} sentences")
        
        # STAGE 1: Semantic Relevance Filtering (was Stage 2)
        print("\n🎯 Stage 1: Semantic Relevance Filtering...")
        relevance_filtered = self._stage2_semantic_relevance_filtering(
            sentences, claim_text, min_relevance_score, max_final_sentences, subject_keywords
        )
        pipeline_results['stage_results']['relevance_filtered'] = {
            'input': len(sentences),
            'output': len(relevance_filtered),
            'filtered': len(sentences) - len(relevance_filtered)
        }
        print(f"✅ Relevance filtering: {len(relevance_filtered)}/{len(sentences)} kept")
        
        # STAGE 2: Entity-Based Filtering (was Stage 3)
        if entities:
            print("\n🏷️ Stage 2: Entity-Based Filtering...")
            entity_filtered = self._stage3_entity_based_filtering(
                relevance_filtered, entities, claim_text, min_entity_score, min_entity_keep)
            pipeline_results['stage_results']['entity_filtered'] = {
                'input': len(relevance_filtered),
                'output': len(entity_filtered),
                'filtered': len(relevance_filtered) - len(entity_filtered)
            }
            print(f"✅ Entity filtering: {len(entity_filtered)}/{len(relevance_filtered)} kept")
        else:
            entity_filtered = relevance_filtered
            pipeline_results['stage_results']['entity_filtered'] = {
                'input': len(relevance_filtered),
                'output': len(entity_filtered),
                'filtered': 0
            }
            print("⏭️ Stage 2: Skipped (no entities)")
        
        # STAGE 3: Contradiction Detection (was Stage 4)
        if self.use_contradiction_detection:
            print("\n⚠️ Stage 3: Contradiction Detection...")
            final_sentences = self._stage4_contradiction_detection(entity_filtered, claim_text, delta=stance_delta)
            pipeline_results['stage_results']['contradiction_filtered'] = {
                'input': len(entity_filtered),
                'output': len(final_sentences),
                'filtered': len(entity_filtered) - len(final_sentences)
            }
            print(f"✅ Contradiction filtering: {len(final_sentences)}/{len(entity_filtered)} kept")
        else:
            final_sentences = entity_filtered
            pipeline_results['stage_results']['contradiction_filtered'] = {
                'input': len(entity_filtered),
                'output': len(final_sentences),
                'filtered': 0
            }
            print("⏭️ Stage 3: Skipped")
        
        # 🔄 Fallback SBERT boosting: ensure at least 5 sentences after Stage 3
        min_required = 5
        if len(final_sentences) < min_required:
            print(f"⚠️  Only {len(final_sentences)} sentences after Stage 3 – applying SBERT fallback to reach {min_required}")

            # Candidates are the sentences that survived entity filtering but were removed by contradiction step
            candidate_pool = [s for s in entity_filtered if s not in final_sentences]

            # If SBERT available, rank candidates by semantic similarity to claim
            extra_chosen = []
            if self.sbert_model is not None:
                try:
                    claim_emb = self.sbert_model.encode([claim_text])[0]
                    claim_emb = claim_emb / np.linalg.norm(claim_emb)

                    scored = []
                    for cand in candidate_pool:
                        sent_text = cand.get('sentence', '')
                        if not sent_text:
                            continue
                        sent_emb = self.sbert_model.encode([sent_text])[0]
                        sent_emb = sent_emb / np.linalg.norm(sent_emb)
                        sim = float(np.dot(claim_emb, sent_emb))
                        scored.append((sim, cand))

                    scored.sort(key=lambda x: x[0], reverse=True)
                    needed = min_required - len(final_sentences)
                    extra_chosen = [c for _, c in scored[:needed]]
                except Exception as e:
                    print(f"⚠️  SBERT fallback failed: {e}")

            # If SBERT unavailable or not enough candidates scored, fallback to relevance_score
            if not extra_chosen and candidate_pool:
                candidate_pool_sorted = sorted(candidate_pool, key=lambda x: x.get('relevance_score', 0), reverse=True)
                needed = min_required - len(final_sentences)
                extra_chosen = candidate_pool_sorted[:needed]

            # Append chosen extras
            if extra_chosen:
                print(f"✅ SBERT fallback added {len(extra_chosen)} extra sentences")
                final_sentences.extend(extra_chosen)

        # Calculate comprehensive statistics (after fallback)
        pipeline_results['final_count'] = len(final_sentences)
        pipeline_results['filtering_statistics'] = self._calculate_filtering_statistics(
            sentences, final_sentences, pipeline_results
        )
        
        print(f"\n🎉 Multi-Stage Filtering Complete!")
        print(f"📊 Final Results: {len(final_sentences)}/{len(sentences)} sentences selected")
        print(f"📈 Overall filtering rate: {(1 - len(final_sentences)/len(sentences))*100:.1f}%")
        
        return {
            'filtered_sentences': final_sentences,
            'pipeline_results': pipeline_results
        }

    def _stage1_quality_filtering(self, sentences: List[Dict], min_quality_score: float) -> List[Dict]:
        """
        🌱 Stage 1: Quality-Based Filtering
        - Length appropriateness
        - Information density
        - Sentence structure
        - Content richness
        """
        quality_filtered = []
        
        for sentence_data in sentences:
            sentence_text = sentence_data.get('sentence', '')
            if not sentence_text:
                continue
            
            # Calculate quality score
            quality_score = self._calculate_sentence_quality(sentence_text)
            
            # Add quality info to sentence
            sentence_data['quality_score'] = quality_score
            sentence_data['quality_analysis'] = self._analyze_sentence_quality(sentence_text)
            
            if quality_score >= min_quality_score:
                quality_filtered.append(sentence_data)
        
        return quality_filtered

    def _stage2_semantic_relevance_filtering(self, sentences: List[Dict], claim_text: str, 
                                           min_relevance_score: float, max_final_sentences: int,
                                           subject_keywords: Optional[Set[str]] = None) -> List[Dict]:
        """
        🎯 Stage 2: Semantic Relevance Filtering
        - SBERT semantic similarity (if available)
        - Keyword overlap
        - Topic coherence
        - Claim-specific relevance
        """
        relevance_filtered = []
        
        for sentence_data in sentences:
            sentence_text = sentence_data.get('sentence', '')
            if not sentence_text:
                continue

            # If subject_keywords provided, require sentence to contain at least one keyword
            if subject_keywords:
                lower_sentence = sentence_text.lower()
                if not any(kw.lower() in lower_sentence for kw in subject_keywords):
                    continue  # Skip sentences without main subject

            # Calculate relevance score
            relevance_score = self._calculate_semantic_relevance(sentence_text, claim_text)

            # Add relevance info to sentence
            sentence_data['relevance_score'] = relevance_score
            sentence_data['relevance_analysis'] = self._analyze_semantic_relevance(sentence_text, claim_text)

            if relevance_score >= min_relevance_score:
                relevance_filtered.append(sentence_data)

        # 🔄 Fallback: if no sentences kept (usually when SBERT is disabled),
        # automatically take top max_final_sentences sentences with highest relevance to avoid empty pipeline.
        if not relevance_filtered:
            print("⚠️  No sentences passed relevance threshold – applying fallback top-K selection")
            sorted_by_rel = sorted(sentences, key=lambda x: x.get('relevance_score', 0), reverse=True)
            relevance_filtered = sorted_by_rel[:max(len(sorted_by_rel)//2, 5)]  # keep at least 5 or top 50%

        return relevance_filtered

    def _stage3_entity_based_filtering(self, sentences: List[Dict], entities: Optional[List[str]], 
                                     claim_text: str, min_entity_score: float, min_entity_keep: int) -> List[Dict]:
        """
        🏷️ Stage 3: Entity-Based Filtering
        - Entity presence and frequency
        - Entity relevance to claim
        - Entity relationship strength
        """
        entity_filtered = []
        
        for sentence_data in sentences:
            sentence_text = sentence_data.get('sentence', '')
            if not sentence_text:
                continue
            
            # Calculate entity-based score
            entity_score = self._calculate_entity_based_score(sentence_text, entities, claim_text)
            
            # Add entity analysis to sentence
            sentence_data['entity_score'] = entity_score
            sentence_data['entity_analysis'] = self._analyze_entity_presence(sentence_text, entities)
            
            # Keep sentences with at least some entity relevance
            if entity_score >= min_entity_score:
                entity_filtered.append(sentence_data)
        
        # If no sentences remain after entity filtering → keep input to avoid empty pipeline
        if not entity_filtered or len(entity_filtered) < min_entity_keep:
            # Keep previous list if too few sentences
            print(f"⚠️  Entity filtering kept {len(entity_filtered)} sentences (<{min_entity_keep}) – relaxing filter")
            # Return top sentences by entity_score or fallback to input
            if entity_filtered:
                sorted_by_ent = sorted(sentences, key=lambda x: x.get('entity_score', 0), reverse=True)
                return sorted_by_ent[:max(len(sentences)//2, min_entity_keep)]
            return sentences
        return entity_filtered

    def _stage4_contradiction_detection(self, sentences: List[Dict], claim_text: str, delta: float = 0.1, suppress_log: bool = False) -> List[Dict]:
        """
        ⚠️ Stage 4: SBERT-based Stance Detection
        Keep only SUPPORT / REFUTE sentences.
        Method:
        1. Calculate SBERT embeddings for claim (v_c) and simple negation ("không " + claim) (v_neg).
        2. For each sentence s, compute cosine(v_c, s) and cosine(v_neg, s).
        3. diff = sim_claim - sim_neg
           • diff >  delta → SUPPORT
           • diff < -delta → REFUTE
           • |diff| ≤ delta → NEI (discard)
        If SBERT unavailable, fallback to heuristic contradiction_score.
        """
        # If NLI model available, prioritize its use
        if hasattr(self, 'use_nli') and self.use_nli and self.nli_pipeline:
            filtered = []
            for sentence_data in sentences:
                sentence_text = sentence_data.get('sentence', '')
                if not sentence_text:
                    continue
                try:
                    result = self.nli_pipeline(f"{sentence_text} </s></s> {claim_text}")[0]
                    label = result['label'].lower()  # entailment / contradiction / neutral
                except Exception:
                    label = 'neutral'
                if label.startswith('entail'):
                    sentence_data['stance'] = 'support'
                    sentence_data['stance_score'] = 1.0
                    filtered.append(sentence_data)
                elif label.startswith('contradict'):
                    sentence_data['stance'] = 'refute'
                    sentence_data['stance_score'] = 1.0
                    filtered.append(sentence_data)
                # neutral skip
            if not filtered and not suppress_log:
                print("⚠️  NLI model found no support/refute – fallback to SBERT method")
            else:
                return filtered

        # If SBERT available, use embedding method
        if self.use_sbert and self.sbert_model:
            claim_embedding = self.sbert_model.encode([claim_text])[0]
            neg_claim_text = "không " + claim_text
            neg_embedding = self.sbert_model.encode([neg_claim_text])[0]
            # normalize
            claim_emb_norm = claim_embedding / np.linalg.norm(claim_embedding)
            neg_emb_norm = neg_embedding / np.linalg.norm(neg_embedding)

            filtered = []
            for sentence_data in sentences:
                sentence_text = sentence_data.get('sentence', '')
                if not sentence_text:
                    continue
                sent_emb = self.sbert_model.encode([sentence_text])[0]
                sent_emb_norm = sent_emb / np.linalg.norm(sent_emb)

                sim_claim = float(np.dot(claim_emb_norm, sent_emb_norm))
                sim_neg   = float(np.dot(neg_emb_norm,   sent_emb_norm))
                diff = sim_claim - sim_neg

                if diff > delta:
                    sentence_data['stance'] = 'support'
                    sentence_data['stance_score'] = diff
                    filtered.append(sentence_data)
                elif diff < -delta:
                    sentence_data['stance'] = 'refute'
                    sentence_data['stance_score'] = -diff
                    filtered.append(sentence_data)
                # else neutral skip

            if not filtered:
                # Try lowering delta threshold to 0.05
                delta_low = 0.05
                for sentence_data in sentences:
                    sentence_text = sentence_data.get('sentence', '')
                    if not sentence_text:
                        continue
                    sent_emb = self.sbert_model.encode([sentence_text])[0]
                    sent_emb_norm = sent_emb / np.linalg.norm(sent_emb)
                    sim_claim = float(np.dot(claim_emb_norm, sent_emb_norm))
                    sim_neg   = float(np.dot(neg_emb_norm,   sent_emb_norm))
                    diff = sim_claim - sim_neg
                    if diff > delta_low:
                        sentence_data['stance'] = 'support'
                        sentence_data['stance_score'] = diff
                        filtered.append(sentence_data)
                    elif diff < -delta_low:
                        sentence_data['stance'] = 'refute'
                        sentence_data['stance_score'] = -diff
                        filtered.append(sentence_data)

            # If still empty, select top 1 support & refute by highest diff to avoid losing stance
            if not filtered:
                scored = []
                for sentence_data in sentences:
                    sentence_text = sentence_data.get('sentence', '')
                    sent_emb = self.sbert_model.encode([sentence_text])[0]
                    sent_emb_norm = sent_emb / np.linalg.norm(sent_emb)
                    sim_claim = float(np.dot(claim_emb_norm, sent_emb_norm))
                    sim_neg   = float(np.dot(neg_emb_norm,   sent_emb_norm))
                    diff = sim_claim - sim_neg
                    scored.append((diff, sentence_data, sim_claim, sim_neg))
                # sort by diff
                scored_sorted = sorted(scored, key=lambda x: x[0])
                if scored_sorted:
                    # most negative diff -> refute
                    diff_neg, sent_neg, _, _ = scored_sorted[0]
                    sent_neg['stance'] = 'refute'
                    sent_neg['stance_score'] = abs(diff_neg)
                    filtered.append(sent_neg)
                    # most positive diff -> support
                    diff_pos, sent_pos, _, _ = scored_sorted[-1]
                    if sent_pos is not sent_neg:
                        sent_pos['stance'] = 'support'
                        sent_pos['stance_score'] = diff_pos
                        filtered.append(sent_pos)

            if not filtered and not suppress_log:
                print("⚠️  SBERT stance detection still found no support/refute – keep previous list")
                return sentences
            return filtered

        # Fallback heuristic if SBERT unavailable
        if not suppress_log:
            print("⏭️  SBERT unavailable – using heuristic contradiction detection")
        contradiction_filtered = []
        for sentence_data in sentences:
            sentence_text = sentence_data.get('sentence', '')
            if not sentence_text:
                continue
            contradiction_score = self._calculate_contradiction_score(sentence_text, claim_text)
            sentence_data['contradiction_score'] = contradiction_score
            if contradiction_score <= 0.3:
                sentence_data['stance'] = 'support'
                sentence_data['stance_score'] = 1 - contradiction_score
                contradiction_filtered.append(sentence_data)
            elif contradiction_score >= 0.7:
                sentence_data['stance'] = 'refute'
                sentence_data['stance_score'] = contradiction_score
                contradiction_filtered.append(sentence_data)

        if not contradiction_filtered:
            if not suppress_log:
                print("⚠️  Heuristic stance detection found no support/refute – keep previous list")
            return sentences
        return contradiction_filtered

    def _stage5_duplicate_removal_and_ranking(self, sentences: List[Dict], 
                                            max_final_sentences: int) -> List[Dict]:
        """
        🔄 Stage 5: Duplicate Removal & Final Ranking
        - Remove semantic duplicates
        - Final confidence scoring
        - Top-N selection
        """
        # Remove exact duplicates
        seen_texts = set()
        unique_sentences = []
        
        for sentence_data in sentences:
            sentence_text = sentence_data.get('sentence', '')
            if sentence_text and sentence_text not in seen_texts:
                unique_sentences.append(sentence_data)
                seen_texts.add(sentence_text)
        
        # Calculate final confidence scores
        for sentence_data in unique_sentences:
            confidence_score = self._calculate_final_confidence_score(sentence_data)
            sentence_data['confidence_score'] = confidence_score
        
        # Sort by confidence score and select top-N
        final_sentences = sorted(unique_sentences, 
                                key=lambda x: x.get('confidence_score', 0), reverse=True)
        
        return final_sentences[:max_final_sentences]

    def _calculate_sentence_quality(self, sentence_text: str) -> float:
        """
        Calculate sentence quality score based on multiple factors
        """
        if not sentence_text:
            return 0.0
        
        # 1. Length appropriateness (5-50 words is optimal)
        words = sentence_text.split()
        word_count = len(words)
        
        if word_count < 3:
            length_score = 0.2
        elif word_count < 5:
            length_score = 0.5
        elif 5 <= word_count <= 50:
            length_score = 1.0
        else:
            length_score = max(0.3, 1.0 - (word_count - 50) * 0.01)
        
        # 2. Information density (meaningful words vs total words)
        meaningful_words = [word for word in words 
                          if len(word) > 2 and word.lower() not in self.stop_words]
        density_score = len(meaningful_words) / max(word_count, 1)
        
        # 3. Sentence structure (has subject-verb-object pattern)
        structure_score = self._calculate_structure_score(sentence_text)
        
        # 4. Content richness (variety of words, entities, etc.)
        richness_score = self._calculate_content_richness(sentence_text)
        
        # Combine scores with weights
        quality_score = (
            length_score * 0.3 +
            density_score * 0.3 +
            structure_score * 0.2 +
            richness_score * 0.2
        )
        
        return min(1.0, quality_score)

    def _calculate_semantic_relevance(self, sentence_text: str, claim_text: str) -> float:
        """
        Calculate semantic relevance between sentence and claim
        """
        if not sentence_text or not claim_text:
            return 0.0
        
        # 1. Keyword overlap
        sentence_words = set(sentence_text.lower().split())
        claim_words = set(claim_text.lower().split())
        
        if not claim_words:
            return 0.0
        
        overlap = len(sentence_words.intersection(claim_words))
        keyword_score = overlap / len(claim_words)
        
        # 2. SBERT semantic similarity (if available)
        semantic_score = 0.0
        if self.use_sbert and self.sbert_model:
            try:
                sentence_embedding = self.sbert_model.encode([sentence_text])
                claim_embedding = self.sbert_model.encode([claim_text])
                
                similarity = np.dot(sentence_embedding[0], claim_embedding[0]) / (
                    np.linalg.norm(sentence_embedding[0]) * np.linalg.norm(claim_embedding[0])
                )
                semantic_score = max(0.0, similarity)
            except Exception as e:
                print(f"⚠️ SBERT similarity calculation failed: {e}")
        
        # 3. Topic coherence (shared concepts)
        coherence_score = self._calculate_topic_coherence(sentence_text, claim_text)
        
        # Combine scores
        if self.use_sbert and semantic_score > 0:
            relevance_score = (
                keyword_score * 0.4 +
                semantic_score * 0.4 +
                coherence_score * 0.2
            )
        else:
            relevance_score = (
                keyword_score * 0.6 +
                coherence_score * 0.4
            )
        
        return min(1.0, relevance_score)

    def _calculate_entity_based_score(self, sentence_text: str, entities: Optional[List[str]], 
                                    claim_text: str) -> float:
        """
        Calculate entity-based relevance score
        """
        if not entities or not sentence_text:
            return 0.0
        
        sentence_lower = sentence_text.lower()
        entity_matches = []
        
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower in sentence_lower:
                entity_matches.append(entity)
        
        if not entity_matches:
            return 0.0
        
        # Entity frequency in sentence
        entity_frequency = len(entity_matches) / len(entities)
        
        # Entity relevance to claim
        claim_entities = [e for e in entities if e.lower() in claim_text.lower()]
        relevant_entities = [e for e in entity_matches if e in claim_entities]
        entity_relevance = len(relevant_entities) / max(len(claim_entities), 1)
        
        # Combined entity score
        entity_score = (entity_frequency * 0.6 + entity_relevance * 0.4)
        
        return min(1.0, entity_score)

    def _calculate_contradiction_score(self, sentence_text: str, claim_text: str) -> float:
        """
        Calculate contradiction score between sentence and claim
        """
        if not sentence_text or not claim_text:
            return 0.0
        
        sentence_lower = sentence_text.lower()
        claim_lower = claim_text.lower()
        
        contradiction_indicators = 0
        total_indicators = 0
        
        # Check for negation indicators
        for negation in self.contradiction_indicators['negation']:
            if negation in sentence_lower:
                contradiction_indicators += 1
            total_indicators += 1
        
        # Check for opposition indicators
        for opposition in self.contradiction_indicators['opposition']:
            if opposition in sentence_lower:
                contradiction_indicators += 1
            total_indicators += 1
        
        # Check for contrast indicators
        for contrast in self.contradiction_indicators['contrast']:
            if contrast in sentence_lower:
                contradiction_indicators += 1
            total_indicators += 1
        
        # Check for disagreement indicators
        for disagreement in self.contradiction_indicators['disagreement']:
            if disagreement in sentence_lower:
                contradiction_indicators += 1
            total_indicators += 1
        
        # Calculate contradiction score
        if total_indicators > 0:
            contradiction_score = contradiction_indicators / total_indicators
        else:
            contradiction_score = 0.0
        
        return min(1.0, contradiction_score)

    def _calculate_final_confidence_score(self, sentence_data: Dict) -> float:
        """
        Calculate final confidence score combining all filtering results
        """
        # Get individual scores
        quality_score = sentence_data.get('quality_score', 0.0)
        relevance_score = sentence_data.get('relevance_score', 0.0)
        entity_score = sentence_data.get('entity_score', 0.0)
        contradiction_score = sentence_data.get('contradiction_score', 0.0)
        original_score = sentence_data.get('score', 0.0)
        
        # Convert contradiction score to agreement score
        agreement_score = 1.0 - contradiction_score
        
        # Calculate weighted confidence score
        confidence_score = (
            quality_score * 0.2 +
            relevance_score * 0.3 +
            entity_score * 0.2 +
            agreement_score * 0.2 +
            original_score * 0.1
        )
        
        return float(min(1.0, confidence_score))

    def _analyze_sentence_quality(self, sentence_text: str) -> Dict:
        """Analyze sentence quality factors"""
        words = sentence_text.split()
        meaningful_words = [word for word in words 
                          if len(word) > 2 and word.lower() not in self.stop_words]
        
        return {
            'word_count': len(words),
            'meaningful_word_count': len(meaningful_words),
            'information_density': len(meaningful_words) / max(len(words), 1),
            'structure_score': self._calculate_structure_score(sentence_text),
            'richness_score': self._calculate_content_richness(sentence_text)
        }

    def _analyze_semantic_relevance(self, sentence_text: str, claim_text: str) -> Dict:
        """Analyze semantic relevance factors"""
        sentence_words = set(sentence_text.lower().split())
        claim_words = set(claim_text.lower().split())
        overlap = sentence_words.intersection(claim_words)
        
        return {
            'keyword_overlap': len(overlap),
            'keyword_overlap_ratio': len(overlap) / max(len(claim_words), 1),
            'shared_keywords': list(overlap),
            'topic_coherence': self._calculate_topic_coherence(sentence_text, claim_text)
        }

    def _analyze_entity_presence(self, sentence_text: str, entities: Optional[List[str]]) -> Dict:
        """Analyze entity presence in sentence"""
        sentence_lower = sentence_text.lower()
        found_entities = [e for e in entities if e.lower() in sentence_lower] if entities else []
        
        return {
            'found_entities': found_entities,
            'entity_count': len(found_entities),
            'entity_coverage': len(found_entities) / max(len(entities), 1) if entities else 0
        }

    def _analyze_contradiction_indicators(self, sentence_text: str) -> Dict:
        """Analyze contradiction indicators in sentence"""
        sentence_lower = sentence_text.lower()
        found_indicators = {}
        
        for category, indicators in self.contradiction_indicators.items():
            found = [ind for ind in indicators if ind in sentence_lower]
            found_indicators[category] = found
        
        return {
            'found_indicators': found_indicators,
            'total_indicators': sum(len(indicators) for indicators in found_indicators.values())
        }

    def _calculate_structure_score(self, sentence_text: str) -> float:
        """Calculate sentence structure score"""
        # Simple heuristic: check for basic sentence structure
        words = sentence_text.split()
        if len(words) < 3:
            return 0.3
        
        # Check for common sentence patterns
        has_verb = any(word.endswith(('là', 'có', 'được', 'cho', 'với')) for word in words)
        has_noun = any(len(word) > 3 for word in words)
        
        if has_verb and has_noun:
            return 0.8
        elif has_verb or has_noun:
            return 0.5
        else:
            return 0.3

    def _calculate_content_richness(self, sentence_text: str) -> float:
        """Calculate content richness score"""
        words = sentence_text.split()
        unique_words = len(set(words))
        
        # Vocabulary diversity
        diversity_score = unique_words / max(len(words), 1)
        
        # Information content (longer words = more information)
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        length_score = min(1.0, avg_word_length / 8.0)  # Normalize to 8 chars
        
        return (diversity_score * 0.6 + length_score * 0.4)

    def _calculate_topic_coherence(self, sentence_text: str, claim_text: str) -> float:
        """Calculate topic coherence between sentence and claim"""
        # Extract key concepts (words longer than 3 characters)
        sentence_concepts = {word.lower() for word in sentence_text.split() if len(word) > 3}
        claim_concepts = {word.lower() for word in claim_text.split() if len(word) > 3}
        
        if not claim_concepts:
            return 0.0
        
        # Calculate concept overlap
        overlap = len(sentence_concepts.intersection(claim_concepts))
        coherence_score = overlap / len(claim_concepts)
        
        return min(1.0, coherence_score)

    def _calculate_filtering_statistics(self, original_sentences: List[Dict], 
                                      final_sentences: List[Dict], 
                                      pipeline_results: Dict) -> Dict:
        """Calculate comprehensive filtering statistics"""
        stats = {
            'overall_filtering_rate': float((1 - len(final_sentences) / len(original_sentences)) * 100),
            'stage_breakdown': pipeline_results['stage_results'],
            'quality_scores': {
                'min': float(min(s.get('quality_score', 0) for s in final_sentences)) if final_sentences else 0,
                'max': float(max(s.get('quality_score', 0) for s in final_sentences)) if final_sentences else 0,
                'avg': float(sum(float(s.get('quality_score', 0)) for s in final_sentences) / len(final_sentences)) if final_sentences else 0
            },
            'relevance_scores': {
                'min': float(min(s.get('relevance_score', 0) for s in final_sentences)) if final_sentences else 0,
                'max': float(max(s.get('relevance_score', 0) for s in final_sentences)) if final_sentences else 0,
                'avg': float(sum(float(s.get('relevance_score', 0)) for s in final_sentences) / len(final_sentences)) if final_sentences else 0
            },
            'confidence_scores': {
                'min': float(min(s.get('confidence_score', 0) for s in final_sentences)) if final_sentences else 0,
                'max': float(max(s.get('confidence_score', 0) for s in final_sentences)) if final_sentences else 0,
                'avg': float(sum(float(s.get('confidence_score', 0)) for s in final_sentences) / len(final_sentences)) if final_sentences else 0
            }
        }
        
        return stats


def integrate_advanced_filtering_with_existing_pipeline(processor, text_graph, claim_text, 
                                                      sentences, entities=None, 
                                                      max_final_sentences=30):
    """
    🔗 Integrate Advanced Filtering with existing pipeline
    """
    # Initialize advanced filter
    advanced_filter = AdvancedDataFilter(
        use_sbert=True,
        use_contradiction_detection=True
    )
    
    # Apply multi-stage filtering
    filtering_results = advanced_filter.multi_stage_filtering_pipeline(
        sentences=sentences,
        claim_text=claim_text,
        entities=entities,
        min_quality_score=0.3,
        min_relevance_score=0.25,
        max_final_sentences=max_final_sentences
    )
    
    filtered_sentences = filtering_results['filtered_sentences']
    pipeline_results = filtering_results['pipeline_results']
    
    # Add filtering metadata to sentences
    for sentence_data in filtered_sentences:
        sentence_data['filtering_metadata'] = {
            'quality_score': sentence_data.get('quality_score', 0),
            'relevance_score': sentence_data.get('relevance_score', 0),
            'entity_score': sentence_data.get('entity_score', 0),
            'contradiction_score': sentence_data.get('contradiction_score', 0),
            'confidence_score': sentence_data.get('confidence_score', 0)
        }
    
    return filtered_sentences, pipeline_results


if __name__ == "__main__":
    # Test the advanced filtering system
    print("🧪 Testing Advanced Data Filtering System...")
    
    # Sample data
    test_sentences = [
        {"sentence": "Việt Nam là một quốc gia ở Đông Nam Á.", "score": 0.8},
        {"sentence": "Thời tiết hôm nay rất đẹp.", "score": 0.3},
        {"sentence": "GDP của Việt Nam tăng trưởng 6.8% trong năm 2023.", "score": 0.9},
        {"sentence": "Cà phê là thức uống phổ biến ở Việt Nam.", "score": 0.7}
    ]
    
    test_claim = "Việt Nam có nền kinh tế tăng trưởng mạnh"
    test_entities = ["Việt Nam", "GDP", "kinh tế", "tăng trưởng"]
    
    # Initialize filter
    filter_system = AdvancedDataFilter()
    
    # Apply filtering
    results = filter_system.multi_stage_filtering_pipeline(
        sentences=test_sentences,
        claim_text=test_claim,
        entities=test_entities,
        max_final_sentences=2
    )
    
    print(f"\n✅ Test completed!")
    print(f"📊 Results: {len(results['filtered_sentences'])} sentences selected")
    print(f"📈 Filtering rate: {results['pipeline_results']['filtering_statistics']['overall_filtering_rate']:.1f}%") 