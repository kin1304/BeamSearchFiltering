#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MINT TextGraph - Beam Search Path Finding
Find paths from claim to sentence nodes using Beam Search
"""

import json
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional
import heapq
from datetime import datetime
import time
from difflib import SequenceMatcher
import networkx as nx


class Path:
    """Represents a path in the graph"""
    
    def __init__(self, nodes: List[str], edges: Optional[List[Tuple[str, str, str]]] = None, score: float = 0.0):
        self.nodes = nodes  # List of node IDs
        self.edges = edges or []  # List of (from_node, to_node, relation)
        self.score = score  # Path evaluation score
        self.claim_words = set()  # Words in claim for comparison
        self.word_matches = set()  # ✅ ADDED: Set of matched words
        self.path_words = set()   # Words in path
        self.entities_visited = set()  # Entities visited
        
    def __lt__(self, other):
        """Compare to sort paths by score"""
        return self.score < other.score
        
    def add_node(self, node_id: str, edge_info: Optional[Tuple[str, str, str]] = None):
        """Add node to path"""
        self.nodes.append(node_id)
        if edge_info:
            self.edges.append(edge_info)
            
    def copy(self):
        """Create a copy of the path"""
        new_path = Path(self.nodes.copy(), self.edges.copy(), self.score)
        new_path.claim_words = self.claim_words.copy()
        new_path.word_matches = self.word_matches.copy()
        new_path.path_words = self.path_words.copy()
        new_path.entities_visited = self.entities_visited.copy()
        return new_path
        
    def get_current_node(self):
        """Get current node (end of path)"""
        return self.nodes[-1] if self.nodes else None
        
    def contains_node(self, node_id: str):
        """Check if path contains this node"""
        return node_id in self.nodes
        
    def to_dict(self):
        """Convert path to dictionary for export"""
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'score': self.score,
            'length': len(self.nodes),
            'claim_words_matched': len(self.claim_words.intersection(self.path_words)),
            'total_claim_words': len(self.claim_words),
            'entities_visited': list(self.entities_visited),
            'path_summary': self._get_path_summary()
        }
        
    def _get_path_summary(self):
        """Create a brief summary of the path"""
        node_types = []
        for node in self.nodes:
            if node.startswith('claim'):
                node_types.append('CLAIM')
            elif node.startswith('word'):
                node_types.append('WORD')
            elif node.startswith('sentence'):
                node_types.append('SENTENCE')
            elif node.startswith('entity'):
                node_types.append('ENTITY')
            else:
                node_types.append('UNKNOWN')
        return ' -> '.join(node_types)


class BeamSearchPathFinder:
    """Beam Search to find paths from claim to sentence nodes"""
    
    def __init__(self, text_graph, beam_width: int = 25, max_depth: int = 30, allow_skip_edge: bool = False):
        self.graph = text_graph
        self.beam_width = beam_width
        self.max_depth = max_depth
        # Allow "jumping" over an intermediate node (2-hop) if needed to expand diversity
        self.allow_skip_edge = allow_skip_edge
        self.claim_words = set()  # Words in claim
        
        # Scoring weights - ✅ IMPROVED WEIGHTS
        self.word_match_weight = 5.0        # Increased from 3.0 to 5.0
        self.semantic_match_weight = 3.0    # ✅ NEW: Semantic similarity
        self.entity_bonus = 2.5             # Increased from 2.0 to 2.5
        self.length_penalty = 0.05          # Decreased from 0.1 to 0.05
        self.sentence_bonus = 4.0           
        self.fuzzy_match_weight = 2.0       # ✅ NEW: Fuzzy string matching
        
        # Stats
        self.paths_explored = 0
        self.sentence_paths_found = 0
        
        # New flag
        self.early_stop_on_sentence = True
        
    def extract_claim_words(self):
        """Extract all words in claim for comparison"""
        claim_words = set()
        
        if self.graph.claim_node:
            # Get all word nodes connected to claim
            for neighbor in self.graph.graph.neighbors(self.graph.claim_node):
                node_data = self.graph.graph.nodes[neighbor]
                if node_data.get('type') == 'word':
                    claim_words.add(node_data.get('text', '').lower())
                    
        self.claim_words = claim_words
        return claim_words
        
    def _calculate_semantic_similarity(self, claim_words, path_words):
        """
        ✅ NEW: Calculate semantic similarity between claim and path words
        Using Jaccard similarity and word overlap
        """
        if not claim_words or not path_words:
            return 0.0
            
        # Jaccard similarity
        intersection = claim_words.intersection(path_words)
        union = claim_words.union(path_words)
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Word overlap ratio
        overlap_ratio = len(intersection) / len(claim_words) if claim_words else 0.0
        
        # Combine scores
        semantic_score = (jaccard * 0.4) + (overlap_ratio * 0.6)
        return semantic_score
        
    def _calculate_fuzzy_similarity(self, claim_text, sentence_text):
        """
        ✅ NEW: Calculate fuzzy string similarity
        """
        if not claim_text or not sentence_text:
            return 0.0
            
        # Normalize texts
        claim_normalized = claim_text.lower().strip()
        sentence_normalized = sentence_text.lower().strip()
        
        # Calculate similarity
        similarity = SequenceMatcher(None, claim_normalized, sentence_normalized).ratio()
        return similarity
        
    def score_path(self, path: Path) -> float:
        """✅ IMPROVED: Calculate score for a path with more metrics"""
        
        if not path.nodes:
            return 0.0
            
        # Get claim text for comparison
        claim_text = ""
        claim_words = set()
        
        for node in path.nodes:
            node_data = self.graph.graph.nodes[node]
            if node_data.get('type') == 'claim':
                claim_text = node_data.get('text', '')
                claim_words = set(claim_text.lower().split())
                break
                
        # Base score
        score = 0.0
        
        # 1. ✅ IMPROVED: Enhanced Word matching score
        path_words = set()
        sentence_texts = []
        
        for node in path.nodes:
            node_data = self.graph.graph.nodes[node]
            node_text = node_data.get('text', '')
            node_type = node_data.get('type', '')
            
            if node_text:
                path_words.update(node_text.lower().split())
                
                # Collect sentence texts for fuzzy matching
                if node_type == 'sentence':
                    sentence_texts.append(node_text)
                
        if claim_words:
            word_matches = claim_words.intersection(path_words)
            word_match_ratio = len(word_matches) / len(claim_words)
            score += word_match_ratio * self.word_match_weight
            path.word_matches = word_matches
            
            # 2. ✅ NEW: Semantic similarity
            semantic_score = self._calculate_semantic_similarity(claim_words, path_words)
            score += semantic_score * self.semantic_match_weight
            
        # 3. ✅ NEW: Fuzzy matching with sentences + Claim entity boost
        if claim_text and sentence_texts:
            max_fuzzy_score = 0.0
            claim_entity_boost = 0.0
            
            # Get claim entities to boost scoring
            claim_entities = self.graph.get_claim_entities() if hasattr(self.graph, 'get_claim_entities') else set()
            
            for sentence_text in sentence_texts:
                fuzzy_score = self._calculate_fuzzy_similarity(claim_text, sentence_text)
                
                # ✅ NEW: Boost for sentences containing claim entities
                entity_boost = 0.0
                if claim_entities:
                    sentence_lower = sentence_text.lower()
                    claim_entity_matches = sum(1 for entity in claim_entities if entity.lower() in sentence_lower)
                    entity_boost = (claim_entity_matches / len(claim_entities)) * 1.0  # Maximum boost 1.0
                
                combined_score = fuzzy_score + entity_boost
                max_fuzzy_score = max(max_fuzzy_score, combined_score)
                claim_entity_boost = max(claim_entity_boost, entity_boost)
                
            score += max_fuzzy_score * self.fuzzy_match_weight
            # Add separate bonus for claim entities
            if claim_entity_boost > 0:
                score += claim_entity_boost * 2.0  # Double weight for claim entity bonus
            
        # 4. ✅ IMPROVED: Entity bonus with higher weight
        entity_count = 0
        for node in path.nodes:
            node_data = self.graph.graph.nodes[node]
            if node_data.get('type') == 'entity':
                entity_count += 1
                
        score += entity_count * self.entity_bonus
        
        # 5. ✅ IMPROVED: Reduce length penalty
        score -= len(path.nodes) * self.length_penalty
        
        # 6. ✅ ADDED: Sentence relevance bonus
        sentence_count = sum(1 for node in path.nodes 
                           if self.graph.graph.nodes[node].get('type') == 'sentence')
        if sentence_count > 0:
            score += sentence_count * 1.5  # Bonus for each sentence in the path
            
        return score
        
    def beam_search(self, start_node: Optional[str] = None) -> List[Path]:
        """
        Perform Beam Search from claim node to sentence nodes
        
        Returns:
            List[Path]: List of the best paths found
        """
        if start_node is None:
            start_node = self.graph.claim_node
            
        if not start_node:
            print("⚠️ No claim node found to start beam search")
            return []
            
        # Extract claim words for scoring
        self.extract_claim_words()
        
        # Prepare graph data for faster lookup
        graph_data = dict(self.graph.graph.nodes(data=True))
        
        # Initialize beam with path from claim node
        beam = [Path([start_node])]
        completed_paths = []  # Paths reaching sentence nodes
        
        print(f"🎯 Starting Beam Search from {start_node}")
        print(f"📊 Beam width: {self.beam_width}, Max depth: {self.max_depth}")
        print(f"💭 Claim words: {self.claim_words}")
        
        for depth in range(self.max_depth):
            if not beam:
                break
                
            print(f"\n🔍 Depth {depth + 1}/{self.max_depth} - Current beam size: {len(beam)}")
            
            new_candidates = []
            
            # Expand each path in the current beam
            for path in beam:
                current_node = path.get_current_node()
                
                # Get all neighbors of the current node
                neighbors = list(self.graph.graph.neighbors(current_node))
                
                for neighbor in neighbors:
                    # Avoid cycles - do not go back to visited nodes
                    if path.contains_node(neighbor):
                        continue
                        
                    # Create new path
                    new_path = path.copy()
                    
                    # Get edge info
                    edge_data = self.graph.graph.get_edge_data(current_node, neighbor)
                    relation = edge_data.get('relation', 'unknown') if edge_data else 'unknown'
                    edge_info = (str(current_node), str(neighbor), str(relation))
                    
                    new_path.add_node(neighbor, edge_info)
                    
                    # Score new path
                    new_path.score = self.score_path(new_path)
                    
                    # Check if it reaches a sentence node
                    neighbor_data = graph_data.get(neighbor, {})
                    if neighbor_data.get('type') == 'sentence':
                        completed_paths.append(new_path)
                        print(f"  ✅ Found path to sentence: {neighbor} (score: {new_path.score:.3f})")
                    else:
                        new_candidates.append(new_path)
                        
            # Select top K candidates for the next beam
            if new_candidates:
                # Sort by score descending and select top beam_width
                new_candidates.sort(key=lambda p: p.score, reverse=True)
                beam = new_candidates[:self.beam_width]
                
                # Debug info
                print(f"  📈 Top scores in beam: {[f'{p.score:.3f}' for p in beam[:5]]}")
            else:
                beam = []
                
        # Combine completed paths and sort by score
        all_paths = completed_paths
        all_paths.sort(key=lambda p: p.score, reverse=True)
        
        print(f"\n🎉 Beam Search completed!")
        print(f"  Found {len(completed_paths)} paths to sentences")
        print(f"  Top path score: {all_paths[0].score:.3f}" if all_paths else "  No paths found")
        
        return all_paths
        
    def find_best_paths(self, max_paths: int = 20) -> List[Path]:
        """
        Find the best paths from claim to sentences
        
        Args:
            max_paths: Maximum number of paths to return
            
        Returns:
            List[Path]: List of paths sorted by score
        """
        start_time = time.time()
        
        # Get claim nodes and sentence nodes  
        claim_nodes = [node for node, data in self.graph.graph.nodes(data=True) 
                      if data.get('type') == 'claim']
        sentence_nodes = [node for node, data in self.graph.graph.nodes(data=True)
                         if data.get('type') == 'sentence']
                         
        if not claim_nodes:
            print("⚠️  No claim nodes found!")
            return []
            
        if not sentence_nodes:
            print("⚠️  No sentence nodes found!")
            return []
            
        print(f"🎯 Found {len(claim_nodes)} claim nodes, {len(sentence_nodes)} sentence nodes")
        
        # Initialize beam with paths from each claim node
        current_beam = []
        for claim_node in claim_nodes:
            initial_path = Path([claim_node], [], 0.0)
            current_beam.append(initial_path)
            
        completed_paths = []
        
        # Beam search main loop
        for depth in range(self.max_depth):
            if not current_beam:
                break
                
            next_beam = []
            
            for path in current_beam:
                current_node = path.nodes[-1]
                
                # Check if the current node is a sentence
                current_node_data = self.graph.graph.nodes[current_node]
                if current_node_data.get('type') == 'sentence':
                    # Reached sentence node - can stop here
                    completed_paths.append(path)
                    self.sentence_paths_found += 1
                    continue  # Do not expand further from sentence node
                    
                # Expand path to neighbors
                for neighbor in self.graph.graph.neighbors(current_node):
                    # Avoid cycles
                    if neighbor in path.nodes:
                        continue
                        
                    # Create new path
                    edge_data = self.graph.graph.get_edge_data(current_node, neighbor, {})
                    edge_label = edge_data.get('label', f"{current_node}->{neighbor}")
                    
                    new_path = Path(
                        path.nodes + [neighbor],
                        path.edges + [edge_label],
                        0.0
                    )
                    
                    # Calculate score for new path
                    new_path.score = self.score_path(new_path)
                    next_beam.append(new_path)
                    self.paths_explored += 1
                    
            # Keep top beam_width paths
            next_beam.sort(key=lambda p: p.score, reverse=True)
            current_beam = next_beam[:self.beam_width]
            
            if self.early_stop_on_sentence and completed_paths:
                break  # Stop immediately when the first sentence is found
            
        # Combine completed paths and current beam
        all_paths = completed_paths + current_beam
        
        # Filter only paths ending at sentence nodes
        sentence_paths = []
        for path in all_paths:
            if path.nodes:
                last_node = path.nodes[-1] 
                last_node_data = self.graph.graph.nodes[last_node]
                if last_node_data.get('type') == 'sentence':
                    sentence_paths.append(path)
                    
        # Sort and get top paths
        sentence_paths.sort(key=lambda p: p.score, reverse=True)
        
        end_time = time.time()
        print(f"⏱️  Beam search completed in {end_time - start_time:.2f}s")
        print(f"📊 Explored {self.paths_explored} paths, found {len(sentence_paths)} sentence paths")
        
        return sentence_paths[:max_paths]
        
    def export_paths_to_file(self, paths: List[Path], output_file: str = None) -> str:
        """
        Export paths to a JSON file for investigation
        
        Args:
            paths: List of paths to export
            output_file: Output file path (if None, will generate one)
            
        Returns:
            str: Path of the file saved
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use absolute path to ensure correct directory
            current_dir = os.getcwd()
            if current_dir.endswith('vncorenlp'):
                # If we're in vncorenlp directory, go back to parent
                current_dir = os.path.dirname(current_dir)
            output_file = os.path.join(current_dir, "output", f"beam_search_paths_{timestamp}.json")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Prepare data for export
        export_data = {
            'search_config': {
                'beam_width': self.beam_width,
                'max_depth': self.max_depth,
                'word_match_weight': self.word_match_weight,
                'entity_bonus': self.entity_bonus,
                'length_penalty': self.length_penalty,
                'sentence_bonus': self.sentence_bonus
            },
            'claim_words': list(self.claim_words),
            'total_paths_found': len(paths),
            'paths': []
        }
        
        # Prepare graph data for node details
        graph_data = dict(self.graph.graph.nodes(data=True))
        
        for i, path in enumerate(paths):
            path_data = path.to_dict()
            
            # Add detailed node info
            path_data['node_details'] = []
            for node_id in path.nodes:
                node_info = graph_data.get(node_id, {})
                path_data['node_details'].append({
                    'id': node_id,
                    'type': node_info.get('type', 'unknown'),
                    'text': node_info.get('text', ''),
                    'pos': node_info.get('pos', ''),
                    'lemma': node_info.get('lemma', '')
                })
                
            export_data['paths'].append(path_data)
            
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
            
        print(f"💾 Exported {len(paths)} paths to: {output_file}")
        return output_file
        
    def export_paths_summary(self, paths: List[Path], output_file: str = None) -> str:
        """
        Export a readable summary of paths
        
        Args:
            paths: List of paths
            output_file: Output file (if None, will generate one)
            
        Returns:
            str: Path of the file saved
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use absolute path to ensure correct directory
            current_dir = os.getcwd()
            if current_dir.endswith('vncorenlp'):
                # If we're in vncorenlp directory, go back to parent
                current_dir = os.path.dirname(current_dir)
            output_file = os.path.join(current_dir, "output", f"beam_search_summary_{timestamp}.txt")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Prepare graph data
        graph_data = dict(self.graph.graph.nodes(data=True))
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("🎯 BEAM SEARCH PATH ANALYSIS\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Search Configuration:\n")
            f.write(f"  Beam Width: {self.beam_width}\n")
            f.write(f"  Max Depth: {self.max_depth}\n")
            f.write(f"  Claim Words: {', '.join(self.claim_words)}\n")
            f.write(f"  Total Paths Found: {len(paths)}\n\n")
            
            for i, path in enumerate(paths[:10]):  # Top 10 paths
                f.write(f"PATH #{i+1} (Score: {path.score:.3f})\n")
                f.write("-" * 40 + "\n")
                
                f.write(f"Length: {len(path.nodes)} nodes\n")
                f.write(f"Word Matches: {len(path.word_matches) if hasattr(path, 'word_matches') else 'None'}\n")
                f.write(f"Entities Visited: {', '.join(path.entities_visited) if path.entities_visited else 'None'}\n")
                f.write(f"Path Type: {path._get_path_summary()}\n\n")
                
                f.write("Detailed Path:\n")
                for j, node_id in enumerate(path.nodes):
                    node_info = graph_data.get(node_id, {})
                    node_type = node_info.get('type', 'unknown').upper()
                    node_text = node_info.get('text', '')[:50]  # Truncate long text
                    
                    prefix = "  START: " if j == 0 else f"  {j:2d}: "
                    f.write(f"{prefix}[{node_type}] {node_text}\n")
                    
                    if j < len(path.edges):
                        edge_info = path.edges[j]
                        f.write(f"       └─ ({edge_info[2]}) ─>\n")
                        
                f.write("\n" + "="*60 + "\n\n")
                
        print(f"📄 Exported paths summary to: {output_file}")
        return output_file

    def multi_level_beam_search(
        self,
        max_levels: int = 3,
        beam_width_per_level: int = 3,
        min_new_sentences: int = 2,   # ❶ ensure each level has ≥ 2 new sentences
        advanced_data_filter=None,
        claim_text: str = "",
        entities=None,
        filter_top_k: int = 2
    ) -> Dict[int, List[Path]]:
        """
        Multi-level beam search: from claim → sentences → related sentences → ...
        
        Args:
            max_levels: Maximum number of levels (k)
            beam_width_per_level: Number of sentences to keep per level
            
        Returns:
            Dict[level, List[Path]]: Sentences by level
        """
        results = {}
        all_found_sentences = set()  # Track sentences found to avoid duplicates
        
        print(f"🎯 Starting Multi-Level Beam Search (max_levels={max_levels}, beam_width={beam_width_per_level})")
        
        # Level 0: Beam search from claim
        print(f"\n📍 LEVEL 0: Claim → Sentences")
        level_0_paths = self.find_best_paths(max_paths=beam_width_per_level)
        level_0_sentences = self._extract_sentence_nodes_from_paths(level_0_paths)
        
        results[0] = level_0_paths
        all_found_sentences.update(level_0_sentences)
        
        print(f"   Found {len(level_0_sentences)} sentences at level 0")
        
        # Levels 1 to k: Beam search from sentences of the previous level
        current_sentence_nodes = level_0_sentences
        
        for level in range(1, max_levels + 1):
            if not current_sentence_nodes:
                print(f"   No sentences to expand from level {level-1}")
                break
                
            print(f"\n📍 LEVEL {level}: Sentences → New Sentences")
            level_paths = []
            new_sentence_nodes = set()
            
            # Beam search from each sentence of the previous level
            for sentence_node in current_sentence_nodes:
                print(f"   Expanding from sentence: {sentence_node}")
                
                # Beam search from this sentence
                sentence_paths = self._beam_search_from_sentence(
                    sentence_node, 
                    max_paths=beam_width_per_level,
                    exclude_sentences=all_found_sentences
                )
                
                # Get new sentences
                new_sentences = self._extract_sentence_nodes_from_paths(sentence_paths)
                new_sentences = [s for s in new_sentences if s not in all_found_sentences]
                
                level_paths.extend(sentence_paths)
                new_sentence_nodes.update(new_sentences)
                
                print(f"     → Found {len(new_sentences)} new sentences")
            
            # Keep top beam_width_per_level best sentences for this level
            if level_paths:
                level_paths.sort(key=lambda p: p.score, reverse=True)
                level_paths = level_paths[:beam_width_per_level]

                # ❷ Get new sentences, remove duplicates
                final_new_sentences = self._extract_sentence_nodes_from_paths(level_paths)
                unique_new = [s for s in final_new_sentences if s not in all_found_sentences]

                # 🔄 Apply AdvancedDataFilter (if provided) to select seeds for the next level
                if advanced_data_filter and claim_text and unique_new:
                    try:
                        raw_sentences = [
                            {"sentence": self.graph.graph.nodes[node]["text"]}
                            for node in unique_new
                        ]
                        filtered = advanced_data_filter.multi_stage_filtering_pipeline(
                            sentences=raw_sentences,
                            claim_text=claim_text,
                            entities=entities or [],
                            max_final_sentences=filter_top_k,
                            min_quality_score=0.25,
                            min_relevance_score=0.2,
                            suppress_log=True
                        )["filtered_sentences"]

                        # Get node-ids corresponding to the remaining sentences after filtering
                        filtered_texts = {s["sentence"] for s in filtered}
                        filtered_nodes = [
                            n for n in unique_new
                            if self.graph.graph.nodes[n]["text"] in filtered_texts
                        ]
                        if filtered_nodes:
                            unique_new = filtered_nodes[:filter_top_k]
                            print(f"   🔍 Advanced filter kept {len(unique_new)} sentences for the next level")
                    except Exception as e:
                        print(f"⚠️  Advanced filter error (level {level}): {e}")

                # ❸ If not enough, get more sentences (not duplicates) from the level_paths list (sorted)
                if len(unique_new) < min_new_sentences:
                    for path in level_paths:
                        for node in path.nodes[::-1]:  # traverse from end of path
                            node_data = self.graph.graph.nodes[node]
                            if node_data.get('type') == 'sentence' and node not in all_found_sentences:
                                unique_new.append(node)
                                if len(unique_new) >= min_new_sentences:
                                    break
                        if len(unique_new) >= min_new_sentences:
                            break

                # ❹ Update results / tracking
                results[level] = level_paths
                all_found_sentences.update(unique_new)
                current_sentence_nodes = unique_new
                
                print(f"   Level {level} final: {len(unique_new)} sentences")
            else:
                print(f"   Level {level}: No new sentences found")
                break
        
        print(f"\n🎉 Multi-Level Search completed! Total levels: {len(results)}")
        return results

    def _extract_sentence_nodes_from_paths(self, paths: List[Path]) -> List[str]:
        """Extract unique sentence node IDs from paths"""
        sentence_nodes = set()
        for path in paths:
            for node in path.nodes:
                node_data = self.graph.graph.nodes.get(node, {})
                if node_data.get('type') == 'sentence':
                    sentence_nodes.add(node)
        return list(sentence_nodes)

    def _beam_search_from_sentence(self, start_sentence: str, max_paths: int = 3, exclude_sentences: Set[str] = None) -> List[Path]:
        """
        Beam search from a sentence node to find related sentences
        
        Args:
            start_sentence: Sentence node to start from
            max_paths: Maximum number of paths
            exclude_sentences: Sentences to exclude (already found before)
        """
        if exclude_sentences is None:
            exclude_sentences = set()
        
        # Initialize beam from the sentence node
        beam = [Path([start_sentence])]
        completed_paths = []
        
        # Reduced depth for sentence-to-sentence search
        max_depth = min(self.max_depth // 2, 15)  # Shorter paths for efficiency
        
        for depth in range(max_depth):
            if not beam:
                break
                
            new_candidates = []
            
            for path in beam:
                current_node = path.get_current_node()
                
                # Expand to neighbors
                for neighbor in self.graph.graph.neighbors(current_node):
                    # Avoid cycles
                    if path.contains_node(neighbor):
                        continue
                    
                    # Create new path
                    new_path = path.copy()
                    edge_data = self.graph.graph.get_edge_data(current_node, neighbor)
                    relation = edge_data.get('relation', 'unknown') if edge_data else 'unknown'
                    edge_info = (current_node, neighbor, relation)
                    
                    new_path.add_node(neighbor, edge_info)
                    new_path.score = self.score_path(new_path)
                    
                    # Check if reached new sentence
                    neighbor_data = self.graph.graph.nodes.get(neighbor, {})
                    if (neighbor_data.get('type') == 'sentence' and 
                        neighbor != start_sentence and  # Not the same as start
                        neighbor not in exclude_sentences):  # Not already found
                        completed_paths.append(new_path)
                    else:
                        new_candidates.append(new_path)
            
            # Keep top candidates
            if new_candidates:
                new_candidates.sort(key=lambda p: p.score, reverse=True)
                beam = new_candidates[:self.beam_width]
            else:
                beam = []
        
        # Return top sentence paths
        completed_paths.sort(key=lambda p: p.score, reverse=True)
        return completed_paths[:max_paths] 
        
    def multi_level_beam_search_from_start_nodes(
        self,
        start_nodes: List[str],
        max_levels: int = 3,
        beam_width_per_level: int = 3,
        min_new_sentences: int = 2,
        advanced_data_filter=None,
        claim_text: str = "",
        entities=None,
        filter_top_k: int = 2
    ) -> Dict[int, List[Path]]:
        """
        Multi-level beam search from specific start nodes (instead of claim)
        
        Args:
            start_nodes: List of sentence node IDs to start from
            max_levels: Maximum number of levels
            beam_width_per_level: Number of sentences to keep per level
            
        Returns:
            Dict[level, List[Path]]: Sentences by level
        """
        results = {}
        all_found_sentences = set(start_nodes)  # Include start nodes to avoid duplicates
        
        print(f"🎯 Starting Multi-Level Beam Search from {len(start_nodes)} start nodes")
        
        # Level 0: Start from the provided sentence nodes
        print(f"\n📍 LEVEL 0: Start Nodes → Expansion")
        
        # Create initial paths from start nodes
        level_0_paths = []
        for start_node in start_nodes:
            initial_path = Path([start_node])
            initial_path.score = 1.0  # Base score for start nodes
            level_0_paths.append(initial_path)
        
        results[0] = level_0_paths
        current_sentence_nodes = start_nodes
        
        print(f"   Starting from {len(current_sentence_nodes)} sentences at level 0")
        
        # Levels 1 to k: Beam search from sentences of the previous level
        for level in range(1, max_levels + 1):
            if not current_sentence_nodes:
                print(f"   No sentences to expand from level {level-1}")
                break
                
            print(f"\n📍 LEVEL {level}: Sentences → New Sentences")
            level_paths = []
            new_sentence_nodes = set()
            
            # Beam search from each sentence of the previous level
            for sentence_node in current_sentence_nodes:
                print(f"   Expanding from sentence: {sentence_node}")
                
                # Beam search from this sentence
                sentence_paths = self._beam_search_from_sentence(
                    sentence_node, 
                    max_paths=beam_width_per_level,
                    exclude_sentences=all_found_sentences
                )
                
                # Get new sentences
                new_sentences = self._extract_sentence_nodes_from_paths(sentence_paths)
                new_sentences = [s for s in new_sentences if s not in all_found_sentences]
                
                level_paths.extend(sentence_paths)
                new_sentence_nodes.update(new_sentences)
                
                print(f"     → Found {len(new_sentences)} new sentences")
            
            # Keep top beam_width_per_level best sentences for this level
            if level_paths:
                level_paths.sort(key=lambda p: p.score, reverse=True)
                level_paths = level_paths[:beam_width_per_level]

                # Get new sentences, remove duplicates
                final_new_sentences = self._extract_sentence_nodes_from_paths(level_paths)
                unique_new = [s for s in final_new_sentences if s not in all_found_sentences]

                # Apply AdvancedDataFilter (if provided) to select seeds for the next level
                if advanced_data_filter and claim_text and unique_new:
                    try:
                        raw_sentences = [
                            {"sentence": self.graph.graph.nodes[node]["text"]}
                            for node in unique_new
                        ]
                        filtered = advanced_data_filter.multi_stage_filtering_pipeline(
                            sentences=raw_sentences,
                            claim_text=claim_text,
                            entities=entities or [],
                            max_final_sentences=filter_top_k,
                            min_quality_score=0.25,
                            min_relevance_score=0.2,
                            suppress_log=True
                        )["filtered_sentences"]

                        # Get node-ids corresponding to the remaining sentences after filtering
                        filtered_texts = {s["sentence"] for s in filtered}
                        filtered_nodes = [
                            n for n in unique_new
                            if self.graph.graph.nodes[n]["text"] in filtered_texts
                        ]
                        if filtered_nodes:
                            unique_new = filtered_nodes[:filter_top_k]
                            print(f"   🔍 Advanced filter kept {len(unique_new)} sentences for the next level")
                    except Exception as e:
                        print(f"⚠️  Advanced filter error (level {level}): {e}")

                # If not enough, get more sentences (not duplicates) from the level_paths list
                if len(unique_new) < min_new_sentences:
                    for path in level_paths:
                        for node in path.nodes[::-1]:  # traverse from end of path
                            node_data = self.graph.graph.nodes[node]
                            if node_data.get('type') == 'sentence' and node not in all_found_sentences:
                                unique_new.append(node)
                                if len(unique_new) >= min_new_sentences:
                                    break
                        if len(unique_new) >= min_new_sentences:
                            break

                # Update results / tracking
                results[level] = level_paths
                all_found_sentences.update(unique_new)
                current_sentence_nodes = unique_new
                
                print(f"   Level {level} final: {len(unique_new)} sentences")
            else:
                print(f"   Level {level}: No new sentences found")
                break
        
        print(f"\n🎉 Multi-Level Search from start nodes completed! Total levels: {len(results)}")
        return results 