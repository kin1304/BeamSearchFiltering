import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from .beam_search import BeamSearchPathFinder
import unicodedata
import re
from difflib import SequenceMatcher
from typing import List, Dict

try:
    from mint.helpers import segment_entity_with_vncorenlp
except ImportError:
    try:
        from process_with_beam_search_fixed import segment_entity_with_vncorenlp
    except ImportError:
        segment_entity_with_vncorenlp = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None

class TextGraph:
    """
    TextGraph class to build and analyze text graphs from context and claim
    
    The graph includes the following node types:
    - Word nodes: contain each word in context and claim
    - Sentence nodes: sentences in context  
    - Claim node: claim value
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.word_nodes = {}
        self.sentence_nodes = {}
        self.claim_node = None
        self.entity_nodes = {}  # Add dictionary to manage entity nodes
        self.claim_entities = set()  # ✅ NEW: Store claim entities for scoring
        
        # POS tag filtering configuration
        self.enable_pos_filtering = True  # Default enabled to reduce noise
        self.important_pos_tags = {
            'N',    # Common nouns
            'Np',   # Proper nouns
            'V',    # Verbs
            'A',    # Adjectives
            'Nc',   # Person nouns
            'M',    # Numbers
            'R',    # Adverbs (can be debated)
            'P'     # Pronouns (can be debated)
        }
        
        # Load environment variables
        load_dotenv()
        self.openai_client = None
        self._init_openai_client()
        
        # Semantic similarity components
        self.phobert_tokenizer = None
        self.phobert_model = None
        self.word_embeddings = {}  # Cache embeddings
        self.embedding_dim = 768  # PhoBERT base dimension (full dimension - no PCA)
        self.faiss_index = None
        self.word_to_index = {}  # Mapping from word -> index in faiss
        self.index_to_word = {}  # Reverse mapping
        
        # Semantic similarity parameters (optimized for full embeddings)
        self.similarity_threshold = 0.85
        self.top_k_similar = 5
        
        self._init_phobert_model()
    
    def set_pos_filtering(self, enable=True, custom_pos_tags=None):
        """
        Configure POS tag filtering for word nodes
        
        Args:
            enable (bool): Enable/disable POS tag filtering feature
            custom_pos_tags (set): Set of POS tags to keep (if None, use default)
        """
        self.enable_pos_filtering = enable
        if custom_pos_tags is not None:
            self.important_pos_tags = set(custom_pos_tags)
    
    def is_important_word(self, word, pos_tag):
        """
        Check if a word is important based on its POS tag
        
        Args:
            word (str): Word to check
            pos_tag (str): POS tag of the word
            
        Returns:
            bool: True if the word is important and should create a word node
        """
        # If POS filtering is not enabled, all words are important
        if not self.enable_pos_filtering:
            return True
            
        # Check if POS tag is in the important list
        return pos_tag in self.important_pos_tags
    
    def add_word_node(self, word, pos_tag=None, lemma=None):
        """Add word node to graph (can filter by POS tag)"""
        # Check if the word is important
        if not self.is_important_word(word, pos_tag):
            return None  # Don't create node for unimportant words
            
        if word not in self.word_nodes:
            node_id = f"word_{len(self.word_nodes)}"
            self.word_nodes[word] = node_id
            self.graph.add_node(node_id, 
                              type="word", 
                              text=word, 
                              pos=pos_tag, 
                              lemma=lemma)
        return self.word_nodes[word]
    
    def add_sentence_node(self, sentence_id, sentence_text):
        """Add sentence node to graph"""
        node_id = f"sentence_{sentence_id}"
        self.sentence_nodes[sentence_id] = node_id
        self.graph.add_node(node_id, 
                          type="sentence", 
                          text=sentence_text)
        return node_id
    
    def add_claim_node(self, claim_text):
        """Add claim node to graph"""
        self.claim_node = "claim_0"
        self.graph.add_node(self.claim_node, 
                          type="claim", 
                          text=claim_text)
        return self.claim_node
    
    def connect_word_to_sentence(self, word_node, sentence_node):
        """Connect word to sentence"""
        self.graph.add_edge(word_node, sentence_node, relation="belongs_to", edge_type="structural")
    
    def connect_word_to_claim(self, word_node, claim_node):
        """Connect word to claim"""
        self.graph.add_edge(word_node, claim_node, relation="belongs_to", edge_type="structural")
    
    def connect_dependency(self, dependent_word_node, head_word_node, dep_label):
        """Connect dependency between two words"""
        self.graph.add_edge(dependent_word_node, head_word_node, 
                          relation=dep_label, edge_type="dependency")
    
    def build_from_vncorenlp_output(self, context_sentences, claim_text, claim_sentences):
        """Build graph from py_vncorenlp output"""
        
        # Add claim node
        claim_node = self.add_claim_node(claim_text)
        
        # Process sentences in context (context_sentences is a dict)
        for sent_idx, sentence_tokens in context_sentences.items():
            sentence_text = " ".join([token["wordForm"] for token in sentence_tokens])
            sentence_node = self.add_sentence_node(sent_idx, sentence_text)
            
            # Dictionary to map index -> word_node_id for creating dependency links
            token_index_to_node = {}
            
            # Add words in sentence
            for token in sentence_tokens:
                word = token["wordForm"]
                pos_tag = token.get("posTag", "")
                lemma = token.get("lemma", "")
                token_index = token.get("index", 0)
                
                word_node = self.add_word_node(word, pos_tag, lemma)
                
                # Only create connection if word_node was created successfully (not filtered)
                if word_node is not None:
                    self.connect_word_to_sentence(word_node, sentence_node)
                    # Save mapping to create dependency links later
                    token_index_to_node[token_index] = word_node
            
            # Create dependency connections between words in the sentence
            for token in sentence_tokens:
                token_index = token.get("index", 0)
                head_index = token.get("head", 0)
                dep_label = token.get("depLabel", "")
                
                # Only create dependency if both dependent and head exist in mapping
                if (head_index > 0 and 
                    token_index in token_index_to_node and 
                    head_index in token_index_to_node):
                    dependent_node = token_index_to_node[token_index]
                    head_node = token_index_to_node[head_index]
                    self.connect_dependency(dependent_node, head_node, dep_label)
        
        # Process words in claim (claim_sentences is also a dict)
        for sent_idx, sentence_tokens in claim_sentences.items():
            # Dictionary to map index -> word_node_id for claim
            claim_token_index_to_node = {}
            
            # Add words
            for token in sentence_tokens:
                word = token["wordForm"]
                pos_tag = token.get("posTag", "")
                lemma = token.get("lemma", "")
                token_index = token.get("index", 0)
                
                word_node = self.add_word_node(word, pos_tag, lemma)
                
                # Only create connection if word_node was created successfully (not filtered)
                if word_node is not None:
                    self.connect_word_to_claim(word_node, claim_node)
                    # Save mapping for dependency links
                    claim_token_index_to_node[token_index] = word_node
            
            # Create dependency connections in claim
            for token in sentence_tokens:
                token_index = token.get("index", 0)
                head_index = token.get("head", 0)
                dep_label = token.get("depLabel", "")
                
                # Only create dependency if both dependent and head exist in mapping
                if (head_index > 0 and 
                    token_index in claim_token_index_to_node and 
                    head_index in claim_token_index_to_node):
                    dependent_node = claim_token_index_to_node[token_index]
                    head_node = claim_token_index_to_node[head_index]
                    self.connect_dependency(dependent_node, head_node, dep_label)
    
    def get_statistics(self):
        """Basic statistics about the graph"""
        word_count = len([n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'word'])
        sentence_count = len([n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'sentence'])
        claim_count = len([n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'claim'])
        entity_count = len([n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'entity'])
        
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "word_nodes": word_count,
            "sentence_nodes": sentence_count,
            "claim_nodes": claim_count,
            "entity_nodes": entity_count
        }
    
    def get_shared_words(self):
        """Find words that appear in both context and claim"""
        shared_words = []
        
        for word_node_id in self.word_nodes.values():
            # Check if word node is connected to both sentence nodes and claim node
            neighbors = list(self.graph.neighbors(word_node_id))
            has_sentence_connection = any(
                self.graph.nodes[neighbor]['type'] == 'sentence' for neighbor in neighbors
            )
            has_claim_connection = any(
                self.graph.nodes[neighbor]['type'] == 'claim' for neighbor in neighbors
            )
            
            if has_sentence_connection and has_claim_connection:
                word_text = self.graph.nodes[word_node_id]['text']
                pos_tag = self.graph.nodes[word_node_id]['pos']
                shared_words.append({
                    'word': word_text,
                    'pos': pos_tag,
                    'node_id': word_node_id
                })
        
        return shared_words
    
    def get_word_frequency(self):
        """Count frequency of each word"""
        word_freq = {}
        for word_node_id in self.word_nodes.values():
            word_text = self.graph.nodes[word_node_id]['text']
            word_freq[word_text] = word_freq.get(word_text, 0) + 1
        return word_freq
    
    def get_dependency_statistics(self):
        """Statistics about dependency relationships"""
        dependency_edges = [
            (u, v, data) for u, v, data in self.graph.edges(data=True) 
            if data.get('edge_type') == 'dependency'
        ]
        
        # Count different types of dependencies
        dep_types = {}
        for u, v, data in dependency_edges:
            dep_label = data.get('relation', 'unknown')
            dep_types[dep_label] = dep_types.get(dep_label, 0) + 1
        
        return {
            "total_dependency_edges": len(dependency_edges),
            "dependency_types": dep_types,
            "most_common_dependencies": sorted(dep_types.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def get_word_dependencies(self, word):
        """Get all dependencies of a word"""
        if word not in self.word_nodes:
            return {"dependents": [], "heads": []}
        
        word_node_id = self.word_nodes[word]
        dependents = []
        heads = []
        
        for neighbor in self.graph.neighbors(word_node_id):
            edge_data = self.graph.edges[word_node_id, neighbor]
            if edge_data.get('edge_type') == 'dependency':
                dep_relation = edge_data.get('relation', '')
                neighbor_word = self.graph.nodes[neighbor]['text']
                
                # Check if word_node_id is head or dependent
                # In an undirected graph in NetworkX, we need to check direction based on semantic
                # Assuming edge is created from dependent -> head
                if (word_node_id, neighbor) in self.graph.edges():
                    heads.append({"word": neighbor_word, "relation": dep_relation})
                else:
                    dependents.append({"word": neighbor_word, "relation": dep_relation})
        
        return {"dependents": dependents, "heads": heads}
    
    def get_detailed_statistics(self):
        """Detailed statistics about the graph"""
        basic_stats = self.get_statistics()
        shared_words = self.get_shared_words()
        word_freq = self.get_word_frequency()
        dep_stats = self.get_dependency_statistics()
        semantic_stats = self.get_semantic_statistics()
        
        # Find the most frequent words
        most_frequent_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate total edges by type
        structural_edges = len([
            (u, v) for u, v, data in self.graph.edges(data=True) 
            if data.get('edge_type') == 'structural'
        ])
        
        entity_structural_edges = len([
            (u, v) for u, v, data in self.graph.edges(data=True) 
            if data.get('edge_type') == 'entity_structural'
        ])
        
        # Statistics about entities
        entity_list = [
            {
                'name': self.graph.nodes[node_id]['text'],
                'type': self.graph.nodes[node_id].get('entity_type', 'ENTITY'),
                'connected_sentences': len([
                    neighbor for neighbor in self.graph.neighbors(node_id) 
                    if self.graph.nodes[neighbor]['type'] == 'sentence'
                ])
            }
            for node_id in self.graph.nodes() 
            if self.graph.nodes[node_id]['type'] == 'entity'
        ]
        
        return {
            **basic_stats,
            "shared_words_count": len(shared_words),
            "shared_words": shared_words,
            "unique_words": len(word_freq),
            "most_frequent_words": most_frequent_words,
            "average_words_per_sentence": basic_stats['word_nodes'] / max(basic_stats['sentence_nodes'], 1),
            "dependency_statistics": dep_stats,
            "structural_edges": structural_edges,
            "dependency_edges": dep_stats["total_dependency_edges"],
            "entity_structural_edges": entity_structural_edges,
            "entities": entity_list,
            "unique_entities": len(entity_list),
            "semantic_statistics": semantic_stats,
            "semantic_edges": semantic_stats["total_semantic_edges"]
        }
    
    def visualize(self, figsize=(15, 10), show_dependencies=True, show_semantic=True):
        """Visualize the graph with separate colors for structural, dependency, entity, and semantic edges"""
        plt.figure(figsize=figsize)
        
        # Define colors for different node types
        node_colors = []
        node_sizes = []
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node]['type']
            if node_type == 'word':
                node_colors.append('lightblue')
                node_sizes.append(200)
            elif node_type == 'sentence':
                node_colors.append('lightgreen')
                node_sizes.append(500)
            elif node_type == 'claim':
                node_colors.append('lightcoral')
                node_sizes.append(600)
            elif node_type == 'entity':
                node_colors.append('gold')
                node_sizes.append(400)
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=2, iterations=100)
        
        # Divide edges by type
        structural_edges = []
        dependency_edges = []
        entity_edges = []
        semantic_edges = []
        
        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get('edge_type', 'structural')
            if edge_type == 'structural':
                structural_edges.append((u, v))
            elif edge_type == 'dependency':
                dependency_edges.append((u, v))
            elif edge_type == 'entity_structural':
                entity_edges.append((u, v))
            elif edge_type == 'semantic':
                semantic_edges.append((u, v))
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, 
                             node_color=node_colors,
                             node_size=node_sizes,
                             alpha=0.8)
        
        # Draw structural edges (word -> sentence/claim)
        if structural_edges:
            nx.draw_networkx_edges(self.graph, pos,
                                 edgelist=structural_edges,
                                 edge_color='gray',
                                 style='-',
                                 width=1,
                                 alpha=0.6)
        
        # Draw entity edges (entity -> sentence)
        if entity_edges:
            nx.draw_networkx_edges(self.graph, pos,
                                 edgelist=entity_edges,
                                 edge_color='orange',
                                 style='-',
                                 width=2,
                                 alpha=0.7)
        
        # Draw semantic edges (word -> word)
        if show_semantic and semantic_edges:
            nx.draw_networkx_edges(self.graph, pos,
                                 edgelist=semantic_edges,
                                 edge_color='purple',
                                 style=':',
                                 width=1.5,
                                 alpha=0.8)
        
        # Draw dependency edges (word -> word)
        if show_dependencies and dependency_edges:
            nx.draw_networkx_edges(self.graph, pos,
                                 edgelist=dependency_edges,
                                 edge_color='red',
                                 style='--',
                                 width=0.8,
                                 alpha=0.7,
                                 arrows=True,
                                 arrowsize=10)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='lightblue', label='Word nodes'),
            mpatches.Patch(color='lightgreen', label='Sentence nodes'),
            mpatches.Patch(color='lightcoral', label='Claim node'),
            mpatches.Patch(color='gold', label='Entity nodes')
        ]
        
        edge_legend = []
        if structural_edges:
            edge_legend.append(plt.Line2D([0], [0], color='gray', label='Structural edges'))
        if entity_edges:
            edge_legend.append(plt.Line2D([0], [0], color='orange', label='Entity edges'))
        if show_semantic and semantic_edges:
            edge_legend.append(plt.Line2D([0], [0], color='purple', linestyle=':', label='Semantic edges'))
        if show_dependencies and dependency_edges:
            edge_legend.append(plt.Line2D([0], [0], color='red', linestyle='--', label='Dependency edges'))
        
        legend_elements.extend(edge_legend)
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        title = f"Text Graph: Words, Sentences, Claim, Entities ({len(self.entity_nodes)} entities)"
        if show_semantic and semantic_edges:
            title += f", Semantic ({len(semantic_edges)} edges)"
        if show_dependencies and dependency_edges:
            title += f", Dependencies ({len(dependency_edges)} edges)"
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def visualize_dependencies_only(self, figsize=(12, 8)):
        """Visualize only dependency graph between words"""
        # Create subgraph with only word nodes and dependency edges
        word_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'word']
        dependency_edges = [
            (u, v) for u, v, data in self.graph.edges(data=True) 
            if data.get('edge_type') == 'dependency'
        ]
        
        if not dependency_edges:
            print("No dependency edges to plot!")
            return
        
        # Create subgraph
        subgraph = self.graph.edge_subgraph(dependency_edges).copy()
        
        plt.figure(figsize=figsize)
        
        # Layout for dependency graph
        pos = nx.spring_layout(subgraph, k=1.5, iterations=100)
        
        # Draw nodes with labels
        nx.draw_networkx_nodes(subgraph, pos, 
                             node_color='lightblue',
                             node_size=300,
                             alpha=0.8)
        
        # Draw edges with labels
        nx.draw_networkx_edges(subgraph, pos,
                             edge_color='red',
                             style='-',
                             width=1.5,
                             alpha=0.7,
                             arrows=True,
                             arrowsize=15)
        
        # Add node labels (words)
        node_labels = {node: self.graph.nodes[node]['text'][:10] 
                      for node in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, node_labels, font_size=8)
        
        # Add edge labels (dependency relations)
        edge_labels = {(u, v): data.get('relation', '') 
                      for u, v, data in subgraph.edges(data=True)}
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels, font_size=6)
        
        plt.title(f"Dependency Graph ({len(dependency_edges)} dependencies)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def save_graph(self, filepath):
        """Save graph to file"""
        # Ensure file is saved in the root directory of the project
        if not os.path.isabs(filepath):
            # Get parent directory of mint directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(project_root, filepath)
        
        # Create a copy of the graph to handle None values
        graph_copy = self.graph.copy()
        
        # Handle None values in node attributes
        for node_id in graph_copy.nodes():
            node_data = graph_copy.nodes[node_id]
            for key, value in node_data.items():
                if value is None:
                    graph_copy.nodes[node_id][key] = ""
        
        # Handle None values in edge attributes
        for u, v in graph_copy.edges():
            edge_data = graph_copy.edges[u, v]
            for key, value in edge_data.items():
                if value is None:
                    graph_copy.edges[u, v][key] = ""
        
        nx.write_gexf(graph_copy, filepath)
        print(f"Graph saved to: {filepath}")
    
    def load_graph(self, filepath):
        """Load graph from file"""
        self.graph = nx.read_gexf(filepath)
        
        # Rebuild node mappings
        self.word_nodes = {}
        self.sentence_nodes = {}
        self.entity_nodes = {}
        self.claim_node = None
        
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            if node_data['type'] == 'word':
                self.word_nodes[node_data['text']] = node_id
            elif node_data['type'] == 'sentence':
                # Extract sentence index from node_id
                sent_idx = int(node_id.split('_')[1])
                self.sentence_nodes[sent_idx] = node_id
            elif node_data['type'] == 'claim':
                self.claim_node = node_id
            elif node_data['type'] == 'entity':
                self.entity_nodes[node_data['text']] = node_id
        
        print(f"Graph loaded from: {filepath}")
    
    def export_to_json(self):
        """Export graph to JSON for easier analysis"""
        graph_data = {
            "nodes": [],
            "edges": [],
            "statistics": self.get_detailed_statistics()
        }
        
        # Export nodes
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            graph_data["nodes"].append({
                "id": node_id,
                "type": node_data["type"],
                "text": node_data["text"],
                "pos": node_data.get("pos", ""),
                "lemma": node_data.get("lemma", "")
            })
        
        # Export edges
        for edge in self.graph.edges():
            edge_data = self.graph.edges[edge]
            graph_data["edges"].append({
                "source": edge[0],
                "target": edge[1],
                "relation": edge_data.get("relation", ""),
                "edge_type": edge_data.get("edge_type", "")
            })
        
        return json.dumps(graph_data, ensure_ascii=False, indent=2)
    
    def _init_openai_client(self):
        """Initialize OpenAI client"""
        try:
            # Try multiple key names for backward compatibility
            api_key = os.getenv('OPENAI_KEY') or os.getenv('OPENAI_API_KEY')
            if api_key and api_key != 'your_openai_api_key_here':
                self.openai_client = OpenAI(api_key=api_key)
                # Only print once globally
                if not hasattr(TextGraph, '_openai_initialized'):
                    print("✅ OpenAI client initialized")
                    TextGraph._openai_initialized = True
            else:
                if not hasattr(self, '_openai_warning_shown'):
                    print("Warning: OPENAI_KEY or OPENAI_API_KEY not found in .env file.")
                    self._openai_warning_shown = True
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
    
    def add_entity_node(self, entity_name, entity_type="ENTITY"):
        """Add entity node to graph"""
        if entity_name not in self.entity_nodes:
            node_id = f"entity_{len(self.entity_nodes)}"
            self.entity_nodes[entity_name] = node_id
            self.graph.add_node(node_id, 
                              type="entity", 
                              text=entity_name,
                              entity_type=entity_type)
        return self.entity_nodes[entity_name]
    
    def connect_entity_to_sentence(self, entity_node, sentence_node):
        """Connect entity to sentence"""
        self.graph.add_edge(entity_node, sentence_node, relation="mentioned_in", edge_type="entity_structural")
    
    def _update_openai_model(self, model=None, temperature=None, max_tokens=None):
        """Update OpenAI model parameters"""
        if model:
            self.openai_model = model
        if temperature is not None:
            self.openai_temperature = temperature  
        if max_tokens is not None:
            self.openai_max_tokens = max_tokens
    
    def extract_entities_with_openai(self, context_text):
        """Extract entities from context using OpenAI GPT-4o-mini"""
        if not self.openai_client:
            print("OpenAI client not initialized. Unable to extract entities.")
            return []
        
        try:
            # Prompt to extract entities including date and quantity
            prompt = f"""
Bạn là một chuyên gia trích xuất thông tin cho hệ thống fact-checking. Hãy trích xuất tất cả các thực thể quan trọng từ văn bản sau, bao gồm CẢ NGÀY THÁNG và SỐ LƯỢNG QUAN TRỌNG.
Quan trọng, chỉ lấy những từ có trong văn bản, không lấy những từ không có trong văn bản. Nếu trích xuất được các từ thì phải để nó giống y như trong văn bản không được thay đổi.

NGUYÊN TẮC TRÍCH XUẤT:
- Lấy TÊN THỰC THỂ THUẦN TÚY + NGÀY THÁNG + SỐ LƯỢNG QUAN TRỌNG
- Loại bỏ từ phân loại không cần thiết: "con", "chiếc", "cái", "người" (trừ khi là phần của tên riêng)
- Giữ nguyên số đo lường có ý nghĩa thực tế
YÊU CẦU:
Chỉ lấy những từ/cụm từ xuất hiện trong văn bản, giữ nguyên chính tả, không tự thêm hoặc sửa đổi.
Với mỗi thực thể, chỉ lấy một lần (không lặp lại), kể cả xuất hiện nhiều lần trong văn bản.
Nếu thực thể là một phần của cụm danh từ lớn hơn (ví dụ: "đoàn cứu hộ Việt Nam"), hãy trích xuất cả cụm danh từ lớn ("đoàn cứu hộ Việt Nam") và thực thể nhỏ bên trong ("Việt Nam").
Không bỏ sót thực thể chỉ vì nó nằm trong cụm từ khác hoặc là một phần của tên dài.

Các loại thực thể CẦN trích xuất:
1. **Tên loài/sinh vật**: "Patagotitan mayorum", "titanosaur", "voi châu Phi"
2. **Địa danh**: "Argentina", "London", "Neuquen", "TP.HCM", "Quận 6"
3. **Địa danh kết hợp**: "Bảo tàng Lịch sử tự nhiên London", "Nhà máy nước Tân Hiệp"
4. **Tên riêng người**: "Nguyễn Văn A", "Phạm Văn Chính", "Sinead Marron"
5. **Tổ chức**: "Bảo tàng Lịch sử tự nhiên", "SAWACO", "Microsoft", "PLO"
6. **Sản phẩm/công nghệ**: "iPhone", "ChatGPT", "PhoBERT", "dịch vụ cấp nước"

7. **NGÀY THÁNG & THỜI GIAN QUAN TRỌNG**:
   - Năm: "2010", "2017", "2022"
   - Ngày tháng: "25-3", "15/4/2023", "ngày 10 tháng 5"
   - Giờ cụ thể: "22 giờ", "6h30", "14:30"
   - Khoảng thời gian: "từ 22 giờ đến 6 giờ", "2-3 ngày"

8. **SỐ LƯỢNG & ĐO LƯỜNG QUAN TRỌNG**:
   - Kích thước vật lý: "37m", "69 tấn", "6m", "180cm"
   - Số lượng có ý nghĩa: "6 con", "12 con", "100 người"  
   - Giá trị tiền tệ: "5 triệu đồng", "$100", "€50"
   - Tỷ lệ phần trăm: "80%", "15%"
   - Nhiệt độ: "25°C", "100 độ"

KHÔNG lấy (số lượng không có ý nghĩa):
- Số thứ tự đơn lẻ: "1", "2", "3" (trừ khi là năm hoặc địa chỉ)
- Từ chỉ số lượng mơ hồ: "nhiều", "ít", "vài", "một số"
- Đơn vị đo đơn lẻ: "mét", "tấn", "kg" (phải có số đi kèm)

Ví dụ INPUT: "6 con titanosaur ở Argentina nặng 69 tấn, được trưng bày tại Bảo tàng Lịch sử tự nhiên London từ năm 2017 lúc 14:30"
Ví dụ OUTPUT: ["titanosaur", "Argentina", "69 tấn", "Bảo tàng Lịch sử tự nhiên London", "2017", "14:30", "6 con"]

Ví dụ INPUT: "SAWACO thông báo cúp nước tại Quận 6 từ 22 giờ ngày 25-3 đến 6 giờ ngày 26-3"
Ví dụ OUTPUT: ["SAWACO", "Quận 6", "22 giờ", "25-3", "6 giờ", "26-3"]

Trả về JSON array: ["entity1", "entity2", "entity3"]

Văn bản:
{context_text}
"""

            # Use parameters from CLI if available
            model = getattr(self, 'openai_model', 'gpt-4o-mini')
            temperature = getattr(self, 'openai_temperature', 0.0)
            max_tokens = getattr(self, 'openai_max_tokens', 1000)

            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=max_tokens
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            
            # Strip markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove '```json'
            if response_text.startswith('```'):
                response_text = response_text[3:]   # Remove '```'
            if response_text.endswith('```'):
                response_text = response_text[:-3]  # Remove ending '```'
            response_text = response_text.strip()
            
            # Try to parse JSON
            try:
                entities = json.loads(response_text)
                if isinstance(entities, list):
                    # Filter out empty strings and duplicates
                    entities = list(set([entity.strip() for entity in entities if entity.strip()]))
                    print(f"Extracted {len(entities)} entities: {entities}")
                    return entities
                else:
                    print(f"Response is not a list: {response_text}")
                    return []
            except json.JSONDecodeError:
                print(f"Failed to parse JSON from OpenAI response: {response_text}")
                return []
                
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return []
    
    def normalize_text(self, text):
        if not text:
            return ""
        # Remove punctuation, convert to lower, remove Vietnamese diacritics
        text = text.lower()
        text = re.sub(r'[\W_]+', ' ', text)  # remove non-alphanumeric characters
        text = ''.join(c for c in unicodedata.normalize('NFD', text)
                      if unicodedata.category(c) != 'Mn')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def fuzzy_in(self, entity, claim_text, threshold=0.8):
        # Fuzzy match: entity appears close to claim_text
        if entity in claim_text:
            return True
        # If entity is a phrase, check each word
        for word in entity.split():
            if word in claim_text:
                return True
        # Fuzzy match for entire string
        ratio = SequenceMatcher(None, entity, claim_text).ratio()
        return ratio >= threshold

    def improved_entity_matching(self, entity, sentence_text, model=None):
        entity_lower = entity.lower()
        sentence_lower = sentence_text.lower()
        # Method 1: Direct matching
        if entity_lower in sentence_lower:
            return True
        # Method 2: Simple space->underscore replacement
        entity_simple_seg = entity.replace(" ", "_").lower()
        if entity_simple_seg in sentence_lower:
            return True
        # Method 3: VnCoreNLP segmentation
        if model and segment_entity_with_vncorenlp:
            try:
                entity_vncorenlp_seg = segment_entity_with_vncorenlp(entity, model).lower()
                if entity_vncorenlp_seg in sentence_lower:
                    return True
            except:
                pass
        # Method 4: Fuzzy matching for partial matches
        entity_words = entity.split()
        if len(entity_words) > 1:
            all_words_found = True
            for word in entity_words:
                word_variants = [
                    word.lower(),
                    word.replace(" ", "_").lower()
                ]
                word_found = any(variant in sentence_lower for variant in word_variants)
                if not word_found:
                    all_words_found = False
                    break
            if all_words_found:
                return True
        return False

    def add_entities_to_graph(self, entities, context_sentences, model=None):
        """Add entities to graph and connect them to sentences with improved matching. If entity appears in claim, connect to claim node."""
        entity_nodes_added = []
        total_connections = 0
        # Get claim text (if there is a claim node)
        claim_text = None
        if hasattr(self, 'claim_node') and self.claim_node and self.claim_node in self.graph.nodes:
            claim_text = self.graph.nodes[self.claim_node]['text']
            claim_text_norm = self.normalize_text(claim_text)
        else:
            claim_text_norm = None
        for entity in entities:
            # Add entity node
            entity_node = self.add_entity_node(entity)
            entity_nodes_added.append(entity_node)
            entity_connections = 0
            # Find sentences containing this entity
            for sent_idx, sentence_node in self.sentence_nodes.items():
                sentence_text = self.graph.nodes[sentence_node]['text']
                if self.improved_entity_matching(entity, sentence_text, model):
                    self.connect_entity_to_sentence(entity_node, sentence_node)
                    entity_connections += 1
                    total_connections += 1
            # Connect entity to claim if entity appears in claim (enhanced: fuzzy comparison)
            # Mark entities appearing in claim with higher weight
            is_claim_entity = False
            if claim_text_norm:
                entity_norm = self.normalize_text(entity)
                if self.fuzzy_in(entity_norm, claim_text_norm, threshold=0.8):
                    self.graph.add_edge(entity_node, self.claim_node, relation="mentioned_in", edge_type="entity_structural")
                    is_claim_entity = True
                    # Mark this entity as appearing in claim for scoring
                    self.graph.nodes[entity_node]['in_claim'] = True
                    self.graph.nodes[entity_node]['claim_importance'] = 2.0  # Higher weight
        # ✅ NEW: Directly connect sentences to claim by similarity
        self._connect_sentences_to_claim_by_similarity(claim_text)
        
        print(f"✅ Added {len(entity_nodes_added)} entity nodes to graph")
        return entity_nodes_added
    
    def _connect_sentences_to_claim_by_similarity(self, claim_text):
        """Directly connect sentences to claim by text similarity"""
        if not claim_text or not self.sentence_nodes:
            return
        
        claim_words = set(self.normalize_text(claim_text).split())
        connections_added = 0
        
        for sent_idx, sentence_node in self.sentence_nodes.items():
            sentence_text = self.graph.nodes[sentence_node]['text']
            sentence_words = set(self.normalize_text(sentence_text).split())
            
            # Calculate word overlap ratio
            overlap = len(claim_words.intersection(sentence_words))
            total_words = len(claim_words.union(sentence_words))
            similarity = overlap / total_words if total_words > 0 else 0.0
            
            # Connect to claim if similarity is high enough
            if similarity >= 0.15:  # Threshold 15%
                self.graph.add_edge(sentence_node, self.claim_node, 
                                  relation="text_similar", 
                                  edge_type="semantic",
                                  similarity=similarity)
                connections_added += 1
        
        print(f"🔗 Connected {connections_added} sentences to claim by text similarity (threshold=0.15)")
    
    def extract_and_add_entities(self, context_text, context_sentences):
        """Main method to extract and add entities to graph"""
        print("Extracting entities from OpenAI...")
        entities = self.extract_entities_with_openai(context_text)
        
        if entities:
            print("Adding entities to graph...")
            entity_nodes = self.add_entities_to_graph(entities, context_sentences)
            print(f"Done! Added {len(entity_nodes)} entities to graph.")
            return entity_nodes
        else:
            print("No entities extracted.")
            return []
    
    def _init_phobert_model(self):
        """Initialize PhoBERT model"""
        try:
            self.phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            self.phobert_model = AutoModel.from_pretrained("vinai/phobert-base")
            # Only print once globally
            if not hasattr(TextGraph, '_phobert_initialized'):
                print("✅ PhoBERT model initialized")
                TextGraph._phobert_initialized = True
        except Exception as e:
            print(f"Error initializing PhoBERT model: {e}")
    
    def get_word_embeddings(self, words):
        """Get embeddings of words"""
        if not self.phobert_tokenizer or not self.phobert_model:
            print("PhoBERT model not initialized. Unable to get embeddings.")
            return None
        
        embeddings = []
        for word in words:
            if word not in self.word_embeddings:
                inputs = self.phobert_tokenizer(word, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.phobert_model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
                self.word_embeddings[word] = embeddings[-1]
            else:
                embeddings.append(self.word_embeddings[word])
        
        return np.array(embeddings)
    
    def get_similarity(self, word1, word2):
        if not cosine_similarity:
            print("cosine_similarity not available.")
            return 0.0
        if word1 not in self.word_embeddings or word2 not in self.word_embeddings:
            print(f"Word '{word1}' or '{word2}' not found in word_embeddings.")
            return 0.0
        embedding1 = self.word_embeddings[word1]
        embedding2 = self.word_embeddings[word2]
        return cosine_similarity([embedding1], [embedding2])[0][0]
    
    def get_similar_words(self, word, top_k=5):
        """Find words with high similarity to the given word"""
        if word not in self.word_embeddings:
            return []
        
        similarities = []
        for other_word in self.word_embeddings.keys():
            if other_word != word:
                similarity = self.get_similarity(word, other_word)
                similarities.append((other_word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [word for word, similarity in similarities[:top_k]]
    
    def get_sentence_embeddings(self, sentences):
        """Get embeddings of sentences"""
        if not self.phobert_tokenizer or not self.phobert_model:
            print("PhoBERT model not initialized. Unable to get embeddings.")
            return None
        
        embeddings = []
        for sentence in sentences:
            inputs = self.phobert_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                outputs = self.phobert_model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        
        return np.array(embeddings)
    
    def get_sentence_similarity(self, sentence1, sentence2):
        """Calculate similarity between two sentences"""
        # Get embeddings for both sentences
        embeddings = self.get_sentence_embeddings([sentence1, sentence2])
        if embeddings is None or len(embeddings) < 2:
            return 0.0
        
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    def build_semantic_similarity_edges(self, use_faiss=True):
        """Build semantic similarity edges between words (without PCA)"""
        print("Starting to build semantic similarity edges...")
        
        # Get all word nodes
        word_nodes = [node_id for node_id in self.graph.nodes() 
                     if self.graph.nodes[node_id]['type'] == 'word']
        
        if len(word_nodes) < 2:
            print("At least 2 word nodes are needed to build semantic edges.")
            return
        
        # Get list of words and POS tags
        words = []
        pos_tags = []
        word_node_mapping = {}
        
        for node_id in word_nodes:
            word = self.graph.nodes[node_id]['text']
            pos = self.graph.nodes[node_id].get('pos', '')
            words.append(word)
            pos_tags.append(pos)
            word_node_mapping[word] = node_id
        
        print(f"Getting embeddings for {len(words)} words...")
        
        # Get embeddings (using full PhoBERT embeddings - no PCA)
        embeddings = self.get_word_embeddings(words)
        if embeddings is None:
            print("Unable to get embeddings.")
            return
        
        print(f"Got embeddings with shape: {embeddings.shape}")
        print("✅ Using full PhoBERT embeddings (768 dim) - NO PCA")
        
        # Build Faiss index (optional)
        if use_faiss:
            print("Building Faiss index with full embeddings...")
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product (for cosine similarity)
            
            # Normalize vectors for cosine similarity
            embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.faiss_index.add(embeddings_normalized.astype(np.float32))
            
            # Create mappings
            self.word_to_index = {word: i for i, word in enumerate(words)}
            self.index_to_word = {i: word for i, word in enumerate(words)}
            print("Faiss index built.")
        else:
            # Normalize embeddings for faster cosine similarity calculation
            embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Find similar words and create edges
        edges_added = 0
        print(f"Finding words similar to threshold={self.similarity_threshold}, top_k={self.top_k_similar}...")
        
        for i, word1 in enumerate(words):
            pos1 = pos_tags[i]
            node1 = word_node_mapping[word1]
            
            if use_faiss and self.faiss_index is not None:
                # Use Faiss to find similar words
                query_vector = embeddings_normalized[i:i+1].astype(np.float32)
                similarities, indices = self.faiss_index.search(query_vector, self.top_k_similar + 1)  # +1 because it includes itself
                
                for j, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                    if idx == i:  # Skip itself
                        continue
                    
                    if similarity < self.similarity_threshold:
                        continue
                    
                    word2 = self.index_to_word[idx]
                    pos2 = pos_tags[idx]
                    node2 = word_node_mapping[word2]
                    
                    # Only create connection between words of the same POS (optional)
                    if pos1 and pos2 and pos1 == pos2:
                        if not self.graph.has_edge(node1, node2):
                            self.graph.add_edge(node1, node2, 
                                              relation="semantic_similar", 
                                              edge_type="semantic",
                                              similarity=float(similarity))
                            edges_added += 1
            else:
                # Use numpy matrix multiplication (faster than sklearn for cosine similarity)
                for j, word2 in enumerate(words):
                    if i >= j:  # Avoid duplicate and self-comparison
                        continue
                    
                    pos2 = pos_tags[j]
                    
                    # Only compare words of the same POS
                    if pos1 and pos2 and pos1 != pos2:
                        continue
                    
                    # Calculate cosine similarity with normalized vectors (faster)
                    similarity = np.dot(embeddings_normalized[i], embeddings_normalized[j])
                    
                    if similarity >= self.similarity_threshold:
                        node2 = word_node_mapping[word2]
                        if not self.graph.has_edge(node1, node2):
                            self.graph.add_edge(node1, node2, 
                                              relation="semantic_similar", 
                                              edge_type="semantic",
                                              similarity=float(similarity))
                            edges_added += 1
        
        print(f"Added {edges_added} semantic similarity edges.")
        return edges_added
    
    def get_semantic_statistics(self):
        """Statistics about semantic edges"""
        semantic_edges = [
            (u, v, data) for u, v, data in self.graph.edges(data=True) 
            if data.get('edge_type') == 'semantic'
        ]
        
        if not semantic_edges:
            return {
                "total_semantic_edges": 0,
                "average_similarity": 0.0,
                "similarity_distribution": {}
            }
        
        similarities = [data.get('similarity', 0.0) for u, v, data in semantic_edges]
        
        return {
            "total_semantic_edges": len(semantic_edges),
            "average_similarity": np.mean(similarities),
            "max_similarity": np.max(similarities),
            "min_similarity": np.min(similarities),
            "similarity_distribution": {
                "0.85-0.90": len([s for s in similarities if 0.85 <= s < 0.90]),
                "0.90-0.95": len([s for s in similarities if 0.90 <= s < 0.95]),
                "0.95-1.00": len([s for s in similarities if 0.95 <= s <= 1.00])
            }
        }
    
    def beam_search_paths(self, beam_width=10, max_depth=6, max_paths=20):
        """
        Find paths from claim to sentence nodes using Beam Search
        
        Args:
            beam_width (int): Width of beam search
            max_depth (int): Maximum depth of path
            max_paths (int): Maximum number of paths to return
            
        Returns:
            List[Path]: List of best paths
        """
        if not self.claim_node:
            print("⚠️ No claim node to perform beam search")
            return []
            
        # Create BeamSearchPathFinder
        path_finder = BeamSearchPathFinder(
            text_graph=self,
            beam_width=beam_width,
            max_depth=max_depth
        )
        
        # Find paths
        paths = path_finder.find_best_paths(max_paths=max_paths)
        
        return paths
    
    def export_beam_search_results(self, paths, output_dir="output", file_prefix="beam_search"):
        """
        Export beam search results to files
        
        Args:
            paths: List of paths from beam search
            output_dir (str): Output directory
            file_prefix (str): Prefix for filenames
            
        Returns:
            tuple: (json_file_path, summary_file_path)
        """
        if not paths:
            print("⚠️ No paths to export")
            return None, None
            
        # Create BeamSearchPathFinder for export
        path_finder = BeamSearchPathFinder(self)
        
        # Export JSON and summary with absolute paths
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure we use the correct directory
        current_dir = os.getcwd()
        if current_dir.endswith('vncorenlp'):
            # If we're in vncorenlp directory, go back to parent
            current_dir = os.path.dirname(current_dir)
        
        json_file = os.path.join(current_dir, output_dir, f"{file_prefix}_{timestamp}.json")
        summary_file = os.path.join(current_dir, output_dir, f"{file_prefix}_summary_{timestamp}.txt")
        
        json_path = path_finder.export_paths_to_file(paths, json_file)
        summary_path = path_finder.export_paths_summary(paths, summary_file)
        
        return json_path, summary_path
    
    def analyze_paths_quality(self, paths):
        """
        Analyze quality of found paths
        
        Args:
            paths: List of paths
            
        Returns:
            dict: Statistics about paths
        """
        if not paths:
            return {
                'total_paths': 0,
                'avg_score': 0,
                'avg_length': 0,
                'paths_to_sentences': 0,
                'paths_through_entities': 0
            }
            
        total_paths = len(paths)
        scores = [p.score for p in paths]
        lengths = [len(p.nodes) for p in paths]
        
        sentences_reached = sum(1 for p in paths if any(
            node.startswith('sentence') for node in p.nodes
        ))
        
        entities_visited = sum(1 for p in paths if p.entities_visited)
        
        return {
            'total_paths': total_paths,
            'avg_score': sum(scores) / total_paths if scores else 0,
            'max_score': max(scores) if scores else 0,
            'min_score': min(scores) if scores else 0,
            'avg_length': sum(lengths) / total_paths if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'paths_to_sentences': sentences_reached,
            'paths_through_entities': entities_visited,
            'sentence_reach_rate': sentences_reached / total_paths if total_paths > 0 else 0,
            'entity_visit_rate': entities_visited / total_paths if total_paths > 0 else 0
        }
    
    def multi_level_beam_search_paths(
        self,
        max_levels: int = 3,
        beam_width_per_level: int = 3,
        max_depth: int = 30,
        allow_skip_edge: bool = False,        # �� turn on/off 2-hops
        min_new_sentences: int = 0,            # already have sentences from previous levels
        advanced_data_filter=None,
        claim_text="",
        entities=None,
        filter_top_k: int = 2
    ) -> Dict[int, List]:
        """
        Multi-level beam search wrapper for TextGraph
        
        Args:
            max_levels: Maximum number of levels
            beam_width_per_level: Number of sentences per level
            max_depth: Maximum depth for beam search
            
        Returns:
            Dict[level, List[Path]]: Results by level
        """
        if not self.claim_node:
            print("⚠️ No claim node to perform multi-level beam search")
            return {}
            
        # Create BeamSearchPathFinder with custom max_depth
        path_finder = BeamSearchPathFinder(
            text_graph=self,
            beam_width=25,
            max_depth=max_depth,
            allow_skip_edge=allow_skip_edge    # 🆕 change parameter
        )
        
        # Run multi-level search
        multi_results = path_finder.multi_level_beam_search(
            max_levels=max_levels,
            beam_width_per_level=beam_width_per_level,
            min_new_sentences=min_new_sentences,
            advanced_data_filter=advanced_data_filter,
            claim_text=claim_text,
            entities=entities,
            filter_top_k=filter_top_k
        )
        
        return multi_results 
        
    def multi_level_beam_search_paths_from_start_nodes(
        self,
        start_nodes: List[str],
        max_levels: int = 3,
        beam_width_per_level: int = 3,
        max_depth: int = 30,
        allow_skip_edge: bool = False,
        min_new_sentences: int = 0,
        advanced_data_filter=None,
        claim_text="",
        entities=None,
        filter_top_k: int = 2
    ) -> Dict[int, List]:
        """
        Multi-level beam search from specific start nodes (instead of claim node)
        
        Args:
            start_nodes: List of node IDs to start search from
            max_levels: Maximum number of levels
            beam_width_per_level: Number of sentences per level
            max_depth: Maximum depth for beam search
            
        Returns:
            Dict[level, List[Path]]: Results by level
        """
        if not start_nodes:
            print("⚠️ No start nodes to perform multi-level beam search")
            return {}
            
        # Create BeamSearchPathFinder with custom max_depth
        path_finder = BeamSearchPathFinder(
            text_graph=self,
            beam_width=25,
            max_depth=max_depth,
            allow_skip_edge=allow_skip_edge
        )
        
        # Run multi-level search from start nodes
        multi_results = path_finder.multi_level_beam_search_from_start_nodes(
            start_nodes=start_nodes,
            max_levels=max_levels,
            beam_width_per_level=beam_width_per_level,
            min_new_sentences=min_new_sentences,
            advanced_data_filter=advanced_data_filter,
            claim_text=claim_text,
            entities=entities,
            filter_top_k=filter_top_k
        )
        
        return multi_results 

    def extract_claim_keywords_with_openai(self, claim_text):
        """Extract important keywords from claim to create additional entities"""
        if not self.openai_client:
            print("OpenAI client not initialized. Unable to extract claim keywords.")
            return []
        
        try:
            prompt = f"""
Bạn là chuyên gia phân tích ngôn ngữ cho hệ thống fact-checking. Hãy trích xuất TẤT CẢ các từ khóa quan trọng từ câu claim dưới đây.

MÔ HÌNH TRÍCH XUẤT:
1. **CHỦ THỂ CHÍNH** (ai/cái gì): tên người, tổ chức, sản phẩm, loài vật, địa danh
2. **HÀNH ĐỘNG/ĐỘNG TỪ** quan trọng: sử dụng, phát triển, tạo ra, giải mã, hiểu, giao tiếp
3. **ĐỐI TƯỢNG/KHÁI NIỆM** quan trọng: công nghệ, khoa học, nghiên cứu, phương pháp
4. **TÍNH CHẤT/TRẠNG THÁI**: mới, hiện đại, tiên tiến, thành công

NGUYÊN TẮC TRÍCH XUẤT:
- Lấy CHÍNH XÁC từ/cụm từ có trong claim
- Lấy cả từ đơn lẻ VÀ cụm từ có ý nghĩa
- Tập trung vào từ khóa có thể fact-check được
- Không thêm từ không có trong claim

VÍ DỤ:
INPUT: "Tận dụng công nghệ mới để hiểu giao tiếp của động vật"
OUTPUT: ["tận dụng", "công nghệ", "công nghệ mới", "hiểu", "giao tiếp", "động vật", "giao tiếp của động vật"]

INPUT: "Thay vì cố gắng dạy chim nói tiếng Anh, các nhà nghiên cứu đang giải mã những gì chúng nói với nhau bằng tiếng chim"
OUTPUT: ["thay vì", "cố gắng", "dạy", "chim", "nói", "tiếng Anh", "nhà nghiên cứu", "giải mã", "tiếng chim", "giao tiếp", "dạy chim nói tiếng Anh", "nhà nghiên cứu giải mã", "chim nói"]

INPUT: "Nhà khoa học Việt Nam phát triển AI để dự báo thời tiết"
OUTPUT: ["nhà khoa học", "Việt Nam", "nhà khoa học Việt Nam", "phát triển", "AI", "dự báo", "thời tiết", "dự báo thời tiết"]

INPUT: "Apple sử dụng chip M1 mới trong MacBook Pro 2021"
OUTPUT: ["Apple", "sử dụng", "chip", "M1", "chip M1", "mới", "MacBook Pro", "2021", "MacBook Pro 2021"]

Trả về JSON array với tất cả keywords quan trọng: ["keyword1", "keyword2", ...]

CLAIM: {claim_text}
"""

            model = getattr(self, 'openai_model', 'gpt-4o-mini')
            temperature = getattr(self, 'openai_temperature', 0.0)
            max_tokens = getattr(self, 'openai_max_tokens', 500)

            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=max_tokens
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Strip markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            try:
                keywords = json.loads(response_text)
                if isinstance(keywords, list):
                    keywords = list(set([kw.strip() for kw in keywords if kw.strip()]))
                    return keywords
                else:
                    return []
            except json.JSONDecodeError:
                return []
                
        except Exception as e:
            return []

    def extract_enhanced_entities_with_openai(self, context_text, claim_text):
        """Enhanced entity extraction: 2 separate prompting approaches then combine"""
        
        print(f"🎯 DEBUG CLAIM TEXT: {claim_text}")
        
        # 🔧 PREPROCESS: Clean up VnCoreNLP format (remove underscores)
        if context_text:
            context_clean = context_text.replace("_", " ").strip()
        else:
            context_clean = ""
            
        claim_clean = claim_text.replace("_", " ").strip() if claim_text else ""
        print(f"🎯 DEBUG CLAIM CLEAN: {claim_clean}")
        
        # 🎯 PROMPTING 1: Extract entities from context + claim (original approach)
        context_claim_entities = []
        if context_clean and len(context_clean.strip()) > 10:
            try:
                # Use improved context entity extraction 
                context_entities = self.extract_context_entities_improved(context_clean)
                # Combine with claim entities extracted separately
                context_claim_entities = context_entities
                # Debug context entities extracted
            except Exception as e:
                print(f"⚠️ Context entity extraction failed: {e}")
                pass
        
        # 🎯 PROMPTING 2: Extract detailed keywords from claim only
        claim_keywords = []
        if claim_clean:
            try:
                claim_keywords = self.extract_claim_keywords_with_openai(claim_clean)
                # Debug claim keywords extracted
            except Exception as e:
                pass
        
        # 🔗 Step 3: Combine two separate arrays then deduplicate
        # Combine and deduplicate
        all_entities = list(set(context_claim_entities + claim_keywords))
        
        # ✅ NEW: Store claim entities for scoring
        self.claim_entities = set(claim_keywords)  # Store claim keywords as claim entities
        # Claim entities saved for scoring boost
        
        # 🆕 Store entities globally for multi-hop reuse
        if not hasattr(self, 'global_entities'):
            self.global_entities = []
        
        # Add new entities to global pool
        new_entities = [e for e in all_entities if e not in self.global_entities]
        self.global_entities.extend(new_entities)
        
        return all_entities

    def extract_context_entities_improved(self, context_text):
        """Extract entities from context with improved prompt and more detail"""
        if not self.openai_client:
            return []
        
        try:
            prompt = f"""
Hãy trích xuất TẤT CẢ thực thể quan trọng từ văn bản tiếng Việt sau đây.

QUY TẮC TRÍCH XUẤT:
1. Chỉ lấy từ/cụm từ CÓ TRONG văn bản
2. Giữ nguyên chính tả như trong văn bản
3. Lấy cả từ đơn lẻ VÀ cụm từ có ý nghĩa

LOẠI THỰC THỂ CẦN LẤY:
✅ Tên người: "Nguyễn Văn A", "John Smith", "Einstein"
✅ Tên tổ chức: "SAWACO", "Microsoft", "Đại học Stanford", "NASA"
✅ Địa danh: "TP.HCM", "Việt Nam", "London", "Quận 1"
✅ Sản phẩm/Công nghệ: "iPhone", "AI", "machine learning", "ChatGPT"
✅ Ngày tháng/Số: "25-3", "2023", "85%", "15 triệu đồng"
✅ Khái niệm khoa học: "nghiên cứu", "phát triển", "công nghệ", "khoa học"
✅ Động vật/Sinh vật: "voi", "chim", "voi châu Phi", "động vật"
✅ Tạp chí/Ấn phẩm: "Nature", "Science", "tạp chí"

VÍ DỤ:
INPUT: "Các nhà khoa học tại Đại học Stanford đã phát triển AI để nghiên cứu voi châu Phi"
OUTPUT: ["nhà khoa học", "Đại học Stanford", "phát triển", "AI", "nghiên cứu", "voi châu Phi", "voi", "châu Phi"]

QUAN TRỌNG: Trả về JSON array, không giải thích thêm.

Văn bản:
{context_text}
"""

            response = self.openai_client.chat.completions.create(
                model=getattr(self, 'openai_model', 'gpt-4o-mini'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000  # Increase token limit
            )
            
            response_text = response.choices[0].message.content.strip()
            print(f"🔍 OpenAI raw response: {response_text[:200]}...")
            
            # Parse JSON
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            entities = json.loads(response_text)
            if isinstance(entities, list):
                entities = [e.strip() for e in entities if e.strip()]
                print(f"📄 Improved context extraction: {len(entities)} entities")
                return entities
            else:
                print(f"❌ Response not a list: {response_text}")
                return []
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error: {e}")
            print(f"❌ Raw response: {response_text}")
            return []
        except Exception as e:
            print(f"❌ Improved context extraction error: {e}")
            return []

    def extract_context_entities_simple(self, context_text):
        """Extract entities from context with simpler prompt"""
        if not self.openai_client:
            return []
        
        try:
            prompt = f"""
Trích xuất tất cả thực thể quan trọng từ văn bản sau. Chỉ lấy những từ/cụm từ có trong văn bản.

LOẠI THỰC THỂ CẦN LẤY:
- Tên người: "Nguyễn Văn A", "John Smith"
- Tên tổ chức/công ty: "SAWACO", "Microsoft", "Đại học Bách Khoa"
- Địa danh: "TP.HCM", "Việt Nam", "Quận 1"
- Sản phẩm/công nghệ: "iPhone", "AI", "ChatGPT"
- Ngày tháng: "25-3", "2023", "tháng 6"
- Số lượng có ý nghĩa: "15 triệu đồng", "69 tấn", "100 người"
- Khái niệm quan trọng: "nghiên cứu", "khoa học", "phát triển"

Trả về JSON array: ["entity1", "entity2", ...]

Văn bản:
{context_text}
"""

            response = self.openai_client.chat.completions.create(
                model=getattr(self, 'openai_model', 'gpt-4o-mini'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=800
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            entities = json.loads(response_text)
            if isinstance(entities, list):
                entities = [e.strip() for e in entities if e.strip()]
                print(f"📄 Simple context extraction: {len(entities)} entities")
                return entities
            return []
            
        except Exception as e:
            print(f"❌ Simple context extraction error: {e}")
            return []

    def get_global_entities(self):
        """Get list of entities collected through multiple extraction attempts"""
        return getattr(self, 'global_entities', [])

    def add_to_global_entities(self, new_entities):
        """Add new entities to global pool"""
        if not hasattr(self, 'global_entities'):
            self.global_entities = []
        
        added = 0
        for entity in new_entities:
            if entity not in self.global_entities:
                self.global_entities.append(entity)
                added += 1
        
        print(f"🌍 Added {added} new entities to global pool (total: {len(self.global_entities)})")
        return added

    def get_claim_entities(self):
        """Get list of claim entities for boosting scoring"""
        return getattr(self, 'claim_entities', set())
    
    def get_sentences_connected_to_claim_entities(self):
        """Get all sentences directly connected to claim entities"""
        if not hasattr(self, 'claim_entities') or not self.claim_entities:
            return []
        
        connected_sentences = set()
        
        # Iterate through all nodes in the graph to find entity nodes with text matching claim entities
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('type') == 'entity':
                entity_text = node_data.get('text', '')
                
                # Check if entity text is in claim entities
                if entity_text in self.claim_entities:
                    # Get all neighbors of entity node
                    for neighbor in self.graph.neighbors(node_id):
                        # If neighbor is a sentence node
                        if neighbor.startswith('sentence_'):
                            sentence_text = self.graph.nodes[neighbor]['text']
                            connected_sentences.add((neighbor, sentence_text))
        
        # Convert to list and sort by sentence index
        result = list(connected_sentences)
        result.sort(key=lambda x: int(x[0].split('_')[1]))  # Sort by sentence index
        
        print(f"🎯 Found {len(result)} sentences directly connected to claim entities")
        return result
    
    def get_sentences_connected_to_claim_by_similarity(self):
        """Get sentences directly connected to claim by text similarity"""
        if not self.claim_node:
            return []
        
        connected_sentences = []
        
        # Get all neighbors of claim node
        for neighbor in self.graph.neighbors(self.claim_node):
            if neighbor.startswith('sentence_'):
                # Check if it's a text similarity connection
                edge_data = self.graph.get_edge_data(neighbor, self.claim_node)
                if edge_data and edge_data.get('relation') == 'text_similar':
                    sentence_text = self.graph.nodes[neighbor]['text']
                    similarity = edge_data.get('similarity', 0.0)
                    connected_sentences.append((neighbor, sentence_text, similarity))
        
        # Sort by similarity score in descending order
        connected_sentences.sort(key=lambda x: x[2], reverse=True)
        
        print(f"🔗 Found {len(connected_sentences)} sentences connected to claim by similarity")
        return connected_sentences
    
    def get_high_confidence_evidence_sentences(self):
        """Get sentences with high confidence: connected to claim entities + similarity to claim"""
        entity_sentences = self.get_sentences_connected_to_claim_entities()
        similarity_sentences = self.get_sentences_connected_to_claim_by_similarity()
        
        # Combine and remove duplicates
        all_sentences = {}
        
        # Add entity-connected sentences with high priority
        for sent_id, sent_text in entity_sentences:
            all_sentences[sent_id] = {
                'text': sent_text,
                'connected_to_entities': True,
                'similarity_score': 0.0,
                'confidence': 'high'  # Entity connection = high confidence
            }
        
        # Add similarity-connected sentences
        for sent_id, sent_text, similarity in similarity_sentences:
            if sent_id not in all_sentences:
                all_sentences[sent_id] = {
                    'text': sent_text,
                    'connected_to_entities': False,
                    'similarity_score': similarity,
                    'confidence': 'medium' if similarity >= 0.25 else 'low'
                }
            else:
                # Update existing with similarity score
                all_sentences[sent_id]['similarity_score'] = similarity
                all_sentences[sent_id]['confidence'] = 'very_high'  # Both entity + similarity
        
        # Convert to sorted list
        result = []
        for sent_id, data in all_sentences.items():
            result.append({
                'sentence_id': sent_id,
                'text': data['text'],
                'connected_to_entities': data['connected_to_entities'],
                'similarity_score': data['similarity_score'],
                'confidence': data['confidence']
            })
        
        # Sort by confidence level then similarity
        confidence_order = {'very_high': 4, 'high': 3, 'medium': 2, 'low': 1}
        result.sort(key=lambda x: (confidence_order[x['confidence']], x['similarity_score']), reverse=True)
        
        print(f"✨ Found {len(result)} high-confidence evidence sentences")
        return result 