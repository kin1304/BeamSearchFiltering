"""
Simplified TextGraph class for beam search pipeline
"""

import networkx as nx
from typing import Dict, List, Set, Optional


class TextGraphSimple:
    """
    Simplified TextGraph class focused on beam search pipeline needs
    
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
        
        # POS tag filtering configuration
        self.enable_pos_filtering = True
        self.important_pos_tags = {
            'N',    # Common nouns
            'Np',   # Proper nouns
            'V',    # Verbs
            'A',    # Adjectives
            'Nc',   # Person nouns
            'M',    # Numbers
            'R',    # Adverbs
            'P'     # Pronouns
        }
    
    def set_pos_filtering(self, enable: bool = True, custom_pos_tags: Optional[Set[str]] = None):
        """Configure POS tag filtering for word nodes"""
        self.enable_pos_filtering = enable
        if custom_pos_tags is not None:
            self.important_pos_tags = set(custom_pos_tags)
    
    def is_important_word(self, word: str, pos_tag: str) -> bool:
        """Check if a word is important based on its POS tag"""
        if not self.enable_pos_filtering:
            return True
        return pos_tag in self.important_pos_tags
    
    def add_word_node(self, word: str, pos_tag: str = "", lemma: str = "") -> Optional[str]:
        """Add word node to graph (can filter by POS tag)"""
        if not self.is_important_word(word, pos_tag):
            return None
            
        if word not in self.word_nodes:
            node_id = f"word_{len(self.word_nodes)}"
            self.word_nodes[word] = node_id
            self.graph.add_node(node_id, 
                              type="word", 
                              text=word, 
                              pos=pos_tag, 
                              lemma=lemma)
        return self.word_nodes[word]
    
    def add_sentence_node(self, sentence_id: int, sentence_text: str) -> str:
        """Add sentence node to graph"""
        node_id = f"sentence_{sentence_id}"
        self.sentence_nodes[sentence_id] = node_id
        self.graph.add_node(node_id, 
                          type="sentence", 
                          text=sentence_text)
        return node_id
    
    def add_claim_node(self, claim_text: str) -> str:
        """Add claim node to graph"""
        self.claim_node = "claim_0"
        self.graph.add_node(self.claim_node, 
                          type="claim", 
                          text=claim_text)
        return self.claim_node
    
    def connect_word_to_sentence(self, word_node: str, sentence_node: str):
        """Connect word to sentence"""
        self.graph.add_edge(word_node, sentence_node, relation="belongs_to", edge_type="structural")
    
    def connect_word_to_claim(self, word_node: str, claim_node: str):
        """Connect word to claim"""
        self.graph.add_edge(word_node, claim_node, relation="belongs_to", edge_type="structural")
    
    def connect_dependency(self, dependent_word_node: str, head_word_node: str, dep_label: str):
        """Connect dependency between two words"""
        self.graph.add_edge(dependent_word_node, head_word_node, 
                          relation=dep_label, edge_type="dependency")
    
    def build_from_vncorenlp_output(self, context_sentences: Dict, claim_text: str, claim_sentences: Dict):
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
    
    def beam_search_paths(self, beam_width: int = 10, max_depth: int = 6, max_paths: int = 20) -> List:
        """
        Find paths from claim to sentence nodes using Beam Search
        
        Args:
            beam_width (int): Width of beam search
            max_depth (int): Maximum depth of path
            max_paths (int): Maximum number of paths to return
            
        Returns:
            List: List of best paths
        """
        if not self.claim_node:
            print("⚠️ No claim node to perform beam search")
            return []
            
        # Import here to avoid circular imports
        from mint.beam_search import BeamSearchPathFinder
        
        # Create BeamSearchPathFinder
        path_finder = BeamSearchPathFinder(
            text_graph=self,
            beam_width=beam_width,
            max_depth=max_depth
        )
        
        # Find paths
        paths = path_finder.find_best_paths(max_paths=max_paths)
        
        return paths 