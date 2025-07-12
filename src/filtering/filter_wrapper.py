"""
Filter wrapper for beam search pipeline
"""

from typing import Dict, List
import contextlib
import io

# Import existing AdvancedDataFilter
try:
    from mint.filtering.advanced_data_filtering import AdvancedDataFilter
except ImportError:
    AdvancedDataFilter = None
    print("Warning: AdvancedDataFilter not available")


class FilterWrapper:
    """
    Wrapper for AdvancedDataFilter to use in pipeline
    """
    
    def __init__(self, use_sbert: bool = False, use_contradiction_detection: bool = False, use_nli: bool = False):
        """
        Initialize filter wrapper
        
        Args:
            use_sbert (bool): Enable SBERT semantic filtering
            use_contradiction_detection (bool): Enable contradiction detection
            use_nli (bool): Enable NLI stance detection
        """
        if AdvancedDataFilter is None:
            print("Warning: AdvancedDataFilter not available")
            self.filter_sys = None
        else:
            self.filter_sys = AdvancedDataFilter(
                use_sbert=use_sbert,
                use_contradiction_detection=use_contradiction_detection,
                use_nli=use_nli
            )
    
    def filter_sentences(self, sentences: List[Dict], claim_text: str, 
                        min_relevance_score: float = 0.15, 
                        max_final_sentences: int = 30) -> Dict:
        """
        Filter sentences using AdvancedDataFilter
        
        Args:
            sentences (List[Dict]): List of sentences to filter
            claim_text (str): Claim text for comparison
            min_relevance_score (float): Minimum relevance score
            max_final_sentences (int): Maximum number of final sentences
            
        Returns:
            Dict: Filtering results
        """
        if self.filter_sys is None:
            print("Warning: No filter system available, returning original sentences")
            return {
                "filtered_sentences": sentences[:max_final_sentences],
                "input_count": len(sentences),
                "output_count": min(len(sentences), max_final_sentences)
            }
        
        # Suppress stdout during filtering
        silent_buf = io.StringIO()
        with contextlib.redirect_stdout(silent_buf):
            results = self.filter_sys.multi_stage_filtering_pipeline(
                sentences=sentences,
                claim_text=claim_text,
                min_relevance_score=min_relevance_score,
                max_final_sentences=max_final_sentences
            )
        
        return results
    
    def is_available(self) -> bool:
        """Check if filter system is available"""
        return self.filter_sys is not None 