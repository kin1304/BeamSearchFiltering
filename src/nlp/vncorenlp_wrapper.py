"""
VnCoreNLP wrapper for text annotation
"""

import os
import sys
from typing import Dict, List

# Add VnCoreNLP directory to path
VNCORENLP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "vncorenlp")
sys.path.append(VNCORENLP_DIR)

try:
    import py_vncorenlp  # type: ignore
except ImportError:
    py_vncorenlp = None
    print("Warning: py_vncorenlp not available")


class VnCoreNLPWrapper:
    """
    Wrapper for VnCoreNLP text annotation
    """
    
    def __init__(self, vncorenlp_dir: str = ""):
        """
        Initialize VnCoreNLP wrapper
        
        Args:
            vncorenlp_dir (str): Path to VnCoreNLP directory
        """
        self.vncorenlp_dir = vncorenlp_dir if vncorenlp_dir else VNCORENLP_DIR
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """Initialize VnCoreNLP model"""
        if py_vncorenlp is None:
            print("Warning: VnCoreNLP not available")
            return
            
        try:
            self.model = py_vncorenlp.VnCoreNLP(
                annotators=["wseg", "pos", "ner", "parse"], 
                save_dir=self.vncorenlp_dir
            )
            print("✅ VnCoreNLP model loaded successfully")
        except Exception as e:
            print(f"Error loading VnCoreNLP model: {e}")
            self.model = None
    
    def annotate_text(self, text: str) -> Dict:
        """
        Annotate text with VnCoreNLP
        
        Args:
            text (str): Text to annotate
            
        Returns:
            Dict: Annotated tokens
        """
        if self.model is None:
            print("Warning: VnCoreNLP model not available")
            return {}
            
        try:
            return self.model.annotate_text(text)
        except Exception as e:
            print(f"Error annotating text: {e}")
            return {}
    
    def is_available(self) -> bool:
        """Check if VnCoreNLP is available"""
        return self.model is not None 