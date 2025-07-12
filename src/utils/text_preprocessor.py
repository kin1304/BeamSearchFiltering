"""
Text preprocessing utilities for beam graph filter pipeline
"""

import re
from typing import List


def clean_text(text: str) -> str:
    """
    Remove line breaks, normalize whitespaces and trim spaces before punctuation.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    # Remove spaces before punctuation marks (, . ; : ! ? )
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    # Remove spaces after "("
    text = re.sub(r"\(\s+", "(", text)
    # Remove spaces before ")"
    text = re.sub(r"\s+\)", ")", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter using regex: break after . ! ? followed by whitespace.
    
    Args:
        text (str): Text to split into sentences
        
    Returns:
        List[str]: List of sentences
    """
    if not text:
        return []
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()] 