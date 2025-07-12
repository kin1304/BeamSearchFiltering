"""
Pipeline module for beam graph filter pipeline
"""

from .beam_filter_pipeline import BeamFilterPipeline, extract_sentences_from_paths

__all__ = [
    'BeamFilterPipeline',
    'extract_sentences_from_paths'
]
