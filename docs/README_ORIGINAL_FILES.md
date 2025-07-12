# 📚 Original Files Documentation

## Overview
This document describes the original files that were refactored into the new modular structure.

## Original Files

### `beam_graph_filter_pipeline.py`

**Original Purpose**: Main pipeline script that combined all functionality

**Key Features**:
- Text preprocessing with VnCoreNLP
- Graph building with TextGraph
- Beam search path finding
- Sentence filtering with AdvancedDataFilter
- CLI interface with argparse

**Original Structure**:
```python
# Main pipeline class
class BeamGraphFilterPipeline:
    def __init__(self, config=None)
    def run_pipeline(self, context, claim)
    def _preprocess_text(self, context, claim)
    def _build_graph(self, context_sentences, claim)
    def _run_beam_search(self, graph)
    def _filter_sentences(self, candidate_sentences, claim)

# CLI interface
def main():
    parser = argparse.ArgumentParser()
    # ... argument parsing
    pipeline = BeamGraphFilterPipeline(config)
    results = pipeline.run_pipeline(context, claim)
```

**Refactored Into**:
- `src/pipeline/beam_filter_pipeline.py`: Main pipeline class
- `src/pipeline/cli.py`: CLI interface
- `src/utils/text_preprocessor.py`: Text preprocessing
- `src/nlp/vncorenlp_wrapper.py`: NLP wrapper
- `src/graph/text_graph_simple.py`: Simplified graph
- `src/filtering/filter_wrapper.py`: Filter wrapper

### `mint/text_graph.py`

**Original Purpose**: Complex TextGraph class with multiple features

**Key Features**:
- NetworkX-based graph structure
- POS tag filtering
- Entity extraction with OpenAI
- Semantic similarity with PhoBERT
- Beam search integration
- Visualization capabilities
- Graph export/import

**Original Methods**:
```python
class TextGraph:
    def __init__(self)
    def set_pos_filtering(self, enable, custom_pos_tags)
    def add_word_node(self, word, pos_tag, lemma)
    def add_sentence_node(self, sentence_id, sentence_text)
    def add_claim_node(self, claim_text)
    def build_from_vncorenlp_output(self, context_sentences, claim_text, claim_sentences)
    def beam_search_paths(self, beam_width, max_depth, max_paths)
    def extract_entities_with_openai(self, context_text)
    def build_semantic_similarity_edges(self, use_faiss)
    def visualize(self, figsize, show_dependencies, show_semantic)
    # ... many more methods
```

**Refactored Into**:
- `src/graph/text_graph_simple.py`: Simplified version focused on pipeline needs
- Original file preserved for advanced features

**Preserved Features**:
- Entity extraction with OpenAI
- Semantic similarity with PhoBERT
- Visualization capabilities
- Graph export/import
- Multi-level beam search
- Enhanced entity matching

### `advanced_data_filtering.py`

**Original Purpose**: Advanced sentence filtering system

**Key Features**:
- SBERT semantic filtering
- Contradiction detection
- NLI stance detection
- Multi-stage filtering pipeline
- Configurable filtering options

**Original Structure**:
```python
class AdvancedDataFilter:
    def __init__(self, use_sbert=False, use_contradiction_detection=False, use_nli=False)
    def multi_stage_filtering_pipeline(self, sentences, claim_text, min_relevance_score=0.15, max_final_sentences=30)
    def filter_by_semantic_similarity(self, sentences, claim_text, min_relevance_score=0.15)
    def filter_by_contradiction_detection(self, sentences, claim_text)
    def filter_by_nli_stance_detection(self, sentences, claim_text)
```

**Refactored Into**:
- `src/filtering/filter_wrapper.py`: Wrapper for existing system
- Original file preserved for direct use

### `mint/beam_search.py`

**Original Purpose**: Beam search algorithm for path finding

**Key Features**:
- Beam search path finding
- Multi-level beam search
- Path scoring and ranking
- Export capabilities

**Original Structure**:
```python
class BeamSearchPathFinder:
    def __init__(self, text_graph, beam_width=10, max_depth=6)
    def find_best_paths(self, max_paths=20)
    def multi_level_beam_search(self, max_levels=3, beam_width_per_level=3)
    def export_paths_to_file(self, paths, filepath)
    def export_paths_summary(self, paths, filepath)
```

**Status**: Preserved as-is, used by refactored modules

### `mint/helpers.py`

**Original Purpose**: Helper functions for text processing

**Key Features**:
- VnCoreNLP integration
- Text segmentation
- Entity segmentation

**Original Functions**:
```python
def segment_entity_with_vncorenlp(entity, model)
def process_text_with_vncorenlp(text, model)
```

**Status**: Preserved as-is, used by refactored modules

## Refactoring Benefits

### 1. **Modularity**
- **Before**: All functionality in single large files
- **After**: Separated into focused modules

### 2. **Maintainability**
- **Before**: Hard to modify specific features
- **After**: Easy to update individual components

### 3. **Testability**
- **Before**: Difficult to test individual components
- **After**: Each module can be tested independently

### 4. **Reusability**
- **Before**: Tightly coupled components
- **After**: Loose coupling, reusable modules

### 5. **Extensibility**
- **Before**: Adding features required modifying large files
- **After**: New features can be added as separate modules

## Migration Guide

### From Original to Refactored

**Original Usage**:
```python
from beam_graph_filter_pipeline import BeamGraphFilterPipeline

pipeline = BeamGraphFilterPipeline()
results = pipeline.run_pipeline(context, claim)
```

**Refactored Usage**:
```python
from src.pipeline.beam_filter_pipeline import BeamFilterPipeline

pipeline = BeamFilterPipeline()
results = pipeline.run_pipeline(context, claim)
```

### CLI Usage

**Original**:
```bash
python beam_graph_filter_pipeline.py context.txt claim.txt output.json
```

**Refactored**:
```bash
python src/pipeline/cli.py context.txt claim.txt output.json
```

### Advanced Features

**For advanced TextGraph features**:
```python
# Use original TextGraph for advanced features
from mint.text_graph import TextGraph

graph = TextGraph()
# ... use advanced features like entity extraction, visualization
```

**For direct AdvancedDataFilter usage**:
```python
# Use original AdvancedDataFilter directly
from advanced_data_filtering import AdvancedDataFilter

filter_sys = AdvancedDataFilter(use_sbert=True)
results = filter_sys.multi_stage_filtering_pipeline(sentences, claim_text)
```

## Backward Compatibility

### Preserved Files
- `mint/text_graph.py`: Full original functionality
- `mint/beam_search.py`: Beam search algorithm
- `mint/helpers.py`: Helper functions
- `advanced_data_filtering.py`: Advanced filtering system

### Compatible Interfaces
- CLI arguments remain the same
- Output format remains the same
- Configuration options remain the same

### Migration Path
1. **Immediate**: Use refactored modules for better organization
2. **Gradual**: Migrate to new modular structure
3. **Advanced**: Use original files for advanced features

## File Dependencies

### Original Dependencies
```
beam_graph_filter_pipeline.py
├── mint/text_graph.py
├── mint/beam_search.py
├── mint/helpers.py
├── advanced_data_filtering.py
└── vncorenlp/ (directory)
```

### Refactored Dependencies
```
src/
├── utils/text_preprocessor.py
├── nlp/vncorenlp_wrapper.py
├── graph/text_graph_simple.py
├── filtering/filter_wrapper.py
└── pipeline/
    ├── beam_filter_pipeline.py
    └── cli.py

mint/ (preserved)
├── text_graph.py
├── beam_search.py
└── helpers.py

advanced_data_filtering.py (preserved)
```

## Performance Comparison

### Original Structure
- **Pros**: Single file, simple deployment
- **Cons**: Hard to maintain, test, and extend

### Refactored Structure
- **Pros**: Modular, maintainable, testable, extensible
- **Cons**: More files, slightly more complex setup

### Performance Impact
- **Runtime**: No significant difference
- **Memory**: Slightly better due to focused modules
- **Development**: Much better due to modularity

## Best Practices

### When to Use Refactored Modules
- **New development**: Use refactored modules
- **Simple use cases**: Use refactored modules
- **Team development**: Use refactored modules
- **Testing**: Use refactored modules

### When to Use Original Files
- **Advanced features**: Use original TextGraph
- **Custom implementations**: Use original files
- **Research purposes**: Use original files
- **Backward compatibility**: Use original files

## Troubleshooting

### Common Migration Issues

**1. Import errors**:
```python
# Old
from beam_graph_filter_pipeline import BeamGraphFilterPipeline

# New
from src.pipeline.beam_filter_pipeline import BeamFilterPipeline
```

**2. Missing dependencies**:
```bash
# Check if all modules are available
ls -la src/
ls -la mint/
ls -la advanced_data_filtering.py
```

**3. Configuration differences**:
```python
# Old configuration format
config = {"beam_width": 40, "max_depth": 120}

# New configuration format (same, but in different module)
config = {"beam_width": 40, "max_depth": 120}
```

### Fallback Options

**If refactored modules fail**:
```python
# Fallback to original
from beam_graph_filter_pipeline import BeamGraphFilterPipeline
pipeline = BeamGraphFilterPipeline()
```

**If specific features needed**:
```python
# Use original TextGraph for advanced features
from mint.text_graph import TextGraph
graph = TextGraph()
# ... use advanced features
``` 