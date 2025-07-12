# 🚀 REFACTORED BEAM GRAPH FILTER PIPELINE

## 📋 Overview

This is a refactored version of the beam graph filter pipeline, organized into clear modules with better separation of concerns.

## 🏗️ Project Structure

```
BeamSearchFillter/
├── src/
│   ├── utils/
│   │   ├── __init__.py
│   │   └── text_preprocessor.py          # Text cleaning & sentence splitting
│   ├── nlp/
│   │   ├── __init__.py
│   │   └── vncorenlp_wrapper.py         # VnCoreNLP integration
│   ├── graph/
│   │   ├── __init__.py
│   │   └── text_graph_simple.py         # Simplified TextGraph for pipeline
│   ├── filtering/
│   │   ├── __init__.py
│   │   └── filter_wrapper.py            # AdvancedDataFilter wrapper
│   └── pipeline/
│       ├── __init__.py
│       ├── beam_filter_pipeline.py       # Main pipeline class
│       └── cli.py                       # CLI interface
├── run_pipeline.py                       # Main script to run pipeline
├── example_usage.py                      # Example usage
└── README_REFACTORED.md                  # This file
```

## 🔧 Key Improvements

### 1. **Modular Design**
- **Utils**: Text preprocessing functions
- **NLP**: VnCoreNLP wrapper for text annotation
- **Graph**: Simplified TextGraph focused on pipeline needs
- **Filtering**: Wrapper for existing AdvancedDataFilter
- **Pipeline**: Main pipeline orchestration

### 2. **Clean Separation of Concerns**
- Each module has a single responsibility
- Easy to test individual components
- Easy to modify or replace components

### 3. **Backward Compatibility**
- Still uses existing `mint.beam_search` and `advanced_data_filtering`
- Same output format as original pipeline
- Same CLI arguments

## 🚀 Usage

### Method 1: Using CLI Script

```bash
# Basic usage
python run_pipeline.py --input raw_test.json --output_dir output

# With custom parameters
python run_pipeline.py \
    --input raw_test.json \
    --output_dir output \
    --min_relevance 0.15 \
    --beam_width 40 \
    --max_depth 120 \
    --max_paths 200 \
    --max_final_sentences 30 \
    --beam_sentences 50

# Beam-only mode (no filtering)
python run_pipeline.py --input raw_test.json --beam_only

# Filter-only mode (no beam search)
python run_pipeline.py --input raw_test.json --filter_only

# Enable advanced features
python run_pipeline.py \
    --input raw_test.json \
    --use_sbert \
    --use_contradiction \
    --use_nli
```

### Method 2: Using Python API

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import BeamFilterPipeline

# Create pipeline
pipeline = BeamFilterPipeline(
    use_sbert=False,
    use_contradiction_detection=False,
    use_nli=False
)

# Process single sample
sample = {
    "context": "Your context text here...",
    "claim": "Your claim text here..."
}

simple_result, detailed_result, raw_count, beam_count, final_count = pipeline.process_sample(
    sample=sample,
    min_relevance=0.15,
    beam_width=40,
    max_depth=120,
    max_paths=200,
    max_final_sentences=30,
    beam_sentences=50
)

print(f"Evidence: {simple_result.get('multi_level_evidence', [])}")

# Process batch
samples = [sample1, sample2, sample3]
results = pipeline.process_batch(
    samples=samples,
    output_dir="output",
    min_relevance=0.15,
    beam_width=40,
    max_depth=120,
    max_paths=200,
    max_final_sentences=30,
    beam_sentences=50
)

print(f"Results saved to: {results['simple_file']}")
```

### Method 3: Run Example

```bash
python example_usage.py
```

## 📊 Output Format

The pipeline produces the same output format as the original:

### Simple Output (`simple_*.json`)
```json
[
  {
    "context": "Original context text",
    "claim": "Original claim text",
    "multi_level_evidence": ["sentence1", "sentence2", "sentence3"]
  }
]
```

### Detailed Output (`detailed_*.json`)
```json
[
  {
    "context": "Original context text",
    "claim": "Original claim text",
    "multi_level_evidence": [
      {
        "sentence": "sentence1",
        "score": 0.85,
        "relevance_score": 0.92
      }
    ],
    "statistics": {
      "beam": {
        "total_paths": 150,
        "unique_sentences": 25
      }
    }
  }
]
```

### Statistics Output (`stats_*.json`)
```json
{
  "total_context_sentences": 1000,
  "total_beam_sentences": 500,
  "total_final_sentences": 150,
  "num_samples": 50,
  "beam_parameters": {
    "beam_width": 40,
    "max_depth": 120,
    "max_paths": 200,
    "beam_sentences": 50
  }
}
```

## 🔧 Configuration

### Pipeline Parameters
- `min_relevance`: Minimum relevance score (default: 0.15)
- `beam_width`: Beam search width (default: 40)
- `max_depth`: Maximum beam search depth (default: 120)
- `max_paths`: Maximum number of paths (default: 200)
- `max_final_sentences`: Maximum final sentences (default: 30)
- `beam_sentences`: Maximum sentences from beam search (default: 50)

### Advanced Features
- `use_sbert`: Enable SBERT semantic filtering
- `use_contradiction_detection`: Enable contradiction detection
- `use_nli`: Enable NLI stance detection

### Modes
- `beam_only`: Use only beam search, no filtering
- `filter_only`: Use only filtering, no beam search

## 🛠️ Development

### Adding New Components

1. **New Text Preprocessor**:
```python
# Add to src/utils/text_preprocessor.py
def new_preprocessing_function(text: str) -> str:
    # Your preprocessing logic
    return processed_text
```

2. **New Filter**:
```python
# Add to src/filtering/filter_wrapper.py
class NewFilter:
    def filter(self, sentences, claim_text):
        # Your filtering logic
        return filtered_sentences
```

3. **New Graph Builder**:
```python
# Add to src/graph/text_graph_simple.py
def new_graph_building_method(self, data):
    # Your graph building logic
    pass
```

### Testing

```python
# Test individual components
from src.utils.text_preprocessor import clean_text, split_sentences
from src.nlp.vncorenlp_wrapper import VnCoreNLPWrapper
from src.graph.text_graph_simple import TextGraphSimple
from src.filtering.filter_wrapper import FilterWrapper

# Test text preprocessing
cleaned = clean_text("Your text here")
sentences = split_sentences(cleaned)

# Test NLP wrapper
nlp = VnCoreNLPWrapper()
tokens = nlp.annotate_text("Your text here")

# Test graph building
graph = TextGraphSimple()
graph.build_from_vncorenlp_output(context_tokens, claim, claim_tokens)

# Test filtering
filter_wrapper = FilterWrapper()
results = filter_wrapper.filter_sentences(sentences, claim_text)
```

## 📝 Migration from Original

The refactored pipeline is **backward compatible** with the original:

1. **Same CLI arguments**: All original arguments work
2. **Same output format**: Identical JSON structure
3. **Same dependencies**: Uses existing `mint.beam_search` and `advanced_data_filtering`

### Migration Steps

1. **Replace original script**:
```bash
# Old way
python beam_graph_filter_pipeline.py --input raw_test.json

# New way
python run_pipeline.py --input raw_test.json
```

2. **Update imports** (if using as library):
```python
# Old way
from mint.text_graph import TextGraph
from advanced_data_filtering import AdvancedDataFilter

# New way
from src.pipeline import BeamFilterPipeline
```

## 🎯 Benefits

1. **Maintainability**: Clear module structure, easy to understand and modify
2. **Testability**: Each component can be tested independently
3. **Extensibility**: Easy to add new features or replace components
4. **Reusability**: Components can be reused in other projects
5. **Documentation**: Better code organization with clear responsibilities

## 🔍 Troubleshooting

### Common Issues

1. **Import errors**: Make sure `src` directory is in Python path
2. **VnCoreNLP not found**: Check `vncorenlp` directory exists
3. **AdvancedDataFilter not found**: Ensure `advanced_data_filtering.py` is available

### Debug Mode

```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)

# Create pipeline with debug
pipeline = BeamFilterPipeline()
```

## 📞 Support

For issues or questions:
1. Check the original `beam_graph_filter_pipeline.py` for reference
2. Review the example usage in `example_usage.py`
3. Check the module documentation in each `__init__.py` file 