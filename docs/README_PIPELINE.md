# 🔄 Pipeline Module Documentation

## Overview
The `pipeline` module contains the main beam filter pipeline and CLI interface for the refactored project.

## Files

### `beam_filter_pipeline.py`

**Purpose**: Main pipeline class that orchestrates the entire beam search and filtering process

**Class**: `BeamFilterPipeline`

#### Constructor
```python
__init__(config: Optional[Dict] = None)
```

**Parameters**:
- `config (Optional[Dict])`: Configuration dictionary (default: None)

**Default Configuration**:
```python
{
    "beam_width": 40,
    "max_depth": 120,
    "max_paths": 200,
    "min_relevance_score": 0.15,
    "max_final_sentences": 30,
    "use_sbert": False,
    "use_contradiction_detection": False,
    "use_nli": False,
    "enable_pos_filtering": True,
    "important_pos_tags": {"N", "Np", "V", "A", "Nc", "M", "R", "P"}
}
```

#### Methods

##### `run_pipeline(context: str, claim: str) -> Dict`
Runs the complete beam search and filtering pipeline.

**Parameters**:
- `context (str)`: Context text
- `claim (str)`: Claim text

**Returns**:
- `Dict`: Pipeline results with structure:
  ```python
  {
      "context": str,                    # Original context
      "claim": str,                      # Original claim
      "context_sentences": List[str],    # Split context sentences
      "candidate_sentences": List[Dict], # Sentences from beam search
      "final_sentences": List[Dict],     # Filtered final sentences
      "pipeline_stats": Dict,            # Pipeline statistics
      "graph_stats": Dict,               # Graph statistics
      "filter_stats": Dict               # Filter statistics
  }
  ```

**Example**:
```python
from src.pipeline.beam_filter_pipeline import BeamFilterPipeline

# Create pipeline
pipeline = BeamFilterPipeline()

# Run pipeline
context = "SAWACO thông báo tạm ngưng cung cấp nước từ 22 giờ đến 4 giờ. Các khu vực bị ảnh hưởng gồm quận 6, 8, 12."
claim = "SAWACO thông báo tạm ngưng cung cấp nước."

results = pipeline.run_pipeline(context, claim)

print(f"Found {len(results['final_sentences'])} relevant sentences")
for sentence in results['final_sentences']:
    print(f"- {sentence['sentence']}")
```

##### `_preprocess_text(context: str, claim: str) -> Tuple[str, str, List[str]]`
Preprocesses input text.

**Parameters**:
- `context (str)`: Raw context text
- `claim (str)`: Raw claim text

**Returns**:
- `Tuple[str, str, List[str]]`: (cleaned_context, cleaned_claim, context_sentences)

##### `_build_graph(context_sentences: List[str], claim: str) -> Tuple[TextGraphSimple, Dict, Dict]`
Builds the text graph from sentences and claim.

**Parameters**:
- `context_sentences (List[str])`: List of context sentences
- `claim (str)`: Claim text

**Returns**:
- `Tuple[TextGraphSimple, Dict, Dict]`: (graph, context_tokens, claim_tokens)

##### `_run_beam_search(graph: TextGraphSimple) -> List[Dict]`
Runs beam search to find candidate sentences.

**Parameters**:
- `graph (TextGraphSimple)`: Built text graph

**Returns**:
- `List[Dict]`: List of candidate sentences from beam search

##### `_filter_sentences(candidate_sentences: List[Dict], claim: str) -> Tuple[List[Dict], Dict]`
Filters candidate sentences using the filter wrapper.

**Parameters**:
- `candidate_sentences (List[Dict])`: Sentences from beam search
- `claim (str)`: Claim text

**Returns**:
- `Tuple[List[Dict], Dict]`: (filtered_sentences, filter_stats)

### `cli.py`

**Purpose**: Command-line interface for the beam filter pipeline

**Script**: `cli.py`

#### Command Line Arguments

```bash
python src/pipeline/cli.py [OPTIONS] CONTEXT_FILE CLAIM_FILE OUTPUT_FILE
```

**Arguments**:
- `CONTEXT_FILE`: Path to file containing context text
- `CLAIM_FILE`: Path to file containing claim text  
- `OUTPUT_FILE`: Path to output JSON file

**Options**:
- `--beam-width INT`: Beam search width (default: 40)
- `--max-depth INT`: Maximum search depth (default: 120)
- `--max-paths INT`: Maximum paths to find (default: 200)
- `--min-relevance FLOAT`: Minimum relevance score (default: 0.15)
- `--max-final-sentences INT`: Maximum final sentences (default: 30)
- `--use-sbert`: Enable SBERT filtering
- `--use-contradiction-detection`: Enable contradiction detection
- `--use-nli`: Enable NLI stance detection
- `--disable-pos-filtering`: Disable POS tag filtering
- `--help`: Show help message

#### Usage Examples

**Basic Usage**:
```bash
python src/pipeline/cli.py context.txt claim.txt output.json
```

**With Custom Parameters**:
```bash
python src/pipeline/cli.py \
    --beam-width 50 \
    --max-depth 100 \
    --min-relevance 0.2 \
    --max-final-sentences 20 \
    context.txt claim.txt output.json
```

**With Advanced Filtering**:
```bash
python src/pipeline/cli.py \
    --use-sbert \
    --use-contradiction-detection \
    --use-nli \
    context.txt claim.txt output.json
```

#### Output Format

The CLI outputs a JSON file with the following structure:

```json
{
    "context": "Original context text",
    "claim": "Original claim text",
    "context_sentences": ["Sentence 1", "Sentence 2", ...],
    "candidate_sentences": [
        {
            "sentence": "Candidate sentence 1",
            "score": 0.85,
            "path": ["claim_0", "word_1", "sentence_0"]
        }
    ],
    "final_sentences": [
        {
            "sentence": "Final sentence 1",
            "relevance_score": 0.92
        }
    ],
    "pipeline_stats": {
        "total_time": 2.5,
        "graph_nodes": 150,
        "graph_edges": 300,
        "beam_paths_found": 25,
        "sentences_filtered": 15
    },
    "graph_stats": {
        "word_nodes": 100,
        "sentence_nodes": 5,
        "dependency_edges": 200
    },
    "filter_stats": {
        "input_count": 25,
        "output_count": 15,
        "filter_time": 1.2
    }
}
```

## Usage in Pipeline

### Python API Usage

```python
from src.pipeline.beam_filter_pipeline import BeamFilterPipeline

# Create pipeline with custom config
config = {
    "beam_width": 50,
    "max_depth": 100,
    "min_relevance_score": 0.2,
    "max_final_sentences": 20,
    "use_sbert": True
}

pipeline = BeamFilterPipeline(config)

# Run pipeline
results = pipeline.run_pipeline(context_text, claim_text)

# Access results
final_sentences = results["final_sentences"]
pipeline_stats = results["pipeline_stats"]

print(f"Pipeline completed in {pipeline_stats['total_time']:.2f}s")
print(f"Found {len(final_sentences)} relevant sentences")
```

### CLI Usage

```bash
# Basic usage
python src/pipeline/cli.py input/context.txt input/claim.txt output/results.json

# With verbose output
python src/pipeline/cli.py \
    --beam-width 60 \
    --max-depth 150 \
    --min-relevance 0.25 \
    --max-final-sentences 25 \
    input/context.txt input/claim.txt output/results.json
```

## Dependencies

### Required
- `typing`: Type hints
- `time`: Time measurement
- `json`: JSON file handling
- `argparse`: Command-line argument parsing

### Internal Dependencies
- `src.utils.text_preprocessor`: Text preprocessing
- `src.nlp.vncorenlp_wrapper`: NLP annotation
- `src.graph.text_graph_simple`: Graph building
- `src.filtering.filter_wrapper`: Sentence filtering
- `mint.beam_search`: Beam search algorithm

## Error Handling

The pipeline includes comprehensive error handling:

1. **File I/O errors**: Graceful handling of missing files
2. **NLP errors**: Fallback when VnCoreNLP is unavailable
3. **Graph building errors**: Error messages for graph construction issues
4. **Beam search errors**: Handling of search algorithm failures
5. **Filtering errors**: Fallback when filtering system is unavailable

**Example**:
```python
try:
    results = pipeline.run_pipeline(context, claim)
    print("Pipeline completed successfully")
except Exception as e:
    print(f"Pipeline failed: {e}")
```

## Performance Considerations

- **Text preprocessing**: Fast, minimal overhead
- **NLP annotation**: Depends on VnCoreNLP availability and text length
- **Graph building**: Scales with number of words and sentences
- **Beam search**: Performance depends on graph connectivity and search parameters
- **Filtering**: Depends on filtering options and number of candidate sentences

## Testing

```python
# Test pipeline initialization
pipeline = BeamFilterPipeline()
assert pipeline.config is not None

# Test with simple data
context = "Test context. Another sentence."
claim = "Test claim."
results = pipeline.run_pipeline(context, claim)

assert "final_sentences" in results
assert "pipeline_stats" in results
```

## Configuration Options

### Pipeline Configuration
```python
config = {
    "beam_width": 40,              # Beam search width
    "max_depth": 120,              # Maximum search depth
    "max_paths": 200,              # Maximum paths to find
    "min_relevance_score": 0.15,   # Minimum relevance for filtering
    "max_final_sentences": 30,     # Maximum final sentences
    "use_sbert": False,            # Enable SBERT filtering
    "use_contradiction_detection": False,  # Enable contradiction detection
    "use_nli": False,              # Enable NLI stance detection
    "enable_pos_filtering": True,  # Enable POS tag filtering
    "important_pos_tags": {"N", "Np", "V", "A", "Nc", "M", "R", "P"}
}
```

### CLI Configuration
```bash
# Performance-focused configuration
python src/pipeline/cli.py \
    --beam-width 30 \
    --max-depth 80 \
    --max-paths 100 \
    --min-relevance 0.3 \
    --max-final-sentences 15 \
    context.txt claim.txt output.json

# Quality-focused configuration
python src/pipeline/cli.py \
    --beam-width 60 \
    --max-depth 150 \
    --max-paths 300 \
    --min-relevance 0.1 \
    --max-final-sentences 50 \
    --use-sbert \
    context.txt claim.txt output.json
```

## Troubleshooting

### Common Issues

1. **VnCoreNLP not available**:
   ```bash
   # Check VnCoreNLP installation
   ls -la vncorenlp/
   python -c "from py_vncorenlp import py_vncorenlp; print('VnCoreNLP available')"
   ```

2. **AdvancedDataFilter not found**:
   ```bash
   # Check if file exists
   ls -la advanced_data_filtering.py
   ```

3. **Memory issues**:
   ```python
   # Reduce parameters for large texts
   config = {
       "beam_width": 20,
       "max_depth": 60,
       "max_paths": 100,
       "max_final_sentences": 15
   }
   ```

4. **Slow performance**:
   ```python
   # Disable advanced features
   config = {
       "use_sbert": False,
       "use_contradiction_detection": False,
       "use_nli": False
   }
   ```

## Limitations

- Depends on external dependencies (VnCoreNLP, AdvancedDataFilter)
- Performance scales with text length and complexity
- Limited to Vietnamese text processing
- No support for batch processing in CLI
- No real-time processing capabilities 