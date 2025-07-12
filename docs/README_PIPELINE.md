# Pipeline Module Documentation

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

## Output Format

Pipeline sẽ xuất ra **3 file output** với format tên file:

```
{input_name}_beam_filtered_{min_relevance}_{timestamp}_detailed.json
{input_name}_beam_filtered_{min_relevance}_{timestamp}_simple.json
{input_name}_beam_filtered_{min_relevance}_{timestamp}_stats.json
```

### 1. Detailed Output (`*_detailed.json`)
Chứa thông tin chi tiết về quá trình xử lý từng sample.

### 2. Simple Output (`*_simple.json`)
Chỉ chứa danh sách evidence sentences.

### 3. Statistics Output (`*_stats.json`)
Thống kê tổng quan về quá trình xử lý.

**Ý nghĩa:**
- `detailed`: Phục vụ phân tích sâu, debug, kiểm tra đường đi, điểm số, v.v.
- `simple`: Dùng cho downstream task, chỉ lấy danh sách evidence cuối cùng.
- `stats`: Theo dõi hiệu suất, số lượng câu, tham số beam, v.v.

**Lưu ý:**
- Tên file sẽ tự động sinh theo input, min_relevance, timestamp để dễ quản lý batch.
- Các script, CLI, Python API đều xuất ra đúng 3 file này.

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

## Performance Considerations

- **Text preprocessing**: Fast, minimal overhead
- **NLP annotation**: Depends on VnCoreNLP availability and text length
- **Graph building**: Scales with number of words and sentences
- **Beam search**: Performance depends on graph connectivity and search parameters
- **Filtering**: Depends on filtering options and number of candidate sentences

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

## Limitations

- Depends on external dependencies (VnCoreNLP, AdvancedDataFilter)
- Performance scales with text length and complexity
- Limited to Vietnamese text processing
- No support for batch processing in CLI
- No real-time processing capabilities 