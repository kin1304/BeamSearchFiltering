# Filtering Module Documentation

## Overview
The `filtering` module contains the wrapper for the existing AdvancedDataFilter, providing a clean interface for sentence filtering in the beam search pipeline.

## Files

### `filter_wrapper.py`

**Purpose**: Wrapper for AdvancedDataFilter to use in pipeline

**Class**: `FilterWrapper`

#### Constructor
```python
__init__(use_sbert: bool = False, use_contradiction_detection: bool = False, use_nli: bool = False)
```

**Parameters**:
- `use_sbert (bool)`: Enable SBERT semantic filtering
- `use_contradiction_detection (bool)`: Enable contradiction detection
- `use_nli (bool)`: Enable NLI stance detection

**Features**:
- Graceful handling of missing AdvancedDataFilter
- Configurable filtering options
- Suppressed stdout during filtering
- Fallback behavior when filter system is unavailable

#### Methods

##### `filter_sentences(sentences: List[Dict], claim_text: str, min_relevance_score: float = 0.15, max_final_sentences: int = 30) -> Dict`
Filters sentences using AdvancedDataFilter.

**Parameters**:
- `sentences (List[Dict])`: List of sentences to filter
- `claim_text (str)`: Claim text for comparison
- `min_relevance_score (float)`: Minimum relevance score (default: 0.15)
- `max_final_sentences (int)`: Maximum number of final sentences (default: 30)

**Returns**:
- `Dict`: Filtering results with structure:
  ```python
  {
      "filtered_sentences": List[Dict],  # Filtered sentences
      "input_count": int,                # Number of input sentences
      "output_count": int                # Number of output sentences
  }
  ```

##### `is_available() -> bool`
Checks if the filter system is available.

**Returns**:
- `bool`: True if AdvancedDataFilter is available and initialized

## Dependencies

### Required
- `typing`: Type hints
- `contextlib`: Context managers for stdout suppression
- `io`: StringIO for capturing output

### Optional
- `advanced_data_filtering.AdvancedDataFilter`: Main filtering system

## Error Handling

The wrapper includes comprehensive error handling:

1. **Missing AdvancedDataFilter**: Graceful fallback with warning
2. **Initialization failure**: Detailed error messages
3. **Filtering errors**: Exception handling with fallback

## Configuration Options

### SBERT Semantic Filtering
```python
# Enable SBERT for semantic relevance filtering
filter_wrapper = FilterWrapper(use_sbert=True)
```

### Contradiction Detection
```python
# Enable contradiction detection
filter_wrapper = FilterWrapper(use_contradiction_detection=True)
```

### NLI Stance Detection
```python
# Enable NLI for stance detection
filter_wrapper = FilterWrapper(use_nli=True)
```

### Combined Features
```python
# Enable all advanced features
filter_wrapper = FilterWrapper(
    use_sbert=True,
    use_contradiction_detection=True,
    use_nli=True
)
```

## Fallback Behavior

When AdvancedDataFilter is not available, the wrapper provides a fallback that returns top N sentences without filtering.

## Performance Considerations

- **SBERT**: Requires sentence-transformers library and model loading
- **Contradiction detection**: Additional processing time
- **NLI**: Requires HuggingFace transformers and model loading
- **Memory usage**: Scales with number of sentences and model sizes

## Integration with AdvancedDataFilter

The wrapper integrates with the existing AdvancedDataFilter to provide the same functionality through a cleaner interface.

## Limitations

- Depends on existing AdvancedDataFilter implementation
- No custom filtering logic in wrapper
- Limited to sentence-level filtering
- No support for custom filtering algorithms 