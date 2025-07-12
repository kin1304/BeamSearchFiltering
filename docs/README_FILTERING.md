# 🔍 Filtering Module Documentation

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

**Example**:
```python
from src.filtering.filter_wrapper import FilterWrapper

# Create filter wrapper
filter_wrapper = FilterWrapper(
    use_sbert=False,
    use_contradiction_detection=False,
    use_nli=False
)

# Sample sentences
sentences = [
    {"sentence": "SAWACO thông báo tạm ngưng cung cấp nước."},
    {"sentence": "Thời gian thực hiện từ 22 giờ đến 4 giờ."},
    {"sentence": "Các khu vực bị ảnh hưởng gồm quận 6, 8, 12."}
]

claim_text = "SAWACO thông báo tạm ngưng cung cấp nước."

# Filter sentences
results = filter_wrapper.filter_sentences(
    sentences=sentences,
    claim_text=claim_text,
    min_relevance_score=0.15,
    max_final_sentences=10
)

print(f"Input: {results['input_count']} sentences")
print(f"Output: {results['output_count']} sentences")
print(f"Filtered sentences: {results['filtered_sentences']}")
```

##### `is_available() -> bool`
Checks if the filter system is available.

**Returns**:
- `bool`: True if AdvancedDataFilter is available and initialized

**Example**:
```python
filter_wrapper = FilterWrapper()

if filter_wrapper.is_available():
    print("✅ AdvancedDataFilter is available")
else:
    print("⚠️ AdvancedDataFilter not available, using fallback")
```

## Usage in Pipeline

The filter wrapper is used in the main pipeline for sentence filtering:

```python
from src.filtering.filter_wrapper import FilterWrapper

# Initialize filter wrapper
filter_wrapper = FilterWrapper(
    use_sbert=False,
    use_contradiction_detection=False,
    use_nli=False
)

# Filter candidate sentences from beam search
results = filter_wrapper.filter_sentences(
    sentences=candidate_sentences,
    claim_text=claim,
    min_relevance_score=min_relevance,
    max_final_sentences=max_final_sentences
)

final_sentences = results["filtered_sentences"]
```

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

**Example**:
```python
filter_wrapper = FilterWrapper()

# Safe filtering
try:
    results = filter_wrapper.filter_sentences(sentences, claim_text)
    if results["output_count"] > 0:
        print("Filtering successful")
    else:
        print("No sentences passed filtering")
except Exception as e:
    print(f"Error during filtering: {e}")
```

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

When AdvancedDataFilter is not available, the wrapper provides a fallback:

```python
# Fallback behavior
if not filter_wrapper.is_available():
    # Returns top N sentences without filtering
    results = filter_wrapper.filter_sentences(sentences, claim_text)
    # Results contain original sentences up to max_final_sentences
```

## Performance Considerations

- **SBERT**: Requires sentence-transformers library and model loading
- **Contradiction detection**: Additional processing time
- **NLI**: Requires HuggingFace transformers and model loading
- **Memory usage**: Scales with number of sentences and model sizes

## Testing

```python
# Test filter wrapper initialization
filter_wrapper = FilterWrapper()
assert isinstance(filter_wrapper.is_available(), bool)

# Test filtering (if available)
if filter_wrapper.is_available():
    sentences = [{"sentence": "Test sentence"}]
    results = filter_wrapper.filter_sentences(sentences, "Test claim")
    assert "filtered_sentences" in results
    assert "input_count" in results
    assert "output_count" in results
```

## Integration with AdvancedDataFilter

The wrapper integrates with the existing AdvancedDataFilter:

```python
# Direct integration
from advanced_data_filtering import AdvancedDataFilter

# Original usage
filter_sys = AdvancedDataFilter(use_sbert=False, use_contradiction_detection=False, use_nli=False)
results = filter_sys.multi_stage_filtering_pipeline(sentences, claim_text, min_relevance_score=0.15)

# Wrapper usage (same functionality)
filter_wrapper = FilterWrapper(use_sbert=False, use_contradiction_detection=False, use_nli=False)
results = filter_wrapper.filter_sentences(sentences, claim_text, min_relevance_score=0.15)
```

## Troubleshooting

### Common Issues

1. **AdvancedDataFilter not found**:
   ```python
   # Check if file exists
   import os
   if os.path.exists("advanced_data_filtering.py"):
       print("AdvancedDataFilter file found")
   else:
       print("AdvancedDataFilter file not found")
   ```

2. **Import errors**:
   ```python
   # Check import
   try:
       from advanced_data_filtering import AdvancedDataFilter
       print("Import successful")
   except ImportError as e:
       print(f"Import failed: {e}")
   ```

3. **Filtering failures**:
   ```python
   # Test with simple data
   filter_wrapper = FilterWrapper()
   if filter_wrapper.is_available():
       results = filter_wrapper.filter_sentences([{"sentence": "test"}], "test")
       print(f"Filtering result: {results}")
   ```

## Limitations

- Depends on existing AdvancedDataFilter implementation
- No custom filtering logic in wrapper
- Limited to sentence-level filtering
- No support for custom filtering algorithms 