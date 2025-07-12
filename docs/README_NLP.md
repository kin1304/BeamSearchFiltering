# 🧠 NLP Module Documentation

## Overview
The `nlp` module handles natural language processing tasks, specifically VnCoreNLP integration for Vietnamese text annotation.

## Files

### `vncorenlp_wrapper.py`

**Purpose**: Wrapper for VnCoreNLP text annotation

**Class**: `VnCoreNLPWrapper`

#### Constructor
```python
__init__(vncorenlp_dir: str = "")
```

**Parameters**:
- `vncorenlp_dir (str)`: Path to VnCoreNLP directory (default: auto-detect)

**Features**:
- Automatic VnCoreNLP directory detection
- Error handling for missing dependencies
- Graceful fallback when VnCoreNLP is not available

#### Methods

##### `_init_model()`
Initializes the VnCoreNLP model with Vietnamese language support.

**Annotations**:
- `wseg`: Word segmentation
- `pos`: Part-of-speech tagging
- `ner`: Named entity recognition
- `parse`: Dependency parsing

**Example**:
```python
from src.nlp.vncorenlp_wrapper import VnCoreNLPWrapper

# Initialize wrapper
nlp = VnCoreNLPWrapper()

# Check if model is available
if nlp.is_available():
    print("✅ VnCoreNLP model loaded successfully")
else:
    print("⚠️ VnCoreNLP not available")
```

##### `annotate_text(text: str) -> Dict`
Annotates Vietnamese text with VnCoreNLP.

**Parameters**:
- `text (str)`: Vietnamese text to annotate

**Returns**:
- `Dict`: Annotated tokens with POS tags, NER, and dependency information

**Example**:
```python
from src.nlp.vncorenlp_wrapper import VnCoreNLPWrapper

nlp = VnCoreNLPWrapper()
text = "SAWACO thông báo tạm ngưng cung cấp nước."

# Annotate text
tokens = nlp.annotate_text(text)

# Result structure:
# {
#   0: [
#     {"wordForm": "SAWACO", "posTag": "Np", "ner": "ORG", "index": 1, "head": 2, "depLabel": "nsubj"},
#     {"wordForm": "thông_báo", "posTag": "V", "ner": "O", "index": 2, "head": 0, "depLabel": "root"},
#     ...
#   ]
# }
```

##### `is_available() -> bool`
Checks if VnCoreNLP model is available.

**Returns**:
- `bool`: True if model is loaded and ready

## Usage in Pipeline

The NLP wrapper is used in the main pipeline for text annotation:

```python
from src.nlp.vncorenlp_wrapper import VnCoreNLPWrapper

# Initialize NLP wrapper
nlp_wrapper = VnCoreNLPWrapper()

# Annotate context and claim
context_tokens = nlp_wrapper.annotate_text(context_clean)
claim_tokens = nlp_wrapper.annotate_text(claim)

# Use annotated tokens for graph building
graph.build_from_vncorenlp_output(context_tokens, claim, claim_tokens)
```

## Dependencies

### Required
- `py_vncorenlp`: Vietnamese NLP toolkit
- `os`: Operating system interface
- `sys`: System-specific parameters

### Optional
- VnCoreNLP JAR file and models in `vncorenlp/` directory

## Error Handling

The wrapper includes comprehensive error handling:

1. **Missing py_vncorenlp**: Graceful fallback with warning
2. **Model loading failure**: Detailed error messages
3. **Annotation errors**: Exception handling with fallback

**Example**:
```python
nlp = VnCoreNLPWrapper()

# Safe annotation
try:
    tokens = nlp.annotate_text("Some text")
    if tokens:
        print("Annotation successful")
    else:
        print("Annotation failed")
except Exception as e:
    print(f"Error during annotation: {e}")
```

## Configuration

### VnCoreNLP Directory Structure
```
vncorenlp/
├── VnCoreNLP-1.2.jar
├── models/
│   ├── wordsegment/
│   ├── postagger/
│   ├── ner/
│   └── parser/
└── py_vncorenlp.py
```

### Environment Variables
- `VNCORENLP_DIR`: Custom path to VnCoreNLP directory

## Testing

```python
# Test initialization
nlp = VnCoreNLPWrapper()
assert nlp.is_available() or not nlp.is_available()  # Should not raise error

# Test annotation (if available)
if nlp.is_available():
    tokens = nlp.annotate_text("Test text")
    assert isinstance(tokens, dict)
```

## Troubleshooting

### Common Issues

1. **VnCoreNLP not found**:
   ```bash
   # Check if directory exists
   ls -la vncorenlp/
   
   # Download VnCoreNLP if needed
   # (Follow VnCoreNLP installation guide)
   ```

2. **Model loading errors**:
   ```python
   # Check model availability
   nlp = VnCoreNLPWrapper()
   print(f"Model available: {nlp.is_available()}")
   ```

3. **Annotation failures**:
   ```python
   # Test with simple text
   tokens = nlp.annotate_text("Test")
   print(f"Tokens: {tokens}")
   ```

## Performance Notes

- Model loading is done once during initialization
- Annotation is performed on-demand
- Memory usage depends on VnCoreNLP model size
- Processing speed depends on text length and complexity 