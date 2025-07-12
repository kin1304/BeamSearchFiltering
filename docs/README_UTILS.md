# 📝 Utils Module Documentation

## Overview
The `utils` module contains text preprocessing utilities used in the beam graph filter pipeline.

## Files

### `text_preprocessor.py`

**Purpose**: Text cleaning and sentence splitting utilities

**Functions**:

#### `clean_text(text: str) -> str`
Cleans and normalizes text for processing.

**Parameters**:
- `text (str)`: Raw text to clean

**Returns**:
- `str`: Cleaned text

**Features**:
- Removes line breaks and extra whitespace
- Normalizes punctuation spacing
- Removes spaces before punctuation marks
- Removes spaces after opening parentheses
- Removes spaces before closing parentheses

**Example**:
```python
from src.utils.text_preprocessor import clean_text

raw_text = "This is a  test   text.\nWith line breaks."
cleaned = clean_text(raw_text)
# Result: "This is a test text. With line breaks."
```

#### `split_sentences(text: str) -> List[str]`
Splits text into sentences using regex patterns.

**Parameters**:
- `text (str)`: Text to split into sentences

**Returns**:
- `List[str]`: List of sentences

**Features**:
- Splits on `.`, `!`, `?` followed by whitespace
- Handles Vietnamese text
- Removes empty sentences
- Strips whitespace from each sentence

**Example**:
```python
from src.utils.text_preprocessor import split_sentences

text = "This is sentence one. This is sentence two! And this is sentence three?"
sentences = split_sentences(text)
# Result: ["This is sentence one", "This is sentence two", "And this is sentence three"]
```

## Usage in Pipeline

The text preprocessor is used in the main pipeline:

```python
from src.utils.text_preprocessor import clean_text, split_sentences

# In pipeline
context_clean = clean_text(context_raw)
raw_sentences = split_sentences(context_clean)
```

## Dependencies
- `re`: Regular expressions for text processing
- `typing`: Type hints

## Testing

```python
# Test text cleaning
assert clean_text("  test  text  .  ") == "test text."
assert clean_text("text ( with ) spaces") == "text (with) spaces"

# Test sentence splitting
assert split_sentences("A. B! C?") == ["A", "B", "C"]
assert split_sentences("") == []
``` 