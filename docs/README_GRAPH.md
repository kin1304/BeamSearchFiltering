# 🕸️ Graph Module Documentation

## Overview
The `graph` module contains the simplified TextGraph class focused on beam search pipeline needs.

## Files

### `text_graph_simple.py`

**Purpose**: Simplified TextGraph class for beam search pipeline

**Class**: `TextGraphSimple`

#### Constructor
```python
__init__()
```

**Features**:
- NetworkX-based graph structure
- POS tag filtering for word nodes
- Support for word, sentence, and claim nodes
- Dependency relationship handling

#### Properties

##### Node Mappings
- `word_nodes`: Dictionary mapping word text to node ID
- `sentence_nodes`: Dictionary mapping sentence index to node ID
- `claim_node`: String ID of the claim node

##### POS Filtering
- `enable_pos_filtering`: Boolean to enable/disable POS filtering
- `important_pos_tags`: Set of important POS tags to keep

**Default important POS tags**:
- `N`: Common nouns
- `Np`: Proper nouns
- `V`: Verbs
- `A`: Adjectives
- `Nc`: Person nouns
- `M`: Numbers
- `R`: Adverbs
- `P`: Pronouns

#### Methods

##### Configuration Methods

###### `set_pos_filtering(enable: bool = True, custom_pos_tags: Optional[Set[str]] = None)`
Configures POS tag filtering for word nodes.

**Parameters**:
- `enable (bool)`: Enable/disable POS filtering
- `custom_pos_tags (Optional[Set[str]])`: Custom set of POS tags

**Example**:
```python
from src.graph.text_graph_simple import TextGraphSimple

graph = TextGraphSimple()

# Disable POS filtering
graph.set_pos_filtering(enable=False)

# Use custom POS tags
graph.set_pos_filtering(enable=True, custom_pos_tags={'N', 'V', 'A'})
```

###### `is_important_word(word: str, pos_tag: str) -> bool`
Checks if a word should be included based on its POS tag.

**Parameters**:
- `word (str)`: Word text
- `pos_tag (str)`: POS tag

**Returns**:
- `bool`: True if word should be included

##### Node Addition Methods

###### `add_word_node(word: str, pos_tag: str = "", lemma: str = "") -> Optional[str]`
Adds a word node to the graph.

**Parameters**:
- `word (str)`: Word text
- `pos_tag (str)`: POS tag
- `lemma (str)`: Lemma form

**Returns**:
- `Optional[str]`: Node ID if added, None if filtered out

**Example**:
```python
node_id = graph.add_word_node("SAWACO", "Np", "SAWACO")
if node_id:
    print(f"Word node added: {node_id}")
else:
    print("Word filtered out")
```

###### `add_sentence_node(sentence_id: int, sentence_text: str) -> str`
Adds a sentence node to the graph.

**Parameters**:
- `sentence_id (int)`: Sentence index
- `sentence_text (str)`: Sentence text

**Returns**:
- `str`: Node ID

**Example**:
```python
node_id = graph.add_sentence_node(0, "SAWACO thông báo tạm ngưng cung cấp nước.")
print(f"Sentence node added: {node_id}")  # sentence_0
```

###### `add_claim_node(claim_text: str) -> str`
Adds a claim node to the graph.

**Parameters**:
- `claim_text (str)`: Claim text

**Returns**:
- `str`: Node ID (always "claim_0")

**Example**:
```python
node_id = graph.add_claim_node("SAWACO thông báo tạm ngưng cung cấp nước.")
print(f"Claim node added: {node_id}")  # claim_0
```

##### Edge Connection Methods

###### `connect_word_to_sentence(word_node: str, sentence_node: str)`
Connects a word node to a sentence node.

**Parameters**:
- `word_node (str)`: Word node ID
- `sentence_node (str)`: Sentence node ID

**Edge Type**: `structural`
**Relation**: `belongs_to`

###### `connect_word_to_claim(word_node: str, claim_node: str)`
Connects a word node to the claim node.

**Parameters**:
- `word_node (str)`: Word node ID
- `claim_node (str)`: Claim node ID

**Edge Type**: `structural`
**Relation**: `belongs_to`

###### `connect_dependency(dependent_word_node: str, head_word_node: str, dep_label: str)`
Connects dependency relationship between words.

**Parameters**:
- `dependent_word_node (str)`: Dependent word node ID
- `head_word_node (str)`: Head word node ID
- `dep_label (str)`: Dependency label

**Edge Type**: `dependency`
**Relation**: Dependency label (e.g., "nsubj", "obj", "root")

##### Graph Building Method

###### `build_from_vncorenlp_output(context_sentences: Dict, claim_text: str, claim_sentences: Dict)`
Builds the complete graph from VnCoreNLP annotated tokens.

**Parameters**:
- `context_sentences (Dict)`: Annotated context sentences
- `claim_text (str)`: Claim text
- `claim_sentences (Dict)`: Annotated claim sentences

**Process**:
1. Adds claim node
2. Processes context sentences:
   - Adds sentence nodes
   - Adds word nodes (with POS filtering)
   - Connects words to sentences
   - Creates dependency edges
3. Processes claim sentences:
   - Adds word nodes
   - Connects words to claim
   - Creates dependency edges

**Example**:
```python
from src.graph.text_graph_simple import TextGraphSimple

graph = TextGraphSimple()

# Build graph from VnCoreNLP output
graph.build_from_vncorenlp_output(context_tokens, claim_text, claim_tokens)

print(f"Graph built with {len(graph.word_nodes)} words, {len(graph.sentence_nodes)} sentences")
```

##### Beam Search Method

###### `beam_search_paths(beam_width: int = 10, max_depth: int = 6, max_paths: int = 20) -> List`
Finds paths from claim to sentence nodes using beam search.

**Parameters**:
- `beam_width (int)`: Width of beam search
- `max_depth (int)`: Maximum depth of path
- `max_paths (int)`: Maximum number of paths to return

**Returns**:
- `List`: List of best paths

**Dependencies**:
- Uses existing `mint.beam_search.BeamSearchPathFinder`

**Example**:
```python
# Find paths from claim to sentences
paths = graph.beam_search_paths(
    beam_width=25,
    max_depth=30,
    max_paths=20
)

print(f"Found {len(paths)} paths")
for path in paths:
    print(f"Path score: {path.score}, Length: {len(path.nodes)}")
```

## Usage in Pipeline

The simplified TextGraph is used in the main pipeline:

```python
from src.graph.text_graph_simple import TextGraphSimple

# Create graph
graph = TextGraphSimple()

# Configure POS filtering
graph.set_pos_filtering(enable=True)

# Build graph from annotated tokens
graph.build_from_vncorenlp_output(context_tokens, claim, claim_tokens)

# Run beam search
paths = graph.beam_search_paths(beam_width=40, max_depth=120, max_paths=200)
```

## Dependencies

### Required
- `networkx`: Graph data structure
- `typing`: Type hints
- `mint.beam_search`: Beam search algorithm (existing)

### Optional
- VnCoreNLP annotated tokens

## Graph Structure

### Node Types
1. **Word nodes**: `word_0`, `word_1`, ...
   - Attributes: `type`, `text`, `pos`, `lemma`
2. **Sentence nodes**: `sentence_0`, `sentence_1`, ...
   - Attributes: `type`, `text`
3. **Claim node**: `claim_0`
   - Attributes: `type`, `text`

### Edge Types
1. **Structural edges**: Word → Sentence/Claim
   - Type: `structural`
   - Relation: `belongs_to`
2. **Dependency edges**: Word → Word
   - Type: `dependency`
   - Relation: Dependency label

## Performance Considerations

- **POS filtering**: Reduces graph size by filtering unimportant words
- **Memory usage**: Scales with number of words and sentences
- **Beam search**: Performance depends on graph connectivity and search parameters

## Testing

```python
# Test graph building
graph = TextGraphSimple()
graph.add_claim_node("Test claim")
graph.add_sentence_node(0, "Test sentence")
word_node = graph.add_word_node("test", "N")

assert graph.claim_node == "claim_0"
assert "sentence_0" in graph.sentence_nodes.values()
assert word_node is not None

# Test POS filtering
graph.set_pos_filtering(enable=True, custom_pos_tags={'N'})
filtered_word = graph.add_word_node("test", "V")
assert filtered_word is None  # Verb filtered out
```

## Limitations

- Simplified compared to original TextGraph
- Focused on beam search pipeline needs
- No entity extraction or semantic similarity
- No visualization capabilities
- No graph export/import features 