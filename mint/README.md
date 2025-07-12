# MINT - Text Graph Library

MINT (Text Graph Library) is a Python library for building and analyzing text graphs from Vietnamese text using py_vncorenlp.

## 🚀 Key Features

### Text Graph Building
- **Word nodes**: Contains each word in context and claim with POS tag, lemma information
- **Sentence nodes**: Sentences in context
- **Claim node**: Claim content to be verified

### Intelligent Analysis
- ✅ Find common words between context and claim
- ✅ Word frequency statistics
- ✅ Graph structure analysis
- ✅ Export data to JSON
- ✅ Save/load graphs

### Visualization
- ✅ Draw graphs with colors distinguishing different node types
- ✅ Interactive graph layout

## 📦 Installation

```bash
pip install py_vncorenlp networkx matplotlib numpy
```

## 🔧 Basic Usage

```python
from mint import TextGraph
import py_vncorenlp

# Initialize model
model = py_vncorenlp.VnCoreNLP(save_dir="vncorenlp")

# Data
context = "Context text..."
claim = "Claim text..."

# Process with py_vncorenlp
context_sentences = model.annotate_text(context)
claim_sentences = model.annotate_text(claim)

# Create graph
text_graph = TextGraph()
text_graph.build_from_vncorenlp_output(context_sentences, claim, claim_sentences)

# Statistics
stats = text_graph.get_detailed_statistics()
print(f"Total nodes: {stats['total_nodes']}")
print(f"Shared words: {stats['shared_words_count']}")

# Visualize graph
text_graph.visualize()
```

## 📊 Main Methods

### Graph Building
- `build_from_vncorenlp_output()`: Build graph from py_vncorenlp output
- `add_word_node()`: Add word node
- `add_sentence_node()`: Add sentence node
- `add_claim_node()`: Add claim node

### Analysis
- `get_statistics()`: Basic statistics
- `get_detailed_statistics()`: Detailed statistics
- `get_shared_words()`: Find common words
- `get_word_frequency()`: Word frequency statistics

### I/O
- `save_graph()`: Save graph to GEXF file
- `load_graph()`: Load graph from file
- `export_to_json()`: Export to JSON

### Visualization
- `visualize()`: Draw graph

## 🎯 Applications for Fact-checking

This library is specifically designed for fact-checking applications:

1. **Semantic Similarity**: Compare similarity between claim and context
2. **Evidence Detection**: Find supporting/contradicting evidence
3. **Linguistic Analysis**: Analyze language structure
4. **Feature Extraction**: Extract features for ML models

## 📈 Extensions

The library is designed to be modular and easily extensible:

- Add new node types (Entity, Relation, etc.)
- Integrate additional NLP tools
- Build custom similarity metrics
- Support additional export/import formats

## 🤝 Contributing

All contributions are welcome! Please create issues or pull requests.

## 📄 License

MIT License 