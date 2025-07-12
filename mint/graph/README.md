# Graph Module - Text Graph Construction & Analysis

## Overview
Module for building and analyzing text graphs from Vietnamese context and claims, using VnCoreNLP for natural language processing.

## Ideas & Theory

### 1. Text Graph Concept
- **Purpose**: Convert text into graph structure to analyze relationships between words, sentences, and claims
- **Theory**: Based on Graph Theory and NLP to create structured representation of text
- **Applications**: Fact-checking, semantic analysis, evidence detection

### 2. Node Types
- **Word Nodes**: Represent each word in context and claim
  - Contains POS tag, lemma information
  - Can be filtered by POS tag to reduce noise
  - Connected to sentence nodes and claim node
- **Sentence Nodes**: Represent each sentence in context
  - Contains complete sentence content
  - Intermediate point connecting words to context
- **Claim Node**: Represents the claim to be verified
  - Target point of analysis process
  - Directly connected to words in claim

### 3. Edge Types
- **Structural Edges**: Connect words to sentences/claims (belongs_to)
- **Dependency Edges**: Connect words to words via dependency parsing
- **Semantic Edges**: Connect words with similar meanings (based on PhoBERT embeddings)

### 4. Semantic Analysis
- **Embedding Model**: PhoBERT (Vietnamese BERT)
- **Similarity Metric**: Cosine similarity
- **Threshold**: 0.85 (adjustable)
- **Purpose**: Find words with similar meanings to enhance connections

## Structure
```
mint/graph/
├── __init__.py
├── text_graph.py      # Main TextGraph class
├── cli.py            # CLI for testing
└── README.md
```

## Terminal Commands

### 1. Test initialization
```bash
python mint/graph/cli.py init
```

### 2. Test graph building
```bash
python mint/graph/cli.py build
```

### 3. Test semantic analysis
```bash
python mint/graph/cli.py semantic
```

### 4. Test beam search
```bash
python mint/graph/cli.py beam
```

### 5. Test data export
```bash
python mint/graph/cli.py export
```

### 6. Test statistics
```bash
python mint/graph/cli.py stats
```

### 7. Test all
```bash
python mint/graph/cli.py all
```

## Parameter Options

### Custom context and claim
```bash
python mint/graph/cli.py build --context "Your context text" --claim "Your claim text"
```

### Custom beam search parameters
```bash
python mint/graph/cli.py beam --beam-width 20 --max-depth 30 --max-paths 10
```

### Test all with custom parameters
```bash
python mint/graph/cli.py all --context "Your context" --claim "Your claim" --beam-width 15 --max-depth 25
```

## Output Files
- `output/graph_test.gexf` - Graph file (can be opened with Gephi)
- `output/graph_test.json` - JSON data
- `output/beam_search_*.json` - Beam search results
- `output/beam_search_*.txt` - Beam search summary

## Help
```bash
python mint/graph/cli.py --help
```

## Real-World Applications

### 1. Fact-Checking
- Compare claims with context
- Find supporting/contradicting evidence
- Analyze reliability

### 2. Semantic Analysis
- Find synonyms
- Analyze semantics
- Discover hidden relationships

### 3. Information Retrieval
- Search for related information
- Rank results
- Query expansion 