# Mint Module - Vietnamese Text Graph & Beam Search Framework

## Overview
A comprehensive framework for building Vietnamese text graphs and searching for evidence using Beam Search, specifically designed for fact-checking and semantic analysis.

## Concept & Theory

### 1. Framework Architecture
- **Purpose**: To create a complete system for analyzing Vietnamese text
- **Theory**: Combines Graph Theory, NLP, and Search Algorithms
- **Innovation**: Adapts international algorithms for Vietnamese language specifics

### 2. Core Components
- **Text Graph Module**: Builds graphs from text
  - Node types: Word, Sentence, Claim
  - Edge types: Structural, Dependency, Semantic
  - Semantic analysis with PhoBERT
- **Beam Search Module**: Finds optimal paths
  - Multi-level search strategy
  - Advanced scoring mechanisms
  - Evidence discovery algorithms
- **Filtering Module**: Advanced data filtering
  - Multi-stage filtering pipeline
  - Semantic relevance filtering
  - Contradiction detection
  - Entity-based filtering

### 3. Vietnamese Language Processing
- **VnCoreNLP Integration**: Professional Vietnamese language processing
- **POS Tagging**: Accurate part-of-speech analysis
- **Dependency Parsing**: Syntactic analysis
- **Word Segmentation**: Vietnamese word tokenization

### 4. Fact-Checking Pipeline
- **Input**: Context text + Claim text
- **Processing**: Graph construction + Beam search
- **Output**: Evidence paths + Confidence scores
- **Applications**: Automated fact-checking, evidence discovery

### 5. Research Contributions
- **Novel Approach**: Combines graph-based and search-based methods
- **Vietnamese Focus**: Optimized for Vietnamese language
- **Scalable Design**: Modular architecture for easy extension
- **Practical Applications**: Real-world fact-checking scenarios

## Structure
```
mint/
├── __init__.py
├── cli.py              # Main CLI interface
├── README.md
├── graph/              # Text Graph Module
│   ├── __init__.py
│   ├── text_graph.py   # Main TextGraph class
│   ├── cli.py         # Graph CLI
│   └── README.md
└── beam_search/        # Beam Search Module
    ├── __init__.py
    ├── beam_search.py  # Main BeamSearchPathFinder class
    ├── helpers.py      # Utility functions
    ├── improved_scoring.py # Advanced scoring
    ├── cli.py         # Beam Search CLI
    └── README.md
```

## Main Terminal Commands

### Graph Commands
```bash
# Test graph functions
python mint/cli.py graph init
python mint/cli.py graph build
python mint/cli.py graph semantic
python mint/cli.py graph beam
python mint/cli.py graph export
python mint/cli.py graph stats
python mint/cli.py graph all
```

### Beam Search Commands
```bash
# Test beam search functions
python mint/cli.py beam init
python mint/cli.py beam basic
python mint/cli.py beam multi
python mint/cli.py beam export
python mint/cli.py beam analyze
python mint/cli.py beam all
```

### Filtering Commands
```bash
# Test filtering functions
python mint/cli.py filtering init
python mint/cli.py filtering quality
python mint/cli.py filtering semantic
python mint/cli.py filtering entity
python mint/cli.py filtering contradiction
python mint/cli.py filtering pipeline
python mint/cli.py filtering individual
python mint/cli.py filtering export
python mint/cli.py filtering all
```

## Direct Terminal Commands

### Graph Module
```bash
python mint/graph/cli.py init
python mint/graph/cli.py build
python mint/graph/cli.py semantic
python mint/graph/cli.py beam
python mint/graph/cli.py export
python mint/graph/cli.py stats
python mint/graph/cli.py all
```

### Beam Search Module
```bash
python mint/beam_search/cli.py init
python mint/beam_search/cli.py basic
python mint/beam_search/cli.py multi
python mint/beam_search/cli.py export
python mint/beam_search/cli.py analyze
python mint/beam_search/cli.py all
```

### Filtering Module
```bash
python mint/filtering/cli.py init
python mint/filtering/cli.py quality
python mint/filtering/cli.py semantic
python mint/filtering/cli.py entity
python mint/filtering/cli.py contradiction
python mint/filtering/cli.py pipeline
python mint/filtering/cli.py individual
python mint/filtering/cli.py export
python mint/filtering/cli.py all
```

## Parameter Options

### Graph with custom parameters
```bash
python mint/cli.py graph build --context "Your context" --claim "Your claim"
python mint/cli.py graph beam --beam-width 20 --max-depth 30 --max-paths 10
```

### Beam Search with custom parameters
```bash
python mint/cli.py beam basic --context "Your context" --claim "Your claim"
python mint/cli.py beam multi --max-levels 5 --beam-width-per-level 5
```

### Filtering with custom parameters
```bash
python mint/cli.py filtering pipeline --claim "Your claim" --context "Your context"
python mint/cli.py filtering entity --entities "entity1,entity2,entity3"
python mint/cli.py filtering pipeline --min-quality 0.4 --min-relevance 0.3 --min-entity 0.1
```

## Output Files
- `output/graph_test.gexf` - Graph file (can be opened with Gephi)
- `output/graph_test.json` - Graph JSON data
- `output/beam_search_*.json` - Beam search results
- `output/paths_test.json` - Paths data (detailed)
- `output/paths_summary.txt` - Paths summary (human-readable)
- `output/filtering_results.json` - Filtering results (detailed)
- `output/filtering_summary.txt` - Filtering summary (human-readable)

## Real-World Applications

### 1. Automated Fact-Checking
- **Claim Verification**: Automatically verify claims from news articles
- **Evidence Discovery**: Find supporting/contradicting evidence
- **Confidence Scoring**: Assess evidence reliability

### 2. Content Analysis
- **Semantic Analysis**: Analyze text semantics
- **Information Extraction**: Extract important information
- **Relationship Discovery**: Discover hidden relationships

### 3. Research Applications
- **Literature Review**: Automated paper review
- **Citation Analysis**: Citation analysis
- **Knowledge Discovery**: Discover knowledge gaps

### 4. Business Applications
- **Market Research**: Sentiment and trend analysis
- **Competitive Intelligence**: Competitor monitoring
- **Risk Assessment**: Risk assessment from news

### 5. Data Quality Applications
- **Data Preprocessing**: Improve training data quality
- **Noise Reduction**: Remove irrelevant content
- **Quality Assessment**: Assess evidence quality

## Technical Features

### 1. Vietnamese NLP
- **VnCoreNLP Integration**: Professional Vietnamese processing
- **POS Tagging**: Accurate part-of-speech analysis
- **Dependency Parsing**: Syntactic analysis
- **Word Segmentation**: Vietnamese word tokenization

### 2. Graph Construction
- **Multi-level Nodes**: Words, sentences, claims
- **Rich Edge Types**: Structural, dependency, semantic
- **Semantic Similarity**: PhoBERT-based embeddings
- **Flexible Filtering**: POS-based word filtering

### 3. Advanced Search
- **Beam Search Algorithm**: Optimal path finding
- **Multi-level Strategy**: Direct and indirect evidence
- **Scoring Mechanisms**: Semantic, structural, evidence scores
- **Evidence Ranking**: Confidence-based ranking

### 4. Export & Visualization
- **GEXF Export**: Compatible with Gephi
- **JSON Export**: Structured data format
- **Path Summary**: Human-readable results
- **Statistics**: Detailed analysis metrics

### 5. Advanced Filtering
- **Multi-Stage Pipeline**: Quality, semantic, entity, contradiction filtering
- **SBERT Models**: Vietnamese and multilingual semantic models
- **NLI Models**: Cross-lingual stance detection
- **Configurable Parameters**: Adjustable thresholds and weights 