# Beam Search Module - Path Finding & Analysis

## Overview
Module implementing Beam Search algorithm to find optimal paths in Text Graph, supporting fact-checking and evidence discovery.

## Ideas & Theory

### 1. Beam Search Algorithm
- **Purpose**: Find best paths from claim to evidence in context
- **Theory**: Based on classic Beam Search algorithm, adapted for graph traversal
- **Advantages**: Balance between result quality and computational efficiency

### 2. Path Finding Strategy
- **Start Node**: Claim node (starting point)
- **Target Nodes**: Sentence nodes (containing evidence)
- **Scoring Function**: Combines semantic similarity, path length, and node importance
- **Beam Width**: Number of path candidates kept at each level

### 3. Multi-Level Approach
- **Level 1**: Find paths directly from claim to sentences
- **Level 2**: Find paths through 1 intermediate node
- **Level 3+**: Find paths through multiple intermediate nodes
- **Purpose**: Discover indirect evidence and complex relationships

### 4. Scoring Mechanisms
- **Semantic Score**: Based on similarity between claim and path
- **Structural Score**: Based on path length and edge types
- **Evidence Score**: Based on importance of target sentence
- **Combined Score**: Weighted combination of above scores

### 5. Evidence Discovery
- **Direct Evidence**: Paths directly from claim to supporting sentences
- **Indirect Evidence**: Paths through intermediate nodes
- **Contradicting Evidence**: Paths to contradicting sentences
- **Neutral Evidence**: Paths to neutral sentences

## Structure
```
mint/beam_search/
├── __init__.py
├── beam_search.py        # Main BeamSearchPathFinder class
├── helpers.py           # Utility functions
├── improved_scoring.py  # Advanced scoring algorithms
├── cli.py              # CLI for testing
└── README.md
```

## Terminal Commands

### 1. Test initialization
```bash
python mint/beam_search/cli.py init
```

### 2. Test basic beam search
```bash
python mint/beam_search/cli.py basic
```

### 3. Test multi-level beam search
```bash
python mint/beam_search/cli.py multi
```

### 4. Test path export
```bash
python mint/beam_search/cli.py export
```

### 5. Test path analysis
```bash
python mint/beam_search/cli.py analyze
```

### 6. Test all
```bash
python mint/beam_search/cli.py all
```

## Parameter Options

### Custom context and claim
```bash
python mint/beam_search/cli.py basic --context "Your context text" --claim "Your claim text"
```

### Custom beam search parameters
```bash
python mint/beam_search/cli.py basic --beam-width 20 --max-depth 30 --max-paths 10
```

### Custom multi-level parameters
```bash
python mint/beam_search/cli.py multi --max-levels 5 --beam-width-per-level 5 --max-depth 25
```

### Test all with custom parameters
```bash
python mint/beam_search/cli.py all --context "Your context" --claim "Your claim" --beam-width 15 --max-depth 25 --max-paths 8
```

## Output Files
- `output/paths_test.json` - Paths data (detailed)
- `output/paths_summary.txt` - Paths summary (human-readable)

## Help
```bash
python mint/beam_search/cli.py --help
```

## Real-World Applications

### 1. Fact-Checking Automation
- **Claim Verification**: Automatically find evidence for claims
- **Evidence Ranking**: Rank evidence by reliability
- **Contradiction Detection**: Detect contradictions

### 2. Information Retrieval
- **Query Expansion**: Expand queries based on semantic similarity
- **Relevance Ranking**: Rank results by relevance
- **Context Understanding**: Understand context for accurate answers

### 3. Knowledge Graph Applications
- **Entity Linking**: Link entities in text
- **Relation Extraction**: Extract relationships
- **Knowledge Discovery**: Discover new knowledge

### 4. Research Applications
- **Literature Review**: Find related papers
- **Citation Analysis**: Analyze citations
- **Research Gap**: Find gaps in research

## Algorithm Parameters

### Beam Search Parameters
- **beam_width**: Number of paths to keep (default: 10)
- **max_depth**: Maximum path depth (default: 15)
- **max_paths**: Number of paths to return (default: 5)

### Multi-Level Parameters
- **max_levels**: Maximum number of levels (default: 3)
- **beam_width_per_level**: Beam width for each level (default: 3)
- **min_new_sentences**: Minimum new sentences (default: 0)

### Scoring Parameters
- **semantic_weight**: Weight for semantic score (default: 0.6)
- **structural_weight**: Weight for structural score (default: 0.3)
- **evidence_weight**: Weight for evidence score (default: 0.1) 