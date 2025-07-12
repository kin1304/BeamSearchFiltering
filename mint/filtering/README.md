# Filtering Module - Advanced Data Filtering & Quality Control

## Overview
This module implements advanced data filtering with multiple strategies, designed to improve the accuracy of classification and fact-checking tasks.

## Concept & Theory

### 1. Multi-Stage Filtering Pipeline
- **Purpose**: Filter and improve data quality through multiple stages
- **Theory**: Combine various filtering methods for optimal results
- **Advantages**: Reduce noise, increase relevance, improve accuracy

### 2. Filtering Stages
- **Stage 1: Quality Filtering**: Filter by sentence quality (length, structure, content)
- **Stage 2: Semantic Relevance**: Filter by semantic relevance to the claim
- **Stage 3: Entity-Based**: Filter based on the presence of entities
- **Stage 4: Contradiction Detection**: Detect and remove contradictions
- **Stage 5: Duplicate Removal**: Remove duplicates and rank results

### 3. Advanced Techniques
- **SBERT Semantic Filtering**: Use sentence transformers for semantic similarity
- **NLI Stance Detection**: Natural Language Inference for stance classification
- **Entity Recognition**: Named Entity Recognition and entity-based scoring
- **Contradiction Indicators**: Pattern matching for contradiction detection

### 4. Scoring Mechanisms
- **Quality Score**: Based on length, structure, content richness
- **Semantic Score**: Cosine similarity with the claim
- **Entity Score**: Entity presence and importance
- **Contradiction Score**: Stance detection and contradiction indicators
- **Final Confidence Score**: Weighted combination of all scores

### 5. Vietnamese Language Support
- **Vietnamese SBERT**: "keepitreal/vietnamese-sbert" model
- **Vietnamese Stop Words**: Custom stop words for Vietnamese
- **Vietnamese Contradiction Patterns**: Contradiction indicators for Vietnamese
- **XLM-R XNLI**: Cross-lingual NLI model supporting Vietnamese

## Structure
```
mint/filtering/
├── __init__.py
├── advanced_data_filtering.py  # Main AdvancedDataFilter class
├── cli.py                     # CLI for testing
└── README.md
```

## Terminal Commands

### 1. Test initialization
```bash
python mint/filtering/cli.py init
```

### 2. Test quality filtering
```bash
python mint/filtering/cli.py quality
```

### 3. Test semantic filtering
```bash
python mint/filtering/cli.py semantic
```

### 4. Test entity filtering
```bash
python mint/filtering/cli.py entity
```

### 5. Test contradiction detection
```bash
python mint/filtering/cli.py contradiction
```

### 6. Test multi-stage pipeline
```bash
python mint/filtering/cli.py pipeline
```

### 7. Test individual functions
```bash
python mint/filtering/cli.py individual
```

### 8. Test export results
```bash
python mint/filtering/cli.py export
```

### 9. Test all
```bash
python mint/filtering/cli.py all
```

## Parameter Options

### Custom claim and context
```bash
python mint/filtering/cli.py pipeline --claim "Your claim" --context "Your context"
```

### Custom entities
```bash
python mint/filtering/cli.py entity --entities "entity1,entity2,entity3"
```

### Custom thresholds
```bash
python mint/filtering/cli.py pipeline --min-quality 0.4 --min-relevance 0.3 --min-entity 0.1
```

### Test all with custom parameters
```bash
python mint/filtering/cli.py all --claim "Your claim" --context "Your context" --entities "entity1,entity2"
```

## Output Files
- `output/filtering_results.json` - Detailed filtering results
- `output/filtering_summary.txt` - Human-readable summary

## Help
```bash
python mint/filtering/cli.py --help
```

## Real-World Applications

### 1. Data Preprocessing
- **Noise Reduction**: Remove irrelevant sentences
- **Quality Improvement**: Improve training data quality
- **Relevance Filtering**: Focus on relevant content

### 2. Fact-Checking Enhancement
- **Evidence Filtering**: Filter evidence by relevance
- **Contradiction Detection**: Detect conflicting information
- **Quality Assessment**: Assess evidence quality

### 3. Information Retrieval
- **Query Refinement**: Improve search results
- **Relevance Ranking**: Rank by relevance
- **Duplicate Removal**: Remove redundant information

### 4. Content Analysis
- **Semantic Analysis**: Analyze semantics
- **Entity Extraction**: Extract important entities
- **Stance Detection**: Detect content stance

## Technical Features

### 1. Multi-Model Support
- **SBERT Models**: Vietnamese and multilingual models
- **NLI Models**: XLM-R XNLI for stance detection
- **Fallback Mechanisms**: Graceful degradation when models are unavailable

### 2. Configurable Parameters
- **Quality Thresholds**: Adjust quality scores
- **Semantic Thresholds**: Adjust relevance scores
- **Entity Thresholds**: Adjust entity importance
- **Stance Delta**: Adjust contradiction detection

### 3. Advanced Analytics
- **Stage Statistics**: Detailed statistics for each stage
- **Filtering Metrics**: Precision, recall, F1 scores
- **Performance Monitoring**: Execution time and resource usage

### 4. Export Capabilities
- **JSON Export**: Structured data format
- **Summary Export**: Human-readable reports
- **Statistics Export**: Detailed analytics

## Algorithm Parameters

### Quality Filtering
- **min_quality_score**: Minimum quality threshold (default: 0.3)
- **length_weight**: Weight for sentence length (default: 0.3)
- **structure_weight**: Weight for sentence structure (default: 0.3)
- **content_weight**: Weight for content richness (default: 0.4)

### Semantic Filtering
- **min_relevance_score**: Minimum semantic relevance (default: 0.25)
- **max_final_sentences**: Maximum sentences to keep (default: 30)
- **similarity_model**: SBERT model to use (default: Vietnamese SBERT)

### Entity Filtering
- **min_entity_score**: Minimum entity importance (default: 0.05)
- **min_entity_keep**: Minimum entities to keep (default: 5)
- **entity_weight**: Weight for entity presence (default: 0.4)

### Contradiction Detection
- **stance_delta**: Stance detection threshold (default: 0.1)
- **contradiction_weight**: Weight for contradiction score (default: 0.6)
- **nli_model**: NLI model for stance detection (default: XLM-R XNLI) 