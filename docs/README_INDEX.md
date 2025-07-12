# 📖 Beam Search Filter Pipeline - Documentation Index

## Overview
This is the complete documentation for the refactored Beam Search Filter Pipeline project. The project has been modularized for better maintainability, testability, and extensibility.

## 📚 Documentation Structure

### 🏗️ Architecture & Design
- **[README_REFACTORED.md](../README_REFACTORED.md)**: Main project overview and architecture
- **[README_ORIGINAL_FILES.md](README_ORIGINAL_FILES.md)**: Documentation of original files and refactoring process
- **[VNCORENLP_SETUP.md](VNCORENLP_SETUP.md)**: VnCoreNLP installation and setup guide

### 🔧 Core Modules
- **[README_UTILS.md](README_UTILS.md)**: Text preprocessing utilities (`src/utils/`)
- **[README_NLP.md](README_NLP.md)**: Natural language processing wrapper (`src/nlp/`)
- **[README_GRAPH.md](README_GRAPH.md)**: Simplified text graph implementation (`src/graph/`)
- **[README_FILTERING.md](README_FILTERING.md)**: Sentence filtering wrapper (`src/filtering/`)
- **[README_PIPELINE.md](README_PIPELINE.md)**: Main pipeline and CLI interface (`src/pipeline/`)

### 🚀 Usage & Examples
- **[README_MAIN_SCRIPTS.md](README_MAIN_SCRIPTS.md)**: Main scripts and usage examples

## 🏗️ Project Structure

```
BeamSearchFillter/
├── src/                          # Refactored modules
│   ├── utils/                    # Text preprocessing
│   │   ├── __init__.py
│   │   └── text_preprocessor.py
│   ├── nlp/                      # NLP wrapper
│   │   ├── __init__.py
│   │   └── vncorenlp_wrapper.py
│   ├── graph/                    # Simplified graph
│   │   ├── __init__.py
│   │   └── text_graph_simple.py
│   ├── filtering/                # Filter wrapper
│   │   ├── __init__.py
│   │   └── filter_wrapper.py
│   └── pipeline/                 # Main pipeline
│       ├── __init__.py
│       ├── beam_filter_pipeline.py
│       └── cli.py
├── mint/                         # Original modules (preserved)
│   ├── __init__.py
│   ├── text_graph.py            # Full-featured graph
│   ├── beam_search.py           # Beam search algorithm
│   ├── helpers.py               # Helper functions
│   └── improved_scoring.py      # Scoring improvements
├── docs/                        # Documentation
│   ├── README_INDEX.md          # This file
│   ├── README_UTILS.md
│   ├── README_NLP.md
│   ├── README_GRAPH.md
│   ├── README_FILTERING.md
│   ├── README_PIPELINE.md
│   ├── README_MAIN_SCRIPTS.md
│   └── README_ORIGINAL_FILES.md
├── run_pipeline.py              # Simple runner script
├── example_usage.py             # Usage examples
├── advanced_data_filtering.py   # Original filter (preserved)
├── beam_graph_filter_pipeline.py # Original pipeline (preserved)
└── README_REFACTORED.md        # Main project README
```

## 🚀 Quick Start

### 1. Basic Usage
```bash
# Run with default settings
python run_pipeline.py

# Use CLI interface
python src/pipeline/cli.py context.txt claim.txt output.json
```

### 2. Python API
```python
from src.pipeline.beam_filter_pipeline import BeamFilterPipeline

pipeline = BeamFilterPipeline()
results = pipeline.run_pipeline(context_text, claim_text)
```

### 3. Custom Configuration
```python
config = {
    "beam_width": 50,
    "max_depth": 100,
    "min_relevance_score": 0.2,
    "max_final_sentences": 20
}

pipeline = BeamFilterPipeline(config)
results = pipeline.run_pipeline(context, claim)
```

## 📋 Module Overview

### 🔧 Utils Module (`src/utils/`)
- **Purpose**: Text preprocessing utilities
- **Key Features**: Text cleaning, sentence splitting
- **Main File**: `text_preprocessor.py`
- **Documentation**: [README_UTILS.md](README_UTILS.md)

### 🧠 NLP Module (`src/nlp/`)
- **Purpose**: VnCoreNLP integration for Vietnamese text
- **Key Features**: Text annotation, POS tagging, dependency parsing
- **Main File**: `vncorenlp_wrapper.py`
- **Documentation**: [README_NLP.md](README_NLP.md)

### 🕸️ Graph Module (`src/graph/`)
- **Purpose**: Simplified text graph for beam search
- **Key Features**: Graph building, POS filtering, dependency handling
- **Main File**: `text_graph_simple.py`
- **Documentation**: [README_GRAPH.md](README_GRAPH.md)

### 🔍 Filtering Module (`src/filtering/`)
- **Purpose**: Wrapper for AdvancedDataFilter
- **Key Features**: Sentence filtering, relevance scoring
- **Main File**: `filter_wrapper.py`
- **Documentation**: [README_FILTERING.md](README_FILTERING.md)

### 🔄 Pipeline Module (`src/pipeline/`)
- **Purpose**: Main pipeline orchestration and CLI
- **Key Features**: Complete pipeline, command-line interface
- **Main Files**: `beam_filter_pipeline.py`, `cli.py`
- **Documentation**: [README_PIPELINE.md](README_PIPELINE.md)

## 🎯 Use Cases

### 1. **Fact-Checking Pipeline**
```python
# Extract relevant sentences for fact-checking
pipeline = BeamFilterPipeline()
results = pipeline.run_pipeline(context, claim)
relevant_sentences = results["final_sentences"]
```

### 2. **Information Retrieval**
```python
# Find sentences related to a specific claim
config = {"beam_width": 60, "max_depth": 150}
pipeline = BeamFilterPipeline(config)
results = pipeline.run_pipeline(context, query)
```

### 3. **Text Analysis**
```python
# Analyze text structure and relationships
from src.graph.text_graph_simple import TextGraphSimple
graph = TextGraphSimple()
# ... build and analyze graph
```

### 4. **Batch Processing**
```bash
# Process multiple files
for file in contexts/*.txt; do
    python src/pipeline/cli.py "$file" claims/$(basename "$file") output/$(basename "$file" .txt).json
done
```

## 🔧 Configuration Options

### Pipeline Configuration
```python
config = {
    "beam_width": 40,              # Beam search width
    "max_depth": 120,              # Maximum search depth
    "max_paths": 200,              # Maximum paths to find
    "min_relevance_score": 0.15,   # Minimum relevance for filtering
    "max_final_sentences": 30,     # Maximum final sentences
    "use_sbert": False,            # Enable SBERT filtering
    "use_contradiction_detection": False,  # Enable contradiction detection
    "use_nli": False,              # Enable NLI stance detection
    "enable_pos_filtering": True,  # Enable POS tag filtering
    "important_pos_tags": {"N", "Np", "V", "A", "Nc", "M", "R", "P"}
}
```

### CLI Options
```bash
python src/pipeline/cli.py \
    --beam-width 50 \
    --max-depth 100 \
    --min-relevance 0.2 \
    --max-final-sentences 20 \
    --use-sbert \
    context.txt claim.txt output.json
```

## 📊 Performance Guidelines

### Small Texts (< 500 words)
```python
config = {
    "beam_width": 30,
    "max_depth": 80,
    "max_paths": 100,
    "min_relevance_score": 0.2,
    "max_final_sentences": 15
}
```

### Medium Texts (500-2000 words)
```python
config = {
    "beam_width": 40,
    "max_depth": 120,
    "max_paths": 200,
    "min_relevance_score": 0.15,
    "max_final_sentences": 30
}
```

### Large Texts (> 2000 words)
```python
config = {
    "beam_width": 50,
    "max_depth": 150,
    "max_paths": 300,
    "min_relevance_score": 0.1,
    "max_final_sentences": 50
}
```

## 🔍 Troubleshooting

### Common Issues

**1. VnCoreNLP not available**:
```bash
# Check installation
ls -la vncorenlp/
python -c "from py_vncorenlp import py_vncorenlp; print('Available')"
```

**2. AdvancedDataFilter not found**:
```bash
# Check if file exists
ls -la advanced_data_filtering.py
```

**3. Memory issues**:
```python
# Reduce parameters
config = {
    "beam_width": 20,
    "max_depth": 60,
    "max_paths": 100,
    "max_final_sentences": 15
}
```

### Error Handling
```python
try:
    results = pipeline.run_pipeline(context, claim)
    print("✅ Pipeline completed successfully")
except Exception as e:
    print(f"❌ Pipeline failed: {e}")
```

## 🧪 Testing

### Quick Test
```bash
# Test basic functionality
python run_pipeline.py

# Test CLI
python src/pipeline/cli.py test_context.txt test_claim.txt test_output.json
```

### Comprehensive Test
```bash
# Test with examples
python example_usage.py
```

## 📈 Migration from Original

### Original Usage
```python
from beam_graph_filter_pipeline import BeamGraphFilterPipeline
pipeline = BeamGraphFilterPipeline()
results = pipeline.run_pipeline(context, claim)
```

### Refactored Usage
```python
from src.pipeline.beam_filter_pipeline import BeamFilterPipeline
pipeline = BeamFilterPipeline()
results = pipeline.run_pipeline(context, claim)
```

### CLI Migration
```bash
# Original
python beam_graph_filter_pipeline.py context.txt claim.txt output.json

# Refactored
python src/pipeline/cli.py context.txt claim.txt output.json
```

## 🔗 Dependencies

### Required
- `networkx`: Graph data structure
- `typing`: Type hints
- `argparse`: Command-line argument parsing
- `mint.beam_search`: Beam search algorithm (existing)
- `advanced_data_filtering`: Filter system (existing)

### Optional
- `py_vncorenlp`: Vietnamese NLP toolkit
- `vncorenlp/`: VnCoreNLP models directory
- `sentence-transformers`: For SBERT filtering
- `transformers`: For NLI and contradiction detection

## 📝 Contributing

### Adding New Features
1. **Create new module** in appropriate `src/` directory
2. **Update documentation** in `docs/` directory
3. **Add tests** for new functionality
4. **Update main pipeline** if needed

### Code Style
- Follow existing module structure
- Use type hints
- Include comprehensive error handling
- Add documentation for new functions

## 📞 Support

### Documentation Issues
- Check relevant module documentation
- Review [README_ORIGINAL_FILES.md](README_ORIGINAL_FILES.md) for migration help
- Consult [README_REFACTORED.md](../README_REFACTORED.md) for architecture overview

### Technical Issues
- Check troubleshooting sections in module documentation
- Verify dependencies are installed
- Test with simple examples first

### Performance Issues
- Review performance guidelines in [README_MAIN_SCRIPTS.md](README_MAIN_SCRIPTS.md)
- Adjust configuration parameters
- Consider using original files for advanced features

---

**Last Updated**: December 2024  
**Version**: 2.0 (Refactored)  
**Maintainer**: Project Team 