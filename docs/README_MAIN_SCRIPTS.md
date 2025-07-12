# 🚀 Main Scripts Documentation

## Overview
The main scripts provide easy-to-use interfaces for running the beam filter pipeline.

## Files

### `run_pipeline.py`

**Purpose**: Simple script to run the beam filter pipeline with default settings

**Usage**:
```bash
python run_pipeline.py
```

**Features**:
- Uses default configuration
- Processes sample data
- Prints results to console
- No command-line arguments required

**Example Output**:
```
🔧 Beam Filter Pipeline - Default Run
=====================================

📝 Input:
Context: [context text]
Claim: [claim text]

🔄 Running pipeline...
✅ Pipeline completed successfully

📊 Results:
- Found relevant sentences
- Graph built successfully
- Beam search completed
- Filtering applied

🎯 Final Sentences:
1. [relevant sentence 1]
2. [relevant sentence 2]
```

**Configuration**:
```python
# Default configuration used
config = {
    "beam_width": 40,
    "max_depth": 120,
    "max_paths": 200,
    "min_relevance_score": 0.15,
    "max_final_sentences": 30,
    "use_sbert": False,
    "use_contradiction_detection": False,
    "use_nli": False,
    "enable_pos_filtering": True
}
```

**Sample Data**:
```python
context = """[context text with multiple sentences]"""

claim = "[claim text]"
```

### `example_usage.py`

**Purpose**: Comprehensive example showing different ways to use the pipeline

**Usage**:
```bash
python example_usage.py
```

**Features**:
- Multiple usage examples
- Different configuration options
- Error handling demonstrations
- Performance comparisons

#### Examples Included

**1. Basic Usage**:
```python
from src.pipeline.beam_filter_pipeline import BeamFilterPipeline

# Create pipeline with default settings
pipeline = BeamFilterPipeline()

# Run pipeline
results = pipeline.run_pipeline(context, claim)
print(f"Found {len(results['final_sentences'])} relevant sentences")
```

**2. Custom Configuration**:
```python
# Custom configuration for better performance
config = {
    "beam_width": 50,
    "max_depth": 100,
    "min_relevance_score": 0.2,
    "max_final_sentences": 20,
    "use_sbert": True
}

pipeline = BeamFilterPipeline(config)
results = pipeline.run_pipeline(context, claim)
```

**3. Error Handling**:
```python
try:
    results = pipeline.run_pipeline(context, claim)
    print("✅ Pipeline completed successfully")
except Exception as e:
    print(f"❌ Pipeline failed: {e}")
```

**4. Performance Comparison**:
```python
# Compare different configurations
configs = [
    {"name": "Fast", "beam_width": 20, "max_depth": 60},
    {"name": "Balanced", "beam_width": 40, "max_depth": 120},
    {"name": "Thorough", "beam_width": 60, "max_depth": 150}
]

for config in configs:
    start_time = time.time()
    pipeline = BeamFilterPipeline(config)
    results = pipeline.run_pipeline(context, claim)
    end_time = time.time()
    
    print(f"{config['name']}: {len(results['final_sentences'])} sentences, {end_time - start_time:.2f}s")
```

**5. Batch Processing**:
```python
# Process multiple context-claim pairs
test_cases = [
    {
        "context": "[context text 1]",
        "claim": "[claim text 1]"
    },
    {
        "context": "[context text 2]",
        "claim": "[claim text 2]"
    }
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n--- Test Case {i} ---")
    results = pipeline.run_pipeline(test_case["context"], test_case["claim"])
    print(f"Found {len(results['final_sentences'])} relevant sentences")
```

## CLI Usage

### Basic CLI Usage

**Using the CLI script**:
```bash
# Create input files
echo "[context text]" > context.txt
echo "[claim text]" > claim.txt

# Run CLI
python src/pipeline/cli.py context.txt claim.txt output.json

# View results
cat output.json
```

**With custom parameters**:
```bash
python src/pipeline/cli.py \
    --beam-width 50 \
    --max-depth 100 \
    --min-relevance 0.2 \
    --max-final-sentences 20 \
    context.txt claim.txt output.json
```

**With advanced filtering**:
```bash
python src/pipeline/cli.py \
    --use-sbert \
    --use-contradiction-detection \
    --use-nli \
    context.txt claim.txt output.json
```

## Output Formats

### Console Output (run_pipeline.py)
```
🔧 Beam Filter Pipeline - Default Run
=====================================

📝 Input:
Context: [context text]
Claim: [claim text]

🔄 Running pipeline...
✅ Pipeline completed successfully

📊 Results:
- Found relevant sentences
- Graph built successfully
- Beam search completed
- Filtering applied

🎯 Final Sentences:
1. [sentence 1]
2. [sentence 2]
...
```

### JSON Output (CLI)
```json
{
    "context": "Original context text",
    "claim": "Original claim text",
    "context_sentences": ["Sentence 1", "Sentence 2"],
    "candidate_sentences": [
        {
            "sentence": "Candidate sentence",
            "score": 0.85,
            "path": ["claim_0", "word_1", "sentence_0"]
        }
    ],
    "final_sentences": [
        {
            "sentence": "Final sentence",
            "relevance_score": 0.92
        }
    ],
    "pipeline_stats": {
        "total_time": 1.5,
        "graph_nodes": 50,
        "graph_edges": 100,
        "beam_paths_found": 20,
        "sentences_filtered": 5
    },
    "graph_stats": {
        "word_nodes": 30,
        "sentence_nodes": 2,
        "dependency_edges": 60
    },
    "filter_stats": {
        "input_count": 15,
        "output_count": 2,
        "filter_time": 1.2
    }
}
```

## Performance Guidelines

### For Small Texts (< 500 words)
```python
config = {
    "beam_width": 30,
    "max_depth": 80,
    "max_paths": 100,
    "min_relevance_score": 0.2,
    "max_final_sentences": 15
}
```

### For Medium Texts (500-2000 words)
```python
config = {
    "beam_width": 40,
    "max_depth": 120,
    "max_paths": 200,
    "min_relevance_score": 0.15,
    "max_final_sentences": 30
}
```

### For Large Texts (> 2000 words)
```python
config = {
    "beam_width": 50,
    "max_depth": 150,
    "max_paths": 300,
    "min_relevance_score": 0.1,
    "max_final_sentences": 50
}
```

## Error Handling

### Common Error Scenarios

**1. VnCoreNLP not available**:
```
⚠️ VnCoreNLP not available, using fallback text processing
```

**2. AdvancedDataFilter not found**:
```
⚠️ AdvancedDataFilter not available, using fallback filtering
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

**4. File not found (CLI)**:
```bash
# Check file existence
ls -la context.txt claim.txt
```

## Testing

### Quick Test
```bash
# Test basic functionality
python run_pipeline.py

# Test CLI
python src/pipeline/cli.py test_context.txt test_claim.txt test_output.json
```

### Comprehensive Test
```bash
# Test with example usage
python example_usage.py
```

## Dependencies

### Required
- All modules in `src/` directory
- `mint.beam_search` (existing)
- `advanced_data_filtering` (existing)

### Optional
- `vncorenlp/` directory with VnCoreNLP models
- `py_vncorenlp` package

## Troubleshooting

### Performance Issues
1. **Reduce beam width and depth**
2. **Disable advanced filtering features**
3. **Use smaller max_final_sentences**

### Memory Issues
1. **Reduce max_paths**
2. **Use smaller beam_width**
3. **Process smaller text chunks**

### Accuracy Issues
1. **Increase min_relevance_score**
2. **Enable SBERT filtering**
3. **Use larger beam_width and max_depth**

## Best Practices

1. **Start with default settings** for new use cases
2. **Adjust parameters based on text size** and complexity
3. **Use CLI for batch processing** and automation
4. **Monitor performance** and adjust accordingly
5. **Test with representative data** before production use 