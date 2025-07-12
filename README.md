# 🔍 Beam Search Filter Pipeline

> **Comprehensive Vietnamese text processing framework for fact-checking and information retrieval using beam search and advanced filtering techniques**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/Docs-Complete-brightgreen.svg)](docs/README_INDEX.md)

## 📋 **Overview**

Beam Search Filter Pipeline là một framework toàn diện cho việc xử lý văn bản tiếng Việt, tập trung vào fact-checking và information retrieval. Framework sử dụng beam search algorithm kết hợp với advanced filtering techniques để tìm và lọc các câu liên quan từ context dựa trên claim.

### **Core Capabilities**

* 🔍 **Vietnamese NLP**: VnCoreNLP integration cho text annotation
* 🕸️ **Graph-based Search**: NetworkX-based text graph với beam search
* 🔄 **Advanced Filtering**: SBERT, contradiction detection, NLI stance detection
* 🏗️ **Modular Design**: Clean, maintainable, và extensible architecture
* 🖥️ **CLI Interface**: Easy-to-use command-line interface
* 🐍 **Python API**: Flexible programming interface

## 🚀 **Quick Start**

### **Task 1: Basic Pipeline Usage**

```bash
# Run with default settings
python run_pipeline.py

# Use CLI interface
python src/pipeline/cli.py context.txt claim.txt output.json
```

### **Task 2: Custom Configuration**

```python
from src.pipeline.beam_filter_pipeline import BeamFilterPipeline

# Custom configuration
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

### **Task 3: Advanced Filtering**

```bash
# Enable advanced filtering features
python src/pipeline/cli.py \
    --use-sbert \
    --use-contradiction-detection \
    --use-nli \
    context.txt claim.txt output.json
```

### **Task 4: Output Files**

Pipeline tạo ra **3 loại output file**:

```bash
# Run pipeline
python run_pipeline.py --input raw_test.json --min_relevance 0.15

# Output files generated:
# • raw_test_beam_filtered_0.15_20241201_143022_detailed.json
# • raw_test_beam_filtered_0.15_20241201_143022_simple.json  
# • raw_test_beam_filtered_0.15_20241201_143022_stats.json
```

**File Types:**
- **Detailed**: Thông tin chi tiết về quá trình xử lý
- **Simple**: Chỉ có danh sách evidence sentences
- **Stats**: Thống kê tổng quan về processing

## 📊 **Performance Guidelines**

| Text Size | Beam Width | Max Depth | Min Relevance | Max Sentences | Expected Time |
|-----------|------------|-----------|---------------|---------------|---------------|
| **Small** (< 500 words) | 30 | 80 | 0.2 | 15 | ~30s |
| **Medium** (500-2000 words) | 40 | 120 | 0.15 | 30 | ~60s |
| **Large** (> 2000 words) | 50 | 150 | 0.1 | 50 | ~120s |

## 🏗️ **Project Architecture**

### **Core Components**

```
src/                           # Refactored modules
├── utils/                     # Text preprocessing utilities
│   └── text_preprocessor.py   # Text cleaning & sentence splitting
├── nlp/                       # NLP wrapper
│   └── vncorenlp_wrapper.py  # VnCoreNLP integration
├── graph/                     # Simplified graph
│   └── text_graph_simple.py  # TextGraph for pipeline
├── filtering/                 # Filter wrapper
│   └── filter_wrapper.py     # AdvancedDataFilter wrapper
└── pipeline/                  # Main pipeline
    ├── beam_filter_pipeline.py # Main pipeline class
    └── cli.py                # CLI interface

mint/                          # Original modules (preserved)
├── text_graph.py             # Full-featured graph
├── beam_search.py            # Beam search algorithm
└── helpers.py                # Helper functions

advanced_data_filtering.py     # Advanced filtering system
```

### **Workflow Architecture**

```
Input Text → Preprocessing → NLP Annotation → Graph Building → Beam Search → Filtering → Output
     ↓              ↓              ↓              ↓              ↓           ↓
  Clean Text   VnCoreNLP    Text Graph    Path Finding   Relevance   Final Sentences
```

## 📚 **Comprehensive Documentation**

* **[📖 Complete Documentation](docs/README_INDEX.md)**: Full documentation index
* **[🏗️ Architecture Overview](README_REFACTORED.md)**: Project architecture details
* **[🧠 VnCoreNLP Setup](docs/VNCORENLP_SETUP.md)**: Vietnamese NLP installation guide
* **[🔧 Module Documentation](docs/)**: Individual module documentation
* **[📚 Original Files](docs/README_ORIGINAL_FILES.md)**: Migration and backward compatibility

## 🎯 **Use Cases**

### **Fact-Checking Pipeline**
```python
# Extract relevant sentences for fact-checking
pipeline = BeamFilterPipeline()
results = pipeline.run_pipeline(context, claim)
relevant_sentences = results["final_sentences"]
```

### **Information Retrieval**
```python
# Find sentences related to a specific claim
config = {"beam_width": 60, "max_depth": 150}
pipeline = BeamFilterPipeline(config)
results = pipeline.run_pipeline(context, query)
```

### **Text Analysis**
```python
# Analyze text structure and relationships
from src.graph.text_graph_simple import TextGraphSimple
graph = TextGraphSimple()
# ... build and analyze graph
```

### **Batch Processing**
```bash
# Process multiple files
for file in contexts/*.txt; do
    python src/pipeline/cli.py "$file" claims/$(basename "$file") output/$(basename "$file" .txt).json
done
```

## 🔧 **Configuration**

### **Environment Setup**

```bash
# Clone repository
git clone <repository-url>
cd BeamSearchFillter

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env and add your OpenAI API key if needed

# Optional: Install VnCoreNLP for Vietnamese processing
# See [VnCoreNLP Setup Guide](docs/VNCORENLP_SETUP.md)
# Or use auto setup: ./setup_vncorenlp.sh
```

### **Pipeline Configuration**

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

### **CLI Options**

```bash
python src/pipeline/cli.py \
    --beam-width 50 \
    --max-depth 100 \
    --min-relevance 0.2 \
    --max-final-sentences 20 \
    --use-sbert \
    context.txt claim.txt output.json
```

## 🛠️ **Advanced Features**

### **Implemented Capabilities**

* ✅ **Vietnamese NLP Processing**: Full VnCoreNLP integration
* ✅ **Graph-based Search**: NetworkX text graph với beam search
* ✅ **Multi-stage Filtering**: SBERT, contradiction, NLI filtering
* ✅ **Modular Architecture**: Clean separation of concerns
* ✅ **CLI Interface**: Complete command-line tools
* ✅ **Python API**: Flexible programming interface
* ✅ **Error Handling**: Comprehensive error handling và fallbacks
* ✅ **Documentation**: Complete documentation với examples

### **Research Applications**

* **Fact-checking Research**: Extract relevant evidence từ large texts
* **Information Retrieval**: Find related sentences cho specific queries
* **Text Analysis**: Analyze text structure và relationships
* **Vietnamese NLP**: Study Vietnamese text processing techniques

## 📈 **Output Formats**

Pipeline tạo ra **3 loại output file** với format tên file: `{input_name}_beam_filtered_{min_relevance}_{timestamp}_{type}.json`

### **1. Detailed Output** (`*_detailed.json`)
Chứa thông tin chi tiết về quá trình xử lý:
```json
[
  {
    "context": "Original context text",
    "claim": "Original claim text",
    "multi_level_evidence": [
      {
        "sentence": "Final sentence 1",
        "relevance_score": 0.92,
        "beam_score": 0.85,
        "path": ["claim_0", "word_1", "sentence_0"]
      }
    ],
    "statistics": {
      "beam": {
        "total_paths": 15,
        "unique_sentences": 25
      }
    }
  }
]
```

### **2. Simple Output** (`*_simple.json`)
Chứa kết quả đơn giản, chỉ có evidence sentences:
```json
[
  {
    "context": "Original context text",
    "claim": "Original claim text",
    "multi_level_evidence": [
      "Final sentence 1",
      "Final sentence 2",
      "Final sentence 3"
    ]
  }
]
```

### **3. Statistics Output** (`*_stats.json`)
Chứa thống kê tổng quan về quá trình xử lý:
```json
{
  "total_context_sentences": 6442,
  "total_beam_sentences": 4933,
  "total_final_sentences": 4933,
  "num_samples": 300,
  "beam_parameters": {
    "beam_width": 80,
    "max_depth": 300,
    "max_paths": 500,
    "beam_sentences": 400
  }
}
```

### **Console Output**
```
🔧 Beam Search Filter Pipeline - Default Run
=====================================

📝 Input:
Context: SAWACO thông báo tạm ngưng cung cấp nước từ 22 giờ đến 4 giờ.
Claim: SAWACO thông báo tạm ngưng cung cấp nước.

🔄 Running pipeline...
✅ Pipeline completed in 2.34s

📊 Results:
- Found 2 relevant sentences
- Graph: 45 nodes, 89 edges
- Beam search: 15 paths found
- Filtering: 15 → 2 sentences

🎯 Final Sentences:
1. SAWACO thông báo tạm ngưng cung cấp nước từ 22 giờ đến 4 giờ.
2. Các khu vực bị ảnh hưởng gồm quận 6, 8, 12.

✅ Done! Output saved to:
   • raw_test_beam_filtered_0.15_20241201_143022_detailed.json
   • raw_test_beam_filtered_0.15_20241201_143022_simple.json
   • raw_test_beam_filtered_0.15_20241201_143022_stats.json
```

## 🔗 **Dependencies**

### **Required**
- `networkx`: Graph data structure
- `typing`: Type hints
- `argparse`: Command-line argument parsing
- `mint.beam_search`: Beam search algorithm (existing)
- `advanced_data_filtering`: Filter system (existing)

### **Optional**
- `py_vncorenlp`: Vietnamese NLP toolkit
- `vncorenlp/`: VnCoreNLP models directory
- `sentence-transformers`: For SBERT filtering
- `transformers`: For NLI and contradiction detection

## 🧪 **Testing**

### **Quick Test**
```bash
# Test basic functionality
python run_pipeline.py

# Test CLI
python src/pipeline/cli.py test_context.txt test_claim.txt test_output.json
```

### **Comprehensive Test**
```bash
# Test with examples
python example_usage.py
```

## 🔍 **Troubleshooting**

### **Common Issues**

**1. VnCoreNLP not available:**
```bash
# Check VnCoreNLP installation
ls -la vncorenlp/
python -c "from py_vncorenlp import py_vncorenlp; print('VnCoreNLP available')"
```

**2. AdvancedDataFilter not found:**
```bash
# Check if file exists
ls -la advanced_data_filtering.py
```

**3. Memory issues:**
```python
# Reduce parameters
config = {
    "beam_width": 20,
    "max_depth": 60,
    "max_paths": 100,
    "max_final_sentences": 15
}
```

## 📝 **Contributing**

Beam Search Filter Pipeline welcomes contributions in:

* **New filtering methods**: Additional sentence filtering algorithms
* **Graph algorithms**: Novel beam search approaches
* **Vietnamese NLP**: Enhanced text processing techniques
* **Evaluation metrics**: Advanced relevance measures
* **Performance optimization**: More efficient processing patterns

### **Code Style**
- Follow existing module structure
- Use type hints
- Include comprehensive error handling
- Add documentation for new functions

## 📞 **Support**

* **Documentation**: [docs/README_INDEX.md](docs/README_INDEX.md)
* **Issues**: Check troubleshooting sections in module docs
* **Migration**: See [docs/README_ORIGINAL_FILES.md](docs/README_ORIGINAL_FILES.md)

## 📜 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

---

🚀 **Happy Vietnamese Text Processing!** Whether you're doing fact-checking research, information retrieval, or Vietnamese NLP development, Beam Search Filter Pipeline provides comprehensive tools for Vietnamese text intelligence research.

## About

> **Comprehensive Vietnamese text processing framework for fact-checking and information retrieval using beam search and advanced filtering techniques**

### Topics

- `nlp` - Natural Language Processing
- `vietnamese` - Vietnamese language processing
- `fact-checking` - Fact verification and validation
- `beam-search` - Graph-based search algorithms
- `text-filtering` - Advanced text filtering techniques

### Resources

- [📖 Complete Documentation](docs/README_INDEX.md)
- [🏗️ Architecture Overview](README_REFACTORED.md)
- [🧠 VnCoreNLP Setup](docs/VNCORENLP_SETUP.md)

### License

MIT license 