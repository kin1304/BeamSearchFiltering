# 🧠 VnCoreNLP Setup Guide

## Overview
VnCoreNLP là toolkit xử lý ngôn ngữ tự nhiên tiếng Việt, được sử dụng trong pipeline để:
- Word segmentation
- POS tagging  
- Named entity recognition
- Dependency parsing

## 📦 Installation

### Quick Setup (Recommended)

**Auto setup script:**
```bash
# Make script executable
chmod +x setup_vncorenlp.sh

# Run auto setup
./setup_vncorenlp.sh
```

**Manual setup:**
### Prerequisites

**1. Install Java (Required)**

VnCoreNLP cần Java để chạy. Kiểm tra Java:

```bash
java -version
```

Nếu chưa có Java, cài đặt:

**macOS:**
```bash
# Install with Homebrew
brew install openjdk@11

# Or download from Oracle
# https://www.oracle.com/java/technologies/downloads/
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install openjdk-11-jdk
```

**Windows:**
- Download từ: https://www.oracle.com/java/technologies/downloads/
- Hoặc sử dụng: https://adoptium.net/

**2. Verify Java Installation**
```bash
java -version
javac -version
echo $JAVA_HOME
```

### Method 1: Automatic Setup (Recommended)

```bash
# Tạo thư mục vncorenlp
mkdir vncorenlp
cd vncorenlp

# Download VnCoreNLP JAR file
wget https://github.com/vncorenlp/VnCoreNLP/raw/master/VnCoreNLP-1.2.jar

# Download models
mkdir models
cd models

# Word segmentation model
wget https://github.com/vncorenlp/VnCoreNLP/raw/master/models/wordsegmenter/vi-vocab
wget https://github.com/vncorenlp/VnCoreNLP/raw/master/models/wordsegmenter/wordsegmenter.rdr

# POS tagging model
wget https://github.com/vncorenlp/VnCoreNLP/raw/master/models/postagger/vi-tagger
wget https://github.com/vncorenlp/VnCoreNLP/raw/master/models/postagger/postagger.rdr

# NER model
wget https://github.com/vncorenlp/VnCoreNLP/raw/master/models/ner/vi-ner
wget https://github.com/vncorenlp/VnCoreNLP/raw/master/models/ner/ner.rdr

# Dependency parsing model
wget https://github.com/vncorenlp/VnCoreNLP/raw/master/models/dep/vi-dep
wget https://github.com/vncorenlp/VnCoreNLP/raw/master/models/dep/dependency.rdr

cd ..
```

### Method 2: Manual Download

1. **Download JAR file:**
   - Go to: https://github.com/vncorenlp/VnCoreNLP/releases
   - Download `VnCoreNLP-1.2.jar`

2. **Download models:**
   - Go to: https://github.com/vncorenlp/VnCoreNLP/tree/master/models
   - Download all files from each subdirectory:
     - `wordsegmenter/`
     - `postagger/`
     - `ner/`
     - `dep/`

3. **Organize files:**
   ```
   vncorenlp/
   ├── VnCoreNLP-1.2.jar
   └── models/
       ├── wordsegmenter/
       │   ├── vi-vocab
       │   └── wordsegmenter.rdr
       ├── postagger/
       │   ├── vi-tagger
       │   └── postagger.rdr
       ├── ner/
       │   ├── vi-ner
       │   └── ner.rdr
       └── dep/
           ├── vi-dep
           └── dependency.rdr
   ```

## 🔧 Configuration

### 1. Update Environment Variables

```bash
# Edit .env file
cp .env.example .env
nano .env

# Add VnCoreNLP path
DEFAULT_VNCORENLP_PATH=./vncorenlp
```

### 2. Install Python Dependencies

```bash
# Install py_vncorenlp
pip install py_vncorenlp

# Or install from source
git clone https://github.com/vncorenlp/VnCoreNLP.git
cd VnCoreNLP
pip install -e .
```

### 3. Test Java + VnCoreNLP Integration

```bash
# Test Java can run JAR file
cd vncorenlp
java -jar VnCoreNLP-1.2.jar -h

# Test with sample text
echo "SAWACO thông báo tạm ngưng cung cấp nước." > test.txt
java -jar VnCoreNLP-1.2.jar -annotators wseg,pos,ner,parse -file test.txt
```

## 🧪 Testing

### Test VnCoreNLP Installation

```python
# Test basic functionality
from py_vncorenlp import py_vncorenlp

# Initialize model
model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "ner", "parse"])

# Test text
text = "SAWACO thông báo tạm ngưng cung cấp nước."
result = model.annotate(text)
print(result)
```

### Test in Pipeline

```python
from src.nlp.vncorenlp_wrapper import VnCoreNLPWrapper

# Test wrapper
nlp = VnCoreNLPWrapper()
if nlp.is_available():
    print("✅ VnCoreNLP is working!")
else:
    print("❌ VnCoreNLP not available")
```

## 🚀 Usage in Pipeline

### Automatic Detection

Pipeline sẽ tự động detect VnCoreNLP:

```python
from src.pipeline.beam_filter_pipeline import BeamFilterPipeline

pipeline = BeamFilterPipeline()
# VnCoreNLP sẽ được sử dụng nếu có sẵn
results = pipeline.run_pipeline(context, claim)
```

### Manual Configuration

```python
# Specify VnCoreNLP path
import os
os.environ['VNCORENLP_DIR'] = './vncorenlp'

# Or in .env file
DEFAULT_VNCORENLP_PATH=./vncorenlp
```

## 🔍 Troubleshooting

### Common Issues

**1. Java not found:**
```bash
# Check Java installation
java -version

# If not found, install Java first
# macOS: brew install openjdk@11
# Ubuntu: sudo apt install openjdk-11-jdk
# Windows: Download from Oracle website
```

**2. VnCoreNLP not found:**
```bash
# Check if directory exists
ls -la vncorenlp/

# Check if JAR file exists
ls -la vncorenlp/VnCoreNLP-1.2.jar

# Check if models exist
ls -la vncorenlp/models/
```

**3. Java memory issues:**
```bash
# Increase Java heap memory
export JAVA_OPTS="-Xmx2g -Xms1g"

# Or set in .env file
JAVA_MEMORY="-Xmx2g -Xms1g"
```

**4. Model loading errors:**
```python
# Test individual components
from py_vncorenlp import py_vncorenlp

try:
    model = py_vncorenlp.VnCoreNLP(annotators=["wseg"])
    print("✅ Word segmentation OK")
except Exception as e:
    print(f"❌ Word segmentation failed: {e}")
    print("Check Java installation and JAR file")
```

**5. Memory issues:**
```python
# Use smaller models or reduce memory usage
import os
os.environ['VNCORENLP_MEMORY'] = '512m'

# Or set Java memory
os.environ['JAVA_OPTS'] = '-Xmx1g -Xms512m'
```

### Performance Optimization

**For large texts:**
```python
# Process in chunks
def process_large_text(text, chunk_size=1000):
    sentences = text.split('.')
    results = []
    for i in range(0, len(sentences), chunk_size):
        chunk = '. '.join(sentences[i:i+chunk_size])
        result = model.annotate(chunk)
        results.extend(result)
    return results
```

## 📊 File Structure

```
vncorenlp/
├── VnCoreNLP-1.2.jar          # Main JAR file (26MB)
└── models/
    ├── wordsegmenter/          # Word segmentation models
    │   ├── vi-vocab
    │   └── wordsegmenter.rdr
    ├── postagger/              # POS tagging models
    │   ├── vi-tagger
    │   └── postagger.rdr
    ├── ner/                    # NER models
    │   ├── vi-ner
    │   └── ner.rdr
    └── dep/                    # Dependency parsing models
        ├── vi-dep
        └── dependency.rdr
```

## 📝 Notes

- **Java requirement**: Java 8+ required
- **Total size**: ~140MB (JAR + models)
- **Download time**: ~5-10 minutes depending on connection
- **Memory usage**: ~512MB-1GB during processing (Java heap)
- **Processing speed**: ~100-500 sentences/second
- **Java heap**: Recommend 2GB+ for large texts

## 🔗 Resources

- **Official Repository**: https://github.com/vncorenlp/VnCoreNLP
- **Documentation**: https://github.com/vncorenlp/VnCoreNLP/wiki
- **Paper**: https://arxiv.org/abs/1803.06052 