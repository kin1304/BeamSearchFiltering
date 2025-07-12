#!/bin/bash

# 🧠 VnCoreNLP Auto Setup Script
# This script automatically downloads and sets up VnCoreNLP

set -e  # Exit on any error

echo "🧠 VnCoreNLP Auto Setup Script"
echo "================================"

# Check Java installation
echo "🔍 Checking Java installation..."
if ! command -v java &> /dev/null; then
    echo "❌ Java not found!"
    echo "Please install Java first:"
    echo "  macOS: brew install openjdk@11"
    echo "  Ubuntu: sudo apt install openjdk-11-jdk"
    echo "  Windows: Download from https://www.oracle.com/java/technologies/downloads/"
    exit 1
fi

echo "✅ Java found: $(java -version 2>&1 | head -n 1)"

# Create vncorenlp directory
echo "📁 Creating vncorenlp directory..."
mkdir -p vncorenlp
cd vncorenlp

# Download VnCoreNLP JAR file
echo "⬇️ Downloading VnCoreNLP JAR file..."
if [ ! -f "VnCoreNLP-1.2.jar" ]; then
    wget https://github.com/vncorenlp/VnCoreNLP/raw/master/VnCoreNLP-1.2.jar
    echo "✅ VnCoreNLP JAR downloaded"
else
    echo "✅ VnCoreNLP JAR already exists"
fi

# Create models directory
echo "📁 Creating models directory..."
mkdir -p models
cd models

# Download models
echo "⬇️ Downloading models..."

# Word segmentation
mkdir -p wordsegmenter
cd wordsegmenter
wget -O vi-vocab https://github.com/vncorenlp/VnCoreNLP/raw/master/models/wordsegmenter/vi-vocab
wget -O wordsegmenter.rdr https://github.com/vncorenlp/VnCoreNLP/raw/master/models/wordsegmenter/wordsegmenter.rdr
cd ..

# POS tagging
mkdir -p postagger
cd postagger
wget -O vi-tagger https://github.com/vncorenlp/VnCoreNLP/raw/master/models/postagger/vi-tagger
wget -O postagger.rdr https://github.com/vncorenlp/VnCoreNLP/raw/master/models/postagger/postagger.rdr
cd ..

# NER
mkdir -p ner
cd ner
wget -O vi-ner https://github.com/vncorenlp/VnCoreNLP/raw/master/models/ner/vi-ner
wget -O ner.rdr https://github.com/vncorenlp/VnCoreNLP/raw/master/models/ner/ner.rdr
cd ..

# Dependency parsing
mkdir -p dep
cd dep
wget -O vi-dep https://github.com/vncorenlp/VnCoreNLP/raw/master/models/dep/vi-dep
wget -O dependency.rdr https://github.com/vncorenlp/VnCoreNLP/raw/master/models/dep/dependency.rdr
cd ..

cd ../..

# Test VnCoreNLP
echo "🧪 Testing VnCoreNLP installation..."
echo "SAWACO thông báo tạm ngưng cung cấp nước." > test.txt

if java -jar vncorenlp/VnCoreNLP-1.2.jar -annotators wseg,pos,ner,parse -file test.txt > /dev/null 2>&1; then
    echo "✅ VnCoreNLP test successful!"
else
    echo "❌ VnCoreNLP test failed!"
    echo "Please check the installation manually"
fi

# Clean up test file
rm -f test.txt

echo ""
echo "🎉 VnCoreNLP setup completed!"
echo "📁 Directory structure:"
echo "   vncorenlp/"
echo "   ├── VnCoreNLP-1.2.jar"
echo "   └── models/"
echo "       ├── wordsegmenter/"
echo "       ├── postagger/"
echo "       ├── ner/"
echo "       └── dep/"
echo ""
echo "📚 Next steps:"
echo "   1. Install Python package: pip install py_vncorenlp"
echo "   2. Test in Python: python -c \"from py_vncorenlp import py_vncorenlp; print('✅ VnCoreNLP ready!')\""
echo "   3. Run pipeline: python run_pipeline.py" 