#!/usr/bin/env python3
"""
Main script to run the refactored beam graph filter pipeline
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline.cli import main

if __name__ == "__main__":
    main() 