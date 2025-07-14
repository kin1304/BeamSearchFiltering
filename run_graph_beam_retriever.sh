#!/bin/bash

python src/pipeline/cli.py \
    --input data_test.json \
    --output_dir output_full_filter \
    --use-sbert \
    --use-contradiction \
    --use-nli \
    --beam-width 80 \
    --max-depth 300 \
    --max-paths 500 \
    --beam-sentences 400 