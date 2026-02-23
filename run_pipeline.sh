#!/bin/bash

# Activate virtual environment
uv sync
source .venv/bin/activate

echo "Starting image processing pipeline..."
echo ""

echo "Step 1: Processing images with inpainting..."
python3 scripts/main.py

echo ""
echo "Step 2: Classifying processed images..."
echo ""

python3 scripts/classify.py --data results | tail -1 # tail -1 to only print the final accuracy
