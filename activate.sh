#!/bin/bash
# Auto-activate venv for this project
if [ -d "./venv" ]; then
    source ./venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ venv directory not found"
fi
