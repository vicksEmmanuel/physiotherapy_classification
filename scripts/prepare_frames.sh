#!/bin/bash
set -e

./scripts/extract_frames.sh
python util/generate_frames.py

# etc.
