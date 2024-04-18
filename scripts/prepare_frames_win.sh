#!/bin/bash
set -e

./scripts/extract_frames_win.sh
python util/generate_frames.py

# etc.
