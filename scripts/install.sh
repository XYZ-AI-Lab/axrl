#!/usr/bin/env bash

set -euo pipefail

pip install -e .

# FIX for https://github.com/pytorch/pytorch/issues/168167
# pip install "nvidia-cudnn-cu12==9.17.1.4"
