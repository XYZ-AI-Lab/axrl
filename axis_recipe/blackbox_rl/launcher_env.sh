#!/usr/bin/env bash

# Blackbox rollouts start many Ray actor and processor processes. Keep common
# native thread pools small so tokenizer/BLAS imports do not exhaust host threads.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# The E2B Python SDK defaults to 20 keepalive connections for its shared async
# HTTP/2 transport. Concurrent OpenHands sandboxes use many envd origins, so use
# a larger pool unless the operator has set a value explicitly.
export E2B_MAX_KEEPALIVE_CONNECTIONS="${E2B_MAX_KEEPALIVE_CONNECTIONS:-512}"
