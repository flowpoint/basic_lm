#!/bin/bash
set -euo pipefail
#python -m venv data/venv
#source data/venv/bin/activate

pip install -U pip packaging wheel setuptools
pip install dataset transformers causal-conv1d

pushd && models/mambda && pip install . || popd
