#!/bin/bash

set -x
set -o pipefail

function create_venv() {
  python3 -m venv venv
  source venv/bin/activate
  pip3 install "maturin[zig]"
}

cargo build --release -p pelorus_c
source venv/bin/activate || create_venv
cd pelorus_py
maturin develop
cd ..
python3 pelorus_c/examples/compare_async_sync.py

