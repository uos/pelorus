#!/bin/bash

set -x 
set -o pipefail

function create_venv() {
  python3 -m venv venv
  source venv/bin/activate
  pip3 install "maturin[zig]"
}

source venv/bin/activate || create_venv

MATURIN_ARGS="build --zig --manifest-path minot/Cargo.toml --target-dir wheels"

maturin $MATURIN_ARGS --target aarch64-unknown-linux-gnu 
maturin $MATURIN_ARGS --target x86_64-unknown-linux-gnu
