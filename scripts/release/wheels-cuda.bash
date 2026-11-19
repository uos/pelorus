#!/usr/bin/env bash
# Release build step, run by .forgejo/workflows/release.yml from the repository root.
set -eo pipefail

ulimit -n 65536
rustup self update
rustup toolchain uninstall stable
rustup toolchain install stable
rustup default stable
rustc --version
maturin build --manifest-path pelorus_cli/Cargo.toml --release --features cuda,rerun --target "$TARGET" --skip-auditwheel
maturin build --manifest-path pelorus_py/Cargo.toml --release --features cuda --target "$TARGET" --skip-auditwheel
