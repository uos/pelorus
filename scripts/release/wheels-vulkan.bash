#!/usr/bin/env bash
# Release build step, run by .forgejo/workflows/release.yml from the repository root.
set -eo pipefail

set -euo pipefail
ulimit -n 65536
apt-get update
apt-get install -y cmake libvulkan-dev patchelf
rustup self update
rustup toolchain uninstall stable
rustup toolchain install stable
rustup default stable
rustc --version
rustup target add "$TARGET"
maturin build --manifest-path pelorus_cli/Cargo.toml --release --features vulkan,rerun --target "$TARGET" --skip-auditwheel
maturin build --manifest-path pelorus_py/Cargo.toml --release --features vulkan --target "$TARGET" --skip-auditwheel
