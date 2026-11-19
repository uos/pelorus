#!/usr/bin/env bash
# Release build step, run by .forgejo/workflows/release.yml from the repository root.
set -eo pipefail

set -euo pipefail
ulimit -n 65536
apt-get install -y cmake
VERSION=$(toml get --raw Cargo.toml workspace.package.version)
rustup self update
rustup toolchain uninstall stable
rustup toolchain install stable
rustup default stable
rustc --version

mkdir -p "dist/${ARCH}"

# Parse cuda tag
IMAGE_TAG="${IMAGE##*:}"
CUDA_PART="${IMAGE_TAG#*-cu}"

# ------- RAW
cargo build -p pelorus_c --release --features cuda
mv target/release/libpelorus.so dist/${ARCH}/
mv target/release/libpelorus.a  dist/${ARCH}/

cargo build -p pelorus_cli --release --features cuda,rerun
mv target/release/pelorus dist/${ARCH}/

# ------- DEB
# -- lib
updatetoml=pelorus_c/Cargo.toml
DEB_NAME="libpelorus-cuda${CUDA_PART}"
LIB_DEB=$DEB_NAME
tmp=$(mktemp)
toml set "$updatetoml" 'package.metadata.deb.name' "$DEB_NAME" >"$tmp"
mv "$tmp" "$updatetoml"

cargo deb --package pelorus_c

# -- dev
DEB_NAME="libpelorus-cuda${CUDA_PART}-dev"
DEV_DEB=$DEB_NAME
updatetoml=pelorus_c/deb-dev/Cargo.toml
tmp=$(mktemp)
toml set "$updatetoml" 'package.metadata.deb.name' "$DEB_NAME" >"$tmp"
mv "$tmp" "$updatetoml"

# - pin dependencies
tmp=$(mktemp)
toml set "$updatetoml" 'package.metadata.deb.depends' "${LIB_DEB} ( = ${VERSION} )" >"$tmp"
mv "$tmp" "$updatetoml"

# Generate packaging artifacts with CMake and copy them into the dev crate
cmake -S pelorus_c -B pelorus_c/build -DCMAKE_INSTALL_PREFIX=/usr
cp pelorus_c/build/pelorus.pc pelorus_c/deb-dev/pelorus.pc
cp pelorus_c/build/pelorusConfig.cmake pelorus_c/deb-dev/pelorusConfig.cmake
cp pelorus_c/build/pelorusConfigVersion.cmake pelorus_c/deb-dev/pelorusConfigVersion.cmake

cargo deb --package libpelorus-dev

# -- bin
DEB_NAME="pelorus-cuda${CUDA_PART}"
updatetoml=pelorus_cli/Cargo.toml
tmp=$(mktemp)
toml set "$updatetoml" 'package.metadata.deb.name' "$DEB_NAME" >"$tmp"
mv "$tmp" "$updatetoml"

cargo deb --package pelorus_cli --features cuda,rerun

# prepare upload
mv target/debian/*.deb dist/${ARCH}/
