#!/usr/bin/env bash
# Release build step, run by .forgejo/workflows/release.yml from the repository root.
set -eo pipefail

set -euo pipefail
ulimit -n 65536
apt-get update
apt-get install -y cmake libvulkan-dev
VERSION=$(toml get --raw Cargo.toml workspace.package.version)
VERSION=$VERSION-1
rustup self update
rustup toolchain uninstall stable
rustup toolchain install stable
rustup default stable
rustc --version
mkdir -p "dist/${ARCH}"

# ------- RAW
cargo build -p pelorus_c --release --features vulkan
mv target/release/libpelorus.so dist/${ARCH}/
mv target/release/libpelorus.a dist/${ARCH}/

cargo build -p pelorus_cli --release --features vulkan,rerun
mv target/release/pelorus dist/${ARCH}/

# ------- DEB
# -- lib
updatetoml=pelorus_c/Cargo.toml
DEB_NAME="libpelorus-vulkan"
LIB_DEB=$DEB_NAME
tmp=$(mktemp)
toml set "$updatetoml" 'package.metadata.deb.name' "$DEB_NAME" >"$tmp"
mv "$tmp" "$updatetoml"
tmp=$(mktemp)
toml set "$updatetoml" 'package.metadata.deb.depends' 'libvulkan1' >"$tmp"
mv "$tmp" "$updatetoml"

cargo deb --package pelorus_c --features vulkan

# -- dev
DEB_NAME="libpelorus-vulkan-dev"
updatetoml=pelorus_c/deb-dev/Cargo.toml
tmp=$(mktemp)
toml set "$updatetoml" 'package.metadata.deb.name' "$DEB_NAME" >"$tmp"
mv "$tmp" "$updatetoml"
tmp=$(mktemp)
toml set "$updatetoml" 'package.metadata.deb.depends' "${LIB_DEB} ( = ${VERSION} )" >"$tmp"
mv "$tmp" "$updatetoml"

cmake -S pelorus_c -B pelorus_c/build -DCMAKE_INSTALL_PREFIX=/usr
cp pelorus_c/build/pelorus.pc pelorus_c/deb-dev/pelorus.pc
cp pelorus_c/build/pelorusConfig.cmake pelorus_c/deb-dev/pelorusConfig.cmake
cp pelorus_c/build/pelorusConfigVersion.cmake pelorus_c/deb-dev/pelorusConfigVersion.cmake

cargo deb --package libpelorus-dev

# -- bin
DEB_NAME="pelorus-vulkan"
updatetoml=pelorus_cli/Cargo.toml
tmp=$(mktemp)
toml set "$updatetoml" 'package.metadata.deb.name' "$DEB_NAME" >"$tmp"
mv "$tmp" "$updatetoml"
tmp=$(mktemp)
toml set "$updatetoml" 'package.metadata.deb.depends' \
  'minot (>= 0.15.2), minot (<< 0.16.0~), libvulkan1' >"$tmp"
mv "$tmp" "$updatetoml"

cargo deb --package pelorus_cli --features vulkan,rerun

mv target/debian/*.deb dist/${ARCH}/
