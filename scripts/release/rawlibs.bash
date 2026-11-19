#!/usr/bin/env bash
# Release build step, run by .forgejo/workflows/release.yml from the repository root.
set -eo pipefail

set -euo pipefail
ulimit -n 65536
apt-get install -y cmake
VERSION=$(toml get --raw Cargo.toml workspace.package.version)
VERSION=$VERSION-1
rustup self update
rustup toolchain uninstall stable
rustup toolchain install stable
rustup default stable
rustc --version
mkdir -p "dist/${ARCH}"

# ------- RAW
cargo build -p pelorus_c --release
mv target/release/libpelorus.so dist/${ARCH}/
mv target/release/libpelorus.a  dist/${ARCH}/

cargo build -p pelorus_cli --features rerun --release
mv target/release/pelorus dist/${ARCH}/

# ------- DEB
# -- lib
updatetoml=pelorus_c/Cargo.toml
DEB_NAME="libpelorus"
LIB_DEB=$DEB_NAME
tmp=$(mktemp)
toml set "$updatetoml" 'package.metadata.deb.name' "$DEB_NAME" >"$tmp"
mv "$tmp" "$updatetoml"

cargo deb --package pelorus_c

# -- dev
DEB_NAME="libpelorus-dev"
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
cmake -S pelorus_c -B pelorus_c/build -DCMAKE_INSTALL_PREFIX=/usr -DPELORUS_BUILD_SHARED=OFF -DPELORUS_BUILD_STATIC=OFF
cp pelorus_c/build/pelorus.pc pelorus_c/deb-dev/pelorus.pc
cp pelorus_c/build/pelorusConfig.cmake pelorus_c/deb-dev/pelorusConfig.cmake
cp pelorus_c/build/pelorusConfigVersion.cmake pelorus_c/deb-dev/pelorusConfigVersion.cmake

cargo deb --package libpelorus-dev

# -- bin
DEB_NAME="pelorus"
updatetoml=pelorus_cli/Cargo.toml
tmp=$(mktemp)
toml set "$updatetoml" 'package.metadata.deb.name' "$DEB_NAME" >"$tmp"
mv "$tmp" "$updatetoml"

cargo deb --package pelorus_cli --features rerun

# prepare upload
mv target/debian/*.deb dist/${ARCH}/
