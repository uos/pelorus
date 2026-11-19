#!/usr/bin/env bash
# Release build step, run by .forgejo/workflows/release.yml from the repository root.
set -eo pipefail

# Install libpelorus
VERSION=$(toml get --raw Cargo.toml workspace.package.version)
VERSION=$VERSION-1
dpkg -i "./deb-artifacts/$ARCH/libpelorus_${VERSION}_${ARCH}.deb"
dpkg -i "./deb-artifacts/$ARCH/libpelorus-dev_${VERSION}_${ARCH}.deb"

# Build
ulimit -n 65536
VERSION=$(toml get --raw Cargo.toml workspace.package.version)
VERSION=$VERSION-1
mkdir -p dist/${ARCH}
rustup self update
rustup toolchain uninstall stable
rustup toolchain install stable
rustup default stable
rustc --version

# - set name
updatetoml=glim_integration/pelorus_glim_odometry/Cargo.toml
DEB_NAME="libpelorus_glim_odometry"
tmp=$(mktemp)
toml set "$updatetoml" 'package.metadata.deb.name' "$DEB_NAME" >"$tmp"
mv "$tmp" "$updatetoml"

WD=$(pwd)
BUILD_DIR="$WD/glim_integration/pelorus_glim_odometry/build"
mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake ..
make -j2
cd $WD
cp "$BUILD_DIR/libodometry_estimation_pelorus.so" dist/${ARCH}/

# ---- CPU
DEB_NAME="libpelorus_glim_odometry"
LIB_DEB="libpelorus"

tmp=$(mktemp)
toml set "$updatetoml" 'package.metadata.deb.name' "$DEB_NAME" >"$tmp"
mv "$tmp" "$updatetoml"
tmp=$(mktemp)
toml set "$updatetoml" 'package.metadata.deb.depends' "${LIB_DEB} ( = ${VERSION} )" >"$tmp"
mv "$tmp" "$updatetoml"
cargo deb --package pelorus_glim_odometry --no-build
mv target/debian/*.deb dist/${ARCH}/

# ---- CUDA 12.6
DEB_NAME="libpelorus_glim_odometry-cuda12-6"
LIB_DEB="libpelorus-cuda12-6"

tmp=$(mktemp)
toml set "$updatetoml" 'package.metadata.deb.name' "$DEB_NAME" >"$tmp"
mv "$tmp" "$updatetoml"
tmp=$(mktemp)
toml set "$updatetoml" 'package.metadata.deb.depends' "${LIB_DEB} ( = ${VERSION} )" >"$tmp"
mv "$tmp" "$updatetoml"

cargo deb --package pelorus_glim_odometry --no-build
mv target/debian/*.deb dist/${ARCH}/

# ---- CUDA 12.8
# GLIM upstream supports CUDA 12.6 and 12.8, so the plugin is only
# offered against those two libpelorus variants even though libpelorus
# itself is also built for 13.1.
DEB_NAME="libpelorus_glim_odometry-cuda12-8"
LIB_DEB="libpelorus-cuda12-8"

tmp=$(mktemp)
toml set "$updatetoml" 'package.metadata.deb.name' "$DEB_NAME" >"$tmp"
mv "$tmp" "$updatetoml"
tmp=$(mktemp)
toml set "$updatetoml" 'package.metadata.deb.depends' "${LIB_DEB} ( = ${VERSION} )" >"$tmp"
mv "$tmp" "$updatetoml"

cargo deb --package pelorus_glim_odometry --no-build
mv target/debian/*.deb dist/${ARCH}/

# ---- Vulkan
DEB_NAME="libpelorus_glim_odometry-vulkan"
LIB_DEB="libpelorus-vulkan"

tmp=$(mktemp)
toml set "$updatetoml" 'package.metadata.deb.name' "$DEB_NAME" >"$tmp"
mv "$tmp" "$updatetoml"
tmp=$(mktemp)
toml set "$updatetoml" 'package.metadata.deb.depends' "${LIB_DEB} ( = ${VERSION} )" >"$tmp"
mv "$tmp" "$updatetoml"

cargo deb --package pelorus_glim_odometry --no-build
mv target/debian/*.deb dist/${ARCH}/
