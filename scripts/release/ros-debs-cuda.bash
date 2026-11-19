#!/usr/bin/env bash
# Release build step, run by .forgejo/workflows/release.yml from the repository root.
set -eo pipefail

ulimit -n 65536
for distro in lyrical jazzy humble; do
if [ -f "/opt/ros/$distro/setup.bash" ]; then
  echo "Sourcing ROS distro: $distro"
  . "/opt/ros/$distro/setup.bash"
  break
fi
done

if [ -z "${ROS_DISTRO:-}" ]; then
  echo "No supported ROS distro found" >&2
  exit 1
fi

rustup self update
rustup toolchain uninstall stable
rustup toolchain install stable
rustup default stable
rustc --version

# mark ROS package with CUDA suffix so the generated .deb is variant-specific
export CUDA_PART="$CUDA"
sed -i "s|<name>pelorus</name>|<name>pelorus-cuda${CUDA_PART}</name>|" package.xml
sed -i "s|project(pelorus)|project(pelorus-cuda${CUDA_PART})|" CMakeLists.txt
sed -i "s/^\[features\]/[[bin]]\nname = \"pelorus-cuda${CUDA_PART}\"\npath = \"src\/main.rs\"\n\n[features]/" pelorus_cli/Cargo.toml

export PELORUS_CUDA=1
bash ./scripts/make-deb.bash
