#!/usr/bin/env bash
# Release build step, run by .forgejo/workflows/release.yml from the repository root.
set -eo pipefail

set -euo pipefail
ulimit -n 65536
apt-get update
apt-get install -y libvulkan-dev

for distro in lyrical jazzy humble; do
  if [ -f "/opt/ros/$distro/setup.bash" ]; then
    echo "Sourcing ROS distro: $distro"
    set +eu
    . "/opt/ros/$distro/setup.bash"
    set -eu
    export ROS_DISTRO="$distro"
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

sed -i 's|<name>pelorus</name>|<name>pelorus-vulkan</name>|' package.xml
sed -i 's|project(pelorus)|project(pelorus-vulkan)|' CMakeLists.txt
sed -i '/^\[features\]/i [[bin]]\nname = "pelorus-vulkan"\npath = "src/main.rs"\n' pelorus_cli/Cargo.toml

export PELORUS_VULKAN=1
bash ./scripts/make-deb.bash
