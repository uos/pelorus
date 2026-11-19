#!/usr/bin/env bash
# Release build step, run by .forgejo/workflows/release.yml from the repository root.
set -eo pipefail

ulimit -n 65536
rustup self update
rustup toolchain uninstall stable
rustup toolchain install stable
rustup default stable
rustc --version

for distro in lyrical jazzy humble; do
if [ -f "/opt/ros/$distro/setup.bash" ]; then
  echo "Sourcing ROS distro: $distro"
  . "/opt/ros/$distro/setup.bash"
  export ROS_DISTRO="$distro"
  break
fi
done

if [ -z "${ROS_DISTRO:-}" ]; then
  echo "No supported ROS distro found" >&2
  exit 1
fi

bash ./scripts/make-deb.bash
