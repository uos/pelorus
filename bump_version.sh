#!/usr/bin/env bash
# Usage: ./bump_version.sh <new-version>
# Example: ./bump_version.sh 0.1.0

set -euo pipefail

NEW_VERSION="${1:-}"
if [[ -z "$NEW_VERSION" ]]; then
    echo "Usage: $0 <new-version>"
    echo "Example: $0 0.1.0"
    exit 1
fi

if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9._-]+)?(\+[a-zA-Z0-9._-]+)?$ ]]; then
    echo "Error: version must be semver format (e.g. 1.2.3 or 1.2.3-alpha.1)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect current version from the workspace Cargo.toml (for logging only)
DETECTED_VERSION="$(grep -m1 '^version = ' Cargo.toml | sed 's/version = "\(.*\)"/\1/')"
echo "Bumping to: $NEW_VERSION (detected current: $DETECTED_VERSION)"

# Portable sed -i
sed_i() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i "" "$@"
    else
        sed -i "$@"
    fi
}

# Update Cargo.toml (workspace version)
sed_i "s/^version = \".*\"/version = \"${NEW_VERSION}\"/" Cargo.toml

# Update workspace member versions in Cargo.toml
sed_i -E "/path = \"[^\"]+\"/s/version = \"[^\"]+\"/version = \"${NEW_VERSION}\"/" Cargo.toml

# Update package.xml
sed_i -E "s|<version>.*</version>|<version>${NEW_VERSION}</version>|" package.xml

# Update pelorus_c/CMakeLists.txt
sed_i -E "s/(project\([a-zA-Z_]+ VERSION )[0-9]+\.[0-9]+\.[0-9]+/\1${NEW_VERSION}/" \
    pelorus_c/CMakeLists.txt

# Update PKGBUILD
sed_i "s/^pkgver=.*/pkgver=${NEW_VERSION}/" \
    packaging/arch-bin/PKGBUILD
sed_i "s/^pkgrel=.*/pkgrel=1/" \
    packaging/arch-bin/PKGBUILD

# Synchronize minot dependency version constraints across packaging manifests
python3 scripts/check-packaging.py --fix

echo ""
echo "Updated files:"
echo "  Cargo.toml"
echo "  package.xml"
echo "  pelorus_c/CMakeLists.txt"
echo "  packaging/arch-bin/PKGBUILD"
echo "  pelorus_cli/Cargo.toml (deb/rpm metadata)"
echo "  pelorus_cli/pyproject.toml"

ROS_IMAGE="docker.io/stelzo/ubuntu-dev:jazzy"
# Use a separate CARGO_HOME for the container to avoid overwriting host shims (macOS vs Linux)
CARGO_HOME_CONTAINER="/tmp/cargo-container-cache"

if command -v docker &>/dev/null; then
    CONTAINER_RUNTIME="docker"
    RUNTIME_FLAGS=""
    MOUNT_SUFFIX=""
elif command -v podman &>/dev/null; then
    CONTAINER_RUNTIME="podman"
    RUNTIME_FLAGS="--userns=keep-id"
    MOUNT_SUFFIX=":z"
else
    echo "Error: neither podman nor docker found" >&2
    exit 1
fi

echo ""
echo "── Re-creating Cargo.lock inside ${ROS_IMAGE} (via ${CONTAINER_RUNTIME}) ────────────────────────"
${CONTAINER_RUNTIME} run --rm \
    --network=host \
    ${RUNTIME_FLAGS} \
    -v "${SCRIPT_DIR}:/workspace${MOUNT_SUFFIX}" \
    -e CARGO_HOME="${CARGO_HOME_CONTAINER}" \
    -w /workspace \
    "${ROS_IMAGE}" \
    bash -c ". /opt/ros/jazzy/setup.bash && rustup default stable && cargo update"

echo ""
echo "── git diff ──────────────────────────────────────────────────────────"
git diff
