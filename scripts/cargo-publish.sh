#!/usr/bin/env bash
# Automatically prepares manifests by stripping unpublished git dependencies (hiroz).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

DRY_RUN=0
ALLOW_DIRTY="--allow-dirty"
EXTRA_ARGS=()

for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN=1
        EXTRA_ARGS+=("--dry-run")
    else
        EXTRA_ARGS+=("$arg")
    fi
done

echo "==> Preparing workspace manifests for crates.io..."

# Ensure we restore Cargo.toml files upon exit/interruption
cleanup() {
    echo "==> Restoring Cargo.toml files..."
    git checkout Cargo.toml prelio_io/Cargo.toml pelorus_cli/Cargo.toml 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Strip unpublished git dependencies (hiroz) from Cargo manifests
python3 - <<'EOF'
def strip_hiroz(content):
    lines = content.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("transforms_io =") and "git" in stripped:
            out.append('transforms_io = "0.3"')
            bracket_depth = line.count("[") - line.count("]")
            brace_depth = line.count("{") - line.count("}")
            while (bracket_depth > 0 or brace_depth > 0) and i + 1 < len(lines):
                i += 1
                line = lines[i]
                bracket_depth += line.count("[") - line.count("]")
                brace_depth += line.count("{") - line.count("}")
            i += 1
            continue

        if any(
            stripped.startswith(prefix)
            for prefix in [
                "hiroz =",
                "hiroz-msgs =",
                "hiroz-protocol =",
                "transforms_io_hiroz =",
                "offline-hiroz =",
                "\"transforms_io_hiroz",
                "\"dep:transforms_io_hiroz",
                "\"dep:hiroz",
                "\"prelio_io/hiroz",
                "\"ros_pointcloud2/hiroz",
            ]
        ):
            bracket_depth = line.count("[") - line.count("]")
            brace_depth = line.count("{") - line.count("}")
            while (bracket_depth > 0 or brace_depth > 0) and i + 1 < len(lines):
                i += 1
                line = lines[i]
                bracket_depth += line.count("[") - line.count("]")
                brace_depth += line.count("{") - line.count("}")
            i += 1
            continue

        out.append(line)
        i += 1

    return "\n".join(out) + "\n"

for path in ["prelio_io/Cargo.toml", "pelorus_cli/Cargo.toml", "Cargo.toml"]:
    with open(path, "r") as f:
        c = f.read()
    c = strip_hiroz(c)
    with open(path, "w") as f:
        f.write(c)
EOF

echo "==> Manifests stripped of unpublished git dependencies."

# Topological publishing order for Pelorus workspace
PACKAGES=(
    "pelorus_core"
    "pelorus_protocol"
    "pelorus_drivers"
    "pelorus_cuda"
    "pelorus_metal"
    "pelorus_vulkan"
    "spole"
    "prelio"
    "pelorus_reg"
    "prelio_io"
    "pelorus"
    "pelorus_cli"
)

echo "==> Publishing ${#PACKAGES[@]} crates in order:"
for pkg in "${PACKAGES[@]}"; do
    echo "    - $pkg"
done
echo ""

for pkg in "${PACKAGES[@]}"; do
    echo "=================================================================="
    echo "==> Publishing $pkg..."
    echo "=================================================================="

    CMD=(cargo publish -p "$pkg" $ALLOW_DIRTY "${EXTRA_ARGS[@]}")
    echo "Running: ${CMD[*]}"

    # Execute publish and capture output to handle already published crates
    set +e
    output=$("${CMD[@]}" 2>&1)
    status=$?
    set -e

    echo "$output"

    if [ $status -ne 0 ]; then
        if echo "$output" | grep -iqE "already exists|already uploaded|already published"; then
            echo "--> Package $pkg is already published at this version. Continuing."
        else
            echo "ERROR: Failed to publish $pkg"
            exit $status
        fi
    else
        echo "==> Successfully published $pkg"
        # When actually publishing, wait for the sparse/registry index to update before downstream crates compile against it
        if [ $DRY_RUN -eq 0 ]; then
            echo "--> Waiting 20s for crates.io index to update..."
            sleep 20
        fi
    fi
    echo ""
done

echo "=================================================================="
echo "==> All ${#PACKAGES[@]} crates published successfully!"
echo "=================================================================="
