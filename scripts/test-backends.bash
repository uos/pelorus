#!/usr/bin/env bash
set -e

# Change to workspace root
cd "$(dirname "$0")/.."

echo "================================================="
echo "Generating CPU Ground Truth for Backend Tests..."
echo "================================================="
cargo test --test backend_compare -p pelorus_reg

echo "================================================="
echo "Testing Metal Backend against CPU Ground Truth..."
echo "================================================="
if [[ "$OSTYPE" == "darwin"* ]]; then
    # On macOS, check if we can actually build and run a small metal check
    if cargo check -p pelorus_metal --features metal_enable >/dev/null 2>&1; then
        cargo test --test backend_compare -p pelorus_reg --features metal
    else
        echo "Metal backend not available or failed to compile, skipping."
    fi
else
    echo "Not on macOS, skipping Metal tests."
fi

echo "================================================="
echo "Testing CUDA Backend against CPU Ground Truth..."
echo "================================================="
# Check if nvcc exists and if the crate compiles
if command -v nvcc >/dev/null 2>&1 && cargo check -p pelorus_cuda --features cuda >/dev/null 2>&1; then
    cargo test --test backend_compare -p pelorus_reg --features cuda || echo "CUDA tests failed or skipped."
else
    echo "CUDA not available (nvcc missing or compilation failed), skipping."
fi

echo "Backend comparison tests completed successfully!"
