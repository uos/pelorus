#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/api-regression-tests.bash [--skip-apt]

Run the same setup sequence as the .gitlab-ci.yml examples-regression job:
  1. install CI system dependencies with apt-get, unless --skip-apt is used
  2. select the stable Rust toolchain
  3. configure, build, and install pelorus_c with CMake
  4. install the Python bindings with maturin develop
  5. run make regression in pelorus_c/examples

Set PELORUS_SKIP_APT=1 to skip apt-get without passing --skip-apt.
EOF
}

skip_apt="${PELORUS_SKIP_APT:-0}"

while (($#)); do
    case "$1" in
        --skip-apt)
            skip_apt=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

sudo_cmd=()
if ((EUID != 0)); then
    if command -v sudo >/dev/null 2>&1; then
        sudo_cmd=(sudo)
    else
        sudo_cmd=()
    fi
fi

install_apt_deps() {
    if [[ "$skip_apt" == "1" ]]; then
        echo "Skipping apt dependency install"
        return
    fi

    if ! command -v apt-get >/dev/null 2>&1; then
        echo "apt-get not found; skipping CI dependency install"
        return
    fi

    if ((EUID != 0)) && ((${#sudo_cmd[@]} == 0)); then
        echo "apt-get dependency install needs root or sudo; rerun as root or use --skip-apt" >&2
        exit 1
    fi

    "${sudo_cmd[@]}" apt-get update
    "${sudo_cmd[@]}" apt-get install --no-install-recommends -y \
        build-essential \
        cmake \
        libpcl-dev \
        libeigen3-dev \
        pkg-config
}

run_rustup_stable() {
    if command -v rustup >/dev/null 2>&1; then
        rustup default stable
    else
        echo "rustup not found; continuing with current Rust toolchain"
    fi
}

build_and_install_pelorus_c() {
    cd "$repo_root/pelorus_c"
    cmake -S . -B build
    cmake --build build -- -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"

    if ((EUID == 0)); then
        cmake --install build
    elif ((${#sudo_cmd[@]} > 0)); then
        "${sudo_cmd[@]}" cmake --install build
    else
        echo "installing pelorus_c needs root or sudo; rerun as root" >&2
        exit 1
    fi
}

install_python_bindings() {
    cd "$repo_root/pelorus_py"
    maturin develop
}

run_regression() {
    make -C "$repo_root/pelorus_c/examples" regression
}

install_apt_deps
run_rustup_stable
build_and_install_pelorus_c
install_python_bindings
run_regression
