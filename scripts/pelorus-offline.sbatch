#!/usr/bin/env bash
#SBATCH --job-name=pelorus-offline
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8

set -euo pipefail

# Submit examples:
#   sbatch scripts/pelorus-offline.sbatch
#   sbatch --gres=gpu:1 --export=ALL,BACKEND=cuda,INSTALL_METHOD=spack,SPACK_SPEC='pelorus+cuda' scripts/pelorus-offline.sbatch
#   sbatch --export=ALL,INSTALL_METHOD=pip,PIP_PACKAGE='pelorus-cli' scripts/pelorus-offline.sbatch
#   sbatch --export=ALL,BAGFILE=/data/bags/run1,CONFIG=pelorus-velodyne.rl scripts/pelorus-offline.sbatch
#
# Optional environment variables:
#   BACKEND=cpu|cuda             Backend to run. Default: cpu.
#   INSTALL_METHOD=path|spack|pip
#                                How to make `pelorus` available. Default: path.
#   PELORUS_BIN=<path-or-name>    Pelorus executable. Default: pelorus.
#   SPACK_SPEC=<spec>             Spec for `spack load`. Default: pelorus or pelorus+cuda.
#   PIP_PACKAGE=<requirement>     Wheel-only pip requirement. Default: pelorus-cli.
#   BAGFILE=<path-or-tag>        Local bagfile path or marina tag.
#   CONFIG=<path>                Ratslang config file. Default: pelorus-velodyne.rl.
#   OUTDIR=<path>                Output directory. Default: slurm_runs/$SLURM_JOB_ID.
#   MINOT_PATH=<path>            Minot executable path. Default: minot.
#   MODULES="<names>"            Optional environment modules to load before running.
#   PRE_COMMAND="<command>"      Optional setup command, e.g. source a cluster env script.

if [[ -n "${MODULES:-}" ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh 2>/dev/null || true
    if type module >/dev/null 2>&1; then
        module load ${MODULES}
    else
        echo "MODULES was set, but the environment module command is unavailable." >&2
        exit 2
    fi
fi

if [[ -n "${PRE_COMMAND:-}" ]]; then
    eval "$PRE_COMMAND"
fi

BACKEND="${BACKEND:-cpu}"
INSTALL_METHOD="${INSTALL_METHOD:-path}"
PELORUS_BIN="${PELORUS_BIN:-pelorus}"
BAGFILE="${BAGFILE:-wachsbleiche:hunter:parking_25}"
CONFIG="${CONFIG:-pelorus-velodyne.rl}"
MINOT_PATH="${MINOT_PATH:-minot}"
OUTDIR="${OUTDIR:-slurm_runs/${SLURM_JOB_ID:-manual}}"

case "$BACKEND" in
    cpu)
        CPU_ONLY_ARGS=(--cpu-only)
        ;;
    cuda)
        CPU_ONLY_ARGS=()
        ;;
    *)
        echo "Unsupported BACKEND '$BACKEND'. Use cpu or cuda." >&2
        exit 2
        ;;
esac

mkdir -p "$OUTDIR" slurm_logs

echo "Pelorus Slurm job"
echo "  job:       ${SLURM_JOB_ID:-manual}"
echo "  host:      $(hostname)"
echo "  backend:   $BACKEND"
echo "  install:   $INSTALL_METHOD"
echo "  bagfile:   $BAGFILE"
echo "  config:    $CONFIG"
echo "  outdir:    $OUTDIR"
echo "  cpus:      ${SLURM_CPUS_PER_TASK:-unset}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Configuration file not found: $CONFIG" >&2
    exit 2
fi

if ! command -v "$MINOT_PATH" >/dev/null 2>&1 && [[ ! -x "$MINOT_PATH" ]]; then
    echo "Minot executable not found: $MINOT_PATH" >&2
    exit 2
fi

case "$INSTALL_METHOD" in
    path)
        ;;
    spack)
        if ! command -v spack >/dev/null 2>&1; then
            echo "spack is not available. Load your site Spack environment first with MODULES or PRE_COMMAND." >&2
            exit 2
        fi
        if [[ -z "${SPACK_SPEC:-}" ]]; then
            if [[ "$BACKEND" == "cuda" ]]; then
                SPACK_SPEC="pelorus+cuda"
            else
                SPACK_SPEC="pelorus~cuda"
            fi
        fi
        echo "Loading Spack spec: $SPACK_SPEC"
        spack load "$SPACK_SPEC"
        ;;
    pip)
        if ! command -v python3 >/dev/null 2>&1; then
            echo "python3 is not available for pip installation." >&2
            exit 2
        fi
        PIP_PACKAGE="${PIP_PACKAGE:-pelorus-cli}"
        if [[ "$BACKEND" == "cuda" && "$PIP_PACKAGE" != *cuda* && "$PIP_PACKAGE" != *cu* ]]; then
            echo "BACKEND=cuda with INSTALL_METHOD=pip requires a CUDA-enabled wheel package." >&2
            echo "Set PIP_PACKAGE to the CUDA wheel requirement published for this cluster, e.g. pelorus-cli-cuda." >&2
            exit 2
        fi
        echo "Installing wheel-only pip package: $PIP_PACKAGE"
        python3 -m pip install --user --only-binary=:all: "$PIP_PACKAGE"
        export PATH="$(python3 -m site --user-base)/bin:$PATH"
        ;;
    *)
        echo "Unsupported INSTALL_METHOD '$INSTALL_METHOD'. Use path, spack, or pip." >&2
        exit 2
        ;;
esac

if ! command -v "$PELORUS_BIN" >/dev/null 2>&1 && [[ ! -x "$PELORUS_BIN" ]]; then
    echo "Pelorus executable not found: $PELORUS_BIN" >&2
    echo "Use INSTALL_METHOD=spack, INSTALL_METHOD=pip, or set PELORUS_BIN to an installed binary." >&2
    exit 2
fi

export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${RAYON_NUM_THREADS:-0}}"
export LC_ALL=C

TRAJECTORY="$OUTDIR/pelorus_${BACKEND}_${SLURM_JOB_ID:-manual}.tum"
RUN_CMD=(
    "$PELORUS_BIN"
    offline
    -c "$CONFIG"
    --minot-path "$MINOT_PATH"
    "$BAGFILE"
    --export-trajectory "$TRAJECTORY"
    "${CPU_ONLY_ARGS[@]}"
)

echo "Running: ${RUN_CMD[*]}"
if command -v srun >/dev/null 2>&1; then
    srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" "${RUN_CMD[@]}"
else
    "${RUN_CMD[@]}"
fi

echo "Trajectory written to: $TRAJECTORY"
