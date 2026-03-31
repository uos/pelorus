#!/usr/bin/env bash
set -euo pipefail

# Use C locale so decimal parsing/formatting is stable across systems.
export LC_ALL=C

# Script: compare-backends-offline.bash
# Purpose: CPU vs Metal trajectory equivalence testing on real bagfile data
# Usage: ./scripts/compare-backends-offline.bash <bagfile-specifier> [config-file]
# Example: ./scripts/compare-backends-offline.bash "$(marina resolve wachsbleiche:hunter:parking_25)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
STRICT_MODE="${STRICT_MODE:-true}"
BAGFILE="${1:?Usage: $0 <bagfile-path> [config-file]}"
CONFIG_FILE="${2:-pelorus.rl}"
RUN_ID=$(date +%s)_$RANDOM
OUTPUT_DIR="trajectory_comparison_${RUN_ID}"

# Threshold configuration (strict mode as per user preference)
APE_ROT_THRESHOLD_DEG=0.05
APE_TRANS_THRESHOLD_M=0.01
RPE_ROT_THRESHOLD_DEG=0.05
RPE_TRANS_THRESHOLD_M=0.01

# Optional: Umeyama scale warning (not hard failure)
UMEYAMA_SCALE_MIN=0.95
UMEYAMA_SCALE_MAX=1.05

# Trajectory filenames
TRAJ_CPU="${OUTPUT_DIR}/pelorus_traj_cpu_${RUN_ID}.tum"
TRAJ_METAL="${OUTPUT_DIR}/pelorus_traj_metal_${RUN_ID}.tum"
TRAJ_CUDA="${OUTPUT_DIR}/pelorus_traj_cuda_${RUN_ID}.tum"
COMPARISON_JSON_METAL="${OUTPUT_DIR}/comparison_metrics_metal_vs_cpu_${RUN_ID}.json"
COMPARISON_JSON_CUDA="${OUTPUT_DIR}/comparison_metrics_cuda_vs_cpu_${RUN_ID}.json"

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

fail() {
    log_error "$*"
    exit 1
}

is_macos() {
    [[ "$(uname -s)" == "Darwin" ]]
}

has_nvidia_hardware() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi -L &> /dev/null && return 0
    fi

    if [[ -d "/proc/driver/nvidia/gpus" ]] && compgen -G "/proc/driver/nvidia/gpus/*" > /dev/null; then
        return 0
    fi

    return 1
}

float_gt() {
    local lhs="$1" rhs="$2"
    awk -v a="$lhs" -v b="$rhs" 'BEGIN { exit !(a > b) }'
}

float_outside_range() {
    local x="$1" min="$2" max="$3"
    awk -v x="$x" -v min="$min" -v max="$max" 'BEGIN { exit !((x < min) || (x > max)) }'
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check sounding availability
    if ! command -v sounding &> /dev/null; then
        fail "sounding not found in PATH. Install via: cargo install sounding"
    fi
    
    # Check marina availability
    if ! command -v marina &> /dev/null; then
        fail "marina not found in PATH"
    fi

    # Check jq availability
    if ! command -v jq &> /dev/null; then
        fail "jq not found in PATH"
    fi

    # Check bagfile exists
    if [[ ! -d "$BAGFILE" ]]; then
        fail "Bagfile not found at: $BAGFILE"
    fi
    
    # Check config file exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        fail "Configuration file not found at: $CONFIG_FILE"
    fi
    
    log_info "All prerequisites met"
}

setup_output_dir() {
    mkdir -p "$OUTPUT_DIR"
    log_info "Output directory: $OUTPUT_DIR"
}

run_backend_test() {
    local backend_name="$1"
    local feature_flag="$2"
    local output_file="$3"

    # Metal backend is supported only on macOS.
    if [[ "$feature_flag" == "metal" ]] && ! is_macos; then
        fail "Metal backend can only be run on macOS"
    fi
    
    log_info "Running Pelorus $backend_name backend..."
    
    local cmd=(cargo run --release --bin pelorus)
    if [[ -n "$feature_flag" ]]; then
        cmd+=(--features "$feature_flag")
    fi
    cmd+=(-- offline -c "$CONFIG_FILE" "$BAGFILE" --export-trajectory "$output_file")
    
    # macOS compatibility: use gtimeout if available, otherwise skip timeout
    local timeout_cmd=""
    if command -v timeout &> /dev/null; then
        timeout_cmd="timeout 900"
    elif command -v gtimeout &> /dev/null; then
        timeout_cmd="gtimeout 900"
    else
        log_warn "timeout/gtimeout not found; running without timeout (may block indefinitely on hang)"
    fi
    
    local exit_code=0
    local run_log="${OUTPUT_DIR}/${backend_name,,}_run.log"
    : > "$run_log"
    if [[ -n "$timeout_cmd" ]]; then
        # Pipe through tee so pelorus sees non-interactive stdout and skips TUI mode.
        $timeout_cmd "${cmd[@]}" 2>&1 | tee -a "$run_log" || exit_code=${PIPESTATUS[0]}
    else
        "${cmd[@]}" 2>&1 | tee -a "$run_log" || exit_code=${PIPESTATUS[0]}
    fi
    
    # Check for timeout exit code (124 from timeout/gtimeout)
    if [[ $exit_code -eq 124 ]]; then
        fail "$backend_name run timed out after 900 seconds"
    elif [[ $exit_code -ne 0 ]]; then
        fail "$backend_name run failed with exit code $exit_code"
    fi
    
    # Verify output file was created
    if [[ ! -f "$output_file" ]]; then
        fail "Trajectory file not created: $output_file"
    fi
    
    local line_count=$(wc -l < "$output_file")
    log_info "$backend_name trajectory: $output_file ($line_count poses)"
    local first_ts last_ts
    first_ts=$(awk 'NF{print $1; exit}' "$output_file")
    last_ts=$(awk 'NF{t=$1} END{print t}' "$output_file")
    log_info "$backend_name trajectory span: ${first_ts:-n/a} -> ${last_ts:-n/a}"
    if grep -qi "reached end of bagfile" "$run_log"; then
        log_info "$backend_name stop reason: reached end of bagfile"
    else
        log_warn "$backend_name stop reason: end-of-bag marker not detected in log"
    fi
}

compare_trajectories() {
    local lhs_label="$1"
    local lhs_traj="$2"
    local rhs_label="$3"
    local rhs_traj="$4"
    local comparison_json="$5"

    log_info "Comparing trajectories with sounding: ${lhs_label} vs ${rhs_label}..."
    
    local lhs_lines rhs_lines
    lhs_lines=$(wc -l < "$lhs_traj")
    rhs_lines=$(wc -l < "$rhs_traj")
    if (( lhs_lines == 0 || rhs_lines == 0 )); then
        fail "Cannot compare empty trajectories"
    fi
    local ratio
    ratio=$(awk -v a="$lhs_lines" -v b="$rhs_lines" 'BEGIN{if (a>b) print a/b; else print b/a}')
    if awk -v r="$ratio" 'BEGIN{exit !(r > 1.02)}'; then
        fail "Trajectory length mismatch too large (${lhs_label}=${lhs_lines}, ${rhs_label}=${rhs_lines})"
    fi

    # Run sounding and capture output (on non-TTY this is JSON)
    local sounding_output
    sounding_output=$(sounding trajectory "$lhs_traj" "$rhs_traj" --json) || {
        fail "sounding comparison failed"
    }
    
    # Save JSON output
    echo "$sounding_output" > "$comparison_json"
    log_info "Metrics saved to: $comparison_json"
    
    # Extract metrics using jq
    local ape_trans_rmse ape_rot_rmse_deg rpe_trans_rmse rpe_rot_rmse_deg umeyama_scale
    
    ape_trans_rmse=$(echo "$sounding_output" | jq -r '.ape_trans_rmse // empty')
    ape_rot_rmse_deg=$(echo "$sounding_output" | jq -r '.ape_rot_rmse_deg // empty')
    rpe_trans_rmse=$(echo "$sounding_output" | jq -r '.rpe_trans_rmse // empty')
    rpe_rot_rmse_deg=$(echo "$sounding_output" | jq -r '.rpe_rot_rmse_deg // empty')
    umeyama_scale=$(echo "$sounding_output" | jq -r '.umeyama_scale // empty')
    
    if [[ -z "$ape_trans_rmse" ]] || [[ -z "$ape_rot_rmse_deg" ]] || [[ -z "$rpe_trans_rmse" ]] || [[ -z "$rpe_rot_rmse_deg" ]]; then
        log_error "Failed to parse sounding output"
        log_error "Output was: $sounding_output"
        fail "sounding metrics extraction failed"
    fi
    
    # Pretty-print metrics
    log_info "========== EQUIVALENCE METRICS =========="
    printf "  APE Translational RMSE: %.4f m (threshold: %.4f m)\n" "$ape_trans_rmse" "$APE_TRANS_THRESHOLD_M"
    printf "  APE Rotational RMSE:    %.4f deg (threshold: %.4f deg)\n" "$ape_rot_rmse_deg" "$APE_ROT_THRESHOLD_DEG"
    printf "  RPE Translational RMSE: %.4f m (threshold: %.4f m)\n" "$rpe_trans_rmse" "$RPE_TRANS_THRESHOLD_M"
    printf "  RPE Rotational RMSE:    %.4f deg (threshold: %.4f deg)\n" "$rpe_rot_rmse_deg" "$RPE_ROT_THRESHOLD_DEG"
    printf "  Umeyama Scale:          %.4f (expected ~1.0)\n" "$umeyama_scale"
    log_info "=========================================="
    
    # Check thresholds
    local all_pass=true
    
    if float_gt "$ape_trans_rmse" "$APE_TRANS_THRESHOLD_M"; then
        log_error "APE Translational RMSE exceeds threshold: $ape_trans_rmse > $APE_TRANS_THRESHOLD_M"
        all_pass=false
    fi
    
    if float_gt "$ape_rot_rmse_deg" "$APE_ROT_THRESHOLD_DEG"; then
        log_error "APE Rotational RMSE exceeds threshold: $ape_rot_rmse_deg > $APE_ROT_THRESHOLD_DEG"
        all_pass=false
    fi
    
    if float_gt "$rpe_trans_rmse" "$RPE_TRANS_THRESHOLD_M"; then
        log_error "RPE Translational RMSE exceeds threshold: $rpe_trans_rmse > $RPE_TRANS_THRESHOLD_M"
        all_pass=false
    fi
    
    if float_gt "$rpe_rot_rmse_deg" "$RPE_ROT_THRESHOLD_DEG"; then
        log_error "RPE Rotational RMSE exceeds threshold: $rpe_rot_rmse_deg > $RPE_ROT_THRESHOLD_DEG"
        all_pass=false
    fi
    
    # Warn on Umeyama scale
    if float_outside_range "$umeyama_scale" "$UMEYAMA_SCALE_MIN" "$UMEYAMA_SCALE_MAX"; then
        log_warn "Umeyama scale outside expected range [${UMEYAMA_SCALE_MIN}, ${UMEYAMA_SCALE_MAX}]: $umeyama_scale"
    fi
    
    return $([ "$all_pass" = true ] && echo 0 || echo 1)
}

main() {
    log_info "=========================================="
    log_info "CPU vs Metal Trajectory Equivalence Test"
    log_info "=========================================="
    log_info "Bagfile: $BAGFILE"
    log_info "Config:  $CONFIG_FILE"
    log_info "Run ID:  $RUN_ID"
    
    check_prerequisites
    setup_output_dir

    local ran_gpu_comparison=false
    
    # Run CPU backend (no features)
    run_backend_test "CPU" "" "$TRAJ_CPU"
    
    # Run Metal backend (macOS only)
    if is_macos; then
        run_backend_test "Metal" "metal" "$TRAJ_METAL"

        if ! compare_trajectories "Metal" "$TRAJ_METAL" "CPU" "$TRAJ_CPU" "$COMPARISON_JSON_METAL"; then
            log_error "Trajectory comparison FAILED (Metal vs CPU)"
            log_error "Equivalence thresholds were exceeded"
            log_info "Trajectories and metrics saved to: $OUTPUT_DIR"
            exit 1
        fi

        ran_gpu_comparison=true
    else
        log_warn "Skipping Metal test (macOS only feature)"
    fi

    # Run CUDA backend only if NVIDIA hardware is present
    if has_nvidia_hardware; then
        run_backend_test "CUDA" "cuda" "$TRAJ_CUDA"

        if ! compare_trajectories "CUDA" "$TRAJ_CUDA" "CPU" "$TRAJ_CPU" "$COMPARISON_JSON_CUDA"; then
            log_error "Trajectory comparison FAILED (CUDA vs CPU)"
            log_error "Equivalence thresholds were exceeded"
            log_info "Trajectories and metrics saved to: $OUTPUT_DIR"
            exit 1
        fi

        ran_gpu_comparison=true
    else
        log_warn "Skipping CUDA test (no NVIDIA hardware detected)"
    fi

    if [[ "$ran_gpu_comparison" != "true" ]]; then
        log_warn "No GPU backend comparison was executed"
        log_info "Outputs saved to $OUTPUT_DIR"
        exit 0
    fi
    
    log_info "Trajectory comparison PASSED"
    log_info "Outputs saved to: $OUTPUT_DIR"
    exit 0
}

# Trap errors for cleanup
trap 'log_error "Script interrupted"; exit 130' INT TERM

main "$@"
