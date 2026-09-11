#!/usr/bin/env bash
set -euo pipefail

# Use C locale so decimal parsing/formatting is stable across systems.
export LC_ALL=C

# Script: compare-backends-offline.bash
# Purpose: CPU vs selected accelerator trajectory equivalence on real bagfile data
# Usage:
#   ./scripts/compare-backends-offline.bash [--backends <list|auto>] <bagfile-path> [config-file]
# Example:
#   ./scripts/compare-backends-offline.bash --backends vulkan "$(marina resolve wachsbleiche:hunter:parking_25)" pelorus-velodyne.rl

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
STRICT_MODE="${STRICT_MODE:-true}"
BACKENDS_SPEC="auto"
BAGFILE=""
CONFIG_FILE="pelorus.rl"
RUN_ID=$(date +%s)_$RANDOM
OUTPUT_DIR="trajectory_comparison_${RUN_ID}"

# Threshold configuration (relaxed for cross-platform numerical precision)
APE_ROT_THRESHOLD_DEG=0.3
APE_TRANS_THRESHOLD_M=0.05
RPE_ROT_THRESHOLD_DEG=0.1
RPE_TRANS_THRESHOLD_M=0.02

# Optional: Umeyama scale warning (not hard failure)
UMEYAMA_SCALE_MIN=0.95
UMEYAMA_SCALE_MAX=1.05

# Trajectory filenames
TRAJ_CPU="${OUTPUT_DIR}/pelorus_traj_cpu_${RUN_ID}.tum"
TRAJ_RAYON="${OUTPUT_DIR}/pelorus_traj_rayon_${RUN_ID}.tum"
TRAJ_METAL="${OUTPUT_DIR}/pelorus_traj_metal_${RUN_ID}.tum"
TRAJ_CUDA="${OUTPUT_DIR}/pelorus_traj_cuda_${RUN_ID}.tum"
TRAJ_VULKAN="${OUTPUT_DIR}/pelorus_traj_vulkan_${RUN_ID}.tum"
COMPARISON_JSON_RAYON="${OUTPUT_DIR}/comparison_metrics_rayon_vs_cpu_${RUN_ID}.json"
COMPARISON_JSON_METAL="${OUTPUT_DIR}/comparison_metrics_metal_vs_cpu_${RUN_ID}.json"
COMPARISON_JSON_CUDA="${OUTPUT_DIR}/comparison_metrics_cuda_vs_cpu_${RUN_ID}.json"
COMPARISON_JSON_VULKAN="${OUTPUT_DIR}/comparison_metrics_vulkan_vs_cpu_${RUN_ID}.json"

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

parse_args() {
    local positional=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --backends)
                shift
                [[ $# -gt 0 ]] || fail "--backends requires a value"
                BACKENDS_SPEC="$1"
                ;;
            --help|-h)
                cat <<EOF
Usage: $0 [--backends <list|auto>] <bagfile-path> [config-file]

Options:
  --backends   Comma-separated list of backends to test against CPU.
               Supported values: rayon,cuda,metal,vulkan
               Special value: auto (default) -> run all available backends.
EOF
                exit 0
                ;;
            *)
                positional+=("$1")
                ;;
        esac
        shift
    done

    [[ ${#positional[@]} -ge 1 ]] || fail "Usage: $0 [--backends <list|auto>] <bagfile-path> [config-file]"
    BAGFILE="${positional[0]}"
    if [[ ${#positional[@]} -ge 2 ]]; then
        CONFIG_FILE="${positional[1]}"
    fi
}

is_valid_backend() {
    case "$1" in
        rayon|cuda|metal|vulkan) return 0 ;;
        *) return 1 ;;
    esac
}

has_vulkan_hardware() {
    local entry dir
    local manifest_files="${VK_DRIVER_FILES:-${VK_ICD_FILENAMES:-}}"

    # Explicit loader configuration takes precedence over standard discovery.
    if [[ -n "$manifest_files" ]]; then
        local IFS=':'
        local configured_manifests
        read -r -a configured_manifests <<< "$manifest_files"
        for entry in "${configured_manifests[@]}"; do
            [[ -f "$entry" ]] && return 0
        done
        return 1
    fi

    local data_dirs="${XDG_DATA_HOME:-${HOME:-}/.local/share}:${XDG_DATA_DIRS:-/usr/local/share:/usr/share}"
    local icd_dirs=(
        /etc/vulkan/icd.d
        /usr/local/etc/vulkan/icd.d
        /opt/homebrew/share/vulkan/icd.d
    )

    local IFS=':'
    local data_dir_list
    read -r -a data_dir_list <<< "$data_dirs"
    for dir in "${data_dir_list[@]}"; do
        [[ -n "$dir" ]] && icd_dirs+=("${dir%/}/vulkan/icd.d")
    done

    for dir in "${icd_dirs[@]}"; do
        compgen -G "$dir/*.json" > /dev/null && return 0
    done

    return 1
}

backend_supported_on_host() {
    case "$1" in
        rayon) return 0 ;;
        metal) is_macos ;;
        cuda) has_nvidia_hardware ;;
        vulkan) has_vulkan_hardware ;;
        *) return 1 ;;
    esac
}

backend_display_name() {
    case "$1" in
        rayon) echo "Rayon" ;;
        metal) echo "Metal" ;;
        cuda) echo "CUDA" ;;
        vulkan) echo "Vulkan" ;;
        *) echo "$1" ;;
    esac
}

resolve_requested_backends() {
    local spec="$1"
    
    local result=""
    
    if [[ "$spec" == "auto" ]]; then
        for b in rayon metal cuda vulkan; do
            if backend_supported_on_host "$b"; then
                if [[ -z "$result" ]]; then
                    result="$b"
                else
                    result="$result $b"
                fi
            fi
        done
        echo "$result"
        return 0
    fi
    
    local IFS=','
    local requested
    read -r -a requested <<< "$spec"
    for raw in "${requested[@]}"; do
        local b
        b="$(echo "$raw" | tr '[:upper:]' '[:lower:]' | xargs)"
        [[ -n "$b" ]] || continue
        is_valid_backend "$b" || fail "Unknown backend '$b' (supported: rayon, cuda, metal, vulkan)"
        # Deduplicate while preserving order
        if [[ " $result " != *" $b "* ]]; then
            if [[ -z "$result" ]]; then
                result="$b"
            else
                result="$result $b"
            fi
        fi
    done
    
    echo "$result"
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

# Route core dumps to a file we can actually reach. `coredumpctl` talks to the
# host journal, so inside a container (the CI runs under act) it never finds the
# core the crash produced and every SIGSEGV is thrown away undiagnosed.
# core_pattern is a host-wide kernel setting, so this only sticks when the
# container is privileged; when it does not, we fall back to coredumpctl.
CORE_DIR=""
SAVED_CORE_PATTERN=""

# core_pattern is global to the kernel, so a container that is privileged enough
# to set it is also changing it for everything else on the host. Put it back.
restore_core_pattern() {
    if [[ -n "$SAVED_CORE_PATTERN" ]]; then
        printf '%s\n' "$SAVED_CORE_PATTERN" > /proc/sys/kernel/core_pattern 2>/dev/null || true
        SAVED_CORE_PATTERN=""
    fi
}

setup_core_capture() {
    if [[ "${PELORUS_CAPTURE_CORES:-0}" != "1" ]]; then
        return
    fi

    ulimit -c unlimited 2>/dev/null || log_warn "Could not raise the core dump size limit"

    if [[ ! -w /proc/sys/kernel/core_pattern ]]; then
        log_warn "/proc/sys/kernel/core_pattern is not writable; relying on coredumpctl"
        return
    fi

    CORE_DIR="$(cd "$OUTPUT_DIR" && pwd)/cores"
    mkdir -p "$CORE_DIR"
    SAVED_CORE_PATTERN="$(cat /proc/sys/kernel/core_pattern 2>/dev/null || true)"
    # %e executable, %p pid. No pipe, so this works without a userspace helper.
    if echo "${CORE_DIR}/core.%e.%p" > /proc/sys/kernel/core_pattern 2>/dev/null; then
        log_info "Core dumps will be written to ${CORE_DIR}"
        trap restore_core_pattern EXIT
    else
        log_warn "Could not set core_pattern; relying on coredumpctl"
        CORE_DIR=""
        SAVED_CORE_PATTERN=""
    fi
}

# The Vulkan loader ignores a requested-but-missing layer without saying a word,
# so probe for the manifest before promising validation.
VULKAN_VALIDATION_MANIFEST=""
vulkan_validation_layer_available() {
    local d
    for d in "${VK_LAYER_PATH:-}" \
             /usr/share/vulkan/explicit_layer.d \
             /usr/local/share/vulkan/explicit_layer.d \
             /etc/vulkan/explicit_layer.d \
             "$HOME/.local/share/vulkan/explicit_layer.d"; do
        [[ -n "$d" && -d "$d" ]] || continue
        local m
        for m in "${d}"/*validation*.json; do
            [[ -e "$m" ]] || continue
            VULKAN_VALIDATION_MANIFEST="$m"
            return 0
        done
    done
    return 1
}

capture_core_backtrace() {
    local backend_name="$1"
    local run_started="$2"

    if [[ "${PELORUS_CAPTURE_CORES:-0}" != "1" ]]; then
        return
    fi
    if ! command -v gdb &> /dev/null; then
        log_warn "Cannot capture $backend_name backtrace: gdb is required"
        return
    fi

    local executable core_file
    executable="$(readlink -f target/release/pelorus)"
    core_file=""

    # Prefer a core the kernel dropped where we asked it to; coredumpctl cannot
    # see it from inside the container.
    if [[ -n "$CORE_DIR" ]]; then
        core_file="$(find "$CORE_DIR" -maxdepth 1 -name 'core.pelorus.*' -newermt "$run_started" \
            -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)"
    fi

    if [[ -z "$core_file" ]]; then
        if ! command -v coredumpctl &> /dev/null; then
            log_warn "No core dump under ${CORE_DIR:-<unset>} and coredumpctl is unavailable"
            return
        fi
        core_file="${OUTPUT_DIR}/pelorus_${backend_name,,}.core"
        log_info "No local core dump; trying coredumpctl..."
        if ! coredumpctl --quiet --since "$run_started" -1 --output "$core_file" dump "$executable"; then
            log_warn "No core dump for $executable was available through coredumpctl"
            return
        fi
    fi

    log_info "Extracting $backend_name backtrace from ${core_file}..."
    gdb -q -batch \
        -ex "set debuginfod enabled off" \
        -ex "set pagination off" \
        -ex "thread apply all backtrace full" \
        "$executable" "$core_file" || true
}

run_backend_test() {
    local backend_name="$1"
    local feature_flag="$2"
    local output_file="$3"
    local no_default_features="${4:-false}"

    # Metal backend is supported only on macOS.
    if [[ "$feature_flag" == "metal" ]] && ! is_macos; then
        fail "Metal backend can only be run on macOS"
    fi
    
    log_info "Running Pelorus $backend_name backend..."
    
    local -a cargo_flags=()
    if [[ "$no_default_features" == "true" ]]; then
        cargo_flags+=(--no-default-features)
    fi
    if [[ -n "$feature_flag" ]]; then
        cargo_flags+=(--features "$feature_flag")
    fi
    local -a pelorus_args=(offline -c "$CONFIG_FILE" "$BAGFILE" --export-trajectory "$output_file")

    # Build before the timed region below. `timeout` is there to catch a hung
    # run, and wrapping `cargo run` would let a cold target dir spend that
    # budget on compilation instead - a slow build and a deadlock then look
    # identical from the outside.
    log_info "Building $backend_name binary..."
    cargo build --release --bin pelorus ${cargo_flags[@]+"${cargo_flags[@]}"}
    local -a cmd=(target/release/pelorus "${pelorus_args[@]}")
    
    # macOS compatibility: use gtimeout if available, otherwise skip timeout
    local timeout_cmd=""
    if command -v timeout &> /dev/null; then
        timeout_cmd="timeout 900"
    elif command -v gtimeout &> /dev/null; then
        timeout_cmd="gtimeout 900"
    else
        log_warn "timeout/gtimeout not found; running without timeout (may block indefinitely on hang)"
    fi
    
    # Vulkan validation layers turn silent API misuse - use-after-free, missing
    # barriers, out-of-bounds descriptor access - into a message at the offending
    # call instead of a segfault somewhere later. Off by default because they
    # cost a lot of runtime; opt in with PELORUS_VULKAN_VALIDATION=1.
    local -a run_env=()
    if [[ "$feature_flag" == *vulkan* && "${PELORUS_VULKAN_VALIDATION:-0}" == "1" ]]; then
        # A layer the loader cannot find is ignored in silence, which is what
        # happened the first time this was switched on: the script announced
        # validation and nothing was actually validating. Check before claiming.
        if ! vulkan_validation_layer_available; then
            log_warn "PELORUS_VULKAN_VALIDATION=1 but VK_LAYER_KHRONOS_validation is not installed"
            log_warn "  install it with: apt-get install -y vulkan-validationlayers"
            log_warn "  running without validation"
        else
            log_info "Enabling Vulkan validation layers (incl. synchronization validation)"
            log_info "  layer manifest: ${VULKAN_VALIDATION_MANIFEST}"
            run_env=(
                env
                # the first is honoured by older loaders, the second by >= 1.3.234
                "VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation"
                "VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation"
                "VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT"
                # make the loader say so if it still cannot load the layer
                "VK_LOADER_DEBUG=warn"
            )
        fi
    fi

    local exit_code=0
    local backend_lower
    backend_lower="$(echo "$backend_name" | tr '[:upper:]' '[:lower:]')"
    local run_log="${OUTPUT_DIR}/${backend_lower}_run.log"
    local run_started
    # `date --iso-8601` is GNU-only; this format works on GNU and BSD date alike.
    run_started="$(date +"%Y-%m-%d %H:%M:%S")"
    : > "$run_log"
    if [[ -n "$timeout_cmd" ]]; then
        # Pipe through tee so pelorus sees non-interactive stdout and skips TUI mode.
        $timeout_cmd ${run_env[@]+"${run_env[@]}"} "${cmd[@]}" 2>&1 | tee -a "$run_log" || exit_code=${PIPESTATUS[0]}
    else
        ${run_env[@]+"${run_env[@]}"} "${cmd[@]}" 2>&1 | tee -a "$run_log" || exit_code=${PIPESTATUS[0]}
    fi
    
    # Check for timeout exit code (124 from timeout/gtimeout)
    if [[ $exit_code -eq 124 ]]; then
        fail "$backend_name run timed out after 900 seconds"
    elif [[ $exit_code -eq 139 ]]; then
        capture_core_backtrace "$backend_name" "$run_started"
        fail "$backend_name run failed with SIGSEGV (exit code 139)"
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
    lhs_lines=$(wc -l < "$lhs_traj" | tr -d ' ')
    rhs_lines=$(wc -l < "$rhs_traj" | tr -d ' ')
    if (( lhs_lines == 0 || rhs_lines == 0 )); then
        fail "Cannot compare empty trajectories"
    fi
    
    # Require exact match - no tolerance
    if (( lhs_lines != rhs_lines )); then
        fail "Trajectory length mismatch: ${lhs_label}=${lhs_lines} vs ${rhs_label}=${rhs_lines} (must be equal)"
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
    parse_args "$@"
    log_info "=========================================="
    log_info "CPU vs Selected Backend Equivalence Test"
    log_info "=========================================="
    log_info "Bagfile: $BAGFILE"
    log_info "Config:  $CONFIG_FILE"
    log_info "Run ID:  $RUN_ID"
    log_info "Backends: $BACKENDS_SPEC"
    
    check_prerequisites
    setup_output_dir
    setup_core_capture

local ran_gpu_comparison=false
local requested_backends_str
requested_backends_str="$(resolve_requested_backends "$BACKENDS_SPEC")"
local requested_backends
read -r -a requested_backends <<< "$requested_backends_str"

# Run CPU backend without rayon (ground truth for rayon equivalency checks)
    run_backend_test "CPU" "offline" "$TRAJ_CPU" "true"
    
    for backend in "${requested_backends[@]}"; do
        local backend_name
        backend_name="$(backend_display_name "$backend")"

        if ! backend_supported_on_host "$backend"; then
            log_warn "Skipping $backend_name test (unsupported on this host)"
            continue
        fi

        local traj_file comparison_json
        case "$backend" in
            rayon)
                traj_file="$TRAJ_RAYON"
                comparison_json="$COMPARISON_JSON_RAYON"
                ;;
            metal)
                traj_file="$TRAJ_METAL"
                comparison_json="$COMPARISON_JSON_METAL"
                ;;
            cuda)
                traj_file="$TRAJ_CUDA"
                comparison_json="$COMPARISON_JSON_CUDA"
                ;;
            vulkan)
                traj_file="$TRAJ_VULKAN"
                comparison_json="$COMPARISON_JSON_VULKAN"
                ;;
            *)
                fail "Unhandled backend: $backend"
                ;;
        esac

        local backend_features
        case "$backend" in
            rayon) backend_features="offline,rayon" ;;
            *) backend_features="offline,$backend" ;;
        esac

        run_backend_test "$backend_name" "$backend_features" "$traj_file" "true"
        if ! compare_trajectories "$backend_name" "$traj_file" "CPU" "$TRAJ_CPU" "$comparison_json"; then
            log_error "Trajectory comparison FAILED ($backend_name vs CPU)"
            log_error "Equivalence thresholds were exceeded"
            log_info "Trajectories and metrics saved to: $OUTPUT_DIR"
            exit 1
        fi
        ran_gpu_comparison=true
    done

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
trap 'restore_core_pattern; log_error "Script interrupted"; exit 130' INT TERM

main "$@"
