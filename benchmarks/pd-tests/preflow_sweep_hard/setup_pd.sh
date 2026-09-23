#!/usr/bin/env bash
# Launch local vLLM-Ascend MooncakeConnectorV1 P/D workers and vLLM Router.
#
# NPU groups are allocated in order: all prefiller groups first, then all
# decoder groups. For example:
#   --prefill-workers 2 --decode-workers 1 --npu-ids 0,1,2 --npus-per-worker 1
#   => prefill[0]=0, prefill[1]=1, decode[0]=2
#
# The router deliberately uses its `nixl` metadata mode. It forwards the
# kv_transfer_params produced by MooncakeConnectorV1; the router's `mooncake`
# mode instead requires the HTTP bootstrap server of upstream MooncakeConnector.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL=""
SERVED_MODEL_NAME=""
NPU_IDS_CSV=""
PREFILL_WORKERS=""
DECODE_WORKERS=""
TP_SIZE=1
PP_SIZE=1
NPUS_PER_WORKER=""
DUMMY_WEIGHTS=false
WORKER_HOST="0.0.0.0"
WORKER_URL_HOST="127.0.0.1"
ROUTER_HOST="127.0.0.1"
ROUTER_PORT=8090
WORKER_PORT_BASE=13700
PREFILL_KV_PORT_BASE=30000
DECODE_KV_PORT_BASE=30100
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=4096
# Prefill can keep several requests and their KV resident, while PREFLOW sends
# only PREFLOW_MAX_NUM_BATCHED_SEQS requests to compute in each scheduler step.
PREFILL_MAX_NUM_SEQS=8
DECODE_MAX_NUM_SEQS=16
GPU_MEMORY_UTILIZATION=0.85
SEED=1024
STARTUP_TIMEOUT=600
OUTPUT_DIR="${RUN_DIR:-${SCRIPT_DIR}/benchmark_output/pd-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR=""
QUEUE_STATS_ENABLED=true
ENABLE_PREFLOW=false
PREFLOW_MAX_FCFS_INFLATION=0.5
PREFLOW_WORK_MODEL=triangular
PREFLOW_MAX_NUM_BATCHED_SEQS=1
ENABLE_PREFIX_CACHING=false
VLLM_BIN="${VLLM_BIN:-vllm}"
VLLM_ROUTER_BIN="${VLLM_ROUTER_BIN:-vllm-router}"

usage() {
    cat <<'EOF'
Usage:
  setup_pd.sh --model MODEL --npu-ids IDS [options]

Required:
  --model PATH                 Model path or Hugging Face model ID.
  --served-model-name NAME     API model name (default: basename of --model).
  --npu-ids ID,ID,...          Comma-separated physical NPU IDs.
  --prefill-workers N          Number of prefill vLLM workers to start.
  --decode-workers N           Number of decode vLLM workers to start.

Worker topology:
  --tp-size N                  Tensor-parallel size per worker (default: 1).
  --pp-size N                  Pipeline-parallel size per worker (default: 1).
  --npus-per-worker N          NPU IDs allocated to each P or D worker.
                               Defaults to TP * PP and must equal TP * PP.
  --dummy-weights              Add --load-format dummy to each vLLM worker.

Serving options:
  --max-model-len N            Maximum model length (default: 4096).
  --max-num-batched-tokens N   Maximum batched tokens (default: 4096).
  --prefill-max-num-seqs N     Resident request/KV slots per prefill worker
                               (default: 8).
  --decode-max-num-seqs N      Maximum sequences per decode worker (default: 16).
  --max-num-seqs N             Deprecated alias for --decode-max-num-seqs.
  --enable-preflow             Enable PREFLOW on PD-prefill workers (default: disabled).
  --preflow-max-fcfs-inflation F
                               Maximum completion-time inflation relative to the
                               frozen FCFS baseline (default: 0.5, meaning 50%).
  --preflow-work-model MODEL   PREFLOW work model: triangular or profiled
                               (default: triangular).
  --preflow-max-num-batched-seqs N
                               Requests computed per PREFLOW scheduler step
                               (default: 1; must not exceed prefill max-num-seqs).
  --enable-prefix-caching      Enable prefix caching on P and D (default: disabled).
  --gpu-memory-utilization F   Device-memory utilization (default: 0.85).
  --seed N                     vLLM seed (default: 1024).

Network and logging options:
  --worker-host HOST           vLLM bind host (default: 0.0.0.0).
  --worker-url-host HOST       Host used by the local router (default: 127.0.0.1).
  --router-host HOST           Router bind host (default: 127.0.0.1).
  --router-port PORT           Router port (default: 8090).
  --worker-port-base PORT      First worker API port (default: 13700).
  --prefill-kv-port-base PORT  First prefill Mooncake handshake port (default: 30000).
  --decode-kv-port-base PORT   First decode Mooncake handshake port (default: 30100).
  --output-dir PATH            Benchmark artifact directory (default:
                               ./benchmark_output/pd-<timestamp>). It contains
                               logs/, queue_stats/, launch_config.yaml, and a
                               process manifest for targeted teardown.
  --log-dir PATH               Override the worker/router log directory.
  --disable-queue-stats        Do not write scheduler queue trace CSV files.
  --startup-timeout SECONDS    Per-worker health-check timeout (default: 600).

The script keeps all services in the background but remains in the foreground
to supervise them. Ctrl-C stops the router and all workers it launched.

Current limitation: MooncakeConnectorV1 requires decode PP=1. Therefore the
shared --pp-size accepted by this launcher must currently be 1.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

require_option_value() {
    local option="$1"
    local value="${2:-}"
    [[ -n "${value}" && "${value}" != --* ]] || die "${option} requires a value"
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_number() {
    [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]
}

while (($#)); do
    case "$1" in
        --model)
            require_option_value "$1" "${2:-}"
            MODEL="$2"
            shift 2
            ;;
        --npu-ids)
            require_option_value "$1" "${2:-}"
            NPU_IDS_CSV="$2"
            shift 2
            ;;
        --served-model-name)
            require_option_value "$1" "${2:-}"
            SERVED_MODEL_NAME="$2"
            shift 2
            ;;
        --prefill-workers)
            require_option_value "$1" "${2:-}"
            PREFILL_WORKERS="$2"
            shift 2
            ;;
        --decode-workers)
            require_option_value "$1" "${2:-}"
            DECODE_WORKERS="$2"
            shift 2
            ;;
        --tp-size)
            require_option_value "$1" "${2:-}"
            TP_SIZE="$2"
            shift 2
            ;;
        --pp-size)
            require_option_value "$1" "${2:-}"
            PP_SIZE="$2"
            shift 2
            ;;
        --npus-per-worker)
            require_option_value "$1" "${2:-}"
            NPUS_PER_WORKER="$2"
            shift 2
            ;;
        --dummy-weights)
            DUMMY_WEIGHTS=true
            shift
            ;;
        --max-model-len)
            require_option_value "$1" "${2:-}"
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --max-num-batched-tokens)
            require_option_value "$1" "${2:-}"
            MAX_NUM_BATCHED_TOKENS="$2"
            shift 2
            ;;
        --prefill-max-num-seqs)
            require_option_value "$1" "${2:-}"
            PREFILL_MAX_NUM_SEQS="$2"
            shift 2
            ;;
        --decode-max-num-seqs)
            require_option_value "$1" "${2:-}"
            DECODE_MAX_NUM_SEQS="$2"
            shift 2
            ;;
        --max-num-seqs)
            require_option_value "$1" "${2:-}"
            echo "warning: --max-num-seqs is deprecated; use --decode-max-num-seqs instead" >&2
            DECODE_MAX_NUM_SEQS="$2"
            shift 2
            ;;
        --enable-preflow)
            ENABLE_PREFLOW=true
            shift
            ;;
        --preflow-max-fcfs-inflation)
            require_option_value "$1" "${2:-}"
            PREFLOW_MAX_FCFS_INFLATION="$2"
            shift 2
            ;;
        --preflow-work-model)
            require_option_value "$1" "${2:-}"
            PREFLOW_WORK_MODEL="$2"
            shift 2
            ;;
        --preflow-max-num-batched-seqs)
            require_option_value "$1" "${2:-}"
            PREFLOW_MAX_NUM_BATCHED_SEQS="$2"
            shift 2
            ;;
        --enable-prefix-caching)
            ENABLE_PREFIX_CACHING=true
            shift
            ;;
        --gpu-memory-utilization)
            require_option_value "$1" "${2:-}"
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --seed)
            require_option_value "$1" "${2:-}"
            SEED="$2"
            shift 2
            ;;
        --worker-host)
            require_option_value "$1" "${2:-}"
            WORKER_HOST="$2"
            shift 2
            ;;
        --worker-url-host)
            require_option_value "$1" "${2:-}"
            WORKER_URL_HOST="$2"
            shift 2
            ;;
        --router-host)
            require_option_value "$1" "${2:-}"
            ROUTER_HOST="$2"
            shift 2
            ;;
        --router-port)
            require_option_value "$1" "${2:-}"
            ROUTER_PORT="$2"
            shift 2
            ;;
        --worker-port-base)
            require_option_value "$1" "${2:-}"
            WORKER_PORT_BASE="$2"
            shift 2
            ;;
        --prefill-kv-port-base)
            require_option_value "$1" "${2:-}"
            PREFILL_KV_PORT_BASE="$2"
            shift 2
            ;;
        --decode-kv-port-base)
            require_option_value "$1" "${2:-}"
            DECODE_KV_PORT_BASE="$2"
            shift 2
            ;;
        --output-dir)
            require_option_value "$1" "${2:-}"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --log-dir)
            require_option_value "$1" "${2:-}"
            LOG_DIR="$2"
            shift 2
            ;;
        --disable-queue-stats)
            QUEUE_STATS_ENABLED=false
            shift
            ;;
        --startup-timeout)
            require_option_value "$1" "${2:-}"
            STARTUP_TIMEOUT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[[ -n "${MODEL}" ]] || die "--model is required"
[[ -n "${NPU_IDS_CSV}" ]] || die "--npu-ids is required"
[[ -n "${PREFILL_WORKERS}" ]] || die "--prefill-workers is required"
[[ -n "${DECODE_WORKERS}" ]] || die "--decode-workers is required"
if [[ -z "${SERVED_MODEL_NAME}" ]]; then
    SERVED_MODEL_NAME="$(basename "${MODEL%/}")"
fi

for value in "$PREFILL_WORKERS" "$DECODE_WORKERS" "$TP_SIZE" "$PP_SIZE" "$MAX_MODEL_LEN" \
    "$MAX_NUM_BATCHED_TOKENS" "$PREFILL_MAX_NUM_SEQS" "$DECODE_MAX_NUM_SEQS" "$SEED" "$STARTUP_TIMEOUT" "$ROUTER_PORT" \
    "$WORKER_PORT_BASE" "$PREFILL_KV_PORT_BASE" "$DECODE_KV_PORT_BASE" "$PREFLOW_MAX_NUM_BATCHED_SEQS"; do
    is_positive_integer "$value" || die "expected a positive integer, got: $value"
done
is_nonnegative_number "$PREFLOW_MAX_FCFS_INFLATION" || \
    die "--preflow-max-fcfs-inflation must be a non-negative number"
[[ "$PREFLOW_WORK_MODEL" == "triangular" || "$PREFLOW_WORK_MODEL" == "profiled" ]] || \
    die "--preflow-work-model must be triangular or profiled"
((PREFLOW_MAX_NUM_BATCHED_SEQS <= PREFILL_MAX_NUM_SEQS)) || \
    die "--preflow-max-num-batched-seqs must not exceed --prefill-max-num-seqs"

if [[ -z "${NPUS_PER_WORKER}" ]]; then
    NPUS_PER_WORKER=$((TP_SIZE * PP_SIZE))
fi
is_positive_integer "$NPUS_PER_WORKER" || die "--npus-per-worker must be a positive integer"
((NPUS_PER_WORKER == TP_SIZE * PP_SIZE)) || die "--npus-per-worker must equal --tp-size * --pp-size"
((PP_SIZE == 1)) || die "MooncakeConnectorV1 currently requires --pp-size 1 because decode PP is unsupported"

command -v "$VLLM_BIN" >/dev/null || die "vLLM command not found: $VLLM_BIN"
command -v "$VLLM_ROUTER_BIN" >/dev/null || die "vLLM Router command not found: $VLLM_ROUTER_BIN"
command -v setsid >/dev/null || die "setsid is required to supervise worker process groups"
command -v curl >/dev/null || die "curl is required for worker health checks"

declare -a NPU_IDS=()
IFS=',' read -r -a raw_npu_ids <<< "$NPU_IDS_CSV"
for npu_id in "${raw_npu_ids[@]}"; do
    npu_id="${npu_id//[[:space:]]/}"
    [[ "$npu_id" =~ ^[0-9]+$ ]] || die "invalid NPU ID: $npu_id"
    NPU_IDS+=("$npu_id")
done

WORKER_COUNT=$((PREFILL_WORKERS + DECODE_WORKERS))
REQUIRED_NPU_COUNT=$((WORKER_COUNT * NPUS_PER_WORKER))
(( ${#NPU_IDS[@]} == REQUIRED_NPU_COUNT )) || die \
    "--npu-ids must contain exactly ${REQUIRED_NPU_COUNT} IDs for ${PREFILL_WORKERS} prefill and ${DECODE_WORKERS} decode workers"

# Engine-core workers are separate processes. Pass absolute artifact paths so
# their working directory cannot redirect queue traces somewhere unexpected.
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd -P)"
if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR="${OUTPUT_DIR}/logs"
fi
mkdir -p "$LOG_DIR"
LOG_DIR="$(cd -- "$LOG_DIR" && pwd -P)"
QUEUE_STATS_DIR="${OUTPUT_DIR}/queue_stats"
PROCESS_MANIFEST="${OUTPUT_DIR}/processes.tsv"
: >"$PROCESS_MANIFEST"
printf 'role\tpid\n' >>"$PROCESS_MANIFEST"
if [[ "$QUEUE_STATS_ENABLED" == true ]]; then
    mkdir -p "$QUEUE_STATS_DIR"
fi

# Prevent a local HTTP proxy from intercepting worker health checks or router
# requests to the loopback vLLM services.
LOCAL_NO_PROXY="localhost,127.0.0.1,::1"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${LOCAL_NO_PROXY}"
export no_proxy="${no_proxy:+${no_proxy},}${LOCAL_NO_PROXY}"

declare -a CHILD_PIDS=()
declare -a PREFILL_URLS=()
declare -a DECODE_URLS=()

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    if ((${#CHILD_PIDS[@]})); then
        echo "Stopping launched router and vLLM workers..." >&2
        for pid in "${CHILD_PIDS[@]}"; do
            kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        done
        for pid in "${CHILD_PIDS[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
    fi
    exit "$exit_code"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

devices_for_group() {
    local start_index="$1"
    local devices="${NPU_IDS[start_index]}"
    local index
    for ((index = start_index + 1; index < start_index + NPUS_PER_WORKER; index++)); do
        devices+=",${NPU_IDS[index]}"
    done
    printf '%s' "$devices"
}

kv_transfer_config() {
    local role="$1"
    local kv_port="$2"
    printf '%s' \
        "{\"kv_connector\":\"MooncakeConnectorV1\",\"kv_role\":\"${role}\",\"kv_port\":\"${kv_port}\",\"kv_connector_extra_config\":{\"prefill\":{\"dp_size\":1,\"tp_size\":${TP_SIZE},\"pp_size\":${PP_SIZE}},\"decode\":{\"dp_size\":1,\"tp_size\":${TP_SIZE},\"pp_size\":${PP_SIZE}}}}"
}

json_escape() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//$'\n'/\\n}"
    value="${value//$'\r'/\\r}"
    value="${value//$'\t'/\\t}"
    printf '%s' "$value"
}

scheduler_additional_config() {
    printf '%s' \
        "{\"scheduler_config\":{\"preflow_config\":{\"enabled\":${ENABLE_PREFLOW},\"max_fcfs_inflation\":${PREFLOW_MAX_FCFS_INFLATION},\"work_model\":\"${PREFLOW_WORK_MODEL}\",\"max_num_batched_seqs\":${PREFLOW_MAX_NUM_BATCHED_SEQS}},\"queue_stats_config\":{\"enabled\":${QUEUE_STATS_ENABLED},\"output_dir\":\"$(json_escape "$QUEUE_STATS_DIR")\"}}}"
}

write_launch_config() {
    {
        printf 'model: "%s"\n' "$(json_escape "$MODEL")"
        printf 'served_model_name: "%s"\n' "$(json_escape "$SERVED_MODEL_NAME")"
        printf 'npu_ids: "%s"\n' "$(json_escape "$NPU_IDS_CSV")"
        printf 'npus_per_worker: %s\n' "$NPUS_PER_WORKER"
        printf 'tensor_parallel_size: %s\n' "$TP_SIZE"
        printf 'pipeline_parallel_size: %s\n' "$PP_SIZE"
        printf 'max_model_len: %s\n' "$MAX_MODEL_LEN"
        printf 'gpu_memory_utilization: %s\n' "$GPU_MEMORY_UTILIZATION"
        printf 'seed: %s\n' "$SEED"
        printf 'preflow_enabled: %s\n' "$ENABLE_PREFLOW"
        printf 'preflow_max_fcfs_inflation: %s\n' "$PREFLOW_MAX_FCFS_INFLATION"
        printf 'preflow_work_model: "%s"\n' "$PREFLOW_WORK_MODEL"
        printf 'preflow_max_num_batched_seqs: %s\n' "$PREFLOW_MAX_NUM_BATCHED_SEQS"
        printf 'async_scheduling: true\n'
        printf 'chunked_prefill: true\n'
        printf 'prefix_caching_enabled: %s\n' "$ENABLE_PREFIX_CACHING"
        printf 'queue_stats_enabled: %s\n' "$QUEUE_STATS_ENABLED"
        printf 'queue_stats_dir: "%s"\n' "$(json_escape "$QUEUE_STATS_DIR")"
        printf 'prefill_workers: %s\n' "$PREFILL_WORKERS"
        printf 'decode_workers: %s\n' "$DECODE_WORKERS"
        printf 'prefill_max_num_seqs: %s\n' "$PREFILL_MAX_NUM_SEQS"
        printf 'decode_max_num_seqs: %s\n' "$DECODE_MAX_NUM_SEQS"
        printf 'max_num_batched_tokens: %s\n' "$MAX_NUM_BATCHED_TOKENS"
        printf 'router_policy: consistent_hash\n'
        printf 'router_port: %s\n' "$ROUTER_PORT"
        printf 'worker_port_base: %s\n' "$WORKER_PORT_BASE"
        printf 'prefill_kv_port_base: %s\n' "$PREFILL_KV_PORT_BASE"
        printf 'decode_kv_port_base: %s\n' "$DECODE_KV_PORT_BASE"
        printf 'process_manifest: "%s"\n' "$(json_escape "$PROCESS_MANIFEST")"
        printf 'worker_log_dir: "%s"\n' "$(json_escape "$LOG_DIR")"
    } >"${OUTPUT_DIR}/launch_config.yaml"
}

start_worker() {
    local worker_type="$1"
    local kv_role="$2"
    local worker_index="$3"
    local devices="$4"
    local api_port="$5"
    local kv_port="$6"
    local log_file="${LOG_DIR}/${worker_type}-${worker_index}-npus-${devices//,/_}-api-${api_port}.log"
    local max_num_seqs="$PREFILL_MAX_NUM_SEQS"
    if [[ "$worker_type" == "decode" ]]; then
        max_num_seqs="$DECODE_MAX_NUM_SEQS"
    fi
    local config
    local additional_config
    config="$(kv_transfer_config "$kv_role" "$kv_port")"
    additional_config="$(scheduler_additional_config)"
    local -a cmd=(
        "$VLLM_BIN" serve "$MODEL"
        --host "$WORKER_HOST"
        --port "$api_port"
        --served-model-name "$SERVED_MODEL_NAME"
        --tensor-parallel-size "$TP_SIZE"
        --pipeline-parallel-size "$PP_SIZE"
        --max-model-len "$MAX_MODEL_LEN"
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
        --max-num-seqs "$max_num_seqs"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --seed "$SEED"
        --kv-transfer-config "$config"
        --async-scheduling
        --enable-chunked-prefill
        --additional-config "$additional_config"
    )

    if [[ "$ENABLE_PREFIX_CACHING" == true ]]; then
        cmd+=(--enable-prefix-caching)
    else
        cmd+=(--no-enable-prefix-caching)
    fi

    if [[ "$DUMMY_WEIGHTS" == true ]]; then
        cmd+=(--load-format dummy)
    fi

    echo "Starting ${worker_type}[${worker_index}] on NPUs ${devices} with max-num-seqs=${max_num_seqs}; log: ${log_file}"
    setsid env \
        -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
        ASCEND_RT_VISIBLE_DEVICES="$devices" "${cmd[@]}" >"$log_file" 2>&1 &
    local child_pid="$!"
    CHILD_PIDS+=("$child_pid")
    printf '%s\t%s\n' "${worker_type}[${worker_index}]" "$child_pid" >>"$PROCESS_MANIFEST"
}

wait_for_health() {
    local name="$1"
    local url="$2"
    local pid="$3"
    local log_file="$4"
    local deadline=$((SECONDS + STARTUP_TIMEOUT))

    until curl --noproxy '*' --fail --silent --show-error --connect-timeout 2 --max-time 5 "${url}/health" \
        >/dev/null 2>&1; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "${name} exited before becoming healthy. Last log lines:" >&2
            tail -n 80 "$log_file" >&2 || true
            exit 1
        fi
        if ((SECONDS >= deadline)); then
            echo "Timed out waiting for ${name} at ${url}/health. Last log lines:" >&2
            tail -n 80 "$log_file" >&2 || true
            exit 1
        fi
        sleep 2
    done
    echo "${name} is healthy at ${url}"
}

declare -a PREFILL_PIDS=()
declare -a DECODE_PIDS=()
declare -a PREFILL_LOGS=()
declare -a DECODE_LOGS=()

write_launch_config

# Launch every prefill worker before launching any decoder worker.
for ((worker_index = 0; worker_index < PREFILL_WORKERS; worker_index++)); do
    npu_start=$((worker_index * NPUS_PER_WORKER))
    devices="$(devices_for_group "$npu_start")"
    api_port=$((WORKER_PORT_BASE + worker_index))
    kv_port=$((PREFILL_KV_PORT_BASE + worker_index * NPUS_PER_WORKER))
    start_worker prefill kv_producer "$worker_index" "$devices" "$api_port" "$kv_port"
    PREFILL_PIDS+=("${CHILD_PIDS[${#CHILD_PIDS[@]} - 1]}")
    PREFILL_URLS+=("http://${WORKER_URL_HOST}:${api_port}")
    PREFILL_LOGS+=("${LOG_DIR}/prefill-${worker_index}-npus-${devices//,/_}-api-${api_port}.log")
done

for ((worker_index = 0; worker_index < DECODE_WORKERS; worker_index++)); do
    npu_start=$(((PREFILL_WORKERS + worker_index) * NPUS_PER_WORKER))
    devices="$(devices_for_group "$npu_start")"
    api_port=$((WORKER_PORT_BASE + PREFILL_WORKERS + worker_index))
    kv_port=$((DECODE_KV_PORT_BASE + worker_index * NPUS_PER_WORKER))
    start_worker decode kv_consumer "$worker_index" "$devices" "$api_port" "$kv_port"
    DECODE_PIDS+=("${CHILD_PIDS[${#CHILD_PIDS[@]} - 1]}")
    DECODE_URLS+=("http://${WORKER_URL_HOST}:${api_port}")
    DECODE_LOGS+=("${LOG_DIR}/decode-${worker_index}-npus-${devices//,/_}-api-${api_port}.log")
done

for ((worker_index = 0; worker_index < PREFILL_WORKERS; worker_index++)); do
    wait_for_health \
        "prefill[${worker_index}]" "${PREFILL_URLS[worker_index]}" "${PREFILL_PIDS[worker_index]}" \
        "${PREFILL_LOGS[worker_index]}"
done
for ((worker_index = 0; worker_index < DECODE_WORKERS; worker_index++)); do
    wait_for_health \
        "decode[${worker_index}]" "${DECODE_URLS[worker_index]}" "${DECODE_PIDS[worker_index]}" \
        "${DECODE_LOGS[worker_index]}"
done

router_log="${LOG_DIR}/router.log"
router_cmd=(
    "$VLLM_ROUTER_BIN"
    --policy consistent_hash
    --vllm-pd-disaggregation
    --kv-connector nixl
    --host "$ROUTER_HOST"
    --port "$ROUTER_PORT"
    --intra-node-data-parallel-size 1
)
for url in "${PREFILL_URLS[@]}"; do
    router_cmd+=(--prefill "$url")
done
for url in "${DECODE_URLS[@]}"; do
    router_cmd+=(--decode "$url")
done

echo "Starting router for ${PREFILL_WORKERS} prefill and ${DECODE_WORKERS} decode workers; log: ${router_log}"
setsid env \
    -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
    "${router_cmd[@]}" >"$router_log" 2>&1 &
router_pid="$!"
CHILD_PIDS+=("$router_pid")
printf 'router\t%s\n' "$router_pid" >>"$PROCESS_MANIFEST"

echo "Router endpoint: http://${ROUTER_HOST}:${ROUTER_PORT}"
echo "Benchmark artifacts: ${OUTPUT_DIR}"
echo "  Logs: ${LOG_DIR}"
if [[ "$QUEUE_STATS_ENABLED" == true ]]; then
    echo "  Queue stats: ${QUEUE_STATS_DIR}"
else
    echo "  Queue stats: disabled"
fi
echo "Press Ctrl-C to stop the router and all launched workers."

wait "$router_pid"
