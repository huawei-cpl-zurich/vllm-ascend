# PREFLOW prefill trace suite

This self-contained suite compares prefill scheduling policies on every bundled
Mooncake and Qwen-Bailian trace. It runs three independent TP4 vLLM servers on
NPUs 0--11, leaves NPUs 12--15 free, uses dummy weights, requests exactly one
output token, and does not launch a decode server or vLLM Router.

## Fixed experiment

- Model: `/data/weights/Qwen3-Coder-30B-A3B-Instruct/`
- Load format: `dummy`
- Three NPU groups: `0,1,2,3`, `4,5,6,7`, `8,9,10,11`
- Tensor parallelism: 4
- Target calibrated fixed-chunk execution load: 0.95
- Maximum resident requests (`max_num_seqs`): 16
- Maximum chunk: 2,048 tokens
- Scheduled prefill batch width: exactly one request
- Output tokens: 1
- Prefix caching: disabled
- Async scheduling: explicitly enabled, matching production deployment. Each
  model-execution batch still contains exactly one prefill request; vLLM may
  keep the next single-request batch in flight to overlap scheduling and NPU
  execution.
- Trace window: contiguous prefix containing at least 1,000 requests and 600
  seconds of calibrated isolated fixed-chunk execution, capped at 20,000
  requests

Every policy configuration runs all eight traces:

- chunked FCFS;
- non-preemptive triangular SJF;
- triangular SRPT;
- PrefillOnly with `lambda=200,500,2000`;
- pure EDF with 120% FCFS-relative slack; and
- hard PREFLOW with 30%, 50%, 80%, and 120% FCFS-relative slack.

This is 11 policy configurations by eight traces, or 88 conditions. The full
hard-PREFLOW slack sweep shows the tradeoff from relaxing the guarantee. The
full PrefillOnly lambda sweep tests whether one hand-tuned aging coefficient
transfers across workloads.

## Run

From this directory:

```bash
python run.py
```

Use `python run.py --plan` to print the 88-condition plan without needing
vLLM, the model, or NPUs. `VLLM_BIN` may point to a non-default vLLM executable.

Run `python status.py` at any time for a read-only progress and ETA snapshot.
It takes one filesystem snapshot from the suite inputs and existing result
metadata; it never polls the servers, reads their logs, or modifies benchmark
output. Add `--unfinished` to hide completed conditions, or `--output-root
PATH` when the run uses a non-default directory.

Results are written to `benchmark_output/`. A condition is complete only when
its status is `completed`, its `summary.json` exists, and every request
succeeded. Running `python run.py` again skips those conditions and retries only
missing or failed work. Each retry gets a new `attempt-NNN` directory, so failed
logs are retained.

## Extend the suite without repeating completed work

The experiment membership is declared in `suite_config.json`. A policy's
runtime identity deliberately excludes its trace coverage. Therefore, adding a
trace to the top-level `traces` list makes policies with `"traces": "all"` run
only that new trace; their completed trace results remain valid. Adding a new
policy object similarly creates only that policy's trace conditions. A policy
may use `"traces": "all"` or an explicit list of trace names. Its
`scheduler_config` is merged into the existing vLLM-Ascend scheduler config.
Because async scheduling is a fixed suite invariant, any newly added scheduler
class must implement vLLM's asynchronous scheduler behavior.

To add and compact another trace:

```bash
python prepare_traces.py --trace azure_chat=/path/to/azure_chat.jsonl
```

Then append `azure_chat` to `suite_config.json` and rerun `python run.py`.
Multiple `--trace NAME=PATH` arguments may be supplied. Relative paths are
resolved under `--source-root`. Use `--force` only when intentionally replacing
an existing derived trace; a changed trace digest invalidates just that trace's
conditions. A changed calibration artifact invalidates all conditions because
it changes load normalization.

Each suite configuration is archived by digest under
`benchmark_output/provenance/suite_configs/`, while `suite_manifest.json` is
refreshed to describe the active matrix.

After a complete run, `run.py` invokes `analyze_results.py` to produce
`aggregate_results.csv`, `fcfs_relative_requests.csv`, and
`fcfs_relative_summary.csv`. The FCFS-relative files pair identical trace
request IDs across policies; they are empirical wall-clock comparisons, not
the hard scheduler's formal triangular-work guarantee.

`run_index.csv` is refreshed during the campaign. Each result contains:

- per-request scheduled and actual arrival times, TTFT, completion latency,
  prompt serialization time, prompt length, and calibrated work;
- a summary with TTFT and client-lag distributions;
- server queue/KV metric samples;
- the exact selected trace window and arrival stretch;
- server and client logs plus scheduler queue-stat traces.

## Load normalization

`calibration/parametric_chunk_cost_model.json` is the bundled dummy-weight TP4
fixed-chunk calibration for the exact model path used by the suite. It is the
same execution model used by the PREFLOW simulator: each request's isolated
service is the sum of its calibrated 2,048-token chunks, including modeled
linear work and launch overhead. For a selected window with total calibrated
service `S`, the client stretches the original timestamp span to `S / 0.95`.
Relative timing, ordering, and bursts within the contiguous trace window are
preserved. PREFLOW's scheduling decisions themselves continue to use only the
triangular work model.

This is intentionally a prefill-node experiment. One output token makes TTFT
the request completion metric while avoiding a router, KV transfer, and a
decode worker. It must not be described as an end-to-end PD benchmark.

## Bundled traces

The `traces/` files preserve each source row's timestamp, input length, and
original row index. Prefix identifiers and source output lengths are omitted
because prefix caching is disabled and output length is fixed to one. Source
and derived SHA-256 digests are recorded in `traces/manifest.json`.

To regenerate them from the original local trace checkout:

```bash
python prepare_traces.py --force
```
