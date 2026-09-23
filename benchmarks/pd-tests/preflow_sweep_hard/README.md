# Hard PREFLOW PD parameter sweep

This directory is a self-contained hard-policy variant of `preflow_sweep`.
Copy the whole directory to the target machine, enter it, and run:

```bash
cd preflow_sweep_hard
python run.py
```

The target environment must provide `vllm`, `vllm-router`, and a vLLM Ascend
checkout containing the hard FCFS-relative PREFLOW policy. Install the workload
client dependencies with:

```bash
python -m pip install -r requirements.txt
```

The default model is
`/data/weights/Qwen3-30B-A3B-Instruct-2507/`. Override it when necessary:

```bash
python run.py --model /path/to/model
```

## Fixed experiment plan

This sweep runs only the startup-profiled `PREFLOWScheduler`; it has no FCFS
or soft-aging deployment. Every deployment sets
`preflow_config.work_model = "profiled"`, and the three deployments set:

```text
preflow_config.max_fcfs_inflation = 0.3  # 30% slack
preflow_config.max_fcfs_inflation = 0.5  # 50% slack
preflow_config.max_fcfs_inflation = 0.8  # 80% slack
```

For each deployment, the harness runs arrival rates `0.5` and `0.6` requests/s,
with decode length fixed to 128 tokens. The low-load `0.4` condition is skipped.
Each condition has 1,000 requests and retains the original workload seed
(`20260728`), MMPP arrival process, prompt distribution, and synthetic-query
generation code. This gives 3 model loads and 6 measured runs.

The topology and remaining controls also match the original sweep:

- one TP=2 prefill worker on NPUs `0,1`;
- one TP=2 decode worker on NPUs `2,3`;
- prefix caching disabled;
- asynchronous scheduling and chunked prefill enabled explicitly;
- `preflow_config.max_num_batched_seqs=1`.

The prefill worker becomes healthy only after startup profiling completes. The
harness therefore allows up to one hour for deployment startup and records the
selected work model in the manifest, deployment metadata, run index, and
launch configuration.

Use `--plan-only` to print the complete matrix without creating files or
launching any process:

```bash
python run.py --plan-only
```

## Outputs and recovery

Each invocation creates a unique directory under
`preflow_sweep_hard/benchmark_output/`. It contains the sweep manifest and
status, lifecycle events, a run index, provenance, launcher logs, queue traces,
workload traces, sampled metrics, request records, and summaries.

Retries use new `attempt-NNN` directories, preserving partial diagnostics.
Resume a failed or interrupted sweep with:

```bash
python run.py --resume /absolute/path/to/the/sweep-root
```

Completed conditions are skipped. Only deployments with unfinished conditions
are loaded again.

To retry only conditions that are explicitly marked `failed`, without running
conditions that have never started, use:

```bash
python run.py \
  --resume /absolute/path/to/the/sweep-root \
  --retry-failed-only
```

This preserves the failed attempt, writes the retry under a new `attempt-NNN`
directory, and leaves completed and pending conditions untouched.

The FCFS/soft-aging `analyze_results.py` from `preflow_sweep` is intentionally
not bundled: it requires paired FCFS and aging results, while this sweep emits
hard-policy-only results. The raw output layout remains the same for subsequent
hard-policy analysis.
