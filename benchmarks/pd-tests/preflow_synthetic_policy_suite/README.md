# Synthetic PREFLOW policy suite

This self-contained suite runs the final scheduler comparison on the same
synthetic MMPP workload used by `preflow_sweep_hard`.

Run it from this directory:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python run.py
```

`ASCEND_RT_VISIBLE_DEVICES` is required and must contain exactly four distinct
comma-separated NPU IDs. Their order determines the topology: the first two
are assigned to prefill and the last two to decode. The script exits before
creating or modifying benchmark output when this validation fails.

No experiment CLI arguments are required. The fixed configuration is:

- one TP=2 prefill instance on the first two visible NPUs;
- one TP=2 decode instance on the last two visible NPUs;
- Qwen3-30B-A3B-Instruct-2507 from
  `/data/weights/Qwen3-30B-A3B-Instruct-2507/`;
- 1,000 requests using workload seed `20260728`;
- MMPP mean arrival rate `0.5` requests/s;
- 128 requested decode tokens;
- 4,096-token chunked prefill;
- eight resident prefill requests, 16 decode sequences, and exactly one
  scheduled prefill request per model-execution step;
- asynchronous scheduling enabled and prefix caching disabled.

The suite runs 11 policy configurations:

- FCFS;
- non-preemptive SJF;
- preemptive SRPT;
- PrefillOnly with `lambda=200`, `500`, and `2000`;
- EDF with 120% FCFS-relative slack;
- hard PREFLOW with 30%, 50%, 80%, and 120% FCFS-relative slack.

The four hard PREFLOW configurations use the startup-profiled cost model.
Their prefill workers become healthy only after profiling completes and the
profiling KV cache is released. The other policy baselines retain the
triangular work model. The old exponential-aging PREFLOW policy is not
included.

## Failure handling and resumption

Each policy gets one initial attempt and at most two retries. A failed attempt
causes the harness to kill the client and every vLLM/router process group
recorded for that attempt before starting a fresh deployment. Cleanup is
strictly scoped to recorded PIDs; the harness never uses a host-wide `pkill`.

After three failed attempts, the policy is marked failed and the suite moves
on. Completed policies are never rerun. Calling `python run.py` again resumes
the fixed output campaign at:

```text
benchmark_output/final-synthetic-policy-suite/
```

Use `python run.py --plan-only` to inspect the matrix without launching it.
To intentionally create a separate campaign, pass `--output-root PATH`.

The output includes per-attempt server/client logs, request results, summaries,
queue traces, a run index, the resolved commands, and source provenance.
