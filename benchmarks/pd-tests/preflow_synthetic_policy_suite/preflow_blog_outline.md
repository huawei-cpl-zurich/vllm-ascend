# PREFLOW blog post structure

## Working title

**PREFLOW: Preempt Long LLM Prefills Without Starving Them**

Alternative: **Fast When Safe: Hard Starvation Protection for Preemptive LLM Prefill**

**Deck:** PREFLOW replaces heuristic aging with an FCFS-relative completion
contract, allowing a prefill worker to prioritize high-value work only while
every accepted request remains protected.

## 1. Hero and TL;DR

Open with the operational tension:

> Chunking makes a long prefill preemptible. It does not tell the scheduler when
> another preemption becomes harmful.

Three headline points:

- Shortest-work policies cut mean latency but can repeatedly delay long prompts.
- Aging reduces starvation empirically, but its parameter has no latency meaning
  and must be retuned when the workload or load changes.
- PREFLOW optimizes efficiency inside a hard, per-request FCFS-relative safety
  envelope controlled by one dimensionless slack parameter.

Place the polished **Figure 5** here as the hero result. Lead with one numerical
comparison:

> At 30% configured slack, PREFLOW reduced mean TTFT by 68.2% while the worst
> matched request was 1.37x its FCFS TTFT. Aggressive PrefillOnly reached 80.4%
> mean improvement, but exposed a 5.62x worst-request slowdown.

## 2. Why prefill scheduling matters in PD systems

Explain the setting before introducing the algorithm:

- Prefill and decode run on separate workers and have different resource needs.
- Long-context traffic mixes prompts from 4K to 100K+ tokens.
- Bursts make queue order consequential.
- FCFS causes head-of-line blocking when one long prompt sits ahead of many
  short prompts.
- Chunked prefill exposes safe execution boundaries, but policy still determines
  which request gets the next chunk.

Suggested visual: a four-request timeline showing FCFS convoying, unrestricted
SRPT repeatedly preempting a long request, and PREFLOW allowing overtaking only
until the long request's slack is exhausted.

## 3. Why current fairness mechanisms are not enough

Use a compact comparison table:

| Existing approach | What it provides | What it cannot provide |
|---|---|---|
| FCFS | No scheduler-induced overtaking | Good mean latency under heterogeneous work |
| SJF/SRPT | Strong mean-latency efficiency | A bound for an individual long request |
| Aging / PrefillOnly | Eventual priority boost | An interpretable or workload-independent bound |
| External SLO scheduling | Protection for supplied deadlines | A deadline when the application provides none |
| **PREFLOW** | Efficiency subject to FCFS-relative feasibility | Absolute latency when inherited congestion is already high |

Core marketing contrast:

- **Soft aging -> hard feasibility constraint**
- **Mixed efficiency/fairness score -> optimize efficiency subject to protection**
- **Workload-specific lambda -> one interpretable relative-slack parameter**
- **Possible starvation -> bounded modeled FCFS-relative inflation**
- **Empirical fairness -> a theorem under explicit modeling assumptions**

## 4. The core idea: turn FCFS into an entitlement

Introduce only the minimum mathematics required to claim the algorithm.

For request `i`, freeze its predicted FCFS completion time from the state it
sees at arrival:

```text
B_i = unfinished work already ahead of i + isolated work of i
D_i = a_i + rho * B_i
```

Explain the semantics:

- `B_i` includes congestion already present when the request arrives.
- `D_i` is immutable; future arrivals cannot enlarge it.
- `rho = 1.3` means at most 30% modeled completion inflation relative to that
  request's FCFS counterfactual.
- The same `rho` automatically creates a tight deadline on an idle worker and a
  realistic deadline during a burst.

Use a small idle-versus-busy example to show why this is more meaningful than a
fixed per-length deadline or aging coefficient.

## 5. How PREFLOW decides whether a preemption is safe

### 5.1 Work and progress

Introduce the analytical work model compactly:

```text
W(n) = n(n + 1) / 2
p_i  = W(prompt_i) - W(initial_history_i)
r_i  = W(prompt_i) - W(current_history_i)
q    = W(h + chunk) - W(h)
```

State that the scheduling framework accepts a different conservative cost
predictor; the synthetic PREFLOW runs use the startup-profiled hardware
cost model described later.

### 5.2 Joint feasibility, not individual urgency

Order unfinished requests by deadline and define EDF-prefix slack:

```text
sigma_k = D_k - t - sum(remaining work in EDF prefix k)
```

If candidate `i` has EDF position `m`, its next chunk is safe exactly when:

```text
q <= min(sigma_k for every prefix k before m)
```

Explain intuitively:

- Running a later-deadline chunk consumes time from every earlier prefix.
- Prefixes containing the candidate also lose the same amount of remaining
  work, so their slack does not change.
- The test therefore measures exactly how much overtaking remains safe.

### 5.3 Optimize inside the safety envelope

Among safe candidates, rank by weighted remaining work:

```text
priority_i = 1 / (p_i * r_i)
```

This favors intrinsically small requests and requests near completion. If the
preferred action is unsafe, try the next candidate; if none is safe, execute an
earliest-deadline rescue action.

Emphasize the modularity:

> The efficiency objective can change without weakening the protection
> contract. Safety is a constraint, not another term in a heuristic score.

Include short pseudocode rather than the full proof.

## 6. Crossing the waiting/running admission boundary

Explain why compute feasibility alone is insufficient in a real serving engine:

- A waiting request also needs a resident slot and enough KV capacity.
- PREFLOW does not predict the exact future KV allocator state.
- It conservatively reserves enough slack for the resident prefill set to drain,
  after which the protected waiting request can be admitted.
- Once admission becomes critical, younger admissions cannot enlarge its
  blocking set.

Keep the full `G_i` drain-envelope derivation in a collapsible technical section
or link to the preprint.

## 7. From modeled work to real hardware

Describe the implementation honestly:

- The proof is expressed in work units; wall-clock protection depends on the
  quality and conservatism of the chunk-cost predictor.
- The benchmarked hard policy profiles fixed-size chunks at startup, releases
  the calibration KV cache, and fits a monotone cost curve over chunk history.
- All scheduling then uses that fitted cost model; no workload trace or
  fleet-level SLO training is required.
- PREFLOW remains optional: the analytical triangular proxy can be used when a
  profiled model is unavailable.

This is the correct place to distinguish the theoretical modeled bound from the
observed end-to-end wall-clock measurements.

## 8. Experimental setup

Present a compact, reproducible setup box:

- Qwen3-30B-A3B-Instruct-2507 architecture with dummy weight loading.
- One TP=2 prefill worker and one TP=2 decode worker on four Ascend NPUs.
- 1,000 requests, one fixed seed, and an MMPP burst process averaging 0.5
  requests/s.
- Prompt lengths: 4K, 8K, 16K, 32K, 65K, and 100K with a long-tailed mix.
- 128 output tokens per request.
- 4,096-token prefill chunks, eight resident prefill requests, and exactly one
  scheduled prefill request per execution step.
- Asynchronous scheduling; prefix caching disabled for this experiment.

Policies:

- FCFS, non-preemptive SJF, and preemptive SRPT.
- PrefillOnly with `lambda` = 200, 500, and 2000.
- EDF with 120% slack as a diagnostic baseline.
- PREFLOW with 30%, 50%, 80%, and 120% slack.

Metrics:

- Mean TTFT benefit relative to FCFS.
- Matched-request TTFT ratio: policy TTFT divided by the same request's FCFS
  TTFT.
- Worst matched-request slowdown.
- Full E2E counterparts including decode.

## 9. Results

### 9.1 Reordering helps—but unconstrained reordering harms

Place **Figure 4: matched-request TTFT-ratio CDFs**.

Main takeaways:

- SJF, SRPT, and aggressive PrefillOnly accelerate most requests.
- Their distributions retain long slowdown tails for individual requests.
- PrefillOnly changes substantially with `lambda`, demonstrating the tuning
  problem rather than solving it.
- PREFLOW provides broad improvements while sharply limiting the harmful
  tail.

### 9.2 PREFLOW moves the benefit/protection frontier

Place **Figure 5: mean TTFT benefit versus worst-request slowdown**.

Use a small result table:

| Policy | Mean TTFT improvement | Worst matched slowdown |
|---|---:|---:|
| PREFLOW 30% | 68.2% | 1.37x |
| PREFLOW 50% | 73.2% | 1.66x |
| PREFLOW 80% | 76.1% | 1.92x |
| PREFLOW 120% | 78.2% | 2.37x |
| PrefillOnly lambda=200 | 80.4% | 5.62x |
| SJF | 81.7% | 6.68x |
| SRPT | 84.6% | 7.52x |

Narrative:

- SRPT is the theoretical mean-prefill-latency optimum, but it creates the
  largest individual harm.
- PREFLOW extracts most of the available mean-latency benefit at a fraction of
  the worst-request slowdown.
- Increasing slack produces an interpretable movement along the tradeoff.
- Aging also moves along a tradeoff, but its `lambda` has no direct protection
  meaning and does not cap the tail.

### 9.3 The advantage survives decode

Place the E2E counterparts **Figures 4b and 5c**, or move Figure 4b to an
appendix if the post must remain compact.

Headline comparison:

- PREFLOW spans 66.6--76.7% mean E2E improvement with 1.43--2.30x worst
  E2E slowdown.
- PrefillOnly `lambda=200` reaches 78.8% improvement with 5.38x worst slowdown.
- Measured SRPT reaches 83.4% improvement with 6.95x worst slowdown.

State explicitly that SRPT is a measured reference here, not a theorem-level
upper bound for full two-stage PD E2E latency.

### 9.4 Constraint validation

Optionally include the hard-constraint validation and slowdown-survival plots.
Report both configured and observed wall-clock bounds. Explain small excesses as
the gap between modeled prefill work and separate noisy wall-clock executions;
do not describe the measured E2E ratios as an exact guarantee.

## 10. What is actually novel

Make the claim explicit and narrow:

1. An online FCFS counterfactual turns observed queue state into a fixed,
   per-request completion entitlement.
2. EDF-prefix feasibility converts that entitlement into an exact safe-chunk
   certificate in the compute model.
3. Efficiency ranking and protection are separated, so the scheduler can be
   aggressive without encoding fairness as an opaque score weight.
4. The contract is relative to inherited congestion and independent of future
   arrivals; it needs neither workload-specific aging nor synthetic per-length
   SLOs.
5. A conservative resident-drain obligation extends protection across the
   waiting/running KV-admission boundary.

Avoid claiming that EDF, SRPT, triangular work, or chunked prefill is itself
novel. The contribution is how these components are connected into an
FCFS-relative scheduling contract.

## 11. Scope and limitations

Use a short, confidence-building section:

- The formal guarantee is in modeled prefill work and requires conservative
  cost estimates.
- It bounds scheduler-induced inflation, not absolute latency inherited from an
  already overloaded queue.
- The synthetic study uses one seed and no prefix caching.
- The policy does not guarantee full decode completion time.
- Hardware failure, permanent capacity loss, and requests that cannot fit on an
  empty worker are outside the contract.

Frame these as deliberate boundaries rather than footnotes hidden at the end.

## 12. Why this matters for serving systems

Connect the result back to deployment:

- Operators choose a meaningful percentage rather than an opaque aging rate.
- The same setting adapts automatically to idle periods, bursts, and different
  request-size mixes.
- Fleet routers can continue making cache/load placement decisions; PREFLOW is
  local within-worker control.
- Real application deadlines remain compatible and should take precedence when
  they exist.

## 13. Conclusion and call to action

Close with a memorable formulation:

> PREFLOW does not ask whether a short request is important enough to bypass a
> long one. It asks whether that bypass is provably safe for everyone already in
> the queue.

Restate the outcome:

- FCFS defines the entitlement.
- The soft score finds the benefit.
- EDF-prefix feasibility enforces the boundary.

Link to the preprint, implementation, benchmark harness, and reproducible
analysis artifacts.

## Recommended length and figure hierarchy

- Main post: 2,500--3,500 words.
- Main figures: conceptual timeline, Figure 4, Figure 5, and Figure 5c.
- Supporting/appendix figures: Figure 4b, hard-constraint validation, slowdown
  survival curves, and queue trajectories.
- Keep the main text to the deadline and safe-chunk equations; link the proof,
  admission derivation, and predictor details to technical appendices.
