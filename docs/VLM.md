# Two-tier vision-language cascade (R-7, U6, WP11)

A 2B model screens every case; an 8B model reasons only on the cases the screen escalates. This is the TAU-R1
design of the 499A report (3.1.4, 5.1.2), run locally on the CPU with llama.cpp.

Code: `surveillance/vlm_cascade.py` (cascade, escalation rule, on-demand process manager), `surveillance/vlm.py` (prompt,
schema, every guard, backends, queue), `surveillance/screening.py` (kinematic pre-screen). Tools: `tools/setup_vlm.sh`,
`tools/vlm_server.sh`, `tools/vlm_check.py`, `tools/vlm_probe.py`. Tests: `tests/test_cascade.py`, `tests/test_vlm.py`.

## Models

| Tier | Model | Files (Q4_K_M weights + F16 projector) | Port |
|---|---|---|---|
| screen (always on) | Qwen3-VL-2B-Instruct | `Qwen3VL-2B-Instruct-Q4_K_M.gguf` 1,107,409,952 B + `mmproj-Qwen3VL-2B-Instruct-F16.gguf` 819,394,848 B | 8081 |
| reason (on demand) | Qwen3-VL-8B-Instruct | `Qwen3VL-8B-Instruct-Q4_K_M.gguf` 5,027,784,800 B + `mmproj-Qwen3VL-8B-Instruct-F16.gguf` 1,159,029,824 B | 8082 |
| fallback single model | Qwen2.5-VL-3B-Instruct | unchanged | 8081 |

SHA-256 (the Hugging Face LFS hashes; `tools/setup_vlm.sh` refuses a file that does not match):

```
089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae  Qwen3VL-2B-Instruct-Q4_K_M.gguf
c3d5afbef5287953acd57b4043d2269456e5761a4eaccb3b71b062996970aea5  mmproj-Qwen3VL-2B-Instruct-F16.gguf
67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2  Qwen3VL-8B-Instruct-Q4_K_M.gguf
ca524100ebf825c9a870db1c580d03879e0da0ab2541697e2458e64891cf9d38  mmproj-Qwen3VL-8B-Instruct-F16.gguf
```

Setup: `./tools/setup_vlm.sh cascade` (resumable, idempotent; `3b` and `binary` select the other parts).
Start the always-on tier: `./tools/vlm_server.sh --tier screen`. The 8B (`--tier reason`) is started by the dashboard
server when a case is escalated; start it by hand only to measure.

## Flow

1. **Pre-screen** (`screening.screen`): free, no model. Drops only an obviously empty V10/V11 candidate.
2. **2B screen**: a short prompt (the rule's one-sentence claim, the photograph list) and a grammar-constrained schema
   `{shows_event: yes|no|cannot_tell, image: 0|1|2, certainty: low|medium|high}`, 48 tokens at most. A 2B model cannot
   calibrate a number (it answered `confidence: 0.0` to everything), so it states a level: low 0.3, medium 0.6, high 0.9.
3. **Escalation rule** (below).
4. **8B reasoner**: the existing phase-aware prompt and schema, and every guard in `vlm.to_explanation` (a "supported"
   whose own observation does not back it up becomes "insufficient"; unparsable output is never a verdict). The cloud
   backends (Anthropic / OpenAI-compatible) can be the reasoner instead (`vlm_reason_backend`).
5. **Result** records every stage: model names, verdicts, latencies, why escalated or not, and who decided
   (`decided_by`: `reasoner` / `screen` / `prescreen`).

### Escalation rule (asymmetric; same costs as `screening.py`)

Dropping a real anomaly costs 10, sending an empty one on costs 1 (one 8B pass and a reviewer's glance).

| Situation | Outcome |
|---|---|
| pre-screen says drop | stop, no model used |
| 2B says yes | escalate |
| 2B says cannot_tell | escalate |
| 2B unreachable or unparsable | escalate |
| 2B says no, certainty below high (0.7) | escalate |
| 2B says no, certainty high, pre-screen score >= 0.6 (strong motion evidence, or any rule-decided V1-V9 case, score 1.0) | escalate |
| 2B says no, certainty high, pre-screen score < 0.6 | **stop: screened out** |

Cost of the rule: the screen rarely saves an 8B pass (in the probes the 2B said "no" with low certainty almost every
time, so everything escalated). That is deliberate: the 2B is the less reliable reader, and nothing downstream sees a
dropped case. Policies, switchable by an administrator: `asymmetric` (above), `balanced` (any confident "no" stops),
`always` (the screen only annotates). `vlm_stop_score` (0.6) and `vlm_no_confidence` (0.7) are policy constants, not
measured: there is no labelled set to calibrate the 2B's miss rate (X5).

A screen-only decision is stored as assessment "insufficient" with `decided_by` "screen"/"prescreen" and a summary that says
so; it is never shown as the reasoner's. If the 8B is needed but cannot run (not installed, not enough free memory, did
not come up, request failed or timed out) the case keeps the screen result with `reasoner_unavailable` set and no verdict.

### On-demand 8B (`OnDemandServer`)

Starts `tools/vlm_server.sh --tier reason` on the first escalated case if `MemAvailable` covers the tier's need
(weights + measured overhead; otherwise the case is marked "reasoner unavailable" with the numbers), waits for `/health`
(`vlm_reason_start_sec`, 240), serves one case at a time (the Explainer thread is single), and stops it after
`vlm_reason_idle_sec` (300) without a case. A server someone else started on that URL is used and never stopped. The
dashboard process stops its 8B at exit.

### Images

For V10/V11 the "contact" frame (both vehicles boxed) goes first, so the default `vlm_max_images = 2` keeps it: contact
frame, then close-up. Other codes keep close-up, scene, extra moments, plate.

## Measured on this machine

i5-11400H (6 cores / 12 threads, AVX-512 + VNNI), 23 GB RAM, llama.cpp b11856, `--load-mode mmap+mlock`,
6 threads, ctx 8192, `--image-min-tokens 1024`, images at 768 px, two photographs per case.

**Caveat on latency.** Every latency below was measured while other jobs of the project (a detector evaluation and a
benchmark, 400-500 % CPU each) were running: the load average was 14-26 on 12 hardware threads. Prompt processing ran at
about 15-60 tokens/s and generation at 1-2 tokens/s, far below an idle machine. Treat the figures as an upper bound for a
busy dashboard host; they were not re-measured on an idle machine.

| | 2B screen | 8B reasoner |
|---|---|---|
| load time (weights mapped, server healthy) | 7-19 s | 11-48 s |
| resident memory (RSS) | 3.4 GB (anon 0.7, locked 1.1, rest mapped file) | 10.0-10.5 GB (anon 5.6, mapped file 4.9; llama.cpp repacks the 4-bit weights into anonymous memory) |
| files | 1.9 GB | 6.2 GB |
| median latency per case, 2 photographs | **84.7 s** (min 75.6, max 102.3, n=5, mean load 15) | **274.6 s** (min 256.8, max 295.8, n=4, mean load 15) |
| same, second run, load 7-9 (probe runs) | 1.6-2 s when the same photographs were seen before (prompt cache); 50-100 s when new | 142-261 s per case incl. a 160 s case at load 7 |

Both tiers resident together: 13.5 GB RSS, `MemAvailable` fell from 16.7 GB to 5.5 GB. The 8B alone needs about 10-11 GB
free to start; `vlm_cascade.TIER_OVERHEAD_GB` encodes the measured overhead. With the old 3B (4.7 GB) resident next to a
camera analysing video the 8B will usually be refused, and the case then says so.

Expected cost: a case that does not escalate costs one 2B pass; an escalated case costs the 2B pass plus one 8B pass
(plus a 10-50 s model load if the 8B was cold).

## Probes (`tools/vlm_probe.py --cascade`)

Evidence is rendered by the tool (ultralytics sample photographs, drawn boxes/paths, a synthetic road); nothing from `data/`.
Run through the cascade with the 2B on :8091 and the 8B started on demand on :8092.

| Probe | Acceptable | Result |
|---|---|---|
| unrelated-evidence (footballers, V1 bus) | insufficient / contradicted | contradicted, by the reasoner |
| wrong-class (bus photo, V6 helmet) | insufficient / contradicted | contradicted |
| plausible-case (bus, V1) | any | contradicted (the bus is at the kerb, not in a lane) |
| v11-boxed-bus-and-people | insufficient / contradicted | contradicted ("stationary bus and pedestrians, not two vehicles") |
| v11-synthetic-contact (two drawn cars) | any | first run: 2B said yes (high), the 8B timed out at the old 300 s limit, so "reasoner unavailable" and no verdict; second run: contradicted ("abstract outlines, no visible vehicles") in 153 s |
| v10-empty-road | insufficient / contradicted | contradicted |
| v10-weak-on-empty-road | insufficient / contradicted | contradicted (the 2B said "no" only at low certainty, so it escalated) |
| v10-path-on-bus-scene | any | contradicted |

**Sycophancy set 3/3 kept**: the footballers probe declined, as did the wrong-class probe. 8/8 probes passed in the first
cascade run.

The timeout finding: two V11 cases hit the 300 s request limit under load and were stored as "reasoner unavailable", which
is the intended behaviour (no invented verdict). The reasoner timeout is now its own setting, `vlm_reason_timeout_sec`
(900). In the second run the reasoner returned a verdict for the V11 boxed case in 164 s.

The first run also exposed that the 8B copied the template's `"confidence": 0.0` placeholder (every verdict showed 0 %);
the template now asks for a number instead. 8B confidence values (often 1.0) are still not calibrated.

## Switching the live machine over

1. Stop the 3B: `ss -ltnp | grep :8081`, then `kill <pid>` for that PID only.
2. `./tools/vlm_server.sh --tier screen` (port 8081, keep running; use `setsid nohup` or a service).
3. Restart the dashboard (`vlm_mode` is `auto`: it uses the cascade when the 2B is up and the 8B is installed, otherwise
   the single-model path on :8081, so nothing breaks if the 2B is down).
4. Optional environment: `VLM_BACKEND` (`auto`/`local`/`cloud`; `cloud` makes Anthropic/OpenAI the reasoner). Everything else
   is in the admin page, tab "Reasoning" (saved in `data/vlm_settings.json`): mode, escalation policy, stop score,
   confident-no level, tier model names and ports (loopback only), idle timeout.
5. Before a busy period, check `./.venv/bin/python tools/vlm_check.py`: it prints per-tier need against available memory.

## Limits

- Verdicts of the 2B and 8B are noisy: on identical photographs with 3-grey-level noise the 2B alternated yes/no, and its
  certainty is mostly "low". There is no labelled set of cases to score the screen or the explanations (X5), so the
  escalation constants are policy, not tuned.
- The pre-screen score is an ordinal strength, not a probability (see `screening.py`).
- Latency figures are from a loaded machine; an escalated case takes minutes on this CPU, not seconds.
- The 8B holds about 10 GB while it runs; it cannot start next to a heavy analysis job.
- Rendered V10/V11 frames are not real collisions or swerves; they test the plumbing and the refusal behaviour, not accuracy.
