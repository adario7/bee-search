# Benchmarking Guide

## Overview

The benchmark suite uses [Criterion.rs](https://bheisler.github.io/criterion.rs/book/),
which collects 100 samples per benchmark, runs Welch's t-test, and tells you
whether a measured change is statistically significant or just noise.

Three benchmark groups run on a fixed, deterministic set of 15 positions
(5 seeds × 3 game phases: opening, midgame, endgame):

| Group | What it measures | Plan items it catches |
|-------|------------------|-----------------------|
| `movegen` | `generate_moves()` calls/second | Items 2, 3, 4 |
| `do_undo` | `do_action + undo_action` round-trips/second | Item 1 |
| `search`  | `best_move()` at fixed depth, cold TT | Items 5, 6, 7 |

---

## Workflow

### 1. Save a baseline before your change

```sh
cargo bench --bench bee_bench -- --save-baseline before
```

This runs all three groups and saves the results under
`build/criterion/<group>/<benchmark>/before/`.

### 2. Make your change

### 3. Run again and compare

```sh
cargo bench --bench bee_bench -- --baseline before
```

Criterion prints a comparison for every benchmark:

```
movegen/midgame/s42     time:   [398.11 µs 399.04 µs 400.01 µs]
                        change: [-18.4% -17.9% -17.4%] (p = 0.00 < 0.05)
                        Performance has improved.

movegen/midgame/s137    time:   [511.22 µs 512.88 µs 514.61 µs]
                        change: [-1.2% -0.8% -0.3%] (p = 0.22 > 0.05)
                        No change in performance detected.
```

`p < 0.05` means the difference is real, not noise.
`p > 0.05` means it is indistinguishable from noise — don't count it.

Note: with 100 samples, Criterion is sensitive enough to flag a 1% change
as significant (p = 0.00). Statistical significance is not the same as
practical significance. For this engine, only care about changes ≥ 5% —
anything smaller is in the noise floor of the CPU itself (thermal throttling,
cache state, OS scheduling).

---

## Running individual groups

```sh
# Only movegen (fast, ~2 min)
cargo bench --bench bee_bench -- movegen

# Only do/undo (fast, ~2 min)
cargo bench --bench bee_bench -- do_undo

# Only search (slow, ~15 min — 10 samples × 60s measurement per position)
cargo bench --bench bee_bench -- search

# One specific benchmark
cargo bench --bench bee_bench -- movegen/midgame
```

---

## Reading the output

Each benchmark prints three numbers: `[low  mean  high]` — the 2.5th
percentile, mean, and 97.5th percentile of the 100 samples. A tight spread
(e.g. `[398 µs  399 µs  400 µs]`) means the measurement is clean.
A wide spread means CPU noise — try closing other apps and re-running.

Criterion also flags statistical outliers automatically.

---

## HTML reports

After any `cargo bench` run, open the HTML report for graphs:

```sh
open build/criterion/bee_bench/report/index.html
```

---

## Notes

- The `search` group uses `sample_size(10)` (not the default 100) because
  each call takes ~1 second. 10 samples is enough for a meaningful confidence
  interval at that timescale.
- Each search sample uses a **cold TT** (fresh `Engine::new()` per iteration)
  so results are reproducible and not influenced by TT warmth from previous runs.
- The positions are generated deterministically from seeded random play and
  never change, so baselines saved today are valid for comparison weeks later.
