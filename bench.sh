#!/usr/bin/env bash
# bench.sh — build, run, save, and compare bee-search benchmarks.
#
# Usage:
#   ./bench.sh                          # run all benchmarks
#   ./bench.sh --no-search              # skip search (faster)
#   ./bench.sh --depth 6 --threads 8   # pass args to the bench binary
#   ./bench.sh --save-as baseline       # save result as a named baseline
#
# Results are saved to bench_results/TIMESTAMP_COMMIT.txt.
# The most recent previous result is used for comparison automatically.
# To compare against a specific file: BENCH_COMPARE=bench_results/foo.txt ./bench.sh

set -eo pipefail

RESULTS_DIR="bench_results"
BENCH_BIN="./build/release/bench"
SAVE_AS=""

# Pull out --save-as NAME if present (not forwarded to the bench binary)
BENCH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --save-as)
            SAVE_AS="$2"; shift 2 ;;
        *)
            BENCH_ARGS+=("$1"); shift ;;
    esac
done

mkdir -p "$RESULTS_DIR"

# ─── build ────────────────────────────────────────────────────────────────────
echo "▶ building..."
cargo build --release --bin bench 2>&1 \
    | grep -E "^error|^warning.*unused|Compiling bench|Finished" \
    || true
echo ""

# ─── run ──────────────────────────────────────────────────────────────────────
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$RESULTS_DIR/${TIMESTAMP}_${COMMIT}.txt"

echo "▶ running bench  (commit=$COMMIT, args: ${BENCH_ARGS[*]:-none})"
echo ""

{
    echo "# commit:  $COMMIT"
    echo "# date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# args:    ${BENCH_ARGS[*]:-}"
    echo ""
    "$BENCH_BIN" ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"} 2>/dev/null
} | tee "$OUTFILE"

# Optionally save under a stable alias (e.g. "baseline")
if [[ -n "$SAVE_AS" ]]; then
    cp "$OUTFILE" "$RESULTS_DIR/${SAVE_AS}.txt"
    echo ""
    echo "▶ also saved as: $RESULTS_DIR/${SAVE_AS}.txt"
fi

# ─── comparison ───────────────────────────────────────────────────────────────
# Use BENCH_COMPARE env var if set, otherwise the most recent previous result.
if [[ -n "${BENCH_COMPARE:-}" ]]; then
    PREV="$BENCH_COMPARE"
else
    PREV=$(ls "$RESULTS_DIR"/*.txt 2>/dev/null \
           | grep -v "$OUTFILE" \
           | sort \
           | tail -1 \
           || true)
fi

if [[ -n "${PREV:-}" && -f "$PREV" ]]; then
    echo ""
    echo "▶ comparison vs $(basename "$PREV")"
    echo ""
    printf "%-26s  %14s  %14s  %10s\n" "metric" "before" "after" "change"
    printf "%-26s  %14s  %14s  %10s\n" \
        "$(printf '%0.s─' {1..26})" \
        "$(printf '%0.s─' {1..14})" \
        "$(printf '%0.s─' {1..14})" \
        "$(printf '%0.s─' {1..10})"

    awk -v threshold=1.0 '
        NR == FNR {
            if (/^BENCH /) before[$2] = $3
            next
        }
        /^BENCH / {
            metric = $2
            after  = $3 + 0
            if (!(metric in before)) next
            b   = before[metric] + 0
            if (b == 0) next
            pct = (after - b) / b * 100
            if      (pct >  threshold) arrow = "▲"
            else if (pct < -threshold) arrow = "▼"
            else                       arrow = " "
            # movegen/do_undo: higher is better; search: higher is better too
            printf "%-26s  %14.1f  %14.1f  %s %+6.1f%%\n", \
                metric, b, after, arrow, pct
        }
    ' "$PREV" "$OUTFILE"
else
    echo ""
    echo "(no previous result found — run again after your next change to see a comparison)"
fi

echo ""
echo "▶ saved: $OUTFILE"
