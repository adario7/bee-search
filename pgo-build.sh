#!/usr/bin/env bash
# pgo-build.sh — build bee-search with Profile-Guided Optimization.
#
# !! CURRENTLY NOT RECOMMENDED ON x86 -- MEASURE BEFORE USING !!
#
# History on `perft 6` from a fresh Base+MLP game (x86, i3-1115G4):
#   * When Action was a 6-byte enum, PGO was a clear win:
#       plain 6.872 G instr / ~3.09 G cycles  ->  PGO 6.492 G instr  (-5.5%)
#   * After Action was packed into 4 bytes, PGO REVERSED into a big loss:
#       plain 6.792 G instr / 2.673 G cycles, IPC 2.54
#       PGO   6.335 G instr / 3.016 G cycles, IPC 2.10   (+12.8% cycles, 0/8 wins)
#     i.e. PGO removed instructions but wrecked IPC -- presumably inlining or
#     block-layout choices that no longer suit the smaller hot loop.
#
# So: keep this script for the ARM deployment target, where it must be
# re-measured from scratch (different core, different tradeoffs), but do not
# assume it helps. Always A/B it against a plain build with `perf stat`.
#
# PGO is NOT enabled for ordinary `cargo build --release`, deliberately: it
# needs a training run, and a stale profile silently pessimises the build.
#
# Requires llvm-profdata whose LLVM version matches rustc's:
#   rustc -vV | grep LLVM        # must match
#   llvm-profdata --version
# (If missing: rustup component add llvm-tools-preview, or install system LLVM.)

set -euo pipefail
cd "$(dirname "$0")"

PGO_DIR="${PGO_DIR:-$PWD/build/pgo}"
CPU="${CPU:-native}"
TRAIN=$(mktemp)
trap 'rm -f "$TRAIN"' EXIT
printf 'newgame Base+MLP\nperft 6\nquit\n' > "$TRAIN"

# Fail early on a version skew rather than producing a mysteriously bad build.
RUSTC_LLVM=$(rustc -vV | sed -n 's/^LLVM version: \([0-9]*\).*/\1/p')
PROFDATA_LLVM=$(llvm-profdata --version | sed -n 's/.*LLVM version \([0-9]*\).*/\1/p' | head -1)
if [[ "$RUSTC_LLVM" != "$PROFDATA_LLVM" ]]; then
    echo "ERROR: llvm-profdata is LLVM $PROFDATA_LLVM but rustc uses LLVM $RUSTC_LLVM." >&2
    echo "       The profile format differs between major versions; aborting." >&2
    exit 1
fi

echo "▶ 1/4  instrumented build"
rm -rf "$PGO_DIR"; mkdir -p "$PGO_DIR"
RUSTFLAGS="-C target-cpu=$CPU -C profile-generate=$PGO_DIR" \
    cargo build --release --bin bee-search 2>&1 | tail -1

echo "▶ 2/4  training run (perft 6)"
./build/release/bee-search < "$TRAIN" > /dev/null

echo "▶ 3/4  merging profiles"
llvm-profdata merge -o "$PGO_DIR/merged.profdata" "$PGO_DIR"/*.profraw

echo "▶ 4/4  optimized build"
RUSTFLAGS="-C target-cpu=$CPU -C profile-use=$PGO_DIR/merged.profdata" \
    cargo build --release --bin bee-search 2>&1 | tail -1

echo
echo "▶ done: ./build/release/bee-search is now PGO-optimized."
echo "  NOTE: any later plain \`cargo build --release\` overwrites it with a non-PGO binary."
