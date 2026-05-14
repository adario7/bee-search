#!/usr/bin/env python3
"""
Compare opening book moves against what the engine would actually play.

For each position in the book:
  - replay the prefix moves into the engine via UHP
  - ask for bestmove → engine_move
  - on mismatch: play each move and call eval to score it
    (negated because after playing a move it's the opponent's turn)
"""

import json
import subprocess
import sys
import os
import argparse
from pathlib import Path


def send(proc, cmd: str) -> list[str]:
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()
    lines = []
    while True:
        line = proc.stdout.readline().rstrip("\n")
        if line in ("ok", "err"):
            break
        lines.append(line)
    return lines


def eval_move(proc, move: str, eval_depth: int) -> int:
    """Play move, eval, undo. Returns score from the perspective of the player who just moved."""
    send(proc, f"play {move}")
    score_lines = send(proc, f"eval {eval_depth}")
    send(proc, "undo")
    return -int(score_lines[-1])  # negate: eval is from opponent's POV after the move


def main():
    parser = argparse.ArgumentParser(description="Compare opening book vs engine")
    parser.add_argument(
        "--book",
        default=Path(__file__).parent / "opening_book.json",
        help="Path to opening_book.json",
    )
    parser.add_argument(
        "--engine",
        default=Path(__file__).parent / "build/release/bee-search",
        help="Path to engine binary",
    )
    parser.add_argument(
        "--time",
        default="0:00:50",
        help="Time budget for bestmove in hh:mm:ss (default: 0:00:15)",
    )
    parser.add_argument(
        "--eval-depth",
        type=int,
        default=0,
        help="Depth for eval after each move (0=static, default: 0)",
    )
    args = parser.parse_args()

    with open(args.book) as f:
        book: dict[str, str] = json.load(f)

    if not os.path.exists(args.engine):
        print(f"Engine not found at {args.engine}")
        print("Build it with:  cd ~/Desktop/Hive/bee-search && cargo build --release")
        sys.exit(1)

    proc = subprocess.Popen(
        [args.engine],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    # consume initial "info" block
    while True:
        line = proc.stdout.readline().rstrip("\n")
        if line == "ok":
            break

    matches = 0
    mismatches = 0

    entries = sorted(book.items(), key=lambda x: (x[0].count(";"), x[0]))
    for prefix, book_move in entries:
        depth_n = prefix.count(";") + 1 if prefix else 0
        prefix_display = repr(prefix) if prefix else '"" (empty board)'

        send(proc, "newgame Base+MLP")
        for move in (prefix.split(";") if prefix else []):
            send(proc, f"play {move}")

        engine_lines = send(proc, f"bestmove time {args.time}")
        engine_move = engine_lines[-1] if engine_lines else ""

        if engine_move == book_move:
            matches += 1
            print(f"depth {depth_n}  MATCH      {book_move!r}")
            print(f"           prefix: {prefix_display}")
        else:
            mismatches += 1
            engine_score = eval_move(proc, engine_move, args.eval_depth)
            book_score = eval_move(proc, book_move, args.eval_depth)
            depth_label = "static" if args.eval_depth == 0 else f"depth {args.eval_depth}"
            print(f"depth {depth_n}  MISMATCH   ({depth_label} eval)")
            print(f"           engine: {engine_move!r:30s} score={engine_score:+d}")
            print(f"           book:   {book_move!r:30s} score={book_score:+d}")
            print(f"           prefix: {prefix_display}")

        print()

    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.wait()

    total = matches + mismatches
    print(f"Results: {matches}/{total} match ({100*matches//total if total else 0}%)")


if __name__ == "__main__":
    main()
