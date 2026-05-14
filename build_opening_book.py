#!/usr/bin/env python3
"""
Build an opening book from a directory of tournament game files.

Each file contains a single game in UHP format:
  GameType;GameResult;TurnString;move1;move2;move3;...

The book maps a move-sequence prefix (";"-joined) to the best next move,
defined as the move that appears in at least THRESHOLD games after that prefix.
Among candidates that meet the threshold, the most frequent one wins.

Output: opening_book.json  —  {move_sequence: best_next_move}
"""

import json
import os
import sys
from collections import defaultdict, Counter


def build_book(games_dir: str, threshold: int = 200) -> dict[str, str]:
    # prefix (";"-joined moves) -> Counter of next moves
    next_move_counts: dict[str, Counter] = defaultdict(Counter)

    files = [
        os.path.join(games_dir, f)
        for f in os.listdir(games_dir)
        if f.endswith(".txt")
    ]

    if not files:
        print(f"No .txt files found in {games_dir}", file=sys.stderr)
        return {}

    for path in files:
        with open(path) as fh:
            line = fh.read().strip()
        if not line:
            continue

        parts = line.split(";")
        # parts[0] = game type, parts[1] = result, parts[2] = turn string
        moves = parts[3:]

        for i, next_move in enumerate(moves):
            prefix = ";".join(moves[:i])
            next_move_counts[prefix][next_move] += 1

    book: dict[str, str] = {}
    for prefix, counter in next_move_counts.items():
        best_move, count = counter.most_common(1)[0]
        if count >= threshold:
            book[prefix] = best_move

    return book


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build Hive opening book from tournament games")
    parser.add_argument(
        "games_dir",
        nargs="?",
        default=os.path.expanduser("~/Downloads/pro_matches/board_data_tournaments"),
        help="Directory containing tournament game .txt files",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=200,
        help="Minimum number of games agreeing on the next move (default: 200)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "opening_book.json"),
        help="Output JSON file path (default: opening_book.json next to this script)",
    )
    args = parser.parse_args()

    print(f"Reading games from: {args.games_dir}")
    print(f"Threshold: {args.threshold}")

    book = build_book(args.games_dir, threshold=args.threshold)

    with open(args.output, "w") as fh:
        json.dump(book, fh, separators=(",", ":"))

    print(f"Book entries: {len(book)}")
    print(f"Written to:   {args.output}")

    if book:
        print("\nSample entries (first 10 by prefix length):")
        for prefix, move in sorted(book.items(), key=lambda x: len(x[0]))[:10]:
            display = f'"{prefix}"' if prefix else '""  (empty board)'
            print(f"  {display} -> {move}")


if __name__ == "__main__":
    main()
