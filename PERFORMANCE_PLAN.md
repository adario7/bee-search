# bee-search Performance Improvement Plan

## Context

This plan is for improving the search speed of the Hive engine at
`~/Desktop/Hive/bee-search`. The goal is maximizing nodes/second in
`best_move()` (multi-threaded, targeting a 5-second budget). Perft
speed is not the goal.

The engine is written in Rust. Key source files:
- `src/board.rs` — board state
- `src/abstractions.rs` — TileSet, OccupancyVec, CachedValue
- `src/movegen.rs` — move generation (impl Board)
- `src/engine.rs` — search (PVS + LMR + Lazy SMP)
- `src/tt.rs` — transposition table
- `src/tile.rs` — grid/tile primitives
- `src/eval.rs` — static evaluation (MLP)

---

## Item 1 — Replace HashMap underworld with fixed-size inline array

**File:** `src/board.rs`

**Priority:** HIGH — affects every do/undo_action call in the search tree,
and every board clone for thread spawning.

**Current code (board.rs:41):**
```rust
pub underworld: HashMap<Tile, Vec<Piece>>,
```

`add_piece()` (board.rs:159) calls:
```rust
self.underworld.entry(tile).or_insert_with(Vec::new).push(*curr);
```
`remove_piece()` (board.rs:186-190) calls:
```rust
if let Some(vec) = self.underworld.get_mut(&tile) {
    let last = vec.pop().unwrap();
    if vec.is_empty() { self.underworld.remove(&tile); }
    ...
}
```

Both involve hash computation, HashMap bucket lookup, and Vec heap
allocation. Board cloning (board.rs is `Clone`) deep-clones the HashMap,
which allocates for every thread spawned in `lazy_smp`.

**What to change:**

In `src/board.rs`, replace the HashMap field with two flat arrays:

```rust
// Replace:
pub underworld: HashMap<Tile, Vec<Piece>>,

// With:
pub underworld: [[Piece; 8]; GRID_SIZE],
pub underworld_depth: [u8; GRID_SIZE],
```

Max stack depth is 8 in practice (beetles can stack at most a few high).
Total memory: `1024 * 8 * 1 + 1024 * 1 = ~9KB` — a flat cache-friendly
block instead of pointer chains.

Update `add_piece()` (board.rs:156-171):
```rust
fn add_piece(&mut self, tile: Tile, piece: Piece) {
    let curr = &mut self.world[tile as usize];
    if curr.is_some() {
        let d = self.underworld_depth[tile as usize] as usize;
        self.underworld[tile as usize][d] = *curr;
        self.underworld_depth[tile as usize] += 1;
        Self::remove_occupancy(&mut self.occupied_tiles, *curr, tile);
    }
    *curr = piece;
    Self::add_occupancy(&mut self.occupied_tiles, *curr, tile);
    if piece.ptype() == Pct::Queen {
        self.queens[piece.color().index()] = Some(tile);
    }
    self.height[tile as usize] += 1;
    self.zobrist_hash ^= self.zobrist(tile, piece, self.height(tile) as u32);
}
```

Update `remove_piece()` (board.rs:173-196):
```rust
fn remove_piece(&mut self, tile: Tile) {
    self.zobrist_hash ^= self.zobrist(tile, self.world[tile as usize], self.height(tile) as u32);
    self.height[tile as usize] -= 1;
    let curr = &mut self.world[tile as usize];
    debug_assert!(curr.is_some());
    if curr.ptype() == Pct::Queen {
        self.queens[curr.color().index()] = None;
    }
    Self::remove_occupancy(&mut self.occupied_tiles, *curr, tile);
    let d = self.underworld_depth[tile as usize];
    if d > 0 {
        let last = self.underworld[tile as usize][d as usize - 1];
        self.underworld_depth[tile as usize] -= 1;
        *curr = last;
        Self::add_occupancy(&mut self.occupied_tiles, *curr, tile);
    } else {
        *curr = Piece::empty();
    }
}
```

Remove `use std::collections::HashMap;` from board.rs:1 if no longer used.

The board is `Clone`-derived (board.rs:35). After this change, cloning is a
plain memcpy of two flat arrays instead of a HashMap deep-clone. This
directly benefits `lazy_smp` (engine.rs:552: `let th_board = board.clone()`).

**Test:** `cargo test test_board_do_undo` and `cargo test test_zobrist_hash`.

---

## Item 2 — Remove the immovable TileSet cache; compute once per generate_movements call

**File:** `src/movegen.rs`, `src/board.rs`, `src/abstractions.rs`

**Priority:** HIGH — `find_cut_vertexes()` is called at every node and
currently clones a 128-byte `[u32; 32]` even on a cache hit.

**Current code (movegen.rs:130-138):**
```rust
pub(crate) fn find_cut_vertexes(&mut self) -> TileSet {
    if let Some(immovable) = self.immovable.get_if_valid(self.get_cached_hash()) {
        return immovable.clone(); // 128-byte memcpy on every cache hit
    }
    let immovable = self.find_cut_vertexes_unmutable();
    self.immovable.update(self.get_cached_hash(), immovable.clone());
    immovable
}
```

The cache stores a `TileSet` tied to the Zobrist hash. It is invalidated by
every `do_action`/`undo_action`. Since the search calls
`generate_movements()` once per node (not multiple times for the same
position with MLP eval), the cache provides no cross-node benefit.
The clone fires on every single call.

**What to change:**

1. Remove the `immovable: CachedValue<TileSet>` field from `Board` (board.rs:58).

2. Remove `CachedValue` and `CacheHash` from `abstractions.rs` if nothing
   else uses them (verify with `grep -r "CachedValue\|CacheHash" src/`).

3. In `generate_movements()` (movegen.rs:643), change:
```rust
// Old:
let mut immovable = self.find_cut_vertexes();

// New:
let mut immovable = self.find_cut_vertexes_unmutable();
```

4. In `generate_movements_n()` (movegen.rs:821), same change.

5. In `generate_movements_by_pct()` (movegen.rs:891), same change.

6. Delete the cached `find_cut_vertexes()` function (movegen.rs:130-138).
   Rename `find_cut_vertexes_unmutable()` to `find_cut_vertexes()` for clarity.

7. Remove `pub(crate) fn get_cached_hash(&self) -> CacheHash` from
   board.rs:306-308 if it becomes unused.

**Impact:** Eliminates a 128-byte clone + Zobrist hash comparison at every
node. Also simplifies the code considerably.

---

## Item 3 — Fix TileSet bit layout for cache locality

**File:** `src/abstractions.rs`

**Priority:** HIGH — TileSet is used in every hot path (slidable_adjacent,
cut vertex DFS, placements, dedup). The bit layout is currently transposed,
causing adjacent tiles to fall into different cache words.

**Current code (abstractions.rs:7-35):**
```rust
const TILESET_NUM_WORDS: usize = GRID_SIZE / 32; // 32
const TILESET_SHIFT: u32 = GRID_SIZE.trailing_zeros() - 5; // 5
const TILESET_MASK: usize = TILESET_NUM_WORDS - 1;          // 31

fn set(&mut self, tile: Tile) {
    // word = tile % 32,  bit = tile / 32  ← TRANSPOSED
    self.table[tile as usize & TILESET_MASK] |= 1 << (tile as u32 >> TILESET_SHIFT);
}
```

With this layout, East-adjacent tiles (offset +1) land in consecutive
words (different cache lines). Standard packing puts East-adjacent tiles
in the same 32-bit word, so checking or setting 6 neighbors typically
touches 1–2 words instead of 6.

**What to change:**

Replace the three bit-manipulation methods with standard packing:

```rust
pub(crate) fn set(&mut self, tile: Tile) {
    self.table[(tile as usize) >> 5] |= 1u32 << ((tile as u32) & 31);
}

pub(crate) fn rem(&mut self, tile: Tile) {
    self.table[(tile as usize) >> 5] &= !(1u32 << ((tile as u32) & 31));
}

pub(crate) fn get(&self, tile: Tile) -> bool {
    (self.table[(tile as usize) >> 5] >> ((tile as u32) & 31)) & 1 != 0
}
```

Remove the now-unused constants `TILESET_SHIFT`, `TILESET_MASK`,
`TILESET_NUM_WORDS` (or keep `TILESET_NUM_WORDS` if used elsewhere).

**This change is a global correctness risk** — every TileSet in the
codebase must use consistent indexing. After the change, run ALL tests:
```
cargo test
```
and run the perft binary against a known-good reference to confirm move
counts are unchanged.

---

## Item 4 — Eliminate duplicate `generate_*_n` counting functions

**File:** `src/movegen.rs`

**Priority:** MEDIUM — doubles maintenance burden and inhibits inlining.
With monomorphization the count path gets Vec::push optimized away entirely.

**Current situation:** Every move generator has two versions:
`generate_walk1` / `generate_walk1_n`, `generate_walk3` /
`generate_walk3_n`, `generate_jumps` / `generate_jumps_n`, etc.
(10 duplicated function pairs, ~500 lines of near-identical logic).

**What to change:**

Define a `MoveSink` trait with a single `push` method in `src/movegen.rs`
(or a new `src/movesink.rs`):

```rust
pub(crate) trait MoveSink {
    fn push(&mut self, action: Action);
}

pub(crate) struct MoveVec<'a>(pub &'a mut Vec<Action>);
impl MoveSink for MoveVec<'_> {
    #[inline(always)]
    fn push(&mut self, action: Action) { self.0.push(action); }
}

pub(crate) struct MoveCount(pub usize);
impl MoveSink for MoveCount {
    #[inline(always)]
    fn push(&mut self, _: Action) { self.0 += 1; }
}
```

Replace each pair, for example:
```rust
// Replace:
fn generate_walk1(&self, hex: Tile, turns: &mut Vec<Action>) { ... }
fn generate_walk1_n(&self, hex: Tile) -> usize { ... }

// With:
fn generate_walk1<S: MoveSink>(&self, hex: Tile, sink: &mut S) { ... }
```

Call site for collecting:
```rust
self.generate_walk1(hex, &mut MoveVec(&mut turns));
```
Call site for counting:
```rust
let mut sink = MoveCount(0);
self.generate_walk1(hex, &mut sink);
let n = sink.0;
```

Start with the simpler functions (`generate_walk1`, `generate_jumps`,
`generate_walk3`), then tackle `generate_walk_all` and `generate_mosquito`.

---

## Item 5 — Persist and decay per-thread history tables across searches

**File:** `src/engine.rs`

**Priority:** MEDIUM — avoids multi-megabyte allocations at the start of
every `best_move()` call.

**Current code (engine.rs:559-560):**
```rust
history_h: vec![0; 2 * (GRID_SIZE + PCT_COUNT) * GRID_SIZE + 1],
countermove: vec![Action::Pass; 2 * (GRID_SIZE + PCT_COUNT) * GRID_SIZE + 1],
```

`GRID_SIZE = 1024`, `PCT_COUNT ≈ 8`, so this is `2 * 1032 * 1024 + 1 ≈
2.1M` entries per table. With `i64` per entry: `~16MB` per thread per
call. With 8 threads: `~256MB` allocated and zeroed at every `best_move()`
call.

**What to change:**

1. Add pre-allocated history and countermove tables to `Engine`:

```rust
pub struct Engine {
    tt: TTable,
    nnodes: AtomicU64,
    qsnodes: AtomicU64,
    reductions: Vec<f32>,
    move_votes: Vec<AtomicU64>,
    // Pre-allocated per-thread tables, reused across best_move() calls
    thread_history:     Vec<Vec<i64>>,
    thread_countermove: Vec<Vec<Action>>,
}
```

2. In `Engine::new()`, pre-allocate for `num_cpus::get()` threads:

```rust
let hist_size = 2 * (GRID_SIZE + PCT_COUNT) * GRID_SIZE + 1;
let max_threads = num_cpus::get();
let thread_history     = (0..max_threads).map(|_| vec![0i64; hist_size]).collect();
let thread_countermove = (0..max_threads).map(|_| vec![Action::Pass; hist_size]).collect();
```

3. In `lazy_smp()` (engine.rs:535), before spawning threads, decay history
   instead of zeroing (preserves quality across successive searches):

```rust
for h in &mut self.thread_history[i] { *h /= 2; }
// countermove is left as-is — stale entries are harmless
```

4. Pass `&mut self.thread_history[i]` and `&mut self.thread_countermove[i]`
   into `ThreadData` as mutable slices instead of fresh owned `Vec`s.

---

## Item 6 — Compact TEntry to reduce TT memory footprint

**File:** `src/tt.rs`

**Priority:** MEDIUM — smaller entries mean more positions fit in L3 cache,
directly increasing TT hit rate during search.

**Current TEntry (tt.rs:15-23):**
```rust
pub struct TEntry {
    pub pv: Action,    // Action enum: ~8 bytes with alignment padding
    pub hash: u32,     // 4 bytes
    pub value: Value,  // i16 — 2 bytes
    pub eval: Eval,    // i16 — 2 bytes
    pub depth: Depth,  // u8 — 1 byte
    pub flag: TTFlag,  // u8 — 1 byte
}
// Actual size: ~16-20 bytes depending on Action alignment
```

`TTable::new(1 << 28)` means 256MB / sizeof(TEntry) entries. Shrinking
TEntry from 20 to 12 bytes increases the number of cached positions by ~67%.

**What to change:**

Pack `Action` into a `u32`. Tile is `u16` (10 bits used in a 32×32 grid),
PieceType fits in 4 bits:

```
Pass:               bits 31-30 = 0b00
Place(tile, ptype): bits 31-30 = 0b01, bits 13-10 = ptype (4 bits), bits 9-0 = tile (10 bits)
Move(from, to):     bits 31-30 = 0b10, bits 19-10 = from (10 bits), bits 9-0 = to (10 bits)
```

Add to `src/tt.rs`:
```rust
#[derive(Default, Copy, Clone, Debug)]
pub struct PackedAction(u32);

impl PackedAction {
    pub fn pack(a: Action) -> Self {
        match a {
            Action::Pass => Self(0),
            Action::Place(tile, ptype) =>
                Self((1 << 30) | ((ptype as u32) << 10) | tile as u32),
            Action::Move(from, to) =>
                Self((2 << 30) | ((from as u32) << 10) | to as u32),
        }
    }
    pub fn unpack(self) -> Action {
        match self.0 >> 30 {
            0 => Action::Pass,
            1 => Action::Place(
                (self.0 & 0x3FF) as Tile,
                unsafe { std::mem::transmute(((self.0 >> 10) & 0xF) as u8) },
            ),
            2 => Action::Move(
                ((self.0 >> 10) & 0x3FF) as Tile,
                (self.0 & 0x3FF) as Tile,
            ),
            _ => unreachable!(),
        }
    }
}
```

New TEntry (exactly 16 bytes, power-of-two aligned):
```rust
#[repr(C)]
pub struct TEntry {
    pub pv:    PackedAction, // u32 — 4 bytes
    pub hash:  u32,          // 4 bytes
    pub value: i16,          // 2 bytes
    pub eval:  i16,          // 2 bytes
    pub depth: u8,           // 1 byte
    pub flag:  u8,           // 1 byte (TTFlag as u8)
    pub _pad:  u16,          // 2 bytes — explicit padding to 16 bytes
}
```

Update all `entry.pv` reads to `entry.pv.unpack()` and all `TTable::put`
calls to store `PackedAction::pack(pv)`.

---

## Item 7 — Move ordering: sort by precomputed integer score

**File:** `src/engine.rs:146-185`

**Priority:** MEDIUM — sort is called at every internal node; the current
closure-based comparator re-evaluates multiple conditions per comparison.

**Current code (engine.rs:149-184):**
```rust
let mut moves = moves.into_iter()
    .map(|mv| MoveInfo { mv,
        quiet: td.board.is_quiet(&mv),
        killer: killers.iter().find(...).map(...).unwrap_or(0),
        hist: td.history_h[Self::hist_index(td.board.color(), mv)] })
    .collect::<Vec<_>>();
moves.sort_by(|a, b| {
    if a.mv == pv { return Ordering::Less; }
    else if b.mv == pv { return Ordering::Greater; }
    if !a.quiet && b.quiet { return Ordering::Less; }
    // ... more branches
});
```

**What to change:**

Assign a single integer score per move during the initial map, then use
`sort_unstable_by_key` (no extra allocation, faster for small lists):

```rust
#[derive(Clone, Copy, Debug)]
struct MoveInfo {
    mv:    Action,
    score: i32,
    quiet: bool, // keep for update_heuristics
}

fn order_moves(
    &self, moves: Vec<Action>, td: &mut ThreadData,
    pv: Option<Action>, killers: &KillerT,
) -> Vec<MoveInfo> {
    let pv_mv      = pv.unwrap_or(Action::Pass);
    let countermove = td.countermove[Self::last_move_index(&td.board)];
    let mut infos: Vec<MoveInfo> = moves.into_iter().map(|mv| {
        let quiet = td.board.is_quiet(&mv);
        let score = if mv == pv_mv {
            i32::MAX
        } else {
            let noisy   = if !quiet { 10_000_000i32 } else { 0 };
            let killer  = killers.iter()
                .find(|(m, _)| *m == mv)
                .map(|(_, i)| *i as i32)
                .unwrap_or(0) * 100_000;
            let hist    = (td.history_h[Self::hist_index(td.board.color(), mv)]
                .clamp(-5_000_000, 5_000_000)) as i32;
            let counter = if mv == countermove { 50_000i32 } else { 0 };
            noisy + killer + hist + counter
        };
        MoveInfo { mv, score, quiet }
    }).collect();
    infos.sort_unstable_by_key(|m| -m.score);
    infos
}
```

Update `update_heuristics()` (engine.rs:187) and `qsearch()` (engine.rs:258)
which currently access `mvi.quiet` and `mvi.killer` — `killer` can be
dropped from `MoveInfo` since `update_heuristics` only needs the best move's
identity, not its ordering metadata. `quiet` must be kept.

---

## Item 8 — Increase generate_walk_all queue size from 16 to 64

**File:** `src/movegen.rs`

**Priority:** LOW-MEDIUM — the current size of 16 could silently overflow
in complex board states where an ant can reach more than 16 tiles.

**Current code (movegen.rs:375):**
```rust
let mut queue = [0; 16];
```

**Change all three occurrences:**

`generate_walk_all` (movegen.rs:375):
```rust
let mut queue = [0u16; 64];
```

`generate_walk_all_n` (movegen.rs:402):
```rust
let mut queue = [0u16; 64];
```

`generate_cc_slidable` (movegen.rs:782):
```rust
let mut queue = [0u16; 128]; // connected component can be larger
```

---

## Item 9 — Remove bitvec dependency and dead TileSet in tile.rs

**Files:** `src/tile.rs`, `Cargo.toml`

**Priority:** LOW — cleanup, reduces compile time.

`src/tile.rs:176-194` defines a second `TileSet` using `bitvec`, but all
hot-path code uses `abstractions::TileSet`. Verify it is unused:

```sh
grep -rn "tile::TileSet\|use.*tile.*TileSet" src/
```

If no results, delete lines 176–194 of `tile.rs` and the `use bitvec::prelude::*;`
import at `tile.rs:20`. Then remove from `Cargo.toml:8`:
```toml
bitvec = "1"  # ← remove
```

---

## Item 10 — Cap qsearch max depth

**File:** `src/engine.rs`

**Priority:** LOW — prevents runaway qsearch at high plies.

**Current code (engine.rs:347):**
```rust
return self.qsearch(td, ply, ply * 2, alpha, beta, killers);
```

`ply * 2` means the qsearch horizon grows with depth, multiplying the
tree size at deeper searches.

**Change:**
```rust
const MAX_QS_EXTRA: Depth = 6;
return self.qsearch(td, ply, ply + MAX_QS_EXTRA, alpha, beta, killers);
```

Tune `MAX_QS_EXTRA` experimentally — start at 4 or 6 and measure
playing strength vs nodes/second.

---

## Item 11 — Aspiration window: reduce retry limit from 99 to 6

**File:** `src/engine.rs:486`

**Priority:** LOW — cosmetic, prevents theoretically unbounded retry loops.

Six doublings of the initial window already reaches ±INF from any
starting point (window starts at ±30; 6 doublings = ±1920 which covers
the full score range of ±32000).

**Current:**
```rust
for i in 0..99 {
```

**Change:**
```rust
for i in 0..6 {
```

---

## Execution order

Do the items in this order (each should be a separate commit with
`cargo test` passing before moving on):

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 1 | HashMap underworld → flat arrays | Medium | High |
| 2 | Remove immovable cache | Low | High |
| 3 | Fix TileSet bit layout | Low | High |
| 4 | Persist history tables | Low | Medium |
| 5 | Compact TEntry | Low | Medium |
| 6 | Move ordering score-based sort | Low | Medium |
| 7 | Unify movegen/count via MoveSink | Large | Medium |
| 8 | Queue size fix | Trivial | Safety |
| 9 | Remove dead TileSet/bitvec | Trivial | Cleanup |
| 10 | Cap qsearch depth | Trivial | Tuning |
| 11 | Aspiration window limit | Trivial | Cosmetic |

After Items 1–3, run `cargo run --bin speed-bench --release` to measure
improvement before continuing.
