use bee_search::board::{Action, Board, GameResult};
use bee_search::engine::{Depth, Engine};
use bee_search::perft::perft;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::seq::IndexedRandom;
use rand::{rngs::StdRng, SeedableRng};
use std::sync::Arc;
use std::time::{Duration, Instant};

const SEEDS: &[u64] = &[1, 2, 3, 4, 5];
const PLIES: &[(usize, &str)] = &[(6, "opening"), (14, "midgame"), (24, "endgame")];
const SEARCH_DEPTH: Depth = 4;
const PERFT_DEPTH: usize = 3;
const WARMUP: Duration = Duration::from_secs(2);
const MEASUREMENT: Duration = Duration::from_secs(5);

/// Deterministic position set — identical across every run and every commit.
fn make_positions() -> Vec<Board> {
    let max = PLIES.last().unwrap().0;
    let mut out = Vec::new();
    for &seed in SEEDS {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut board = Board::new();
        for ply in 0..=max {
            if board.game_result() != GameResult::InProgress {
                break;
            }
            if PLIES.iter().any(|(p, _)| *p == ply) {
                out.push(board.clone());
            }
            let moves = board.generate_moves();
            let mv = if moves.is_empty() {
                Action::Pass
            } else {
                *moves.choose(&mut rng).unwrap()
            };
            board.do_action(mv);
        }
    }
    out
}

// ─── movegen ─────────────────────────────────────────────────────────────────
// Measures generate_moves() calls per second.
// Exercises: cut vertex DFS, TileSet ops, placement/movement generation.
fn bench_movegen(c: &mut Criterion) {
    let mut group = c.benchmark_group("movegen");
    group.warm_up_time(WARMUP);
    group.measurement_time(MEASUREMENT);
    let mut positions = make_positions();
    group.bench_function("all", |b| {
        b.iter(|| {
            for board in &mut positions {
                black_box(board.generate_moves());
            }
        })
    });
    group.finish();
}

// ─── do/undo ─────────────────────────────────────────────────────────────────
// Measures do_action + undo_action round-trips per second.
// Exercises: HashMap/array underworld, OccupancyVec, Zobrist hash update.
// This is what each node in the search tree pays in board-manipulation cost.
fn bench_do_undo(c: &mut Criterion) {
    let mut group = c.benchmark_group("do_undo");
    group.warm_up_time(WARMUP);
    group.measurement_time(MEASUREMENT);

    let mut positions_and_moves: Vec<(Board, Action)> = make_positions()
        .into_iter()
        .filter_map(|mut board| {
            let moves = board.generate_moves();
            moves.first().map(|&mv| (board, mv))
        })
        .collect();

    group.bench_function("all", |b| {
        b.iter(|| {
            for (board, mv) in &mut positions_and_moves {
                board.do_action(*mv);
                board.undo_action();
            }
        })
    });
    group.finish();
}

// ─── search ──────────────────────────────────────────────────────────────────
// Measures best_move() at a fixed depth with a cold TT per sample.
// Exercises: TT hit rate, move ordering, eval (MLP), Lazy SMP threading.
// Each sample is slow (~1s), so Criterion uses fewer iterations but still
// produces a proper confidence interval.
fn bench_search(c: &mut Criterion) {
    let threads = 1;
    let mut group = c.benchmark_group("search");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(20));

    let positions = make_positions();
    group.bench_function("all", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                for board in &positions {
                    // Fresh engine every iteration: cold TT, reproducible.
                    let engine = Arc::new(Engine::new());
                    let mut b = board.clone();
                    let t = Instant::now();
                    engine.clone().best_move(
                        &mut b,
                        SEARCH_DEPTH,
                        Duration::from_secs(300),
                        threads,
                        false,
                        None
                    );
                    total += t.elapsed();
                }
            }
            total
        })
    });
    group.finish();
}

// ─── perft ───────────────────────────────────────────────────────────────────
// Measures perft(depth=1) for all positions.
// This is a proxy for movegen + do/undo performance.
fn bench_perft(c: &mut Criterion) {
    let mut group = c.benchmark_group("perft");
    group.warm_up_time(WARMUP);
    group.measurement_time(MEASUREMENT);
    let mut positions = make_positions();
    group.bench_function("all", |b| {
        b.iter(|| {
            for board in &mut positions {
                black_box(perft(board, PERFT_DEPTH));
            }
        })
    });
    group.finish();
}

criterion_group!(benches, bench_movegen, bench_do_undo, bench_search, bench_perft);
criterion_main!(benches);
