use bee_search::board::{Action, Board, GameResult};
use bee_search::engine::{Depth, Engine};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::seq::IndexedRandom;
use rand::{rngs::StdRng, SeedableRng};
use std::sync::Arc;
use std::time::{Duration, Instant};

const SEEDS: &[u64] = &[42, 137, 256, 1337, 9999];
const PLIES: &[(usize, &str)] = &[(6, "opening"), (14, "midgame"), (24, "endgame")];
const SEARCH_DEPTH: Depth = 5;

/// Deterministic position set — identical across every run and every commit.
fn make_positions() -> Vec<(String, Board)> {
    let max = PLIES.last().unwrap().0;
    let mut out = Vec::new();
    for &seed in SEEDS {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut board = Board::new();
        for ply in 0..=max {
            if board.game_result() != GameResult::InProgress {
                break;
            }
            if let Some(&(_, phase)) = PLIES.iter().find(|(p, _)| *p == ply) {
                out.push((format!("{phase}/s{seed}"), board.clone()));
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
    for (id, mut board) in make_positions() {
        group.bench_function(id, |b| {
            b.iter(|| black_box(board.generate_moves()))
        });
    }
    group.finish();
}

// ─── do/undo ─────────────────────────────────────────────────────────────────
// Measures do_action + undo_action round-trips per second.
// Exercises: HashMap/array underworld, OccupancyVec, Zobrist hash update.
// This is what each node in the search tree pays in board-manipulation cost.

fn bench_do_undo(c: &mut Criterion) {
    let mut group = c.benchmark_group("do_undo");
    for (id, mut board) in make_positions() {
        let moves = board.generate_moves();
        if moves.is_empty() {
            continue;
        }
        // Use the first legal move — deterministic, always valid to undo.
        let mv = moves[0];
        group.bench_function(id, |b| {
            b.iter(|| {
                board.do_action(mv);
                board.undo_action();
            })
        });
    }
    group.finish();
}

// ─── search ──────────────────────────────────────────────────────────────────
// Measures best_move() at a fixed depth with a cold TT per sample.
// Exercises: TT hit rate, move ordering, eval (MLP), Lazy SMP threading.
// Each sample is slow (~1s), so Criterion uses fewer iterations but still
// produces a proper confidence interval.

fn bench_search(c: &mut Criterion) {
    let threads = num_cpus::get();
    let mut group = c.benchmark_group("search");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    for (id, board) in make_positions() {
        group.bench_function(id, |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    // Fresh engine every iteration: cold TT, reproducible.
                    let engine = Arc::new(Engine::new());
                    let mut b = board.clone();
                    let t = Instant::now();
                    engine.clone().best_move(
                        &mut b,
                        SEARCH_DEPTH,
                        Duration::from_secs(300),
                        threads,
                    );
                    total += t.elapsed();
                }
                total
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_movegen, bench_do_undo, bench_search);
criterion_main!(benches);
