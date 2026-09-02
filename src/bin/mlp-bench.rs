use bee_search::board::{Action, Board, GameResult};
use bee_search::engine::{Depth, Engine};
use bee_search::eval_mlp::{mlp_inference, set_accumulation, take_accumulated_features, take_accumulated_layer_zero_stats};
use clap::Parser;
use std::hint::black_box;
use rand::seq::IndexedRandom;
use rand::{rngs::StdRng, SeedableRng};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(author, version, about = "MLP feature accumulation & inference throughput benchmark", long_about = None)]
struct Args {
    /// Search depth to evaluate on each position
    #[arg(short, long, default_value_t = 8)]
    depth: Depth,

    /// Number of MLP inference iterations to execute in benchmark loop
    #[arg(short, long, default_value_t = 10_000_000)]
    iterations: usize,
}

fn generate_positions() -> Vec<(usize, Board)> {
    let target_plies = [5, 6, 7, 19, 20, 21];
    let max_ply = 21;
    let seed = 1u64;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut board = Board::new();
    let mut out = Vec::new();

    for ply in 0..=max_ply {
        if board.game_result() != GameResult::InProgress {
            break;
        }
        if target_plies.contains(&ply) {
            out.push((ply, board.clone()));
        }
        let moves = board.generate_moves();
        let mv = if moves.is_empty() {
            Action::Pass
        } else {
            *moves.choose(&mut rng).unwrap()
        };
        board.do_action(mv);
    }
    out
}

fn main() {
    let args = Args::parse();
    println!("=== MLP FEATURE ACCUMULATION & INFERENCE BENCHMARK ===");
    println!("Search depth: {}", args.depth);
    println!("Target inference iterations K: {}\n", args.iterations);

    let positions = generate_positions();
    println!("Selected Positions:");
    for (ply, board) in &positions {
        let piece_count = board.occupied_tiles[0].len() + board.occupied_tiles[1].len();
        let category = if *ply <= 10 { "Early Game" } else { "Mid Game" };
        println!("  - Ply {:2} ({:<10}): {} pieces on board", ply, category, piece_count);
    }
    println!();

    // Phase 1: Search & Feature Harvesting
    println!("Running search at depth {} to harvest feature vectors...", args.depth);
    set_accumulation(true);
    let engine = Arc::new(Engine::new());
    for (ply, board) in &positions {
        let mut b = board.clone();
        print!("  Searching ply {:2}... ", ply);
        let start = Instant::now();
        let (value, _) = engine.clone().best_move(
            &mut b,
            args.depth,
            Duration::from_secs(300),
            1,
            false,
            None,
        );
        println!("done ({:.2?}) -> {}", start.elapsed(), value);
    }
    set_accumulation(false);

    let accumulated = take_accumulated_features();
    let zero_stats = take_accumulated_layer_zero_stats();
    let n = accumulated.len();
    println!("\nTotal accumulated feature vectors N = {}", n);

    if zero_stats.count > 0 {
        println!("\n=== LAYER INPUT ZERO FEATURE ANALYSIS ===");
        println!("Inferences evaluated: {}", zero_stats.count);
        for layer_idx in 0..5 {
            let size = zero_stats.layer_sizes[layer_idx];
            let avg_z = zero_stats.avg_zeros(layer_idx);
            let pct = zero_stats.zero_percentage(layer_idx);
            println!(
                "  Layer {} ({:3} inputs): avg {:6.2} / {:3} zeros ({:6.2}%)",
                layer_idx + 1,
                size,
                avg_z,
                size,
                pct
            );
        }
        println!("=========================================");
    }

    if n < 2 {
        println!("Error: Not enough feature vectors accumulated to run benchmark.");
        return;
    }

    // Phase 2: Adjacent Feature Difference Analysis
    let mut total_diffs: u64 = 0;
    let total_elements: u64 = (n - 1) as u64 * Board::TOTAL_FN as u64;
    for i in 0..n - 1 {
        let a = &accumulated[i];
        let b = &accumulated[i + 1];
        let diff = a.iter().zip(b.iter()).filter(|(&x, &y)| x != y).count() as u64;
        total_diffs += diff;
    }
    let avg_diff = total_diffs as f64 / (n - 1) as f64;
    let pct_diff = (total_diffs as f64 / total_elements as f64) * 100.0;

    println!("\n=== ADJACENT FEATURE DIFFERENCE ANALYSIS ===");
    println!("Adjacent transitions evaluated: {}", n - 1);
    println!("Average changed features per adjacent transition: {:.2} / {} ({:.2}%)", avg_diff, Board::TOTAL_FN, pct_diff);
    println!("============================================\n");

    // Phase 3: MLP Inference Loop Benchmark
    println!("Running MLP inference benchmark for K = {} iterations...", args.iterations);
    let k = args.iterations;

    let start_bench = Instant::now();
    for i in 0..k {
        let x = &accumulated[i % n];
        black_box(mlp_inference(black_box(x)));
    }
    let elapsed = start_bench.elapsed();

    let secs = elapsed.as_secs_f64();
    let ns_per_inf = (secs * 1e9) / k as f64;
    let inf_per_sec = k as f64 / secs;

    println!("\n=== MLP INFERENCE BENCHMARK RESULTS ===");
    println!("Total Inferences (K):     {}", k);
    println!("Total Wall-Clock Time:    {:.4} s", secs);
    println!("Throughput:               {:.2} M inf/s", inf_per_sec / 1e6);
    println!("Latency per inference:    {:.2} ns", ns_per_inf);
    println!("=========================================");
}
