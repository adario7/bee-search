use bee_search::board::{Action, Board, GameResult};
use bee_search::engine::{Depth, Engine};
use bee_search::eval::Eval;
use bee_search::eval_gt::{
    gt_inference, set_accumulation, take_accumulated_graphs, TokenCache, GT_NUM_TOKENS,
};
use clap::Parser;
use rand::seq::IndexedRandom;
use rand::{rngs::StdRng, SeedableRng};
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Graph Transformer (GT) feature accumulation & inference throughput benchmark",
    long_about = None
)]
struct Args {
    /// Search depth to evaluate on each position during harvesting
    #[arg(short, long, default_value_t = 8)]
    depth: Depth,

    /// Number of GT inference iterations to execute in benchmark loop
    #[arg(short, long, default_value_t = 200_000)]
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
    println!("============================================================");
    println!("    GRAPH TRANSFORMER (GT) CPU INFERENCE BENCHMARK");
    println!("============================================================");
    println!("Search depth:                {}", args.depth);
    println!("Target inference iterations: {}\n", args.iterations);

    let positions = generate_positions();
    println!("Selected Positions:");
    for (ply, board) in &positions {
        let piece_count = board.occupied_tiles[0].len() + board.occupied_tiles[1].len();
        let category = if *ply <= 10 { "Early Game" } else { "Mid Game" };
        println!("  - Ply {:2} ({:<10}): {} pieces on board", ply, category, piece_count);
    }
    println!();

    // Phase 1: Search & Graph Harvesting
    println!("Running search at depth {} to harvest token graph transitions...", args.depth);
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
        println!("done ({:.2?}) -> score: {}", start.elapsed(), value);
    }
    set_accumulation(false);

    let graphs = take_accumulated_graphs();
    let n = graphs.len();
    println!("\nTotal accumulated search-tree graphs N = {}", n);

    if n < 2 {
        println!("Error: Not enough token graphs accumulated to run benchmark.");
        return;
    }

    // Phase 2: Search Tree Token Difference & In-Hand Analysis
    let mut total_changed_tokens: u64 = 0;
    let mut total_pieces_on_board: u64 = 0;

    for i in 0..n {
        let mut count_on_board = 0;
        for tok in 0..GT_NUM_TOKENS {
            if graphs[i].features[tok][0] > 0.5 {
                count_on_board += 1;
            }
        }
        total_pieces_on_board += count_on_board;

        if i < n - 1 {
            let a = &graphs[i].features;
            let b = &graphs[i + 1].features;
            let mut diff_tokens = 0;
            for tok in 0..GT_NUM_TOKENS {
                if a[tok] != b[tok] {
                    diff_tokens += 1;
                }
            }
            total_changed_tokens += diff_tokens;
        }
    }

    let avg_on_board = total_pieces_on_board as f64 / n as f64;
    let avg_changed_tokens = total_changed_tokens as f64 / (n - 1) as f64;
    let pct_changed = (avg_changed_tokens / GT_NUM_TOKENS as f64) * 100.0;
    let in_hand_pct = (1.0 - (avg_on_board / GT_NUM_TOKENS as f64)) * 100.0;

    println!("\n=== SEARCH TREE TOKEN TRANSITION ANALYSIS ===");
    println!("Transitions analyzed:                   {}", n - 1);
    println!("Average pieces on board:                {:.2} / {} ({:.1}% in-hand)", avg_on_board, GT_NUM_TOKENS, in_hand_pct);
    println!("Average changed tokens per transition:  {:.2} / {} tokens ({:.2}%)", avg_changed_tokens, GT_NUM_TOKENS, pct_changed);
    println!("=============================================\n");

    let k = args.iterations;

    // Phase 3: Benchmark Uncached GT Inference (Full recomputation from scratch)
    println!("1. Benchmarking UNCACHED GT inference (from scratch) for K = {}...", k);
    let start_uncached = Instant::now();
    let mut dummy_uncached: Eval = 0;
    let fn2_dummy = [0.0f32; 104];
    for i in 0..k {
        let g = &graphs[i % n];
        dummy_uncached ^= black_box(gt_inference(black_box(&g.features), black_box(&g.adj), black_box(&fn2_dummy)));
    }
    let elapsed_uncached = start_uncached.elapsed();
    let secs_uncached = elapsed_uncached.as_secs_f64();
    let us_per_uncached = (secs_uncached * 1e6) / k as f64;
    let inf_per_sec_uncached = k as f64 / secs_uncached;

    // Phase 4: Benchmark Cached GT Inference (TokenCache updating dirty tokens)
    println!("2. Benchmarking CACHED GT inference (TokenCache) for K = {}...", k);
    let mut cache = TokenCache::new();
    let start_cached = Instant::now();
    let mut dummy_cached: Eval = 0;
    for i in 0..k {
        let g = &graphs[i % n];
        dummy_cached ^= black_box(cache.update_and_infer(black_box(&g.features), black_box(&g.adj), black_box(&fn2_dummy)));
    }
    let elapsed_cached = start_cached.elapsed();
    let secs_cached = elapsed_cached.as_secs_f64();
    let us_per_cached = (secs_cached * 1e6) / k as f64;
    let inf_per_sec_cached = k as f64 / secs_cached;

    let speedup = secs_uncached / secs_cached;

    // Prevent compiler from optimizing away loops
    black_box(dummy_uncached);
    black_box(dummy_cached);

    println!("\n============================================================");
    println!("             GRAPH TRANSFORMER BENCHMARK RESULTS            ");
    println!("============================================================");
    println!("Iterations (K):                 {}", k);
    println!("------------------------------------------------------------");
    println!("UNCACHED Inference (Scratch):");
    println!("  - Total Wall-Clock Time:      {:.4} s", secs_uncached);
    println!("  - Throughput:                 {:.2} K inf/s", inf_per_sec_uncached / 1e3);
    println!("  - Latency per Inference:      {:.2} us ({:.0} ns)", us_per_uncached, us_per_uncached * 1e3);
    println!("------------------------------------------------------------");
    println!("CACHED Inference (TokenCache):");
    println!("  - Total Wall-Clock Time:      {:.4} s", secs_cached);
    println!("  - Throughput:                 {:.2} K inf/s", inf_per_sec_cached / 1e3);
    println!("  - Latency per Inference:      {:.2} us ({:.0} ns)", us_per_cached, us_per_cached * 1e3);
    println!("------------------------------------------------------------");
    println!("SPEEDUP (Cached vs Uncached):   {:.2}x", speedup);
    println!("============================================================\n");
}
