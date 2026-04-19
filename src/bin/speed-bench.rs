use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::IndexedRandom;
use rand::{rngs::StdRng, SeedableRng}; use std::sync::Arc;
// Added for seeded RNG and random choice
use std::time::{Duration, Instant};

use bee_search::board::{Action, Board, GameResult}; //
use bee_search::engine::{Engine, Depth}; //

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args { // TODO: num threads
    /// Skip the first s moves
    #[arg(long, default_value_t = 10)]
    skip: u32,

    /// Number of moves to simulate
    #[arg(short, long, default_value_t = 20)]
    n: u32,

    /// Engine search depth
    #[arg(short, long, default_value_t = 4)]
    d: Depth,

    /// Random seed for move selection
    #[arg(short, long, default_value_t = 0)] // Added seed argument
    s: u64,

    #[arg(long, default_value_t=false)]
    movegen_test: bool,

    #[arg(long, default_value_t=1)]
    num_runs: u32,
}

fn main() {
    let args = Args::parse();

    let engine = Arc::new(Engine::new());
    let mut total_think_time = Duration::new(0, 0);
    let mut total_nodes = 0;
    // Initialize seeded RNG for predictable random moves
    let mut rng = StdRng::seed_from_u64(args.s); // Uses the -s seed

    println!(
        "Starting simulation: skip={}, n={}, d={}, seed={}",
        args.skip, args.n, args.d, args.s
    );

    // Setup progress bar
    let pb = ProgressBar::new(args.n as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .expect("Failed to create progress bar style")
        .progress_chars("#>-"));

    if args.movegen_test {

        let mut board ;
        for _game in 0..args.n {
            board = Board::new();

            for _move_i in 0..200 {
                let game_status = board.game_result();
                if game_status != GameResult::InProgress {
                    break;
                }

                let start_t = Instant::now();
                for _ in 0..1000 {
                    total_nodes += board.generate_moves_n() as u64;
                }
                total_think_time += Instant::now() - start_t;

                let mut legal_moves = board.generate_moves(); 
                legal_moves.sort(); // Sort moves for consistent ordering

                let chosen_action = if legal_moves.is_empty() {
                    //pb.println("No legal moves available, playing Pass.");
                    Action::Pass // Play pass if no moves available
                } else {
                    // Select a random move using the seeded RNG
                    *legal_moves.choose(&mut rng).unwrap() // unwrap is safe here due to is_empty check
                };
                // Apply the *randomly selected* move
                board.do_action(chosen_action); 
            }
            pb.inc(1);
        }
        pb.finish_with_message("Simulation complete with total think time ");
        println!("Total think time {:.3}", total_think_time.as_secs_f64());
        return;
    }

    // Do a series of different runs, then take the average of the times
    // Should still work well with random seeding 
    for run_id in 0..args.num_runs { 
        let mut board = Board::new();
        println!("Simulating run: {}", run_id);
        pb.reset();
        for move_i in 0..(args.skip + args.n) {
            let game_status = board.game_result(); //
            if game_status != GameResult::InProgress { //
                pb.println(format!("Game ended early. Result: {:?}", game_status));
                break; // Stop simulation if game is over
            }

            if move_i >= args.skip {
                // --- Compute best move (for timing) but don't use it ---
                let start_time = Instant::now();
                let max_time_per_move = Duration::from_secs(3600); // 1 hour, effectively unlimited for depth search
                // This call is primarily for timing and exercising the engine/TT logic.
                // The engine's internal eprintln will still show computed best move info [cite: 117]
                let _computed_best_action = engine.clone().best_move(&mut board, args.d, max_time_per_move, 1, true); 
                let think_time = start_time.elapsed();
                total_think_time += think_time;
                total_nodes += engine.last_nnodes();
                pb.inc(1); // Increment progress bar
            }
            // ----------------------------------------------------------


            // --- Generate legal moves and play a random one ---
            let mut legal_moves = board.generate_moves(); 
            legal_moves.sort(); // Sort moves for consistent ordering

            let chosen_action = if legal_moves.is_empty() {
                pb.println("No legal moves available, playing Pass.");
                Action::Pass // Play pass if no moves available
            } else {
                // Select a random move using the seeded RNG
                *legal_moves.choose(&mut rng).unwrap() // unwrap is safe here due to is_empty check
            };
            pb.println(format!("Playing move: {:?}", chosen_action));

            // Apply the *randomly selected* move
            board.do_action(chosen_action); 
        }
    }

    pb.finish_with_message("Simulation complete."); // Finish progress bar

    println!("Total number of runs {}", args.num_runs);
    println!("Average engine nodes: {}", total_nodes / args.num_runs as u64);
    println!("Average engine think time: {:.3} seconds", total_think_time.as_secs_f64() / args.num_runs as f64);
    println!("Average speed: {:.3} knodes/s", total_nodes as f64 / total_think_time.as_secs_f64() / 1000.0);
}
