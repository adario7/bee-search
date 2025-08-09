use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::IndexedRandom;
use rand::{rngs::StdRng, SeedableRng}; // Added for seeded RNG and random choice
use std::time::{Duration, Instant};

use bee_search::board::{Action, Board, GameResult}; //
use bee_search::engine::{Engine, Depth}; //

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Skip the first s moves
    #[arg(long, default_value_t = 10)]
    skip: u32,

    /// Number of moves to simulate
    #[arg(short, long, default_value_t = 20)]
    n: u32,

    /// Engine search depth
    #[arg(short, long, default_value_t = 4)]
    d: Depth,

    /// Clear transposition table at each move
    #[arg(short, long, default_value_t = false)]
    c: bool,

    /// Random seed for move selection
    #[arg(short, long, default_value_t = 0)] // Added seed argument
    s: u64,

    #[arg(long, default_value_t=false)]
    movegen_test: bool,
}

fn main() {
    let args = Args::parse();

    let mut board = Board::new(); //
    let mut engine = Engine::new(); //
    let mut total_think_time = Duration::new(0, 0);
    let mut total_nodes = 0;
    // Initialize seeded RNG for predictable random moves
    let mut rng = StdRng::seed_from_u64(args.s); // Uses the -s seed

    println!(
        "Starting simulation: skip={}, n={}, d={}, c={}, seed={}",
        args.skip, args.n, args.d, args.c, args.s
    );

    // Setup progress bar
    let pb = ProgressBar::new(args.n as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .expect("Failed to create progress bar style")
        .progress_chars("#>-"));

    if args.movegen_test {
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

    for move_i in 0..(args.skip + args.n) {
        let game_status = board.game_result(); //
        if game_status != GameResult::InProgress { //
            pb.println(format!("Game ended early. Result: {:?}", game_status));
            break; // Stop simulation if game is over
        }

        if move_i >= args.skip {
            if args.c {
                // Assuming the TTable struct used by Engine has a clear method
                engine.clear_tt();
                 pb.println(" (TT Cleared)"); // Use pb.println to avoid messing up the bar
            }

            // --- Compute best move (for timing) but don't use it ---
            let start_time = Instant::now();
            let max_time_per_move = Duration::from_secs(3600); // 1 hour, effectively unlimited for depth search
            // This call is primarily for timing and exercising the engine/TT logic.
            // The engine's internal eprintln will still show computed best move info [cite: 117]
            let _computed_best_action = engine.best_move(&mut board, args.d, max_time_per_move); 
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

    pb.finish_with_message("Simulation complete."); // Finish progress bar

    println!("Total engine nodes: {}", total_nodes);
    println!("Total engine think time: {:.3} seconds", total_think_time.as_secs_f64());
}
