use bee_search::board::Board;
use bee_search::engine::Engine;
use bee_search::piece::Color;
use bee_search::uhp::Uhp;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rand::rng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Ultra-fast native Rust game replay, position extraction, and engine evaluation"
)]
struct Args {
    #[arg(short, long, default_values_t = vec!["logs/results.json".to_string()])]
    results: Vec<String>,

    #[arg(short = 'o', long, default_value = "logs/evaluations.json")]
    evals: String,

    #[arg(short, long, default_value_t = 6)]
    depth: u8,

    #[arg(short, long, default_value_t = 60)]
    timeout: u64,

    #[arg(short = 'j', long, default_value_t = 0)]
    jobs: usize,

    #[arg(long, default_value_t = false)]
    reevaluate: bool,

    #[arg(long, default_value_t = usize::MAX)]
    cap: usize,

    #[arg(long, default_value_t = 1000)]
    chunk_size: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EvaluationEntry {
    pub engine: String,
    pub depth: u8,
    pub position: String,
    pub evaluation: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub static_eval: Option<i32>,
    pub winner: f64,
}

#[derive(Deserialize, Debug)]
struct JsonResultItem {
    #[serde(default)]
    winner: String,
    #[serde(default)]
    final_gamestate: String,
}

struct ParsedGame {
    winner: Option<Color>, // None = Draw
    gametype: String,
    moves: Vec<String>,
}

fn parse_winner_str(s: &str) -> Option<Option<Color>> {
    let lower = s.to_ascii_lowercase();
    if lower.contains("white") {
        Some(Some(Color::White))
    } else if lower.contains("black") {
        Some(Some(Color::Black))
    } else if lower.contains("draw") {
        Some(None)
    } else {
        None
    }
}

fn load_games_from_file(path: &str) -> Result<Vec<ParsedGame>, String> {
    if !Path::new(path).exists() {
        return Err(format!("File not found: {}", path));
    }

    let mut games = Vec::new();

    if path.ends_with(".json") {
        let file = File::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
        let reader = BufReader::new(file);
        let items: Vec<JsonResultItem> = serde_json::from_reader(reader)
            .map_err(|e| format!("Failed to parse JSON in {}: {}", path, e))?;

        for item in items {
            let parts: Vec<&str> = item.final_gamestate.split(';').collect();
            if parts.len() < 4 || parts[0] != "Base+MLP" {
                continue;
            }
            if let Some(winner) = parse_winner_str(&item.winner) {
                let gametype = parts[0].to_string();
                let moves = parts[3..].iter().map(|&s| s.to_string()).collect();
                games.push(ParsedGame {
                    winner,
                    gametype,
                    moves,
                });
            }
        }
    } else {
        let file = File::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Read error: {}", e))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split(';').collect();
            if parts.len() >= 4 && parts[0] == "Base+MLP" {
                if let Some(winner) = parse_winner_str(parts[1]) {
                    let gametype = parts[0].to_string();
                    let moves = parts[3..].iter().map(|&s| s.to_string()).collect();
                    games.push(ParsedGame {
                        winner,
                        gametype,
                        moves,
                    });
                }
            }
        }
    }

    Ok(games)
}

struct PositionWinAccumulator {
    total_score: f64,
    count: usize,
}

fn get_positions_from_results(
    results_paths: &[String],
    cap: usize,
) -> (Vec<String>, HashMap<String, f64>) {
    let mut group_positions: Vec<Vec<String>> = Vec::new();
    let mut win_accum: HashMap<String, PositionWinAccumulator> = HashMap::new();
    let mut rng = rng();

    for path in results_paths {
        println!("Extracting positions from results in {}...", path);
        let mut games = match load_games_from_file(path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Warning: {}", e);
                continue;
            }
        };

        games.shuffle(&mut rng);
        if cap > 0 && games.len() > cap {
            games.truncate(cap);
        }

        let mut accepted_games = 0;
        let mut discarded_games = 0;
        let mut accepted_positions = 0;
        let mut discarded_positions = 0;
        let mut file_positions: Vec<String> = Vec::new();
        let mut file_pos_set: HashSet<String> = HashSet::new();

        for game in &games {
            let mut board = match Board::parse_game_string(&game.gametype) {
                Ok(b) => b,
                Err(_) => {
                    discarded_games += 1;
                    continue;
                }
            };

            accepted_games += 1;
            let mut move_hist: HashMap<String, usize> = HashMap::new();

            for move_str in &game.moves {
                let count = move_hist.entry(move_str.clone()).or_insert(0);
                *count += 1;
                if *count >= 5 {
                    discarded_positions += 1;
                    break;
                }

                let action = match board.parse_action(move_str) {
                    Ok(a) => a,
                    Err(_) => {
                        discarded_positions += 1;
                        break;
                    }
                };

                if !board.is_legal(action) {
                    discarded_positions += 1;
                    break;
                }

                board.do_action(action);
                let pos_str = board.game_string();

                let turn_color = board.color();
                let relative = match game.winner {
                    Some(w) if w == turn_color => 1.0,
                    Some(w) if w != turn_color => 0.0,
                    _ => 0.5,
                };

                let entry = win_accum
                    .entry(pos_str.clone())
                    .or_insert(PositionWinAccumulator {
                        total_score: 0.0,
                        count: 0,
                    });
                entry.total_score += relative;
                entry.count += 1;

                if file_pos_set.insert(pos_str.clone()) {
                    file_positions.push(pos_str);
                }
                accepted_positions += 1;
            }
        }

        println!(
            "Accepted games: {}, Discarded games: {}",
            accepted_games, discarded_games
        );
        println!(
            "Accepted positions: {}, Discarded positions: {}",
            accepted_positions, discarded_positions
        );

        file_positions.shuffle(&mut rng);
        group_positions.push(file_positions);
    }

    // Interleave positions across groups
    let mut positions: Vec<String> = Vec::new();
    let mut positions_set: HashSet<String> = HashSet::new();
    let k = group_positions.len();
    let mut indices = vec![0usize; k];

    while indices
        .iter()
        .enumerate()
        .any(|(i, &idx)| idx < group_positions[i].len())
    {
        for i in 0..k {
            while indices[i] < group_positions[i].len() {
                let p = &group_positions[i][indices[i]];
                indices[i] += 1;
                if positions_set.insert(p.clone()) {
                    positions.push(p.clone());
                    break;
                }
            }
        }
    }

    println!("Extracted {} unique positions.", positions.len());

    let mut win_prob: HashMap<String, f64> = HashMap::new();
    for (pos, stats) in win_accum {
        win_prob.insert(pos, stats.total_score / stats.count as f64);
    }

    (positions, win_prob)
}

fn load_existing_evals(path: &str) -> HashMap<String, EvaluationEntry> {
    if !Path::new(path).exists() {
        return HashMap::new();
    }
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return HashMap::new(),
    };
    let reader = BufReader::new(file);
    let items: Vec<EvaluationEntry> = match serde_json::from_reader(reader) {
        Ok(it) => it,
        Err(_) => return HashMap::new(),
    };

    let mut map = HashMap::new();
    for item in items {
        map.insert(item.position.clone(), item);
    }
    map
}

fn save_evaluations_atomic(path: &str, evals: &[EvaluationEntry]) -> Result<(), String> {
    let tmp_path = format!("{}.tmp", path);
    if let Some(parent) = Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let file = File::create(&tmp_path)
        .map_err(|e| format!("Failed to create {}: {}", tmp_path, e))?;
    let mut writer = BufWriter::new(file);

    serde_json::to_writer_pretty(&mut writer, evals)
        .map_err(|e| format!("JSON serialization error: {}", e))?;
    writer
        .flush()
        .map_err(|e| format!("Flush error on {}: {}", tmp_path, e))?;

    std::fs::rename(&tmp_path, path)
        .map_err(|e| format!("Failed to rename {} to {}: {}", tmp_path, path, e))?;

    Ok(())
}

fn main() {
    let args = Args::parse();
    let num_threads = if args.jobs == 0 {
        num_cpus::get()
    } else {
        args.jobs
    };

    let engine_name = Uhp::engine_name();
    println!("Engine: {}", engine_name);

    let t0 = Instant::now();
    let (positions, win_prob) = get_positions_from_results(&args.results, args.cap);

    let mut evals_map = load_existing_evals(&args.evals);
    println!("Num loaded existing evals: {}", evals_map.len());

    let mut positions_to_eval = Vec::new();
    let mut num_present_diff_params = 0;
    let mut num_missing = 0;

    for pos in &positions {
        if let Some(existing) = evals_map.get(pos) {
            let is_diff = existing.depth != args.depth || existing.engine != engine_name;
            if is_diff {
                num_present_diff_params += 1;
                if args.reevaluate {
                    positions_to_eval.push(pos.clone());
                }
            }
        } else {
            num_missing += 1;
            positions_to_eval.push(pos.clone());
        }
    }

    println!("Present but with different params: {}", num_present_diff_params);
    println!("Missing ones to evaluate: {}", num_missing);

    // Update winner probabilities for all known positions
    for pos in &positions {
        if let Some(entry) = evals_map.get_mut(pos) {
            if let Some(&wp) = win_prob.get(pos) {
                entry.winner = wp;
            }
        }
    }

    let all_evals: Vec<EvaluationEntry> = evals_map.values().cloned().collect();
    if let Err(e) = save_evaluations_atomic(&args.evals, &all_evals) {
        eprintln!("Warning: {}", e);
    } else {
        println!("Updated existing winner probabilities in {}", args.evals);
    }

    if positions_to_eval.is_empty() {
        println!("No new positions to evaluate. All done in {:.2?}!", t0.elapsed());
        return;
    }

    println!(
        "Evaluating {} positions across {} worker threads (depth: {}, timeout: {}s)...",
        positions_to_eval.len(),
        num_threads,
        args.depth,
        args.timeout
    );

    let pb = ProgressBar::new(positions_to_eval.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} pos ({per_sec}, ETA: {eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    let engine = Arc::new(Engine::new());
    let evals_map_mutex = Arc::new(Mutex::new(evals_map));
    let next_idx = Arc::new(AtomicUsize::new(0));
    let processed_count = Arc::new(AtomicUsize::new(0));
    let positions_arc = Arc::new(positions_to_eval);
    let win_prob_arc = Arc::new(win_prob);

    let mut handles = Vec::new();
    for _ in 0..num_threads {
        let engine_clone = Arc::clone(&engine);
        let evals_map_mutex = Arc::clone(&evals_map_mutex);
        let next_idx = Arc::clone(&next_idx);
        let processed_count = Arc::clone(&processed_count);
        let positions_arc = Arc::clone(&positions_arc);
        let win_prob_arc = Arc::clone(&win_prob_arc);
        let pb_clone = pb.clone();
        let depth = args.depth;
        let timeout = Duration::from_secs(args.timeout);
        let engine_name = engine_name.clone();
        let evals_path = args.evals.clone();
        let chunk_size = args.chunk_size;

        let handle = std::thread::spawn(move || {
            loop {
                let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                if idx >= positions_arc.len() {
                    break;
                }

                let pos_str = &positions_arc[idx];
                let mut board = match Board::parse_game_string(pos_str) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("Error parsing board for position {}: {:?}", pos_str, e);
                        pb_clone.inc(1);
                        continue;
                    }
                };

                let static_eval = board.static_eval() as i32;
                let eval_score = if depth == 0 {
                    static_eval
                } else {
                    let score = engine_clone
                        .clone()
                        .best_move(&mut board, depth, timeout, 1, false, None)
                        .0;
                    score as i32
                };

                let wp = win_prob_arc.get(pos_str).copied().unwrap_or(0.5);
                let entry = EvaluationEntry {
                    engine: engine_name.clone(),
                    depth,
                    position: pos_str.clone(),
                    evaluation: eval_score,
                    static_eval: Some(static_eval),
                    winner: wp,
                };

                {
                    let mut guard = evals_map_mutex.lock().unwrap();
                    guard.insert(pos_str.clone(), entry);
                }

                pb_clone.inc(1);
                let done = processed_count.fetch_add(1, Ordering::Relaxed) + 1;

                if done % chunk_size == 0 {
                    let snapshot: Vec<EvaluationEntry> = {
                        let guard = evals_map_mutex.lock().unwrap();
                        guard.values().cloned().collect()
                    };
                    let _ = save_evaluations_atomic(&evals_path, &snapshot);
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        let _ = handle.join();
    }
    pb.finish_with_message("Evaluation complete!");

    let final_evals: Vec<EvaluationEntry> = {
        let guard = evals_map_mutex.lock().unwrap();
        guard.values().cloned().collect()
    };

    if let Err(e) = save_evaluations_atomic(&args.evals, &final_evals) {
        eprintln!("Error saving final evaluations: {}", e);
    } else {
        println!(
            "Saved {} total evaluations to {} in {:.2?}!",
            final_evals.len(),
            args.evals,
            t0.elapsed()
        );
    }
}
