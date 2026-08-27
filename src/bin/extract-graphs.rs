use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use bee_search::board::Board;
use bee_search::piece::Color;
use bee_search::graph_nn::{NUM_TOKENS, TOKEN_FEAT_DIM};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BinaryGraphRecord {
    pub features: [[f32; TOKEN_FEAT_DIM]; NUM_TOKENS],
    pub adj: [[u8; NUM_TOKENS]; NUM_TOKENS],
    pub fn2_features: [f32; Board::FN2],
    pub eval: f32,
    pub winner: f32,
    pub static_eval: f32,
    pub turn_num: u16,
    pub _pad: [u8; 2],
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Extract graph representations from evaluations dataset into binary format")]
struct Args {
    #[arg(short, long, default_value = "logs/evaluations.json")]
    evals: String,

    #[arg(short, long, default_value = "logs/position_graphs.bin")]
    out: String,

    #[arg(short, long, default_value_t = 0)]
    threads: usize,

    #[arg(short, long)]
    limit: Option<usize>,

    #[arg(long, default_value_t = 6000.0)]
    clip: f32,
}

struct RawEvalEntry {
    position: String,
    evaluation: f32,
    winner: f32,
    static_eval: f32,
}

fn parse_evaluations_json_stream(path: &str, limit: Option<usize>) -> Vec<RawEvalEntry> {
    println!("Reading evaluations from {}...", path);
    let file = File::open(path).unwrap_or_else(|e| panic!("Failed to open {}: {}", path, e));
    let reader = BufReader::with_capacity(16 * 1024 * 1024, file);

    let mut entries = Vec::new();

    // Regex matchers for fast field extraction from JSON
    let pos_re = Regex::new(r#""position"\s*:\s*"([^"]+)""#).unwrap();
    let eval_re = Regex::new(r#""evaluation"\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)"#).unwrap();
    let win_re = Regex::new(r#""winner"\s*:\s*([0-9]+(?:\.[0-9]+)?)"#).unwrap();
    let static_re = Regex::new(r#""static_eval"\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)"#).unwrap();

    let mut cur_pos: Option<String> = None;
    let mut cur_eval: Option<f32> = None;
    let mut cur_win: Option<f32> = None;
    let mut cur_static: Option<f32> = None;

    for line_res in reader.lines() {
        let line = match line_res {
            Ok(l) => l,
            Err(_) => continue,
        };

        if let Some(caps) = pos_re.captures(&line) {
            cur_pos = Some(caps[1].replace("\\\\", "\\"));
        }
        if let Some(caps) = eval_re.captures(&line) {
            cur_eval = caps[1].parse::<f32>().ok();
        }
        if let Some(caps) = win_re.captures(&line) {
            cur_win = caps[1].parse::<f32>().ok();
        }
        if let Some(caps) = static_re.captures(&line) {
            cur_static = caps[1].parse::<f32>().ok();
        }

        if line.contains('}') {
            if let Some(pos) = cur_pos.take() {
                let eval = cur_eval.take().unwrap_or(0.0);
                let win = cur_win.take().unwrap_or(0.0);
                let st = cur_static.take().unwrap_or(0.0);

                entries.push(RawEvalEntry {
                    position: pos,
                    evaluation: eval,
                    winner: win,
                    static_eval: st,
                });

                if let Some(lim) = limit {
                    if entries.len() >= lim {
                        break;
                    }
                }
            } else {
                cur_eval = None;
                cur_win = None;
                cur_static = None;
            }
        }
    }

    println!("Parsed {} valid evaluation entries from JSON", entries.len());
    entries
}

fn main() {
    let args = Args::parse();
    assert_eq!(
        std::mem::size_of::<BinaryGraphRecord>(),
        2112,
        "BinaryGraphRecord size must be exactly 2112 bytes"
    );

    let num_threads = if args.threads == 0 {
        num_cpus::get()
    } else {
        args.threads
    };

    let entries = parse_evaluations_json_stream(&args.evals, args.limit);
    let total_entries = entries.len();
    if total_entries == 0 {
        eprintln!("No positions found in {}", args.evals);
        return;
    }

    if let Some(parent) = Path::new(&args.out).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).unwrap();
        }
    }

    let out_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&args.out)
        .unwrap_or_else(|e| panic!("Failed to create {}: {}", args.out, e));

    let mut writer = BufWriter::with_capacity(32 * 1024 * 1024, out_file);

    println!(
        "Extracting graphs for {} positions using {} threads -> {}...",
        total_entries, num_threads, args.out
    );

    let pb = ProgressBar::new(total_entries as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({per_sec}, ETA {eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    let (tx, rx) = crossbeam::channel::bounded::<(usize, BinaryGraphRecord)>(num_threads * 64);

    let entries_arc = Arc::new(entries);
    let next_idx = Arc::new(AtomicUsize::new(0));

    let mut workers = Vec::new();
    for _ in 0..num_threads {
        let entries_c = Arc::clone(&entries_arc);
        let next_idx_c = Arc::clone(&next_idx);
        let tx_c = tx.clone();
        let clip_val = args.clip;

        let handle = thread::spawn(move || {
            loop {
                let idx = next_idx_c.fetch_add(1, Ordering::Relaxed);
                if idx >= entries_c.len() {
                    break;
                }

                let entry = &entries_c[idx];
                let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut board = Board::parse_game_string(&entry.position).ok()?;
                    let moves = board.generate_moves();
                    let other = board.other_moves();
                    let tg = board.get_token_graph_fast(&moves, &other);
                    let is_white = board.color() == Color::White;
                    let eval_white = if is_white { entry.evaluation } else { -entry.evaluation };
                    let win_white = if is_white { entry.winner } else { 1.0 - entry.winner };
                    let static_white = if is_white { entry.static_eval } else { -entry.static_eval };
                    let clamped_eval = eval_white.clamp(-clip_val, clip_val);
                    let clamped_win = win_white.clamp(0.0, 1.0);

                    // Compute absolute FN2 features (White: 0..FN, Black: FN..FN2)
                    let fn2_raw = board.features_fn2_fast(&moves, &other);
                    let mut fn2_white = [0.0f32; Board::FN2];
                    if is_white {
                        for i in 0..Board::FN2 {
                            fn2_white[i] = fn2_raw[i] as f32;
                        }
                    } else {
                        // Swap White and Black blocks to ensure White-first absolute perspective
                        for i in 0..Board::FN {
                            fn2_white[i] = fn2_raw[Board::FN + i] as f32;
                            fn2_white[Board::FN + i] = fn2_raw[i] as f32;
                        }
                    }

                    Some(BinaryGraphRecord {
                        features: tg.features,
                        adj: tg.adj,
                        fn2_features: fn2_white,
                        eval: clamped_eval,
                        winner: clamped_win,
                        static_eval: static_white,
                        turn_num: board.turn_num as u16,
                        _pad: [0, 0],
                    })
                }));

                if let Ok(Some(record)) = res {
                    if tx_c.send((idx, record)).is_err() {
                        break;
                    }
                }
            }
        });
        workers.push(handle);
    }
    drop(tx); // Drop main tx so rx terminates when workers are done

    let mut processed = 0;
    while let Ok((_idx, record)) = rx.recv() {
        let slice = unsafe {
            std::slice::from_raw_parts(
                &record as *const BinaryGraphRecord as *const u8,
                std::mem::size_of::<BinaryGraphRecord>(),
            )
        };
        writer.write_all(slice).unwrap();
        processed += 1;
        if processed % 1000 == 0 {
            pb.set_position(processed as u64);
        }
    }

    for h in workers {
        h.join().unwrap();
    }

    writer.flush().unwrap();
    pb.finish_with_message("Done");

    let file_size_mb = (processed * std::mem::size_of::<BinaryGraphRecord>()) as f64 / (1024.0 * 1024.0);
    println!(
        "Successfully extracted {} position graphs ({:.2} MB) into {}",
        processed, file_size_mb, args.out
    );
}
