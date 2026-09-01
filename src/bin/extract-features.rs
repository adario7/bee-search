use bee_search::board::Board;
use bee_search::piece::Color;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

#[derive(Parser, Debug)]
#[command(author, version, about = "Fast multi-threaded Sparse CSR feature extractor for MLP training")]
struct Args {
    #[arg(short, long, default_value = "logs/evaluations.json")]
    evals: String,

    #[arg(short, long, default_value = "logs/features_mlp.bin.zst")]
    out: String,

    #[arg(short, long, default_value_t = 0)]
    threads: usize,

    #[arg(short, long)]
    limit: Option<usize>,

    #[arg(long, default_value_t = 6000.0)]
    clip: f32,

    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    augment: bool,

    #[arg(long, default_value_t = 1)]
    compression_level: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CsrHeader {
    pub magic: [u8; 4],   // b"BCSR"
    pub version: u32,     // 1
    pub num_samples: u32, // N
    pub feature_dim: u32, // 2152
    pub total_nnz: u64,   // total non-zero entries across all samples
}

#[derive(Clone, Debug)]
pub struct SparseSample {
    pub eval: f32,
    pub winner: f32,
    pub static_eval: f32,
    pub turn_num: u16,
    pub indices: Vec<u16>,
    pub values: Vec<i16>,
}

struct RawEvalEntry {
    position: String,
    evaluation: f32,
    winner: f32,
    static_eval: f32,
}

fn parse_evaluations_json_stream(path: &str, limit: Option<usize>) -> Vec<RawEvalEntry> {
    let actual_path = if !Path::new(path).exists() && Path::new("/home/alessandro/local/logs/evaluations.json").exists() {
        "/home/alessandro/local/logs/evaluations.json"
    } else {
        path
    };
    println!("Reading evaluations from {}...", actual_path);
    let file = File::open(actual_path).unwrap_or_else(|e| panic!("Failed to open {}: {}", actual_path, e));
    let reader = BufReader::with_capacity(16 * 1024 * 1024, file);

    let mut entries = Vec::new();

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

fn features_to_sparse(features: &[i16; Board::TOTAL_FN]) -> (Vec<u16>, Vec<i16>) {
    let mut indices = Vec::with_capacity(128);
    let mut values = Vec::with_capacity(128);
    for (i, &val) in features.iter().enumerate() {
        if val != 0 {
            indices.push(i as u16);
            values.push(val);
        }
    }
    (indices, values)
}

fn main() {
    let args = Args::parse();

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

    let expected_records = total_entries * if args.augment { 2 } else { 1 };
    println!(
        "Extracting Sparse CSR features (dim={}) for {} positions (augment={}, target {} records) using {} threads -> {}...",
        Board::TOTAL_FN,
        total_entries,
        args.augment,
        expected_records,
        num_threads,
        args.out
    );

    let pb = ProgressBar::new(expected_records as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({per_sec}, ETA {eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    let (tx, rx) = crossbeam::channel::bounded::<(usize, Vec<SparseSample>)>(num_threads * 64);

    let entries_arc = Arc::new(entries);
    let next_idx = Arc::new(AtomicUsize::new(0));

    let mut workers = Vec::new();
    for _ in 0..num_threads {
        let entries_c = Arc::clone(&entries_arc);
        let next_idx_c = Arc::clone(&next_idx);
        let tx_c = tx.clone();
        let clip_val = args.clip;
        let augment = args.augment;

        let handle = thread::spawn(move || {
            loop {
                let idx = next_idx_c.fetch_add(1, Ordering::Relaxed);
                if idx >= entries_c.len() {
                    break;
                }

                let entry = &entries_c[idx];
                let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut board = Board::parse_game_string(&entry.position).ok()?;
                    let features = board.features_slow();

                    let is_white = board.color() == Color::White;
                    let eval_white = if is_white {
                        entry.evaluation
                    } else {
                        -entry.evaluation
                    };
                    let win_white = if is_white {
                        entry.winner
                    } else {
                        1.0 - entry.winner
                    };
                    let static_white = if is_white {
                        entry.static_eval
                    } else {
                        -entry.static_eval
                    };
                    let clamped_eval = eval_white.clamp(-clip_val, clip_val);
                    let clamped_win = win_white.clamp(0.0, 1.0);

                    let (indices1, values1) = features_to_sparse(&features);
                    let rec1 = SparseSample {
                        eval: clamped_eval,
                        winner: clamped_win,
                        static_eval: static_white,
                        turn_num: board.turn_num as u16,
                        indices: indices1,
                        values: values1,
                    };

                    let mut records = Vec::with_capacity(if augment { 2 } else { 1 });
                    records.push(rec1);

                    if augment {
                        let features_swapped = Board::swap_features_color(&features);
                        let (indices2, values2) = features_to_sparse(&features_swapped);
                        let rec2 = SparseSample {
                            eval: -clamped_eval,
                            winner: 1.0 - clamped_win,
                            static_eval: -static_white,
                            turn_num: board.turn_num as u16,
                            indices: indices2,
                            values: values2,
                        };
                        records.push(rec2);
                    }

                    Some(records)
                }));

                if let Ok(Some(records)) = res {
                    if tx_c.send((idx, records)).is_err() {
                        break;
                    }
                }
            }
        });
        workers.push(handle);
    }
    drop(tx);

    let mut all_offsets: Vec<u64> = Vec::with_capacity(expected_records + 1);
    let mut all_evals: Vec<f32> = Vec::with_capacity(expected_records);
    let mut all_winners: Vec<f32> = Vec::with_capacity(expected_records);
    let mut all_static_evals: Vec<f32> = Vec::with_capacity(expected_records);
    let mut all_turn_nums: Vec<u16> = Vec::with_capacity(expected_records);
    let mut all_indices: Vec<u16> = Vec::with_capacity(expected_records * 80);
    let mut all_values: Vec<i16> = Vec::with_capacity(expected_records * 80);

    all_offsets.push(0);
    let mut current_offset: u64 = 0;
    let mut processed = 0;

    while let Ok((_idx, records)) = rx.recv() {
        for record in records {
            all_evals.push(record.eval);
            all_winners.push(record.winner);
            all_static_evals.push(record.static_eval);
            all_turn_nums.push(record.turn_num);

            current_offset += record.indices.len() as u64;
            all_offsets.push(current_offset);

            all_indices.extend_from_slice(&record.indices);
            all_values.extend_from_slice(&record.values);

            processed += 1;
        }
        if processed % 1000 == 0 {
            pb.set_position(processed as u64);
        }
    }

    for h in workers {
        h.join().unwrap();
    }
    pb.finish_with_message("Done extraction, writing CSR archive...");

    let num_samples = all_evals.len() as u32;
    let total_nnz = all_indices.len() as u64;
    let header = CsrHeader {
        magic: *b"BCSR",
        version: 1,
        num_samples,
        feature_dim: Board::TOTAL_FN as u32,
        total_nnz,
    };

    let out_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&args.out)
        .unwrap_or_else(|e| panic!("Failed to create {}: {}", args.out, e));

    let buf_writer = BufWriter::with_capacity(32 * 1024 * 1024, out_file);
    let use_zstd = args.out.ends_with(".zst") || args.out.ends_with(".zstd");

    let mut writer: Box<dyn Write> = if use_zstd {
        let encoder = zstd::Encoder::new(buf_writer, args.compression_level).unwrap();
        Box::new(encoder.auto_finish())
    } else {
        Box::new(buf_writer)
    };

    // Write Header
    let header_slice = unsafe {
        std::slice::from_raw_parts(
            &header as *const CsrHeader as *const u8,
            std::mem::size_of::<CsrHeader>(),
        )
    };
    writer.write_all(header_slice).unwrap();

    // Write Offsets [u64; num_samples + 1]
    let offsets_slice = unsafe {
        std::slice::from_raw_parts(
            all_offsets.as_ptr() as *const u8,
            all_offsets.len() * std::mem::size_of::<u64>(),
        )
    };
    writer.write_all(offsets_slice).unwrap();

    // Write Targets
    let evals_slice = unsafe {
        std::slice::from_raw_parts(
            all_evals.as_ptr() as *const u8,
            all_evals.len() * std::mem::size_of::<f32>(),
        )
    };
    writer.write_all(evals_slice).unwrap();

    let winners_slice = unsafe {
        std::slice::from_raw_parts(
            all_winners.as_ptr() as *const u8,
            all_winners.len() * std::mem::size_of::<f32>(),
        )
    };
    writer.write_all(winners_slice).unwrap();

    let statics_slice = unsafe {
        std::slice::from_raw_parts(
            all_static_evals.as_ptr() as *const u8,
            all_static_evals.len() * std::mem::size_of::<f32>(),
        )
    };
    writer.write_all(statics_slice).unwrap();

    let turns_slice = unsafe {
        std::slice::from_raw_parts(
            all_turn_nums.as_ptr() as *const u8,
            all_turn_nums.len() * std::mem::size_of::<u16>(),
        )
    };
    writer.write_all(turns_slice).unwrap();

    // Write Indices [u16; total_nnz]
    let indices_slice = unsafe {
        std::slice::from_raw_parts(
            all_indices.as_ptr() as *const u8,
            all_indices.len() * std::mem::size_of::<u16>(),
        )
    };
    writer.write_all(indices_slice).unwrap();

    // Write Values [i16; total_nnz]
    let values_slice = unsafe {
        std::slice::from_raw_parts(
            all_values.as_ptr() as *const u8,
            all_values.len() * std::mem::size_of::<i16>(),
        )
    };
    writer.write_all(values_slice).unwrap();

    writer.flush().unwrap();
    drop(writer);

    let csr_uncompressed_bytes = std::mem::size_of::<CsrHeader>()
        + (all_offsets.len() * 8)
        + (all_evals.len() * 4)
        + (all_winners.len() * 4)
        + (all_static_evals.len() * 4)
        + (all_turn_nums.len() * 2)
        + (all_indices.len() * 2)
        + (all_values.len() * 2);

    let uncompressed_mb = csr_uncompressed_bytes as f64 / (1024.0 * 1024.0);
    let dense_equivalent_mb = (processed * (Board::TOTAL_FN * 2 + 16)) as f64 / (1024.0 * 1024.0);
    let disk_size_mb = std::fs::metadata(&args.out).map(|m| m.len() as f64 / (1024.0 * 1024.0)).unwrap_or(0.0);
    let avg_nnz = total_nnz as f64 / num_samples.max(1) as f64;

    println!(
        "Successfully extracted {} records (avg nnz: {:.1} / {}, CSR uncompressed: {:.2} MB [dense was {:.2} MB], on disk: {:.2} MB, compression: {:.1}x) into {}",
        processed,
        avg_nnz,
        Board::TOTAL_FN,
        uncompressed_mb,
        dense_equivalent_mb,
        disk_size_mb,
        uncompressed_mb / disk_size_mb.max(0.001),
        args.out
    );
}
