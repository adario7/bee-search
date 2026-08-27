use clap::Parser;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use regex::Regex;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

use bee_search::board::Board;
use bee_search::graph_nn::NUM_TOKENS;
use bee_search::piece::Color;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, default_value = "/home/alessandro/local/logs/evaluations.json")]
    evals: String,

    #[arg(short, long, default_value_t = 10)]
    count: usize,

    #[arg(short, long, default_value_t = 42)]
    seed: u64,
}

fn main() {
    let args = Args::parse();
    println!("Reading evaluations from {}...", args.evals);
    let file = File::open(&args.evals).unwrap_or_else(|e| panic!("Failed to open {}: {}", args.evals, e));
    let reader = BufReader::with_capacity(32 * 1024 * 1024, file);

    let pos_re = Regex::new(r#""position"\s*:\s*"([^"]+)""#).unwrap();
    let eval_re = Regex::new(r#""evaluation"\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)"#).unwrap();
    let win_re = Regex::new(r#""winner"\s*:\s*([0-9]+(?:\.[0-9]+)?)"#).unwrap();
    let static_re = Regex::new(r#""static_eval"\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)"#).unwrap();

    let mut raw_positions = Vec::new();
    let mut cur_pos = None;
    let mut cur_eval = None;
    let mut cur_win = None;
    let mut cur_static = None;

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

                raw_positions.push((pos, eval, win, st));
            } else {
                cur_eval = None;
                cur_win = None;
                cur_static = None;
            }
        }
    }

    println!("Total positions in dataset: {}", raw_positions.len());
    let mut rng = StdRng::seed_from_u64(args.seed);

    let mut indices: Vec<usize> = (0..raw_positions.len()).collect();
    indices.shuffle(&mut rng);

    let mut out_file = File::create("logs/sampled_10_positions.json").unwrap();
    writeln!(out_file, "[").unwrap();

    let mut sampled_count = 0;
    for &idx in &indices {
        let (pos_str, eval_raw, win_raw, static_raw) = &raw_positions[idx];
        if let Ok(mut board) = Board::parse_game_string(pos_str) {
            let moves = board.generate_moves();
            if moves.is_empty() {
                continue;
            }
            let is_white = board.color() == Color::White;
            let truth_eval_white = if is_white { *eval_raw } else { -*eval_raw };
            let truth_win_white = if is_white { *win_raw } else { 1.0 - *win_raw };
            let truth_static_white = if is_white { *static_raw } else { -*static_raw };

            let mlp_feat = board.features_fast(&moves);
            let other = board.other_moves();
            let tg = board.get_token_graph_fast(&moves, &other);

            let comma = if sampled_count > 0 { "," } else { "" };
            writeln!(out_file, "  {comma}{{").unwrap();
            writeln!(out_file, "    \"idx\": {idx},").unwrap();
            writeln!(out_file, "    \"turn_num\": {},", board.turn_num).unwrap();
            writeln!(out_file, "    \"is_white\": {is_white},").unwrap();
            writeln!(out_file, "    \"truth_eval\": {truth_eval_white},").unwrap();
            writeln!(out_file, "    \"truth_winner\": {truth_win_white},").unwrap();
            writeln!(out_file, "    \"truth_static_eval\": {truth_static_white},").unwrap();

            // MLP features (232-dim)
            let mlp_str = mlp_feat.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
            writeln!(out_file, "    \"mlp_features\": [{mlp_str}],").unwrap();

            // GT features (28 x 8)
            writeln!(out_file, "    \"gt_features\": [").unwrap();
            for tok in 0..NUM_TOKENS {
                let row_str = tg.features[tok].iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
                let end_comma = if tok < NUM_TOKENS - 1 { "," } else { "" };
                writeln!(out_file, "      [{row_str}]{end_comma}").unwrap();
            }
            writeln!(out_file, "    ],").unwrap();

            // GT adj (28 x 28)
            writeln!(out_file, "    \"gt_adj\": [").unwrap();
            for i in 0..NUM_TOKENS {
                let row_str = tg.adj[i].iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
                let end_comma = if i < NUM_TOKENS - 1 { "," } else { "" };
                writeln!(out_file, "      [{row_str}]{end_comma}").unwrap();
            }
            writeln!(out_file, "    ]").unwrap();
            writeln!(out_file, "  }}").unwrap();

            sampled_count += 1;
            if sampled_count >= args.count {
                break;
            }
        }
    }

    writeln!(out_file, "]").unwrap();
    println!("Successfully dumped {} sampled positions to logs/sampled_10_positions.json", sampled_count);
}
