use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, usize};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Command, Stdio};
use bee_search::movegen::CutVertexes;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use regex::Regex;

use bee_search::board::{Board, GameResult};
use bee_search::perft;

const DEPTH: usize = 4;

fn main() {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    let mut rng = StdRng::seed_from_u64(seed);


    let mut args = env::args().skip(1);
    let engine_path = args.next().expect("Usage: <program> <engine_executable> [args...]");
    let engine_args: Vec<_> = args.collect();

    let mut child = Command::new(engine_path)
        .args(&engine_args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to spawn engine");

    let engine_stdin = child.stdin.take().unwrap();
    let engine_stdout = child.stdout.take().unwrap();
    let mut writer = BufWriter::new(engine_stdin);
    let mut reader = BufReader::new(engine_stdout);

    // Start a new game
    read_lines_until_ok(&mut reader); // "info" response
    run_command(&mut writer, &mut reader, "newgame Base+MLP");

    let mut board = Board::new();

    loop {
        if board.game_result() != GameResult::InProgress {
            println!("Game over: {:?}", board.game_result());
            break;
        }

        // Run perft command and parse engine's perft(5) value
        let engine_perft = run_perft_command(&mut writer, &mut reader, DEPTH);
        let our_perft = perft::perft(&mut board, DEPTH);
        if our_perft == engine_perft {
            println!("Perft match");
        } else {
            println!("Perft mismatch: {} != {}", our_perft, engine_perft);
            find_mismatch(&mut board, &mut writer, &mut reader, DEPTH);
            break;
        }

        // Play a random move
        let moves = board.generate_moves();
        let move_index = moves.len() * rng.random::<f64>() as usize;
        let m = moves[move_index];
        // Send the move to the engine
        run_command(&mut writer, &mut reader, &format!("play {}", board.action_to_string(m)));
        board.do_action(m);
    }

    writeln!(writer, "quit").unwrap();
    writer.flush().unwrap();
}

fn find_mismatch(board: &mut Board, writer: &mut impl Write, reader: &mut impl BufRead, depth: usize) {
    if depth == 1 {
        let my_moves = board.generate_moves().into_iter().map(|m| board.action_to_string(m)).collect::<Vec<_>>();
        let their_output = run_command(writer, reader, "validmoves");
        let their_moves = their_output.first().unwrap().split(';').map(|s| s.to_owned()).collect::<Vec<_>>();
        let my_set: HashSet<_> = my_moves.iter().collect();
        let their_set: HashSet<_> = their_moves.iter().collect();
        let only_in_mine: Vec<_> = my_set.difference(&their_set).collect();
        let only_in_theirs: Vec<_> = their_set.difference(&my_set).collect();
        println!();
        println!("In position:");
        println!("{}", board.game_string());
        println!();
        println!("Move count: {} vs {}", my_moves.len(), their_moves.len());
        println!();
        println!("Generated only be me:   {:?}", only_in_mine);
        println!("Generated only by them: {:?}", only_in_theirs);
        return;
    }
    let moves = board.generate_moves();
    for &m in &moves {
        run_command(writer, reader, &format!("play {}", board.action_to_string(m)));
        board.do_action(m);
        let our_perft = perft::perft(board, depth - 1);
        let engine_perft = run_perft_command(writer, reader, depth - 1);
        if our_perft != engine_perft {
            println!("Mismatch at depth {} on move {:?}: {} != {}", depth, m, our_perft, engine_perft);
            find_mismatch(board, writer, reader, depth - 1);
            return;
        }
        board.undo_action();
        run_command(writer, reader, "undo");
    }
    panic!("Failed to find a mismatch");
}

fn run_perft_command(writer: &mut impl Write, reader: &mut impl BufRead, depth: usize) -> usize {
    parse_perft_value(&run_command(writer, reader, &format!("perft {}", depth)), depth)
        .expect("Failed to parse perft")
}

fn run_command<W: Write, R: BufRead>(writer: &mut W, reader: &mut R, command: &str) -> Vec<String> {
    println!("> {}", command);
    writeln!(writer, "{}", command).unwrap();
    writer.flush().unwrap();
    read_lines_until_ok(reader)
}

fn read_lines_until_ok<R: BufRead>(reader: &mut R) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).unwrap() == 0 {
            break;
        }
        let trimmed = line.trim().to_owned();
        println!("< {}", trimmed);
        if trimmed == "ok" {
            break;
        }
        lines.push(trimmed);
    }
    lines
}

fn parse_perft_value(lines: &[String], depth: usize) -> Option<usize> {
    let pattern = format!(r"perft\({}\)\s*=\s*([\d.,]+)", depth);
    let re = Regex::new(&pattern).ok()?;
    for line in lines {
        if let Some(caps) = re.captures(line) {
            let num_str = caps.get(1)?.as_str().replace(".", "").replace(",", "");
            return num_str.parse::<usize>().ok();
        }
    }
    None
}
