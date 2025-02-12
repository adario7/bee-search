use crate::board::{Board, GameResult};

fn perft0(board: &mut Board, depth: usize) -> usize {
    if depth == 0 {
        return 1;
    }
    let result = board.game_result();
    if result != GameResult::InProgress {
        return 1;
    }
    let moves = board.generate_moves();
    if depth == 1 {
        return moves.len();
    }
    moves.iter().map(|m| {
        board.do_action(*m);
        let c = perft0(board, depth - 1);
        board.undo_action();
        c
    }).sum()
}

pub fn perft(board: &mut Board, depth: usize) {
    for d in 0..=depth {
        let start = std::time::Instant::now();
        let c = perft0(board, d);
        let elapsed = start.elapsed();
        let kns = c as f64 / elapsed.as_secs_f64() / 1000.0;
        println!("{}\t\t{}\t\t{}ms\t\t{:.0}KN/s", d, c, elapsed.as_millis(), kns);
    }
}
