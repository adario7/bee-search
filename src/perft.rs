use crate::{board::{ActionList, Board, GameResult}};

pub fn perft(board: &mut Board, depth: usize) -> usize {
    // One reused ActionList per recursion level (indexed by remaining
    // depth), allocated once for the whole traversal instead of a fresh
    // ~1.5KB ActionList being constructed and copied out of generate_moves()
    // at every node.
    let mut bufs: Vec<ActionList> = (0..=depth).map(|_| ActionList::new()).collect();
    perft_rec(board, depth, &mut bufs)
}

fn perft_rec(board: &mut Board, depth: usize, bufs: &mut [ActionList]) -> usize {
    if depth == 0 {
        return 1;
    }
    let result = board.game_result();
    if result != GameResult::InProgress {
        return 1;
    }
    board.generate_moves_into(&mut bufs[depth]);
    let n = bufs[depth].len();
    if depth == 1 {
        return n;
    }
    let mut total = 0;
    for i in 0..n {
        let m = bufs[depth][i];
        board.do_action(m);
        total += perft_rec(board, depth - 1, bufs);
        board.undo_action();
    }
    total
}

pub fn display_perft(board: &mut Board, depth: usize) {
    for d in 0..=depth {
        let start = std::time::Instant::now();
        let c = perft(board, d);
        let elapsed = start.elapsed();
        let kns = c as f64 / elapsed.as_secs_f64() / 1000.0;
        println!("{}\t\t{}\t\t{}ms\t\t{:.0}KN/s", d, c, elapsed.as_millis(), kns);
    }
}
