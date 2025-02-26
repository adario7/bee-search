use crate::{board::{Action, Board, GameResult}, eval::Eval};
use std::time::{Duration, Instant};

pub type Depth = u8;
const INF: Eval = 32500;
const WIN: Eval = 32000;

fn mate_in(ply: Depth) -> Eval {
    WIN + 99 - ply as Eval
}

pub struct Engine {
    nnodes: u64,
    deadline: Instant
}

impl Engine {
    pub fn new() -> Self {
        Self {
            nnodes: 0,
            deadline: Instant::now()
        }
    }


    fn minimax(&mut self, board: &mut Board, ply: Depth, depth: Depth, alpha0: Eval, beta: Eval) -> Option<(Eval, Option<Action>)> {
        if Instant::now() > self.deadline {
            return None;
        }
        self.nnodes += 1;

        let result = board.game_result();
        if result == GameResult::Draw {
            return Some((0, None));
        } else if let GameResult::Winner(color) = result {
            return if color == board.color() {
                Some((mate_in(ply), None))
            } else {
                Some((-mate_in(ply), None))
            };
        }
        if depth == 0 {
            return Some((board.static_eval(), None))
        }

        let mut alpha = alpha0;
        let mut best_move: Option<(Eval, Action)> = None;
        for mv in board.generate_moves() {
            board.do_action(mv);
            let opt = self.minimax(board, ply + 1, depth - 1, -beta, -alpha);
            board.undo_action();
            let (value, _) = opt?;
            let value = -value;
            if best_move.is_none_or(|(v, _)| value > v) {
                best_move = Some((value, mv));
            }
            alpha = alpha.max(value);
            if alpha >= beta {
                break;
            }
        }
        best_move.map(|(value, mv)| (value, Some(mv)))
    }

    fn iterative_deepening(&mut self, board: &mut Board, max_depth: Depth) -> Action {
        let mut incumbent = *board.generate_moves().first().unwrap();
        for depth in 1..=max_depth {
            let opt = self.minimax(board, 0, depth, -INF, INF);
            if let Some((score, Some(mv))) = opt {
                eprintln!("depth: {}, score: {}, move: {}, nodes: {}", depth, score, board.action_to_string(mv), self.nnodes);
                incumbent = mv;
            }
        }
        incumbent
    }

    pub fn best_move(&mut self, board: &mut Board, max_depth: Depth, max_time: Duration) -> Action {
        self.nnodes = 0;
        self.deadline = Instant::now() + max_time;
        self.iterative_deepening(board, max_depth)
    }
}
