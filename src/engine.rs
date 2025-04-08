use crate::{board::{Action, Board, GameResult}, eval::{Eval, Value}, tt::{TTFlag, TTable}};
use std::time::{Duration, Instant};

pub type Depth = u8;
const INF: Eval = 32500;
const WIN: Eval = 32000;

fn mate_in(ply: Depth) -> Eval {
    WIN + 99 - ply as Eval
}

pub struct Engine {
    tt: TTable,
    nnodes: u64,
    deadline: Instant
}

impl Engine {
    pub fn new() -> Self {
        Self {
            nnodes: 0,
            deadline: Instant::now(),
            tt: TTable::new(1 << 24), // TODO: make this configurable
        }
    }


    fn minimax(&mut self, board: &mut Board, ply: Depth, depth: Depth, mut alpha0: Value, mut beta: Value) -> Option<Eval> {
        // out of time case
        if Instant::now() > self.deadline {
            return None;
        }
        self.nnodes += 1;

        // terminal poisiton case
        let result = board.game_result();
        if result == GameResult::Draw {
            return Some(0);
        } else if let GameResult::Winner(color) = result {
            return if color == board.color() {
                Some(mate_in(ply))
            } else {
                Some(-mate_in(ply))
            };
        }

        let entry = self.tt.get(board.zobrist_hash);
        //let pv = entry.map(|e| e.pv); // use the PV even if below depth
        if let Some(entry) = entry {
            if entry.depth >= depth {
                // improve out bound
                if entry.flag == TTFlag::LowerBound {
                    alpha0 = alpha0.max(entry.value);
                } else if entry.flag == TTFlag::UpperBound {
                    beta = beta.min(entry.value);
                }
                // return the exact value if this is not the root
                if ply != 0 && (entry.flag == TTFlag::Exact || alpha0 >= beta) {
                    return Some(entry.value);
                }
            }
        }
        let eval = entry.map(|e| e.eval)
            .unwrap_or_else(|| board.static_eval());

        // depth cutoff case
        if depth == 0 {
            return Some(eval);
        }

        let mut alpha = alpha0;
        let mut best_move: Option<(Value, Action)> = None;
        for mv in board.generate_moves() {
            board.do_action(mv);
            let opt = self.minimax(board, ply + 1, depth - 1, -beta, -alpha);
            board.undo_action();
            let value = opt?;
            let value = -value;
            if best_move.is_none_or(|(v, _)| value > v) {
                best_move = Some((value, mv));
            }
            alpha = alpha.max(value);
            if alpha >= beta {
                break;
            }
        }

        if let Some((value, mv)) = best_move {
            self.tt.put(board.zobrist_hash, alpha0, beta, mv, value, eval, depth);
        }

        best_move.map(|(value, _)| value)
    }

    fn iterative_deepening(&mut self, board: &mut Board, max_depth: Depth) -> Action {
        let mut incumbent = *board.generate_moves().first().unwrap_or(&Action::Pass);
        for depth in 1..=max_depth {
            let start = Instant::now();
            let score = self.minimax(board, 0, depth, -INF, INF);
            let option = self.tt.get(board.zobrist_hash);
            let elapsed = start.elapsed();
            if score.is_none() {
                eprintln!("depth {}: ran out of time", depth);
                continue;
            }
            if let Some(entry) = option {
                incumbent = entry.pv;
                eprintln!("depth {}: score={}, move={}, nodes={}, time={}ms", depth, entry.value, board.action_to_string(incumbent), self.nnodes, elapsed.as_millis());
            } else {
                eprintln!("depth {}: could not find matching tt entry", depth);
            }
        }
        incumbent
    }

    pub fn best_move(&mut self, board: &mut Board, max_depth: Depth, max_time: Duration) -> Action {
        self.nnodes = 0;
        self.deadline = Instant::now() + max_time;
        self.iterative_deepening(board, max_depth)
    }

    pub fn clear_tt(&mut self) {
        self.tt.clear();
    }
}
