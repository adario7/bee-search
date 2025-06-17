use crate::{board::{Action, Board, GameResult}, eval::{Eval, Value}, piece::Color, piece_type::PCT_COUNT, tile::{Direction, GRID_SIZE}, tt::{TTFlag, TTable}};
use std::{cmp::Ordering, time::{Duration, Instant}, u64, usize};

pub type Depth = u8;
const INF: Eval = 32500;
const WIN: Eval = 32000;

fn mate_in(ply: Depth) -> Eval {
    WIN + 99 - ply as Eval
}

pub struct Engine {
    tt: TTable,
    nnodes: u64,
    deadline: Instant,
    // indexed by color, (from or piece type), to; pass is last entry
    history_h: Vec<i64>
}

#[derive(Clone, Copy, Debug)]
struct MoveInfo {
    mv: Action,
    quiet: bool,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            nnodes: 0,
            deadline: Instant::now(),
            tt: TTable::new(1 << 24), // TODO: make this configurable
            history_h: vec![0; 2 * (GRID_SIZE + PCT_COUNT) * GRID_SIZE + 1],
        }
    }

    fn hist_index(c: Color, mv: Action) -> usize {
        let ci = c as usize * (GRID_SIZE + PCT_COUNT) * GRID_SIZE;
        match mv {
            Action::Place(pct, to)
                => ci + to as usize * (GRID_SIZE + PCT_COUNT) + GRID_SIZE + pct as usize,
            Action::Move(from, to)
                => ci + to as usize * (GRID_SIZE + PCT_COUNT) + from as usize,
            Action::Pass
                => 2 * (GRID_SIZE + PCT_COUNT) * GRID_SIZE
        }
    }

    fn terminal_score(&self, board: &Board, ply: Depth) -> Option<Value> {
        let result = board.game_result();
        match result {
            GameResult::InProgress => None,
            GameResult::Draw => Some(0),
            GameResult::Winner(color) => {
                if color == board.color() {
                    Some(mate_in(ply))
                } else {
                    Some(-mate_in(ply))
                }
            }
        }
    }

    // a move is quiet if it does not attack the opponent's queen
    fn is_quiet(board: &Board, action: &Action) -> bool {
        if let Action::Move(_, to) = action {
            let queen = board.queens[board.color().other() as usize];
            if let Some(target) = queen {
                if *to == target {
                    return false;
                }
                for d in Direction::all() {
                    if *to + *d == target {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn ordered_moves(&self, board: &mut Board, pv: Option<Action>) -> Vec<MoveInfo> {
        let pv = pv.unwrap_or(Action::Pass);
        let mut moves = board.generate_moves().into_iter()
            .map(|mv| MoveInfo { mv, quiet: Self::is_quiet(board, &mv) })
            .collect::<Vec<_>>();
        moves.sort_by(|a, b| {
            // 1) PV
            if a.mv == pv {
                return Ordering::Less;
            } else if b.mv == pv {
                return Ordering::Greater;
            }
            // 2) attacks on the queen
            if !a.quiet && b.quiet {
                return Ordering::Less;
            } else if a.quiet && !b.quiet {
                return Ordering::Greater;
            }
            // 3) history heuristic
            let ha = self.history_h[Self::hist_index(board.color(), a.mv)];
            let hb = self.history_h[Self::hist_index(board.color(), b.mv)];
            return hb.cmp(&ha);
        });
        moves
    }

    // principal variation search
    fn pvs(&mut self, board: &mut Board, nply: Depth, ndepth: Depth, alpha: Value, beta: Value, is_first_move: bool) -> Option<Value> {
        if is_first_move{
            // search the first move with the full window
            self.minimax(board, nply, ndepth, -beta, -alpha).map(|v| -v)
        } else {
            // search the next moves with a null window to prove it is <= alpha
            let mut score = -self.minimax(board, nply, ndepth, -(alpha+1), -alpha)?;
            // if the null window search fails, search again with a full window
            if score > alpha && beta > alpha + 1 {
                score = -self.minimax(board, nply, ndepth, -beta, -alpha)?;
            }
            Some(score)
        }
    }

    fn minimax(&mut self, board: &mut Board, ply: Depth, depth: Depth, mut alpha0: Value, mut beta: Value) -> Option<Value> {
        // out of time case
        if Instant::now() > self.deadline {
            return None;
        }
        self.nnodes += 1;

        // terminal poisiton case
        let terminal_score = self.terminal_score(board, ply);
        if terminal_score.is_some() {
            return terminal_score;
        }

        let entry = self.tt.get(board.zobrist_hash);
        let pv = entry.map(|e| e.pv); // use the PV even if below depth
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
        let mut best_move: Option<(Value, MoveInfo)> = None;
        let moves = self.ordered_moves(board, pv);
        let mut explored_quiet = 0;
        for mvi in moves.iter() {
            let mv = mvi.mv;
            let first_move = mv == moves[0].mv;
            board.do_action(mv);
            let opt = self.pvs(board, ply + 1, depth - 1, alpha, beta, first_move);
            board.undo_action();
            let value = opt?;
            if best_move.is_none_or(|(v, _)| value > v) {
                best_move = Some((value, *mvi));
            }
            alpha = alpha.max(value);
            if mvi.quiet {
                explored_quiet += 1;
            }
            if alpha >= beta {
                break;
            }
        }

        // update history heuristic
        if let Some((_, mvi)) = best_move {
            let mv = mvi.mv;
            if alpha >= beta && mvi.quiet && Some(mv) != pv {
                let bonus = depth as i64 * depth as i64;
                self.history_h[Self::hist_index(board.color(), mv)] += bonus;
                for prev in moves.iter() {
                    if prev.mv == mv {
                        break;
                    }
                    if prev.quiet {
                        self.history_h[Self::hist_index(board.color(), prev.mv)] -= bonus / (explored_quiet - 1);
                    }
                }
            }
        }

        // update transposition table
        if let Some((value, mvi)) = best_move {
            self.tt.put(board.zobrist_hash, alpha0, beta, mvi.mv, value, eval, depth);
        }

        best_move.map(|(value, _)| value)
    }

    fn aspiration_search(&mut self, board: &mut Board, depth: Depth) -> Option<Value> {
        // guess the value of the search will be around the previous result
        let guess = self.tt.get(board.zobrist_hash)
            .map(|e| e.value);
        const W: i64 = 70;
        let mut alpha = guess.map(|v| v as i64 - W).unwrap_or(-INF as i64);
        let mut beta = guess.map(|v| v as i64 + W).unwrap_or(INF as i64);
        for i in 0.. {
            let value = self.minimax(board, 0, depth, alpha as Value, beta as Value)?;
            // when search fails, grow the window exponentially
            if value as i64 <= alpha {
                alpha = (alpha - W * (1 << i)).min(value as i64 - 1).max(-INF as i64);
            } else if value as i64 >= beta {
                beta = (beta + W * (1 << i)).max(value as i64 + 1).min(INF as i64);
            } else {
                return Some(value);
            }
        }
        panic!();
    }

    fn iterative_deepening(&mut self, board: &mut Board, max_depth: Depth) -> Action {
        let mut incumbent = *board.generate_moves().first().unwrap_or(&Action::Pass);
        for depth in 1..=max_depth {
            let start = Instant::now();
            let score = self.aspiration_search(board, depth);
            let option = self.tt.get(board.zobrist_hash);
            let elapsed = start.elapsed();
            if score.is_none() {
                eprintln!("depth {}: ran out of time", depth);
                break;
            }
            let score = score.unwrap();
            if let Some(entry) = option {
                incumbent = entry.pv;
                eprintln!("depth {}: score={}, move={}, nodes={}, time={}ms", depth, score, board.action_to_string(incumbent), self.nnodes, elapsed.as_millis());
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

    pub fn last_nnodes(&self) -> u64 {
        self.nnodes
    }
}
