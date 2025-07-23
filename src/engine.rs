use crate::{board::{Action, Board, GameResult}, eval::{Eval, Value}, piece::Color, piece_type::PCT_COUNT, tile::GRID_SIZE, tt::{TTFlag, TTable}};
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
    qsnodes: u64,
    deadline: Instant,
    // heremoves are indexed by color, (from or piece type), to; pass is last entry
    // map move -> history score 
    history_h: Vec<i64>,
    // map move -> countermove producing a beta cutoff
    countermove: Vec<Action>
}

#[derive(Clone, Copy, Debug)]
struct MoveInfo {
    mv: Action,
    quiet: bool,
    killer: u8,
    hist: i64
}

const KILLER_N: usize = 2;
type KillerT = [(Action, u8); KILLER_N];

impl Engine {
    pub fn new() -> Self {
        Self {
            nnodes: 0,
            qsnodes: 0,
            deadline: Instant::now(),
            tt: TTable::new(1 << 28), // TODO: make this configurable
            history_h: vec![0; 2 * (GRID_SIZE + PCT_COUNT) * GRID_SIZE + 1],
            countermove: vec![Action::Pass; 2 * (GRID_SIZE + PCT_COUNT) * GRID_SIZE + 1],
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

    fn last_move_index(b: &Board) -> usize {
        let last_move = b.turn_history.last().unwrap_or(&Action::Pass);
        Self::hist_index(b.color().other(), *last_move)
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

    fn fail_high(&self, soft: Eval, hard: Eval) -> Eval {
        debug_assert!(soft >= hard);
        soft
    }

    fn fail_low(&self, soft: Eval, hard: Eval) -> Eval {
        debug_assert!(soft <= hard);
        soft
    }

    fn order_moves(&self, moves: Vec<Action>, board: &mut Board, pv: Option<Action>, killers: &KillerT) -> Vec<MoveInfo> {
        let pv = pv.unwrap_or(Action::Pass);
        let countermove = self.countermove[Self::last_move_index(board)];
        let mut moves = moves.into_iter()
            .map(|mv| MoveInfo { mv,
                quiet: board.is_quiet(&mv),
                killer: killers.iter().find(|(m, _)| *m == mv).map(|(_, i)| *i).unwrap_or(0),
                hist: self.history_h[Self::hist_index(board.color(), mv)] })
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
            // 3) killer moves
            if a.killer != b.killer {
                return b.killer.cmp(&a.killer);
            }
            // 4) countermove
            if a.mv == countermove {
                return Ordering::Less;
            } else if b.mv == countermove {
                return Ordering::Greater;
            }
            // 5) history heuristic
            if a.hist != b.hist {
                return b.hist.cmp(&a.hist);
            }
            return Ordering::Equal;
        });
        moves
    }

    fn ordered_moves(&self, board: &mut Board, pv: Option<Action>, killers: &KillerT) -> Vec<MoveInfo> {
        self.order_moves(board.generate_moves(), board, pv, killers)
    }

    fn update_heuristics(&mut self, board: &Board, depth: Depth, alpha: Value, beta: Value, killers: &mut KillerT, pv: Option<Action>, mvi: MoveInfo, moves: &[MoveInfo], explored_quiet: i64) {
        let mv = mvi.mv;
        if alpha >= beta && mvi.quiet && Some(mv) != pv {
            // remember killer move
            if let Some((_, cnt)) = killers.iter_mut().find(|(m, _)| *m == mv) {
                *cnt = *cnt + 1;
            } else {
                let (kmove, cnt) = killers.iter_mut().min_by_key(|(_, i)| *i).unwrap();
                *kmove = mv;
                *cnt = 1;
            }
            // remember countermove
            self.countermove[Self::last_move_index(board)] = mv;
            // add history bonus
            let bonus = (1 as i64) << depth;
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

    // principal variation search
    fn pvs(&mut self, board: &mut Board, nply: Depth, ndepth: Depth, alpha: Value, beta: Value, killers: &mut KillerT, full_search: bool) -> Option<Value> {
        if full_search {
            // search the first move with the full window
            self.minimax(board, nply, ndepth, -beta, -alpha, killers).map(|v| -v)
        } else {
            // search the next moves with a null window to prove it is <= alpha
            let mut score = -self.minimax(board, nply, ndepth, -(alpha+1), -alpha, killers)?;
            // if the null window search fails, search again with a full window
            if score > alpha && beta > 1 + alpha {
                score = -self.minimax(board, nply, ndepth, -beta, -alpha, killers)?;
            }
            Some(score)
        }
    }

    fn qsearch(&mut self, board: &mut Board, ply: Depth, max_ply: Depth, mut alpha0: Value, mut beta: Value, killers: &mut KillerT) -> Option<Value> {
        const QS_DEPTH: Depth = 0; // qsearch is alwyas considered at depth 0, lower then any nomrmal search depth

        // out of time case
        if Instant::now() > self.deadline {
            return None;
        }
        self.nnodes += 1;
        self.qsnodes += 1;

        // terminal poisiton case
        let terminal_score = self.terminal_score(board, ply);
        if terminal_score.is_some() {
            return terminal_score;
        }

        let entry = self.tt.get(board.zobrist_hash);
        let pv = entry.map(|e| e.pv); // use the PV even if below depth
        if let Some(entry) = entry {
            if entry.depth >= QS_DEPTH {
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
        let mut moves: Option<Vec<Action>> = None;
        let eval_was_missing = entry.map(|e| e.eval).is_none();
        let eval = entry.and_then(|e| e.eval)
            .unwrap_or_else(|| {
                let mv = board.generate_moves();
                let len = mv.len();
                moves = Some(mv);
                board.static_eval_fast(len)
            });

        // stand pat: return immediately if the static eval is good enough, to avoid searching all non-quiet moves
        let mut alpha = alpha0;
        alpha = alpha.max(eval);
        if alpha >= beta {
            if eval_was_missing {
                self.tt.put_eval(board.zobrist_hash, eval);
            }
            return Some(self.fail_high(alpha, beta));
        }

        // depth cutoff case
        if ply >= max_ply {
            return Some(eval);
        }

        let mut moves = moves.unwrap_or_else(|| board.generate_moves());
        let mut child_klr = Default::default();
        let mut best_move: Option<(Value, MoveInfo)> = None;
        moves.retain(|mv| board.is_noisy(mv)); // qsearch only considers noisy moves
        let moves = self.order_moves(moves, board, pv, killers);
        for mvi in moves.iter() {
            let mv = mvi.mv;
            board.do_action(mv);
            let opt = self.qsearch(board, ply + 1, max_ply, -beta, -alpha, &mut child_klr).map(|v| -v);
            board.undo_action();
            let value = opt?;
            if best_move.is_none_or(|(v, _)| value > v) {
                best_move = Some((value, *mvi));
            }
            alpha = alpha.max(value);
            if alpha >= beta {
                break;
            }
        }

        // update transposition table
        if let Some((value, mvi)) = best_move {
            self.tt.put(board.zobrist_hash, alpha0, beta, mvi.mv, value, Some(eval), QS_DEPTH);
        }

        Some(alpha)
    }


    fn minimax(&mut self, board: &mut Board, ply: Depth, depth: Depth, mut alpha0: Value, mut beta: Value, killers: &mut KillerT) -> Option<Value> {
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

        // depth cutoff case
        if depth == 0 {
            return self.qsearch(board, ply, ply * 2, alpha0, beta, killers);
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

        let mut child_klr = Default::default();
        let mut alpha = alpha0;
        let mut best_move: Option<(Value, MoveInfo)> = None;
        let moves = self.ordered_moves(board, pv, killers);
        let mut explored_quiet = 0;
        for mvi in moves.iter() {
            let mv = mvi.mv;
            let first_move = mv == moves[0].mv || depth <= 2;
            board.do_action(mv);
            let opt = self.pvs(board, ply + 1, depth - 1, alpha, beta, &mut child_klr, first_move);
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
            self.update_heuristics(board, depth, alpha, beta, killers, pv, mvi, &moves, explored_quiet);
        }

        // update transposition table
        if let Some((value, mvi)) = best_move {
            self.tt.put(board.zobrist_hash, alpha0, beta, mvi.mv, value, None, depth);
        }

        Some(alpha)
    }

    fn aspiration_search(&mut self, board: &mut Board, depth: Depth) -> Option<Value> {
        // guess the value of the search will be around the previous result
        let guess = self.tt.get(board.zobrist_hash)
            .map(|e| e.value);
        const W: i64 = 70;
        let mut alpha = guess.map(|v| v as i64 - W).unwrap_or(-INF as i64);
        let mut beta = guess.map(|v| v as i64 + W).unwrap_or(INF as i64);
        let mut root_klr = Default::default();
        for i in 0.. {
            let value = self.minimax(board, 0, depth, alpha as Value, beta as Value, &mut root_klr)?;
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
        let mut prev_nodes = self.nnodes;
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
                eprintln!("depth {}: score={}, move={}, nodes={}, time={}ms", depth, score, board.action_to_string(incumbent), self.nnodes - prev_nodes, elapsed.as_millis());
            } else {
                eprintln!("depth {}: could not find matching tt entry", depth);
            }
            prev_nodes = self.nnodes;
        }
        incumbent
    }

    pub fn best_move(&mut self, board: &mut Board, max_depth: Depth, max_time: Duration) -> Action {
        let start = Instant::now();
        self.nnodes = 0;
        self.qsnodes = 0;
        self.tt.nwrite = 0;
        self.tt.ncollisions = 0;
        self.deadline = start + max_time;
        let r = self.iterative_deepening(board, max_depth);
        let elapsed = start.elapsed().as_secs_f64();
        eprintln!("explored {} nodes in {:.2}s -> {:.3} knodes/s, in qsearch={:.1}%", self.nnodes, elapsed, self.nnodes as f64 / elapsed / 1000.0, 100.0 * self.qsnodes as f64 / self.nnodes as f64);
        eprintln!("tt collisions: {}/{}, {:.2}%", self.tt.ncollisions, self.tt.nwrite, 100.0 * self.tt.ncollisions as f64 / self.tt.nwrite as f64);
        r
    }

    pub fn clear_tt(&mut self) {
        self.tt.clear();
    }

    pub fn last_nnodes(&self) -> u64 {
        self.nnodes
    }
}
