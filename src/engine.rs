use crate::{board::{Action, Board, GameResult}, eval::{Eval, Value}, piece::Color, piece_type::PCT_COUNT, tile::GRID_SIZE, tt::{TEntry, TTFlag, TTable}};
use std::{cmp::Ordering, sync::{atomic::{self, AtomicBool, AtomicU64, AtomicU8}, Arc}, time::{Duration, Instant}, u64, usize};
use crossbeam::thread;

pub type Depth = u8;
const INF: Eval = 32500;
const WIN: Eval = 32000;

fn mate_in(ply: Depth) -> Eval {
    WIN + 99 - ply as Eval
}

pub struct Engine {
    tt: TTable,
    nnodes: AtomicU64,
    qsnodes: AtomicU64,
    // LMR table
    reductions: Vec<f32>
}

pub struct ThreadData {
    id: usize,
    board: Board,
    abort: Arc<AtomicBool>,
    completed_depth: Arc<AtomicU8>,
    deadline: Instant,
    history_h: Vec<i64>,
    countermove: Vec<Action>,
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

struct PendingMove<'a> {
    td: &'a mut ThreadData,
}
impl ThreadData {
    fn play_pending(&mut self, mv: Action) -> PendingMove {
        self.board.do_action(mv);
        PendingMove { td: self }
    }
}
impl Drop for PendingMove<'_> {
    fn drop(&mut self) {
        self.td.board.undo_action();
    }
}

impl Engine {
    pub fn new() -> Self {
        let reductions = (0..1024).map(|i| if i == 0 { 0.0 } else {
            0.68 * (i as f32).ln()
        }).collect();
        Self {
            nnodes: AtomicU64::new(0),
            qsnodes: AtomicU64::new(0),
            tt: TTable::new(1 << 28), // TODO: make this configurable
            reductions
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

    /*fn fail_low(&self, soft: Eval, hard: Eval) -> Eval {
        debug_assert!(soft <= hard);
        soft
    }*/

    fn order_moves(&self, moves: Vec<Action>, td: &mut ThreadData, pv: Option<Action>, killers: &KillerT) -> Vec<MoveInfo> {
        let pv = pv.unwrap_or(Action::Pass);
        let countermove = td.countermove[Self::last_move_index(&td.board)];
        let mut moves = moves.into_iter()
            .map(|mv| MoveInfo { mv,
                quiet: td.board.is_quiet(&mv),
                killer: killers.iter().find(|(m, _)| *m == mv).map(|(_, i)| *i).unwrap_or(0),
                hist: td.history_h[Self::hist_index(td.board.color(), mv)] })
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
            // 4) history heuristic
            if a.hist != b.hist {
                return b.hist.cmp(&a.hist);
            }
            // 5) countermove
            if a.mv == countermove {
                return Ordering::Less;
            } else if b.mv == countermove {
                return Ordering::Greater;
            }
            return Ordering::Equal;
        });
        moves
    }

    fn update_heuristics(&self, td: &mut ThreadData, depth: Depth, alpha: Value, beta: Value, killers: &mut KillerT, pv: Option<Action>, mvi: MoveInfo, moves: &[MoveInfo], explored_quiet: i64) {
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
            td.countermove[Self::last_move_index(&td.board)] = mv;
            // add history bonus
            let bonus = (1 as i64) << depth;
            td.history_h[Self::hist_index(td.board.color(), mv)] += bonus;
            for prev in moves.iter() {
                if prev.mv == mv {
                    break;
                }
                if prev.quiet {
                    td.history_h[Self::hist_index(td.board.color(), prev.mv)] -= bonus / (explored_quiet - 1);
                }
            }
        }
    }

    // principal variation search with LMR
    fn pvs(&self, td: &mut ThreadData, ply: Depth, depth: Depth, alpha: Value, beta: Value, killers: &mut KillerT, move_index: usize) -> Option<Value> {
        let nply = ply + 1;
        let ndepth = depth - 1;
        if move_index == 0 || depth < 2 {
            // search the first move with the full window
            self.minimax(td, nply, ndepth, -beta, -alpha, killers).map(|v| -v)
        } else {
            // search the next moves with a null window to prove it is <= alpha
            let r = self.reductions[move_index] * self.reductions[depth as usize] + 1.05;
            let d = (ndepth as f32 - r).max(1.0).min(ndepth as f32).ceil() as Depth;
            let mut score = -self.minimax(td, nply, d, -(alpha+1), -alpha, killers)?;
            // if the reduced null window search fails, search again with a full window
            if score > alpha && (beta > 1 + alpha || d < ndepth) {
                score = -self.minimax(td, nply, ndepth, -beta, -alpha, killers)?;
            }
            Some(score)
        }
    }

    fn eval_with_caches(&self, board: &mut Board, entry: Option<TEntry>, moves: &mut Option<Vec<Action>>) -> Eval {
        entry.and_then(|e| e.eval)
            .unwrap_or_else(|| {
                let mv = board.generate_moves();
                let len = mv.len();
                *moves = Some(mv);
                let e = board.static_eval_fast(len);
                self.tt.put_eval(board.zobrist_hash, e);
                e
            })
    }

    fn qsearch(&self, td: &mut ThreadData, ply: Depth, max_ply: Depth, mut alpha0: Value, mut beta: Value, killers: &mut KillerT) -> Option<Value> {
        const QS_DEPTH: Depth = 0; // qsearch is alwyas considered at depth 0, lower then any nomrmal search depth

        // out of time case
        if Instant::now() > td.deadline {
            return None;
        }
        self.nnodes.fetch_add(1, atomic::Ordering::Relaxed);
        self.qsnodes.fetch_add(1, atomic::Ordering::Relaxed);

        // terminal poisiton case
        let terminal_score = self.terminal_score(&td.board, ply);
        if terminal_score.is_some() {
            return terminal_score;
        }

        let entry = self.tt.get(td.board.zobrist_hash);
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
        let eval = self.eval_with_caches(&mut td.board, entry, &mut moves);

        // stand pat: return immediately if the static eval is good enough, to avoid searching all non-quiet moves
        let mut alpha = alpha0;
        alpha = alpha.max(eval);
        if alpha >= beta {
            return Some(self.fail_high(alpha, beta));
        }

        // depth cutoff case
        if ply >= max_ply {
            return Some(eval);
        }

        let mut moves = moves.unwrap_or_else(|| td.board.generate_moves());
        let mut child_klr = Default::default();
        let mut best_move: Option<(Value, MoveInfo)> = None;
        moves.retain(|mv| td.board.is_noisy(mv)); // qsearch only considers noisy moves
        let moves = self.order_moves(moves, td, pv, killers);
        for mvi in moves.iter() {
            let mv = mvi.mv;
            let pending = td.play_pending(mv);
            let opt = self.qsearch(pending.td, ply + 1, max_ply, -beta, -alpha, &mut child_klr).map(|v| -v);
            drop(pending);
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
            self.tt.put(td.board.zobrist_hash, alpha0, beta, mvi.mv, value, Some(eval), QS_DEPTH);
        }

        Some(alpha)
    }


    fn minimax(&self, td: &mut ThreadData, ply: Depth, depth: Depth, mut alpha0: Value, mut beta: Value, killers: &mut KillerT) -> Option<Value> {
        let initial_hash = td.board.zobrist_hash;
        // out of time case
        if Instant::now() > td.deadline || td.abort.load(atomic::Ordering::Relaxed) {
            return None;
        }
        self.nnodes.fetch_add(1, atomic::Ordering::Relaxed);

        // depth cutoff case
        if depth == 0 {
            return self.qsearch(td, ply, ply * 2, alpha0, beta, killers);
        }

        // terminal poisiton case
        let terminal_score = self.terminal_score(&td.board, ply);
        if terminal_score.is_some() {
            debug_assert!(ply > 0);
            return terminal_score;
        }

        // tt lookup
        let entry = self.tt.get(td.board.zobrist_hash);
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

        let mut moves: Option<Vec<Action>> = None;

        // null move pruning
        const NMR: Depth = 2;
        if depth > NMR && ply > 0 {
            let eval = self.eval_with_caches(&mut td.board, entry, &mut moves);
            if eval >= beta {
                let r = NMR + (depth - NMR) / 3;
                let mut nm_klr = Default::default();
                let pending = td.play_pending(Action::Pass);
                let value = -self.minimax(pending.td, ply+1, depth - r, -beta, -beta + 1, &mut nm_klr)?;
                drop(pending);
                if value >= beta {
                    return Some(self.fail_high(value, beta));
                }
            }
        }

        let moves = moves.unwrap_or_else(|| td.board.generate_moves());
        let moves = self.order_moves(moves, td, pv, killers);
        let mut alpha = alpha0;
        let mut best_move: Option<(Value, MoveInfo)> = None;
        let mut child_klr = Default::default();
        let mut explored_quiet = 0;
        for (move_idx, mvi) in moves.iter().enumerate() {
            let mv = mvi.mv;
            let pending = td.play_pending(mv);
            let opt = self.pvs(pending.td, ply, depth, alpha, beta, &mut child_klr, move_idx);
            drop(pending);
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
            self.update_heuristics(td, depth, alpha, beta, killers, pv, mvi, &moves, explored_quiet);
        }

        // update transposition table
        if let Some((value, mvi)) = best_move {
            self.tt.put(td.board.zobrist_hash, alpha0, beta, mvi.mv, value, None, depth);
        }

        debug_assert!(td.board.zobrist_hash == initial_hash);
        Some(alpha)
    }

    fn aspiration_search(&self, td: &mut ThreadData, depth: Depth) -> Option<Value> {
        if td.abort.load(atomic::Ordering::Relaxed) { return None; }
        // guess the value of the search will be around the previous result
        let guess = self.tt.get(td.board.zobrist_hash)
            .map(|e| e.value);
        let w: i64 = 70 + td.id as i64;
        let mut alpha = guess.map(|v| v as i64 - w).unwrap_or(-INF as i64);
        let mut beta = guess.map(|v| v as i64 + w).unwrap_or(INF as i64);
        let mut root_klr = Default::default();
        for i in 0.. {
            let initial_hash = td.board.zobrist_hash;
            let value = self.minimax(td, 0, depth, alpha as Value, beta as Value, &mut root_klr)?;
            debug_assert!(td.board.zobrist_hash == initial_hash);
            // when search fails, grow the window exponentially
            if value as i64 <= alpha {
                alpha = (alpha - w * (1 << i)).min(value as i64 - 1).max(-INF as i64);
            } else if value as i64 >= beta {
                beta = (beta + w * (1 << i)).max(value as i64 + 1).min(INF as i64);
            } else {
                return Some(value);
            }
        }
        panic!();
    }

    fn iterative_deepening(&self, td: &mut ThreadData, max_depth: Depth) {
        for depth in 1..=max_depth {
            let start = Instant::now();
            let score = self.aspiration_search(td, depth);
            let option = self.tt.get(td.board.zobrist_hash);
            if score.is_none() {
                break;
            }

            let mut prev = td.completed_depth.load(atomic::Ordering::Relaxed);
            while depth > prev {
                match td.completed_depth.compare_exchange(prev, depth, atomic::Ordering::SeqCst, atomic::Ordering::Relaxed) {
                    Ok(_) => {
                        if let Some(entry) = option {
                            let score = score.unwrap();
                            let elapsed = start.elapsed();
                            let nnodes = self.nnodes.load(atomic::Ordering::Relaxed);
                            eprintln!("[#{}] depth {}: score={}, move={}, nodes={}, time={}ms", td.id, depth, score, td.board.action_to_string(entry.pv), nnodes, elapsed.as_millis());
                        } else {
                            eprintln!("[#{}] depth {}: could not find matching tt entry", td.id, depth);
                        }
                        break;
                    }
                    Err(actual) => prev = actual,
                }
            }

        }
        td.abort.store(true, atomic::Ordering::Relaxed);
    }

    fn lazy_smp(self: Arc<Self>, board: &Board, max_depth: Depth, deadline: Instant, num_threads: usize) -> (Value, Action) {
        self.tt.clear_one(board.zobrist_hash); // make sure the root TT slot is available
    
        thread::scope(|scope| {
            let abort = Arc::new(AtomicBool::new(false));
            let compl = Arc::new(AtomicU8::new(0));
            for i in 0..num_threads {
                let th_engine = self.clone();
                let th_abort = abort.clone();
                let th_compl = compl.clone();
                scope.spawn(move |_| {
                    let mut td = ThreadData {
                        id: i,
                        board: board.clone(),
                        abort: th_abort,
                        completed_depth: th_compl,
                        deadline,
                        history_h: vec![0; 2 * (GRID_SIZE + PCT_COUNT) * GRID_SIZE + 1],
                        countermove: vec![Action::Pass; 2 * (GRID_SIZE + PCT_COUNT) * GRID_SIZE + 1],
                    };
                    th_engine.iterative_deepening(&mut td, max_depth);
                });
            }
        })
        .unwrap();

        let option = self.tt.get(board.zobrist_hash);
        if let Some(entry) = option {
            if board.is_legal(entry.pv) {
                return (entry.value, entry.pv);
            }
            eprintln!("root tt entry is not legal");
        } else {
            eprintln!("could not find matching tt entry");
        }
        (0, *board.generate_moves().first().unwrap_or(&Action::Pass))
    }

    pub fn best_move(self: Arc<Self>, board: &Board, max_depth: Depth, max_time: Duration, num_threads: usize) -> (Value, Action) {
        let start = Instant::now();
        self.nnodes.store(0, atomic::Ordering::Relaxed);
        self.qsnodes.store(0, atomic::Ordering::Relaxed);
        let deadline = start + max_time;
        let r = self.clone().lazy_smp(board, max_depth, deadline, num_threads);
        let elapsed = start.elapsed().as_secs_f64();
        let nnodes = self.nnodes.load(atomic::Ordering::Relaxed);
        let qsnodes = self.qsnodes.load(atomic::Ordering::Relaxed);
        eprintln!("[{} th] explored {} nodes in {:.2}s -> {:.3} knodes/s, in qsearch={:.1}%", num_threads, nnodes, elapsed, nnodes as f64 / elapsed / 1000.0, 100.0 * qsnodes as f64 / nnodes as f64);
        r
    }

    pub fn clear_tt(&mut self) {
        self.tt.clear();
    }

    pub fn last_nnodes(&self) -> u64 {
        self.nnodes.load(atomic::Ordering::Relaxed)
    }
}
