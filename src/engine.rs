use crate::{board::{Action, Board, GameResult}, eval::{Eval, Value}, piece::Color, piece_type::PCT_COUNT, tile::GRID_SIZE, tt::{TEntry, TTFlag, TTable}};
use std::{cmp::Ordering, sync::{atomic::{self, AtomicBool, AtomicU64, AtomicU8}, Arc}, time::{Duration, Instant}, u64, usize};
use crossbeam::thread;

pub type Depth = u8;
const INF: Eval = 32500;
const WIN: Eval = 32000;

fn mate_in(ply: Depth) -> Eval {
    WIN + 99 - ply as Eval
}

fn mated_in(ply: Depth) -> Eval {
    -mate_in(ply)
}

fn display_eval(e: Eval) -> String {
    if e > WIN {
        format!("+#{}", WIN + 99 - e)
    } else if e < -WIN {
        format!("-#{}", e + WIN + 99)
    } else {
        format!("{:+}", e)
    }
}

pub struct Engine {
    tt: TTable,
    nnodes: AtomicU64,
    qsnodes: AtomicU64,
    // LMR table
    reductions: Vec<f32>,
    move_votes: Vec<AtomicU64>,
}

pub struct ThreadData {
    id: usize,
    board: Board,
    abort: Arc<AtomicBool>,
    completed_depth: Arc<AtomicU8>,
    deadline: Instant,
    history_h: Vec<i64>,
    countermove: Vec<Action>,
    local_nodes: u64,
    last_vote: Option<(Action, u64)>,
}

#[derive(Clone, Copy, Debug)]
struct MoveInfo {
    mv: Action,
    quiet: bool,
    killer: u8,
    hist: i64
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NodeType {
    Pv,
    All,
    Cut,
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
            0.55 * (i as f32).ln()
        }).collect();
        let move_votes = (0..(GRID_SIZE + PCT_COUNT) * GRID_SIZE + 1).map(|_| AtomicU64::new(0)).collect();
        Self {
            nnodes: AtomicU64::new(0),
            qsnodes: AtomicU64::new(0),
            tt: TTable::new(1 << 28), // TODO: make this configurable
            reductions,
            move_votes
        }
    }

    fn vote_index(mv: Action) -> usize {
        match mv {
            Action::Place(pct, to) => to as usize * (GRID_SIZE + PCT_COUNT) + GRID_SIZE + pct as usize,
            Action::Move(from, to) => to as usize * (GRID_SIZE + PCT_COUNT) + from as usize,
            Action::Pass => (GRID_SIZE + PCT_COUNT) * GRID_SIZE
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
    fn pvs(&self, td: &mut ThreadData, nt: NodeType, ply: Depth, depth: Depth, alpha: Value, beta: Value, killers: &mut KillerT, move_index: usize) -> Option<Value> {
        let nply = ply + 1;
        let ndepth = depth - 1;
        let child_t = match (nt, move_index) {
            (NodeType::Pv, 0) => NodeType::Pv,
            (NodeType::Pv, _) => NodeType::Cut,
            (NodeType::Cut, 0) => NodeType::All,
            (NodeType::Cut, _) => NodeType::Cut,
            (NodeType::All, _) => NodeType::Cut
        };
        if move_index == 0 || depth < 2 {
            // search the first move with the full window
            self.minimax(td, child_t, nply, ndepth, -beta, -alpha, killers).map(|v| -v)
        } else {
            // search the next moves with a null window to prove it is <= alpha
            let r = 0.8
                + self.reductions[move_index] * self.reductions[depth as usize]
                + match nt { NodeType::Pv => -1.0, NodeType::Cut => 2.3, _ => 0.0  };
            let d = (ndepth as f32 - r).max(1.0).min(ndepth as f32).round() as Depth;
            let mut score = -self.minimax(td,  child_t, nply, d, -(alpha+1), -alpha, killers)?;
            // if the reduced null window search fails, search again with a full window
            if score > alpha && (beta > 1 + alpha || d < ndepth) {
                let re_t = match nt {
                    NodeType::Pv => NodeType::Pv,
                    NodeType::All => NodeType::Cut,
                    NodeType::Cut => NodeType::All
                };
                score = -self.minimax(td, re_t, nply, ndepth, -beta, -alpha, killers)?;
            }
            Some(score)
        }
    }

    fn eval_with_caches(&self, board: &mut Board, entry: Option<TEntry>, moves: &mut Option<Vec<Action>>) -> Eval {
        entry.map(|e| e.eval).unwrap_or_else(|| {
            let mv = board.generate_moves();
            let e = board.static_eval_fast(&mv);
            *moves = Some(mv);
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
        td.local_nodes += 1;

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
            self.tt.put(td.board.zobrist_hash, alpha0, beta, mvi.mv, value, eval, QS_DEPTH);
        }

        Some(alpha)
    }


    fn minimax(&self, td: &mut ThreadData, nt: NodeType, ply: Depth, depth: Depth, mut alpha: Value, mut beta: Value, killers: &mut KillerT) -> Option<Value> {
        let initial_hash = td.board.zobrist_hash;
        let (window_a, window_b) = (alpha, beta);
        // out of time case
        if Instant::now() > td.deadline || td.abort.load(atomic::Ordering::Relaxed) {
            return None;
        }
        self.nnodes.fetch_add(1, atomic::Ordering::Relaxed);
        td.local_nodes += 1;

        // depth cutoff case
        if depth == 0 {
            return self.qsearch(td, ply, ply * 2, alpha, beta, killers);
        }

        // move count pruning
        if ply != 0 {
            alpha = alpha.max(mated_in(ply));
            beta = beta.min(mate_in(ply + 1));
            if alpha >= beta {
                return Some(self.fail_high(alpha, beta));
            }
        }

        // terminal poisiton case
        let terminal_score = self.terminal_score(&td.board, ply);
        if terminal_score.is_some() {
            debug_assert!(ply > 0);
            return terminal_score;
        }

        // tt lookup
        let mut entry = self.tt.get(td.board.zobrist_hash);
        let mut pv = entry.map(|e| e.pv); // use the PV even if below depth
        if let Some(entry) = entry {
            if entry.depth >= depth {
                // improve our bound
                if entry.flag == TTFlag::LowerBound {
                    alpha = alpha.max(entry.value);
                } else if entry.flag == TTFlag::UpperBound {
                    beta = beta.min(entry.value);
                }
                // return the exact value if this is not the root
                if ply != 0 && alpha >= beta {
                    return Some(self.fail_high(alpha, beta));
                }
                if ply != 0 && entry.flag == TTFlag::Exact {
                    return Some(entry.value);
                }
            }
        }

        // internal iterative deepening
        if depth >= 4 && pv.is_none() {
            const IIDR: Depth = 2;
            let r = IIDR + (depth - IIDR) / 3;
            self.minimax(td, nt, ply, depth - r, alpha, beta, killers);
            entry = self.tt.get(td.board.zobrist_hash);
            pv = entry.map(|e| e.pv);
        }

        // static eval
        let mut moves: Option<Vec<Action>> = None;
        let eval = self.eval_with_caches(&mut td.board, entry, &mut moves);

        // razoring
        if nt != NodeType::Pv && depth <= 6 && (eval as i32) < (alpha as i32 - 500 - 200 * depth as i32 * depth as i32) {
            return self.qsearch(td, ply, ply * 2, alpha, beta, killers)
        }

        // futility pruning
        if depth <= 4 && eval as i64 > beta as i64 + 150 + 100 * depth as i64 && eval.abs() < 6000 {
            return Some(self.fail_high(eval, beta));
        }

        // null move pruning
        const NMR: Depth = 2;
        if nt == NodeType::Cut && depth > NMR && eval >= beta + 50 {
            if eval >= beta {
                let r = NMR + (depth - NMR) / 3;
                let mut nm_klr = Default::default();
                let pending = td.play_pending(Action::Pass);
                let value = -self.minimax(pending.td, NodeType::All, ply+1, depth - r, -beta, -beta + 1, &mut nm_klr)?;
                drop(pending);
                if value >= beta {
                    return Some(self.fail_high(value, beta));
                }
            }
        }

        let alpha0 = alpha; // alpha0 is the alpha before searching moves
        let moves = moves.unwrap_or_else(|| td.board.generate_moves());
        let moves = self.order_moves(moves, td, pv, killers);
        let mut best_move: Option<(Value, MoveInfo)> = None;
        let mut child_klr = Default::default();
        let mut explored_quiet = 0;
        for (move_idx, mvi) in moves.iter().enumerate() {
            let mv = mvi.mv;
            let pending = td.play_pending(mv);
            let opt = self.pvs(pending.td, nt, ply, depth, alpha, beta, &mut child_klr, move_idx);
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
            self.tt.put(td.board.zobrist_hash, alpha0, beta, mvi.mv, value, eval, depth);
        }

        // cast vote on the best move
        if ply == 0 && alpha > window_a && alpha < window_b {
            if let Some((mv, cnt)) = td.last_vote {
                self.move_votes[Self::vote_index(mv)].fetch_sub(cnt, atomic::Ordering::Relaxed);
            }
            td.last_vote = if let Some((_, mvi)) = best_move {
                let (mv, cnt) = (mvi.mv, (depth as f64 * (td.local_nodes as f64).sqrt()).round() as u64);
                self.move_votes[Self::vote_index(mv)].fetch_add(cnt, atomic::Ordering::Relaxed);
                Some((mv, cnt))
            } else {
                None
            };
        }

        debug_assert!(td.board.zobrist_hash == initial_hash);
        Some(alpha)
    }

    fn aspiration_search(&self, td: &mut ThreadData, depth: Depth) -> Option<Value> {
        if td.abort.load(atomic::Ordering::Relaxed) { return None; }
        // guess the value of the search will be around the previous result
        let guess = self.tt.get(td.board.zobrist_hash)
            .map(|e| e.value);
        let w: i64 = 30 + td.id as i64;
        let mut alpha = guess.map(|v| v as i64 - w).unwrap_or(-INF as i64);
        let mut beta = guess.map(|v| v as i64 + w).unwrap_or(INF as i64);
        let mut root_klr = Default::default();
        for i in 0..99 {
            let initial_hash = td.board.zobrist_hash;
            let value = self.minimax(td, NodeType::Pv, 0, depth, alpha as Value, beta as Value, &mut root_klr)?;
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
        eprintln!("!!! [th={}] aspiration search failed", td.id);
        debug_assert!(false);
        Some(0)
    }

    fn iterative_deepening(&self, td: &mut ThreadData, max_depth: Depth, verbose: bool) {
        for depth in 1..=max_depth {
            let start = Instant::now();
            let score = self.aspiration_search(td, depth);
            let option = self.tt.get(td.board.zobrist_hash);
            if score.is_none() {
                break;
            }

            if verbose {
                let mut prev = td.completed_depth.load(atomic::Ordering::Relaxed);
                while depth > prev {
                    match td.completed_depth.compare_exchange(prev, depth, atomic::Ordering::SeqCst, atomic::Ordering::Relaxed) {
                        Ok(_) => {
                            if let Some(entry) = option {
                                let score = score.unwrap();
                                let elapsed = start.elapsed();
                                let nnodes = self.nnodes.load(atomic::Ordering::Relaxed);
                                eprintln!("[#{}] depth {}: score={}, move={}, nodes={}, time={}ms", td.id, depth, display_eval(score), td.board.action_to_string(entry.pv), nnodes, elapsed.as_millis());
                            } else {
                                eprintln!("[#{}] depth {}: could not find matching tt entry", td.id, depth);
                            }
                            break;
                        }
                        Err(actual) => prev = actual,
                    }
                }
            }
        }
        td.abort.store(true, atomic::Ordering::Relaxed);
    }

    fn lazy_smp(self: Arc<Self>, board: &mut Board, max_depth: Depth, deadline: Instant, num_threads: usize, verbose: bool, abort_signal: Option<Arc<AtomicBool>>) -> (Value, Action) {
        let root_tt = self.tt.get(board.zobrist_hash);
        if verbose {
            eprintln!("root TT entry has depth {}", root_tt.map_or(0, |e| e.depth));
        }
        self.tt.clear_one(board.zobrist_hash); // make sure the root TT slot is available

        // reset move votes
        let root_moves = board.generate_moves();
        for &mv in &root_moves {
            self.move_votes[Self::vote_index(mv)].store(0, atomic::Ordering::Relaxed);
        }
    
        thread::scope(|scope| {
            let abort = abort_signal.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
            let compl = Arc::new(AtomicU8::new(0));
            for i in 0..num_threads {
                let th_engine = self.clone();
                let th_abort = abort.clone();
                let th_compl = compl.clone();
                let th_board = board.clone();
                scope.spawn(move |_| {
                    let mut td = ThreadData {
                        id: i,
                        board: th_board,
                        abort: th_abort,
                        completed_depth: th_compl,
                        deadline,
                        history_h: vec![0; 2 * (GRID_SIZE + PCT_COUNT) * GRID_SIZE + 1],
                        countermove: vec![Action::Pass; 2 * (GRID_SIZE + PCT_COUNT) * GRID_SIZE + 1],
                        local_nodes: 0,
                        last_vote: None,
                    };
                    th_engine.iterative_deepening(&mut td, max_depth, verbose);
                });
            }
        })
        .unwrap();

        let value = self.tt.get(board.zobrist_hash)
            .map(|e| e.value)
            .unwrap_or(0);
        let mut root_votes: Vec<_> = root_moves.into_iter()
            .map(|m| (m, self.move_votes[Self::vote_index(m)].load(atomic::Ordering::Relaxed)))
            .collect();
        root_votes.sort_by_key(|(_, cnt)| -(*cnt as i64));
        if verbose {
            let tot_votes = root_votes.iter().map(|(_, cnt)| *cnt).sum::<u64>();
            eprint!("Votes: ");
            for &(mv, cnt) in &root_votes {
                if cnt == 0 { continue; }
                let pct = 100.0 * cnt as f64 / tot_votes as f64;
                eprint!("{}={:.1}% ", board.action_to_string(mv), pct);
            }
            eprintln!();
        }
        let best_move = root_votes.first().map(|(mv, _)| *mv).unwrap_or(Action::Pass);
        (value, best_move)
    }

    pub fn best_move(self: Arc<Self>, board: &mut Board, max_depth: Depth, max_time: Duration, num_threads: usize, verbose: bool, abort: Option<Arc<AtomicBool>>) -> (Value, Action) {
        let start = Instant::now();
        self.nnodes.store(0, atomic::Ordering::Relaxed);
        self.qsnodes.store(0, atomic::Ordering::Relaxed);
        let deadline = start + max_time;
        let r = self.clone().lazy_smp(board, max_depth, deadline, num_threads, verbose, abort);
        let elapsed = start.elapsed().as_secs_f64();
        let nnodes = self.nnodes.load(atomic::Ordering::Relaxed);
        let qsnodes = self.qsnodes.load(atomic::Ordering::Relaxed);
        if verbose {
            eprintln!("[{} th] explored {} nodes in {:.4}s -> {:.3} knodes/s, in qsearch={:.1}%", num_threads, nnodes, elapsed, nnodes as f64 / elapsed / 1000.0, 100.0 * qsnodes as f64 / nnodes as f64);
            eprint!("pv: ");
            let mut bb = board.clone();
            for _ in 0..16 {
                let mv = self.tt.get(bb.zobrist_hash).map(|e| e.pv).unwrap_or(Action::Pass);
                if mv == Action::Pass { break; }
                eprint!("{};", bb.action_to_string(mv));
                bb.do_action(mv);
            }
            eprintln!();
        }
        r
    }

    pub fn clear_tt(&mut self) {
        self.tt.clear();
    }

    pub fn get_pv(&self, board: &Board) -> Option<Action> {
        self.tt.get(board.zobrist_hash).map(|e| e.pv)
    }

    pub fn last_nnodes(&self) -> u64 {
        self.nnodes.load(atomic::Ordering::Relaxed)
    }
}
