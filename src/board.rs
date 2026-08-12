use std::collections::hash_map::DefaultHasher;
use std::default::Default;
use std::hash::Hasher;
use crate::tile::{adjacent, Direction, Tile, GRID_SIZE};
use crate::piece::{Color, Piece};
use crate::piece_type::{Pct, PieceType};
use crate::abstractions::*;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnderPiece {
    pub tile: Tile,
    pub piece: Piece,
}
impl Default for UnderPiece {
    fn default() -> Self {
        Self {
            tile: 0,
            piece: Piece::empty(),
        }
    }
}

// Backed by `MaybeUninit` rather than `[Action; 256]`: constructing a fresh
// ActionList used to write all 256 slots (~1.5KB) up front via `[Action::Pass; 256]`
// even though only the first `len` are ever read (everything derefs to
// `&moves[..len]`). That happened on every single `generate_moves()` call -
// millions of times during search/perft - for pure write-then-never-read work.
// With MaybeUninit, `new()` is a no-op and only pushed slots are written.
#[derive(Clone, Copy)]
pub struct ActionList {
    moves: [std::mem::MaybeUninit<Action>; 256],
    len: usize,
}
impl ActionList {
    pub fn new() -> Self {
        // SAFETY: an array of `MaybeUninit<T>` has no validity invariant on
        // its elements, so leaving it uninitialized here is always sound;
        // only `moves[..len]`, which `push` always initializes, is ever read.
        Self { moves: unsafe { std::mem::MaybeUninit::uninit().assume_init() }, len: 0 }
    }
    pub fn push(&mut self, action: Action) {
        debug_assert!(self.len < 256);
        self.moves[self.len] = std::mem::MaybeUninit::new(action);
        self.len += 1;
    }
    /// Append one `Place` per entry of `bugs`, all on the same tile.
    ///
    /// Exists because `push` was ~33% of perft runtime: each call separately
    /// re-read `len`, bounds-checked, stored one element and wrote `len` back,
    /// serialising a read-modify-write per move. Placements dominate (~93% of
    /// all pushes) and every placement on a given tile shares the same encoded
    /// prefix, differing only in the low 3 piece-type bits - so they can be
    /// written as one fixed-length slice loop with a single `len` update,
    /// which LLVM can vectorize.
    #[inline]
    pub fn push_places(&mut self, tile: Tile, bugs: &[PieceType]) {
        let n = bugs.len();
        debug_assert!(self.len + n <= 256);
        debug_assert!((tile as usize) < GRID_SIZE);
        let base = (ACT_TAG_PLACE << ACT_TAG_SHIFT) | ((tile as u32 & ACT_TILE_MASK) << 3);
        let start = self.len;
        let dst = &mut self.moves[start..start + n];
        for (slot, &bug) in dst.iter_mut().zip(bugs) {
            *slot = std::mem::MaybeUninit::new(Action(base | (bug as u32 & ACT_PCT_MASK)));
        }
        self.len = start + n;
    }

    pub fn swap_remove(&mut self, index: usize) -> Action {
        debug_assert!(index < self.len);
        // SAFETY: index and len-1 are both < self.len, hence initialized.
        let val = unsafe { self.moves[index].assume_init() };
        self.moves[index] = self.moves[self.len - 1];
        self.len -= 1;
        val
    }
    pub fn retain<F>(&mut self, mut f: F) where F: FnMut(&Action) -> bool {
        let mut i = 0;
        while i < self.len {
            // SAFETY: i < self.len, hence initialized.
            let keep = f(unsafe { self.moves[i].assume_init_ref() });
            if !keep {
                self.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn clear(&mut self) {
        self.len = 0;
    }
    fn as_slice(&self) -> &[Action] {
        // SAFETY: `push` is the only way to grow `len`, and it always
        // initializes the slot it claims, so moves[..len] is initialized.
        unsafe { std::slice::from_raw_parts(self.moves.as_ptr() as *const Action, self.len) }
    }
    fn as_mut_slice(&mut self) -> &mut [Action] {
        // SAFETY: see as_slice.
        unsafe { std::slice::from_raw_parts_mut(self.moves.as_mut_ptr() as *mut Action, self.len) }
    }
}
impl std::ops::Deref for ActionList {
    type Target = [Action];
    fn deref(&self) -> &[Action] { self.as_slice() }
}
impl std::ops::DerefMut for ActionList {
    fn deref_mut(&mut self) -> &mut [Action] { self.as_mut_slice() }
}
impl<'a> IntoIterator for &'a ActionList {
    type Item = &'a Action;
    type IntoIter = std::slice::Iter<'a, Action>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}
impl IntoIterator for ActionList {
    type Item = Action;
    type IntoIter = std::vec::IntoIter<Action>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().to_vec().into_iter()
    }
}

static ZOBRIST_TABLE: OnceLock<[u64; GRID_SIZE * 2]> = OnceLock::new();
static PLAYER_HASH: u64 = 0xc851ba955a512175;

// A move, packed into 4 bytes (was a 6-byte enum).
//
// Bit layout:
//   bits 21..20  tag: 0 = Place, 1 = Move, 2 = Pass
//   Place        bits 12..3 = tile (10 bits), bits 2..0 = piece type (3 bits)
//   Move         bits 19..10 = from (10 bits), bits 9..0 = to (10 bits)
//   Pass         payload zero
//
// A Tile needs only 10 bits (GRID_SIZE == 1024) and a PieceType only 3, so
// everything fits with room to spare.
//
// ORDERING IS LOAD-BEARING: the tag sits in the high bits and each payload
// packs its fields most-significant-first, so the derived integer `Ord`
// reproduces *exactly* the ordering the old `#[derive(Ord)]` enum had - all
// Place < all Move < Pass, and lexicographic by field within a variant.
// `speed-bench` sorts move lists and then samples with a seeded RNG, so a
// different order would silently change that benchmark's games.
//
// Why 4 bytes rather than 6: on aarch64 (the deployment target) address
// computation can only scale by powers of two, so a 6-byte element forced an
// extra `add x,x,x,lsl #1` + `lsl #1` pair at every indexed access - 59 such
// sequences in the library. At 4 bytes it folds into `[base, idx, lsl #2]`.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Action(u32);

const ACT_TAG_SHIFT: u32 = 20;
const ACT_TAG_PLACE: u32 = 0;
const ACT_TAG_MOVE: u32 = 1;
const ACT_TAG_PASS: u32 = 2;
const ACT_TILE_MASK: u32 = 0x3FF; // 10 bits: GRID_SIZE == 1024
const ACT_PCT_MASK: u32 = 0x7; // 3 bits: 8 piece types

/// Unpacked view of an [`Action`], for matching. `Action::kind()` decodes
/// into this; it is a temporary that optimizes away in hot paths.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ActionKind {
    Place(Tile, PieceType),
    Move(Tile, Tile),
    Pass,
}

// Deliberately named like the old enum variants (`Action::Place`,
// `Action::Move`, `Action::Pass`) rather than snake_case constructors, so
// that switching from an enum to this packed struct did not churn ~47
// otherwise-untouched call sites. The lint allows below are the price.
#[allow(non_snake_case, non_upper_case_globals)]
impl Action {
    pub const Pass: Action = Action(ACT_TAG_PASS << ACT_TAG_SHIFT);

    #[inline]
    pub fn Place(tile: Tile, pct: PieceType) -> Action {
        debug_assert!((tile as usize) < GRID_SIZE);
        Action((ACT_TAG_PLACE << ACT_TAG_SHIFT)
            | ((tile as u32 & ACT_TILE_MASK) << 3)
            | (pct as u32 & ACT_PCT_MASK))
    }

    #[inline]
    pub fn Move(from: Tile, to: Tile) -> Action {
        debug_assert!((from as usize) < GRID_SIZE && (to as usize) < GRID_SIZE);
        Action((ACT_TAG_MOVE << ACT_TAG_SHIFT)
            | ((from as u32 & ACT_TILE_MASK) << 10)
            | (to as u32 & ACT_TILE_MASK))
    }

    #[inline]
    fn tag(self) -> u32 { self.0 >> ACT_TAG_SHIFT }

    #[inline]
    pub fn is_pass(self) -> bool { self.tag() == ACT_TAG_PASS }
    #[inline]
    pub fn is_move(self) -> bool { self.tag() == ACT_TAG_MOVE }
    #[inline]
    pub fn is_place(self) -> bool { self.tag() == ACT_TAG_PLACE }

    /// Destination of a Move. Only meaningful when `is_move()`.
    #[inline]
    pub fn move_to(self) -> Tile { (self.0 & ACT_TILE_MASK) as Tile }
    /// Origin of a Move. Only meaningful when `is_move()`.
    #[inline]
    pub fn move_from(self) -> Tile { ((self.0 >> 10) & ACT_TILE_MASK) as Tile }
    /// Target tile of a Place. Only meaningful when `is_place()`.
    #[inline]
    pub fn place_tile(self) -> Tile { ((self.0 >> 3) & ACT_TILE_MASK) as Tile }
    /// Piece type of a Place. Only meaningful when `is_place()`.
    #[inline]
    pub fn place_pct(self) -> PieceType { PieceType::from_index((self.0 & ACT_PCT_MASK) as u8) }

    #[inline]
    pub fn kind(self) -> ActionKind {
        match self.tag() {
            ACT_TAG_PLACE => ActionKind::Place(self.place_tile(), self.place_pct()),
            ACT_TAG_MOVE => ActionKind::Move(self.move_from(), self.move_to()),
            _ => ActionKind::Pass,
        }
    }
}

impl Default for Action {
    // Must stay Pass: `KillerT` and `countermove` are built with
    // `Default::default()` and rely on the empty slot meaning "no move".
    #[inline]
    fn default() -> Self { Action::Pass }
}

impl std::fmt::Debug for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.kind(), f)
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum GameResult {
    InProgress,
    Winner(Color),
    Draw
}

#[derive(Clone)]
pub struct Board {
    pub gametype: String,
    // the piece on each tile of the board, for stacks: to topmost piece
    pub world: [Piece; GRID_SIZE],
    // stacked pieces
    pub underworld: [UnderPiece; 32],
    pub underworld_size: u8,
    // remaining pieces still to be placed
    pub placeable: [[u8; 8]; 2],
    // position of the queens
    pub queens: [Option<Tile>; 2],
    // list of occupied tiles for each player
    pub occupied_tiles: [OccupancyVec; 2],
    // number of plies played
    pub turn_num: usize,
    // turn history
    pub turn_history: Vec<Action>,
    // placeable tiles for both players
    // a tile is placeable for a player if he can put a piece that is not present on the board on it
    pub tiles_placeable: [TileSet; 2],
    // how many pieces are place on each tile
    pub height: [u8; GRID_SIZE],
    // cache of pieces that are not movable
    pub immovable: CachedValue<TileSet>,
    // zobrist data
    pub zobrist_table: &'static [u64; GRID_SIZE * 2],
    pub zobrist_hash: u64,
    pub zobrist_history: Vec<u64>,
}

pub const TOT_QTY: [u8; 8] = [1, 3, 2, 3, 2, 1, 1, 1];

impl Board {
    pub fn new() -> Self {
        Self::new_mlp(1, 1, 1)
    }

    pub fn new_mlp(m: u8, l: u8, p: u8) -> Self {
        let zobrist_table = ZOBRIST_TABLE.get_or_init(|| {
            let mut table = [0u64; GRID_SIZE * 2];
            let mut hasher = DefaultHasher::new();
            for (i, entry) in table.iter_mut().enumerate() {
                hasher.write_usize(i);
                *entry = hasher.finish();
            }
            table
        });

        Board {
            gametype: format!("Base{}",
                if m + l + p > 0 {
                    format!("+{}{}{}",
                        if m > 0 {"M"} else {""},
                        if l > 0 {"L"} else {""},
                        if p > 0 {"P"} else {""},
                    )
                } else {
                    "".to_string()
                }),
            world: [Piece::empty(); GRID_SIZE],
            underworld: [UnderPiece::default(); 32],
            underworld_size: 0,
            placeable: [[1, 3, 2, 3, 2, m, l, p], [1, 3, 2, 3, 2, m, l, p]], //TODO: doesn't consider values of TOT_QTY
            queens: [None, None],
            occupied_tiles: [OccupancyVec::new(), OccupancyVec::new()],
            turn_num: 0,
            turn_history: Vec::new(),
            tiles_placeable: [TileSet::new(), TileSet::new()],
            height: [0; GRID_SIZE],
            immovable: CachedValue::new(),
            zobrist_table,
            zobrist_hash: 0,
            zobrist_history: Vec::new(),
        }
    }

    pub fn color(&self) -> Color {
        Color::from_index((self.turn_num & 1) as u8)
    }

    pub fn tile(&self, tile: Tile) -> Piece {
        // SAFETY: every Tile value flowing through this codebase is
        // produced either by TILE_ZERO or by `Add<Direction>`, which always
        // masks with GRID_MASK (< GRID_SIZE) - never a raw, unmasked value.
        debug_assert!((tile as usize) < GRID_SIZE);
        unsafe { *self.world.get_unchecked(tile as usize) }
    }

    pub fn occupied(&self, tile: Tile) -> bool {
        // Equivalent to `self.height[tile] > 0` by invariant (a tile has a
        // piece on top iff its height is nonzero), but reads `world`
        // instead of a second, separate 1024-entry array - `world` is
        // already what most movegen call sites touch for the same tile.
        self.tile(tile).is_some()
    }

    pub fn queen_required(&self) -> bool {
        self.turn_num > 5 && self.placeable[self.color().index()][PieceType::Queen as usize] > 0
    }

    pub fn height(&self, tile: Tile) -> i32 {
        // SAFETY: see tile() above - Tile is always < GRID_SIZE by construction.
        (unsafe { *self.height.get_unchecked(tile as usize) }) as i32
    }

    pub fn is_stacked(&self, tile: Tile) -> bool {
        self.height(tile) > 1
    }

    fn zobrist(&self, t: Tile, p: Piece, h: u32) -> u64 {
        let hash = self.zobrist_table[((t as usize) << 1) | (p.color() as usize)];
        hash.rotate_left((h<<3) | (p.ptype() as u32))
    }

    fn add_occupancy(occupied_hexes: &mut [OccupancyVec; 2], p: Piece, t: Tile) {
        occupied_hexes[p.color().index()].push(t);
    }

    fn remove_occupancy(occupied_hexes: &mut [OccupancyVec; 2], p: Piece, t: Tile) {
        let vec = &mut occupied_hexes[p.color().index()];
        vec.remove(t);
    }

    pub fn underworld_at(&self, tile: Tile) -> Vec<Piece> {
        let mut res = Vec::new();
        for i in 0..self.underworld_size as usize {
            if self.underworld[i].tile == tile {
                res.push(self.underworld[i].piece);
            }
        }
        res
    }

    fn add_piece(&mut self, tile: Tile, piece: Piece) {
        let curr = &mut self.world[tile as usize];
        if curr.is_some() {
            debug_assert!((self.underworld_size as usize) < 32);
            self.underworld[self.underworld_size as usize] = UnderPiece { tile, piece: *curr };
            self.underworld_size += 1;
            Self::remove_occupancy(&mut self.occupied_tiles, *curr, tile);
        }
        *curr = piece;
        Self::add_occupancy(&mut self.occupied_tiles, *curr, tile);
        if piece.ptype() == Pct::Queen {
            self.queens[piece.color().index()] = Some(tile);
        }

        self.height[tile as usize] += 1;

        self.zobrist_hash ^= self.zobrist(tile, piece, self.height(tile) as u32);
    }

    fn remove_piece(&mut self, tile: Tile) {

        self.zobrist_hash ^= self.zobrist(tile, self.world[tile as usize], self.height(tile) as u32);

        self.height[tile as usize] -= 1;

        let curr = &mut self.world[tile as usize];
        debug_assert!(curr.is_some());

        if curr.ptype() == Pct::Queen {
            self.queens[curr.color().index()] = None;
        }
        Self::remove_occupancy(&mut self.occupied_tiles, *curr, tile);

        if self.height[tile as usize] > 0 {
            let mut found = false;
            for i in (0..self.underworld_size as usize).rev() {
                if self.underworld[i].tile == tile {
                    *curr = self.underworld[i].piece;
                    for j in i..(self.underworld_size as usize - 1) {
                        self.underworld[j] = self.underworld[j + 1];
                    }
                    self.underworld_size -= 1;
                    Self::add_occupancy(&mut self.occupied_tiles, *curr, tile);
                    found = true;
                    break;
                }
            }
            debug_assert!(found);
        } else {
            *curr = Piece::empty();
        }
    }

    fn do_move(&mut self, from: Tile, to: Tile) {
        let piece = self.tile(from);
        debug_assert!(piece.is_some());
        debug_assert_ne!(from, to);
        self.remove_piece(from);
        self.add_piece(to, piece);
    }

    pub fn make_next_piece(&self, pct: PieceType) -> Piece {
        let num = 1 + TOT_QTY[pct as usize] - self.placeable[self.color().index()][pct as usize];
        debug_assert!(0 < num && num < 4);
        Piece::make(self.color(), pct, num)
    }

    /// very slow!
    pub fn is_legal(&mut self, action: Action) -> bool {
        self.generate_moves().contains(&action)
    }

    /// assumes the action is legal
    pub fn do_action(&mut self, action: Action) {
        match action.kind() {
            ActionKind::Place(tile, piece_type) => {
                let piece = self.make_next_piece(piece_type);
                self.add_piece(tile, piece);
                self.placeable[self.color().index()][piece_type as usize] -= 1;
            }
            ActionKind::Move(from, to) => self.do_move(from, to),
            ActionKind::Pass => {}
        }
        self.turn_num += 1;
        self.turn_history.push(action);

        self.zobrist_hash ^= PLAYER_HASH;
        self.zobrist_history.push(self.zobrist_hash);
    }

    pub fn undo_action(&mut self) {
        let action = self.turn_history.pop().unwrap();
        self.turn_num -= 1;

        self.zobrist_hash ^= PLAYER_HASH;
        self.zobrist_history.pop();
        match action.kind() {
            ActionKind::Place(tile, piece_type) => {
                debug_assert!(self.world[tile as usize].ptype() == piece_type);
                self.remove_piece(tile);
                self.placeable[self.color().index()][piece_type as usize] += 1;
            }
            ActionKind::Move(from, to) => self.do_move(to, from),
            ActionKind::Pass => {}
        }
    }

    fn queen_surround(&self, color: Color) -> usize {
        match self.queens[color as usize] {
            None => 0,
            Some(tile) => adjacent(tile).into_iter().filter(|&t| self.tile(t).is_some()).count()
        }
    }

    fn count_repetitions(&self) -> usize {
        if self.zobrist_history.len() <= 10 { return 1; }
        1 + self.zobrist_history
            .iter().rev().step_by(4).skip(1).take(8)
            .filter(|&&hash| hash == self.zobrist_hash)
            .count()
    }

    pub fn game_result(&self) -> GameResult {
        let repetitions = self.count_repetitions();
        if repetitions >= 3 {
            return GameResult::Draw;
        }
        let sw = self.queen_surround(Color::White);
        let sb = self.queen_surround(Color::Black);
        match (sw, sb) {
            (6, 6) => GameResult::Draw,
            (6, _) => GameResult::Winner(Color::Black),
            (_, 6) => GameResult::Winner(Color::White),
            (_, _) => GameResult::InProgress,
        }
    }


    // a move is noisy if attacks the opponent's queen
    pub fn is_noisy(self: &Board, action: &Action) -> bool {
        // Hot: called once per move during ordering. Uses the direct
        // accessors rather than `kind()` so no ActionKind is materialized.
        if action.is_move() {
            let to = action.move_to();
            let queen = self.queens[self.color().other() as usize];
            if let Some(target) = queen {
                if to == target {
                    return true;
                }
                for d in Direction::all() {
                    if to + *d == target {
                        return true;
                    }
                }
            }
        }
        false
    }

    // a move is quiet if it does not attack the opponent's queen
    pub fn is_quiet(self: &Board, action: &Action) -> bool {
        !self.is_noisy(action)
    }

    pub fn all_occupied_tiles(&self) -> impl Iterator<Item = Tile> + '_ {
        self.occupied_tiles[self.color().index()].iter().copied().chain(self.occupied_tiles[self.color().other().index()].iter().copied())
    }
}

#[cfg(test)]
mod test {
    use rand::Rng;
    use super::*;
    use crate::tile::{Direction, TILE_ZERO};
    use indicatif::ProgressBar;

    #[test]
    fn test_board_do_undo() {
        let mut board = Board::new();
        let a = TILE_ZERO;
        let b = TILE_ZERO + 1;
        board.do_action(Action::Place(a, PieceType::Queen));
        assert_eq!(board.placeable[Color::White.index()][PieceType::Queen as usize], 0);
        board.do_action(Action::Place(b, PieceType::Queen));
        assert_eq!(board.placeable[Color::Black.index()][PieceType::Queen as usize], 0);
        board.do_action(Action::Move(a, b));
        assert_eq!(board.world[a as usize], Piece::empty());
        assert_eq!(board.world[b as usize], Piece::make(Color::White, PieceType::Queen, 1));
        assert_eq!(board.underworld_at(b)[0], Piece::make(Color::Black, PieceType::Queen, 1));
        board.undo_action();
        assert_eq!(board.world[a as usize], Piece::make(Color::White, PieceType::Queen, 1));
        assert_eq!(board.world[b as usize], Piece::make(Color::Black, PieceType::Queen, 1));
        board.undo_action();
        assert!(board.world[b as usize].is_none());
        assert_eq!(board.placeable[Color::Black.index()][PieceType::Queen as usize], 1);
    }

    #[test]
    fn test_board_result() {
        let mut board = Board::new();
        let a = TILE_ZERO;
        let b = TILE_ZERO + Direction::E + Direction::E + Direction::E;
        board.do_action(Action::Place(a, PieceType::Queen));
        board.do_action(Action::Place(b, PieceType::Queen));
        assert_eq!(board.game_result(), GameResult::InProgress);
        for &d in Direction::all() {
            let pct = Pct::iter_all().filter(|&p| board.placeable[0][p as usize] > 0).next().unwrap();
            board.do_action(Action::Place(b + d, pct));
            println!("{} {:?} {:?}", board.turn_num, d, pct);
            if board.turn_num == 1+6*2 {
                assert_eq!(board.game_result(), GameResult::Winner(Color::White));
            } else {
                assert_eq!(board.game_result(), GameResult::InProgress);
            }
            board.do_action(Action::Place(a + d, pct));
        }
        assert_eq!(board.game_result(), GameResult::Draw);
        board.do_action(Action::Move(b + Direction::E, a + Direction::W + Direction::W + Direction::W));
        assert_eq!(board.game_result(), GameResult::Winner(Color::Black));
    }

    fn compare_boards(b1: &mut Board, b2: &mut Board) -> bool {
        for i in 0..2 {
            for tile in b1.occupied_tiles[i].iter() {
                if b1.height(*tile) != b2.height(*tile){
                    return false;
                }

                if b1.underworld_at(*tile) != b2.underworld_at(*tile) {
                    return false;
                }

                if b1.world[*tile as usize] != b2.world[*tile as usize] {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn test_zobrist_hash() {
        //Non riesce a testare permutazioni di pedine sulla stessa casella in base all'altezza, in teoria dovrebbe essere diverso

        let mut b1 = Board::new();
        let mut b2 = Board::new();


        let depth = 50;
        let num_runs = 100;
        let num_tries = 100;
        let mut bar = ProgressBar::new(num_runs);

        let mut rng = rand::rng();

        for _run in 0..num_runs {
            bar.inc(1);
            for _turn in 0..depth {

                if b1.game_result() != GameResult::InProgress {
                    break;
                }

                let moves = b1.generate_moves();

                if moves.len() > 0 {

                    for _i in 0..num_tries {
                        let m1 = rng.random_range(0..moves.len());
                        let m2 = rng.random_range(0..moves.len());

                        b1.do_action(moves[m1]);
                        b2.do_action(moves[m2]);

                        assert_eq!(compare_boards(&mut b1, &mut b2), b1.zobrist_hash == b2.zobrist_hash);

                        b1.undo_action();
                        b2.undo_action();
                    }

                    let mov = rng.random_range(0..moves.len());
                    b1.do_action(moves[mov]);
                    b2.do_action(moves[mov]);
                }else {
                    b1.do_action(Action::Pass);
                    b2.do_action(Action::Pass);
                }
                assert!(compare_boards(&mut b1, &mut b2));
                assert!(b1.zobrist_hash == b2.zobrist_hash);
            }

            while b1.turn_num > 0 {
                b1.undo_action();
                b2.undo_action();
                assert!(compare_boards(&mut b1, &mut b2));
                assert!(b1.zobrist_hash == b2.zobrist_hash);
            }
        }

        let num_runs2 = 100;
        let depth2 = 200;

        bar = ProgressBar::new(num_runs2);
        for _run in 0..num_runs2 {
            bar.inc(1);
            for _turn in 0..depth2 {
                if b1.game_result() != GameResult::InProgress || b2.game_result() != GameResult::InProgress {
                    break;
                }

                let moves1 = b1.generate_moves();
                if moves1.len() > 0 {
                    let mov1 = rng.random_range(0..moves1.len());
                    b1.do_action(moves1[mov1]);
                }else {
                    b1.do_action(Action::Pass);
                }
                let moves2 = b2.generate_moves();
                if moves2.len() > 0 {
                    let mov2 = rng.random_range(0..moves2.len());
                    b2.do_action(moves2[mov2]);
                }else {
                    b2.do_action(Action::Pass);
                }
                assert_eq!(compare_boards(&mut b1, &mut b2), b1.zobrist_hash == b2.zobrist_hash);
            }

            while b1.turn_num > 0 {
                b1.undo_action();
            }
            while b2.turn_num > 0 {
                b2.undo_action();
            }
        }
    }
}

#[cfg(test)]
mod action_repr_test {
    use super::*;
    #[test]
    fn action_is_four_bytes_and_roundtrips() {
        assert_eq!(std::mem::size_of::<Action>(), 4, "Action must be 4 bytes");
        assert_eq!(std::mem::size_of::<ActionList>(), 4 * 256 + 8);
        // Default must remain Pass.
        assert_eq!(Action::default(), Action::Pass);
        assert!(Action::Pass.is_pass());
        // Exhaustive round-trip over the whole encodable space.
        for t in 0..GRID_SIZE as Tile {
            for p in 0..8u8 {
                let pct = PieceType::from_index(p);
                let a = Action::Place(t, pct);
                assert!(a.is_place());
                assert_eq!(a.kind(), ActionKind::Place(t, pct));
            }
            for &f in &[0u16, 1, 511, 1022, 1023] {
                let a = Action::Move(f, t);
                assert!(a.is_move());
                assert_eq!(a.kind(), ActionKind::Move(f, t));
                assert_eq!(a.move_from(), f);
                assert_eq!(a.move_to(), t);
            }
        }
    }

    #[test]
    fn ordering_matches_the_old_enum() {
        // Old derived Ord: all Place < all Move < Pass; within a variant,
        // lexicographic by field. Rebuild that order independently and
        // compare against the packed type's Ord.
        let mut packed = Vec::new();
        let mut reference = Vec::new();
        for t in (0..GRID_SIZE as Tile).step_by(37) {
            for p in 0..8u8 {
                packed.push(Action::Place(t, PieceType::from_index(p)));
                reference.push((0u8, t, p as u16));
            }
            for f in (0..GRID_SIZE as Tile).step_by(101) {
                packed.push(Action::Move(f, t));
                reference.push((1u8, f, t));
            }
        }
        packed.push(Action::Pass);
        reference.push((2u8, 0, 0));

        let mut by_packed: Vec<usize> = (0..packed.len()).collect();
        by_packed.sort_by_key(|&i| packed[i]);
        let mut by_reference: Vec<usize> = (0..reference.len()).collect();
        by_reference.sort_by_key(|&i| reference[i]);
        assert_eq!(by_packed, by_reference, "packed Ord diverges from the old enum ordering");
    }
}
