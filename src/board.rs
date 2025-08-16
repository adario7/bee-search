use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::default::Default;
use std::hash::Hasher;
#[cfg(feature = "gnn")]
use crate::graph_nn::GnnEvaluator;

use crate::tile::{adjacent, Direction, Tile, GRID_SIZE};
use crate::piece::{Color, Piece};
use crate::piece_type::{Pct, PieceType};
use crate::abstractions::*;
use std::sync::OnceLock;

static ZOBRIST_TABLE: OnceLock<[u64; GRID_SIZE * 2]> = OnceLock::new();
static PLAYER_HASH: u64 = 0xc851ba955a512175;

#[cfg(feature = "gnn")]
static GNN_PATH: &str = "models/hive_gnn.onnx";

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub enum Action {
    Place(Tile, PieceType),
    Move(Tile, Tile),
    #[default]
    Pass,
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
    // stacked pieces, from bottom to top
    pub underworld: HashMap<Tile, Vec<Piece>>, // aka "sottobosco"
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

    pub zobrist_table: &'static [u64; GRID_SIZE * 2],
    pub zobrist_hash: u64,
    pub zobrist_history: Vec<u64>,

    pub height: [u8; GRID_SIZE],
    #[cfg(feature = "gnn")]
    pub gnn: std::sync::Arc<GnnEvaluator>,

    pub immovable: CachedValue<TileSet>,
}

pub const TOT_QTY: [u8; 8] = [1, 3, 2, 3, 2, 1, 1, 1];

impl Board {
    pub fn new() -> Self {

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
            gametype: "Base+MLP".to_string(),
            world: [Piece::empty(); GRID_SIZE],
            underworld: HashMap::new(),
            placeable: [TOT_QTY, TOT_QTY],
            queens: [None, None],
            occupied_tiles: [OccupancyVec::new(), OccupancyVec::new()],
            turn_num: 0,
            turn_history: Vec::new(),
            tiles_placeable: [TileSet::new(), TileSet::new()],
            zobrist_table,
            zobrist_hash: 1,
            zobrist_history: Vec::new(),
            height: [0; GRID_SIZE],
            #[cfg(feature = "gnn")]
            gnn: std::sync::Arc::new(GnnEvaluator::new(GNN_PATH).expect("Failed to initialize GnnEvaluator")),
            immovable: CachedValue::new(),
        }
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
            underworld: HashMap::new(),
            placeable: [[1, 3, 2, 3, 2, m, l, p], [1, 3, 2, 3, 2, m, l, p]], //TODO: doesn't consider values of TOT_QTY
            queens: [None, None],
            occupied_tiles: [OccupancyVec::new(), OccupancyVec::new()],
            turn_num: 0,
            turn_history: Vec::new(),
            tiles_placeable: [TileSet::new(), TileSet::new()],
            zobrist_table,
            zobrist_hash: 0,
            zobrist_history: Vec::new(),
            height: [0; GRID_SIZE],
            #[cfg(feature = "gnn")]
            gnn: std::sync::Arc::new(GnnEvaluator::new(GNN_PATH).expect("Failed to initialize GnnEvaluator")),
            immovable: CachedValue::new(),
        }
    }

    pub fn color(&self) -> Color {
        Color::from_index((self.turn_num & 1) as u8)
    }

    pub fn tile(&self, tile: Tile) -> Piece {
        self.world[tile as usize]
    }

    pub fn occupied(&self, tile: Tile) -> bool {
        self.height[tile as usize] > 0
    }

    pub fn queen_required(&self) -> bool {
        self.turn_num > 5 && self.placeable[self.color().index()][PieceType::Queen as usize] > 0 
    }

    pub fn height(&self, tile: Tile) -> i32 {
        self.height[tile as usize] as i32
    } 

    pub fn is_stacked(&self, tile: Tile) -> bool {
        self.height[tile as usize] > 1
    }

    fn zobrist(&self, t: Tile, p: Piece, h: u32) -> u64 {
        let hash = self.zobrist_table[((t as usize) << 1) | (p.color() as usize)];
        hash.rotate_left((h<<3) | (p.ptype() as u32))
    }

    fn add_occupancy(occupied_hexes: &mut [OccupancyVec; 2], p: Piece, t: Tile) {
        let vec = &mut occupied_hexes[p.color().index()];
        if !vec.contains(&t) {
            vec.push(t);
        }
    }

    fn remove_occupancy(occupied_hexes: &mut [OccupancyVec; 2], p: Piece, t: Tile) {
        let vec = &mut occupied_hexes[p.color().index()];
        vec.remove(t);
    }

    fn add_piece(&mut self, tile: Tile, piece: Piece) {
        let curr = &mut self.world[tile as usize];
        if curr.is_some() {
            self.underworld.entry(tile).or_insert_with(Vec::new).push(*curr);
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
        if let Some(vec) = self.underworld.get_mut(&tile) {
            let last = vec.pop().unwrap();
            if vec.is_empty() {
                self.underworld.remove(&tile);
            }
            *curr = last;
            Self::add_occupancy(&mut self.occupied_tiles, *curr, tile); // the uncovered piece may have a differe color
        } else {
            *curr = Piece::empty();
        }
    }

    fn do_move(&mut self, from: Tile, to: Tile) {
        let piece = self.world[from as usize];
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
        match action {
            Action::Place(tile, piece_type) => {
                let piece = self.make_next_piece(piece_type);
                self.add_piece(tile, piece);
                self.placeable[self.color().index()][piece_type as usize] -= 1;
            }
            Action::Move(from, to) => self.do_move(from, to),
            Action::Pass => {}
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
        match action {
            Action::Place(tile, piece_type) => {
                debug_assert!(self.world[tile as usize].ptype() == piece_type);
                self.remove_piece(tile);
                self.placeable[self.color().index()][piece_type as usize] += 1;
            }
            Action::Move(from, to) => self.do_move(to, from),
            Action::Pass => {}
        }
    }

    fn queen_surround(&self, color: Color) -> usize {
        match self.queens[color as usize] {
            None => 0,
            Some(tile) => adjacent(tile).into_iter().filter(|&t| self.tile(t).is_some()).count()
        }
    }

    pub fn game_result(&self) -> GameResult {
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
        if let Action::Move(_, to) = action {
            let queen = self.queens[self.color().other() as usize];
            if let Some(target) = queen {
                if *to == target {
                    return true;
                }
                for d in Direction::all() {
                    if *to + *d == target {
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

    pub fn get_cached_hash(&self) -> CacheHash {
        CacheHash {zobrist_hash: self.zobrist_hash, board_color: self.color()}
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
        assert_eq!(*board.underworld[&b].first().unwrap(), Piece::make(Color::Black, PieceType::Queen, 1));
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

                if let Some(vec1) = b1.underworld.get(&tile) {
                    if let Some(vec2) = b2.underworld.get(&tile){
                        for j in 0..vec1.len() {
                            if vec1[j] != vec2[j] {
                                return false;
                            }
                        }
                    }
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
