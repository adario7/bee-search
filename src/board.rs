use std::collections::{HashMap, HashSet};

use crate::tile::{adjacent, Tile, GRID_SIZE};
use crate::piece::{Color, Piece};
use crate::piece_type::{Pct, PieceType};
use crate::abstractions::*;

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
    WhiteWins,
    BlackWins,
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
    pub occupied_tiles: [Vec<Tile>; 2],
    // number of plies played
    pub turn_num: usize,
    // turn history
    pub turn_history: Vec<Action>,
    // placeable tiles for both players
    // a tile is placeable for a player if he can put a piece that is not present on the board on it
    pub tiles_placeable: [TileBitmask; 2],
}

const TOT_QTY: [u8; 8] = [1, 3, 2, 3, 2, 1, 1, 1];

impl Board {
    pub fn new() -> Self {
        Board {
            gametype: "Base+MLP".to_string(),
            world: [Piece::empty(); GRID_SIZE],
            underworld: HashMap::new(),
            placeable: [TOT_QTY, TOT_QTY],
            queens: [None, None],
            occupied_tiles: [Vec::new(), Vec::new()],
            turn_num: 0,
            turn_history: Vec::new(),
            tiles_placeable: [TileBitmask::new(false), TileBitmask::new(false)],
        }
    }

    pub fn new_mlp(M: u8, L: u8, P: u8) -> Self {
        Board {
            gametype: format!(
                "Base{}",
                if M + L + P > 0 {
                    format!("+{}{}{}",
                        if M > 0 {"M"} else {""},
                        if L > 0 {"L"} else {""},
                        if P > 0 {"P"} else {""},
                    )
                }else {"".to_string()}
            ),
            world: [Piece::empty(); GRID_SIZE],
            underworld: HashMap::new(),
            placeable: [[1, 3, 2, 3, 2, M, L, P], [1, 3, 2, 3, 2, M, L, P]], //TODO: doesn't consider values of TOT_QTY
            queens: [None, None],
            occupied_tiles: [Vec::new(), Vec::new()],
            turn_num: 0,
            turn_history: Vec::new(),
            tiles_placeable: [TileBitmask::new(false), TileBitmask::new(false)],
        }
    }

    pub fn color(&self) -> Color {
        Color::from_index((self.turn_num & 1) as u8)
    }

    pub fn tile(&self, tile: Tile) -> Piece {
        self.world[tile as usize]
    }

    pub fn queen_required(&self) -> bool {
        self.turn_num > 5 && self.placeable[self.color().index()][PieceType::Queen as usize] > 0 
    }

    pub fn height(&self, tile: Tile) -> i32 {
        if let Some(vec) = self.underworld.get(&tile) {
            return (vec.len() + 1) as i32;
        }
        return (self.world[tile as usize] != Piece::empty()) as i32;
    } 

    fn add_occupancy(occupied_hexes: &mut [Vec<Tile>; 2], p: Piece, t: Tile) {
        let vec = &mut occupied_hexes[p.color().index()];
        if !vec.contains(&t) {
            vec.push(t);
        }
    }

    fn remove_occupancy(occupied_hexes: &mut [Vec<Tile>; 2], p: Piece, t: Tile) {
        let vec = &mut occupied_hexes[p.color().index()];
        vec.swap_remove(vec.iter().position(|&x| x == t).unwrap());
    }

    fn add_piece(&mut self, tile: Tile, piece: Piece) {
        let curr = &mut self.world[tile as usize];
        if curr.is_some() {
            self.underworld.entry(tile).or_insert_with(Vec::new).push(*curr);
        }
        *curr = piece;
        Self::add_occupancy(&mut self.occupied_tiles, *curr, tile);
        if piece.ptype() == Pct::Queen {
            self.queens[piece.color().index()] = Some(tile);
        }
    }

    fn remove_piece(&mut self, tile: Tile) {
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
        self.add_piece(to, piece);
        self.remove_piece(from);
    }

    pub fn make_next_piece(&self, pct: PieceType) -> Piece {
        let num = 1 + TOT_QTY[pct as usize] - self.placeable[self.color().index()][pct as usize];
        debug_assert!(0 < num && num < 4);
        Piece::make(self.color(), pct, num)
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
    }

    pub fn undo_action(&mut self) {
        let action = self.turn_history.pop().unwrap();
        self.turn_num -= 1;
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
            (6, _) => GameResult::BlackWins,
            (_, 6) => GameResult::WhiteWins,
            (_, _) => GameResult::InProgress,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tile::{Direction, TILE_ZERO};

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
                assert_eq!(board.game_result(), GameResult::WhiteWins);
            } else {
                assert_eq!(board.game_result(), GameResult::InProgress);
            }
            board.do_action(Action::Place(a + d, pct));
        }
        assert_eq!(board.game_result(), GameResult::Draw);
        board.do_action(Action::Move(b + Direction::E, a + Direction::W + Direction::W + Direction::W));
        assert_eq!(board.game_result(), GameResult::BlackWins);
    }
}
