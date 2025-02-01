use std::collections::HashMap;

use crate::tile::{Tile, GRID_SIZE, TILE_ZERO};
use crate::piece::{Color, Piece};
use crate::piece_type::PieceType;
use crate::abstractions::*;

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub enum Action {
    Place(Tile, PieceType),
    Move(Tile, Tile),
    #[default]
    Pass,
}

#[derive(Clone)]
pub struct Board {
    // the piece on each tile of the board, for stacks: to topmost piece
    pub world: [Piece; GRID_SIZE],
    // stacked pieces, from bottom to top
    pub underworld: HashMap<Tile, Vec<Piece>>, // aka "sottobosco"
    // remaining pieces still to be placed
    pub placeable: [[u8; 8]; 2],
    // position of the queens
    pub queens: [Option<Tile>; 2],
    // list of occupied tiles for each player
    pub occupied_hexes: [Vec<Tile>; 2],

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
    fn new() -> Self {
        Board {
            world: [Piece::empty(); GRID_SIZE],
            underworld: HashMap::new(),
            placeable: [TOT_QTY, TOT_QTY],
            queens: [None, None],
            occupied_hexes: [Vec::new(), Vec::new()],
            turn_num: 0,
            turn_history: Vec::new(),
            tiles_placeable: [TileBitmask::new(true), TileBitmask::new(true)],
        }
    }

    pub fn color(&self) -> Color {
        Color::from_index((self.turn_num & 1) as u8)
    }

    pub fn queen_required(&self) -> bool {
        self.turn_num > 5 && self.placeable[self.color().index()][PieceType::Queen as usize] > 0 
    }

    fn add_piece(&mut self, tile: Tile, piece: Piece) {
        let curr = &mut self.world[tile as usize];
        if curr.is_some() {
            self.underworld.entry(tile).or_insert_with(Vec::new).push(*curr);
        }
        *curr = piece;
    }

    fn remove_piece(&mut self, tile: Tile) {
        let curr = &mut self.world[tile as usize];
        if let Some(vec) = self.underworld.get_mut(&tile) {
            let last = vec.pop().unwrap();
            *curr = last;
            if vec.is_empty() {
                self.underworld.remove(&tile);
            }
        } else {
            *curr = Piece::empty();
        }
    }

    pub fn do_move(&mut self, action: Action) {
        match action {
            Action::Place(tile, piece_type) => {
                let num = 1 + TOT_QTY[piece_type as usize] - self.placeable[self.color().index()][piece_type as usize];
                debug_assert!(0 < num && num < 4);
                let piece = Piece::make(self.color(), piece_type, num);
                self.add_piece(tile, piece);
                self.placeable[self.color().index()][piece_type as usize] -= 1;
            }
            Action::Move(from, to) => {
                let piece = self.world[from as usize];
                debug_assert!(piece.is_some());
                self.add_piece(to, piece);
                self.remove_piece(from);
            }
            Action::Pass => {}
        }
        self.turn_num += 1;
    }


}

#[test]
fn test_board() {
    let mut board = Board::new();
    let a = TILE_ZERO;
    let b = TILE_ZERO + 1;
    board.do_move(Action::Place(a, PieceType::Queen));
    board.do_move(Action::Place(b, PieceType::Queen));
    board.do_move(Action::Move(a, b));
    assert_eq!(board.world[a as usize], Piece::empty());
    assert_eq!(board.world[b as usize], Piece::make(Color::White, PieceType::Queen, 1));
    assert_eq!(*board.underworld[&b].first().unwrap(), Piece::make(Color::Black, PieceType::Queen, 1));
    assert_eq!(board.placeable[Color::White.index()][PieceType::Queen as usize], 0);
}
