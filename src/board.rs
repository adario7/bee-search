use std::collections::HashMap;

use crate::tile::{Tile, GRID_SIZE, TILE_ZERO};
use crate::piece::{Color, Piece};
use crate::piece_type::PieceType;

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
    world: [Piece; GRID_SIZE],
    // stacked pieces, from bottom to top
    underworld: HashMap<Tile, Vec<Piece>>, // aka "sottobosco"
    // remaining pieces still to be placed
    placable: [[u8; 8]; 2],
    // position of the queens
    queens: [Option<Tile>; 2],
    // list of occupied tiles for each player
    occupied_hexes: [Vec<Tile>; 2],

    // number of plies played
    turn_num: usize,
    // turn history
    turn_history: Vec<Action>,
}

const TOT_QTY: [u8; 8] = [1, 3, 2, 3, 2, 1, 1, 1];

impl Board {
    fn new() -> Self {
        Board {
            world: [Piece::empty(); GRID_SIZE],
            underworld: HashMap::new(),
            placable: [TOT_QTY, TOT_QTY],
            queens: [None, None],
            occupied_hexes: [Vec::new(), Vec::new()],
            turn_num: 0,
            turn_history: Vec::new(),
        }
    }

    fn color(&self) -> Color {
        Color::from_index((self.turn_num & 1) as u8)
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

    fn do_move(&mut self, from: Tile, to: Tile) {
        let piece = self.world[from as usize];
        debug_assert!(piece.is_some());
        self.add_piece(to, piece);
        self.remove_piece(from);
    }

    /// assumes the action is legal
    pub fn do_action(&mut self, action: Action) {
        match action {
            Action::Place(tile, piece_type) => {
                let num = 1 + TOT_QTY[piece_type as usize] - self.placable[self.color().index()][piece_type as usize];
                debug_assert!(0 < num && num < 4);
                let piece = Piece::make(self.color(), piece_type, num);
                self.add_piece(tile, piece);
                self.placable[self.color().index()][piece_type as usize] -= 1;
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
                self.placable[self.color().index()][piece_type as usize] += 1;
            }
            Action::Move(from, to) => self.do_move(to, from),
            Action::Pass => {}
        }
    }
}

#[test]
fn test_board_do_undo() {
    let mut board = Board::new();
    let a = TILE_ZERO;
    let b = TILE_ZERO + 1;
    board.do_action(Action::Place(a, PieceType::Queen));
    assert_eq!(board.placable[Color::White.index()][PieceType::Queen as usize], 0);
    board.do_action(Action::Place(b, PieceType::Queen));
    assert_eq!(board.placable[Color::Black.index()][PieceType::Queen as usize], 0);
    board.do_action(Action::Move(a, b));
    assert_eq!(board.world[a as usize], Piece::empty());
    assert_eq!(board.world[b as usize], Piece::make(Color::White, PieceType::Queen, 1));
    assert_eq!(*board.underworld[&b].first().unwrap(), Piece::make(Color::Black, PieceType::Queen, 1));
    board.undo_action();
    assert_eq!(board.world[a as usize], Piece::make(Color::White, PieceType::Queen, 1));
    assert_eq!(board.world[b as usize], Piece::make(Color::Black, PieceType::Queen, 1));
    board.undo_action();
    assert!(board.world[b as usize].is_none());
    assert_eq!(board.placable[Color::Black.index()][PieceType::Queen as usize], 1);
}
