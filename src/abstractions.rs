use crate::tile::{Tile, GRID_SIZE};
use crate::board::Action;
use std::collections::HashSet;


#[derive(Clone)]
pub struct TileBitmask{
    set : [bool; GRID_SIZE],
}

impl TileBitmask{
    
    pub fn new(value: bool) -> Self{
        TileBitmask{
            set: [value; GRID_SIZE],
        }
    }

    pub fn is_on(&self, tile : Tile) -> bool {
        return self.set[tile as usize];
    }

    pub fn set_bit(&mut self, tile: Tile, value: bool){
        self.set[tile as usize] = value;
    }
}

pub struct ActionContainer {
    pub moves : Vec<Action>,
    hashed_moves : HashSet<i32>,
}

const GRID_SIZE_32: i32 = GRID_SIZE as i32;

impl ActionContainer {
    
    pub fn new() -> Self {
        ActionContainer {
            moves: Vec::new(),
            hashed_moves: HashSet::new(),
        }
    }

    pub fn push(&mut self, action: Action) {

        let hash = match action {
            Action::Place(tile, piece ) => {
                GRID_SIZE_32 * (piece as i32) + (tile as i32)
            },
            Action::Move(start, end) => {
                GRID_SIZE_32 * GRID_SIZE_32 + GRID_SIZE_32 * (start as i32) + (end as i32)
            },
            _ => {-1}
        };

        if !self.hashed_moves.contains(&hash) {
            self.moves.push(action);
            self.hashed_moves.insert(hash);
        }
    }   

}


