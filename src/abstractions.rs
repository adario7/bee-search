
use crate::tile::{Tile, GRID_SIZE};


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

