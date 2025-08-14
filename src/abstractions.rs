use crate::piece::Color;

use crate::tile::{Tile, GRID_SIZE};
use crate::board::{Action, Board};
use std::collections::HashSet;

const TILESET_NUM_WORDS: usize = GRID_SIZE / 32;
const TILESET_SHIFT: u32 = GRID_SIZE.trailing_zeros() - 5;
const TILESET_MASK: usize = TILESET_NUM_WORDS - 1;

#[derive(Clone)]
pub struct TileSet {
    table: [u32; TILESET_NUM_WORDS],
}

impl TileSet {
    pub(crate) fn new() -> TileSet {
        TileSet { table: [0; TILESET_NUM_WORDS] }
    }

    pub(crate) fn set(&mut self, tile: Tile) {
        self.table[tile as usize & TILESET_MASK] |= 1 << (tile as u32 >> TILESET_SHIFT);
    }

    pub(crate) fn get(&self, tile: Tile) -> bool {
        (self.table[tile as usize & TILESET_MASK] >> (tile as u32 >> TILESET_SHIFT)) & 1 != 0
    }

    pub fn copy(&self) -> TileSet {
        TileSet {table: self.table}
    }
}

pub struct ActionContainer {
    pub moves: Vec<Action>,
    hashed_moves: HashSet<i32>,
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

#[derive(Debug, Clone)]
pub struct CacheHash {
    pub zobrist_hash: u64,
    pub board_color: Color,
}

impl PartialEq<CacheHash> for CacheHash {
    fn eq(&self, other: &CacheHash) -> bool {
        self.zobrist_hash == other.zobrist_hash && self.board_color == other.board_color
    }
}

#[derive(Debug, Clone)]
pub struct CachedValue<T> {
    cached_result: Option<T>,
    cached_hash: Option<CacheHash>,
}

impl<T> CachedValue<T>
where
    T: Clone,
{
    pub fn new() -> Self {
        Self {
            cached_result: None,
            cached_hash: None,
        }
    }

    pub fn get_or_compute<F>(&mut self, current_hash: CacheHash, board: &Board, compute_fn: F) -> T
    where
        F: FnOnce(&Board) -> T,
    {
        if let (Some(ref cached_result), Some(ref cached_hash)) = (&self.cached_result, &self.cached_hash) {
            if *cached_hash == current_hash {
                return cached_result.clone();
            }
        }

        let new_result = compute_fn(board);
        self.cached_result = Some(new_result.clone());
        self.cached_hash = Some(current_hash);
        new_result
    }
}


