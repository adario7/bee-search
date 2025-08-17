use crate::piece::Color;

use crate::tile::{Tile, GRID_SIZE};
use crate::board::Action;
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

    pub(crate) fn rem(&mut self, tile: Tile) {
        self.table[tile as usize & TILESET_MASK] &= !(1 << (tile as u32 >> TILESET_SHIFT));
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
pub struct CachedValue<T>(Option<(CacheHash, T)>);

impl<T> CachedValue<T>
where
    T: Clone,
{
    pub fn new() -> Self {
        Self(None)
    }

    pub fn get_if_valid(&self, current_hash: CacheHash) -> Option<&T> {
        if let Some((cached_hash, cached_result)) = &self.0 {
            if *cached_hash == current_hash {
                return Some(cached_result);
            }
        }
        None
    }

    pub fn update(&mut self, current_hash: CacheHash, value: T) {
        self.0 = Some((current_hash, value));
    }

    pub fn is_valid(&self, current_hash: CacheHash) -> bool {
        self.0.as_ref().map_or(false, |x| x.0 == current_hash)
    }

}

#[derive(Clone)]
pub struct OccupancyVec {
    pub occupants: Vec<Tile>,
    pub occupants_set: TileSet,
}

impl OccupancyVec {
    pub fn new() -> Self {
        OccupancyVec { 
            occupants: Vec::new(),
            occupants_set: TileSet::new(),
        }
    }

    pub fn contains(&self, tile: &Tile) -> bool {
        return self.occupants_set.get(*tile);
    }

    pub fn push(&mut self, tile: Tile) {
        self.occupants_set.set(tile);
        self.occupants.push(tile);
    }

    pub fn remove(&mut self, tile: Tile) {
        self.occupants_set.rem(tile);
        self.occupants.swap_remove(self.occupants.iter().position(|&x| x == tile).unwrap());
    }

    pub fn iter(&self) -> std::slice::Iter<'_, u16> {
        return self.occupants.iter();
    }

    pub fn first(&self) -> Option<&u16> {
        return self.occupants.first();
    }

    pub fn len(&self) -> usize {
        return self.occupants.len();
    }

}

