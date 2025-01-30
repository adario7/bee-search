// Board representation: wrapping grid of tile locations.
// Rows wrap around, and each row wraps to the next row.
// It's like a spiral around a torus.
//
// A 4x4 example:
//
//         11  12  13  14  15
//           \ / \ / \ / \ /
//       15 - 0 - 1 - 2 - 3 - 4
//         \ / \ / \ / \ / \
//      3 - 4 - 5 - 6 - 7 - 8
//       \ / \ / \ / \ / \
//    7 - 8 - 9 -10 -11 -12
//     \ / \ / \ / \ / \
// 11 -12 -13 -14 -15 - 0
//     / \ / \ / \ / \
//    0   1   2   3   4

use std::ops::Add;
use bitvec::prelude::*;

// tile index
pub type Tile = u16;

// we only have 28 pieces, but use 32 to have a POT allowing GRID_MASK to be used instead of modulos
pub const ROW_SIZE: Tile = 32;

pub const GRID_SIZE: usize = ROW_SIZE as usize * ROW_SIZE as usize;
pub const GRID_MASK: Tile = (GRID_SIZE as Tile).wrapping_sub(1);
pub const TILE_ZERO: Tile = ROW_SIZE / 2 * (ROW_SIZE + 1);

#[repr(u16)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum Direction {
    NW = (GRID_MASK & (ROW_SIZE + 1).wrapping_neg()) as u16,
    NE = (GRID_MASK & ROW_SIZE.wrapping_neg()) as u16,
    W = GRID_MASK as u16 & 1u16.wrapping_neg(),
    E = 1,
    SW = ROW_SIZE as u16,
    SE = ROW_SIZE as u16 + 1,
}

impl Add<Direction> for Tile {
    type Output = Tile;

    fn add(self, rhs: Direction) -> Tile {
        self.wrapping_add(rhs as Tile) & GRID_MASK
    }
}

impl Direction {
    pub fn all() -> &'static [Direction; 6] {
        &[Direction::NW, Direction::NE, Direction::E, Direction::SE, Direction::SW, Direction::W]
    }
}

pub fn adjacent(hex: Tile) -> [Tile; 6] {
    [
        hex + Direction::NW,
        hex + Direction::NE,
        hex + Direction::E,
        hex + Direction::SE,
        hex + Direction::SW,
        hex + Direction::W,
    ]
}

#[test]
fn test_direction() {
    // Reversibility from any position.
    for t in 0..=GRID_MASK {
        assert_eq!(t, t + Direction::NE + Direction::SW);
        assert_eq!(t, t + Direction::E + Direction::W);
        assert_eq!(t, t + Direction::SE + Direction::NW);
    }

    // Exercise the known wrapping properties.
    // Two axes traverse the entire space before wrapping.
    let mut t = TILE_ZERO;
    for _ in 1..GRID_SIZE {
        t = t + Direction::E;
        assert_ne!(t, TILE_ZERO);
    }
    t = t + Direction::E;
    assert_eq!(t, TILE_ZERO);

    t = TILE_ZERO;
    for _ in 1..GRID_SIZE {
        t = t + Direction::NW;
        assert_ne!(t, TILE_ZERO);
    }
    t = t + Direction::NW;
    assert_eq!(t, TILE_ZERO);

    // One axis only traverses one row.
    t = TILE_ZERO;
    for _ in 1..ROW_SIZE {
        t = t + Direction::NE;
        assert_ne!(t, TILE_ZERO);
    }
    t = t + Direction::NE;
    assert_eq!(t, TILE_ZERO);
}

// 2D borad location
pub type Loc = (i8, i8);

pub fn loc_to_tile(loc: Loc) -> Tile {
    TILE_ZERO.wrapping_add(ROW_SIZE.wrapping_mul(loc.1 as Tile)).wrapping_add(loc.0 as Tile)
}

pub fn tile_to_loc(tile: Tile) -> Loc {
    let mut x = (tile.wrapping_sub(TILE_ZERO - ROW_SIZE / 2) / ROW_SIZE) as i8;
    if x > (ROW_SIZE / 2 - 1) as i8 {
        x -= ROW_SIZE as i8;
    }
    let mut y = (tile.wrapping_sub(TILE_ZERO) % ROW_SIZE) as i8;
    if y > (ROW_SIZE / 2 - 1) as i8 {
        y -= ROW_SIZE as i8;
    }
    (y, x)
}

#[test]
fn test_hex_loc() {
    let r = (ROW_SIZE / 2) as i8;
    for x in -r..r {
        for y in -r..r {
            let l = (x, y);
            let t = loc_to_tile(l);
            assert_eq!(l, tile_to_loc(t), "{} != {},{}", t, l.0, l.1);
        }
    }
    for i in 0..GRID_SIZE as Tile {
        let t = i as Tile;
        let l = tile_to_loc(t);
        assert_eq!(t, loc_to_tile(l), "{} != {},{}", i, l.0, l.1);
    }
}


pub struct TileSet(BitArr!(for GRID_SIZE, in u32));

impl TileSet {
    pub fn new() -> Self {
        Self(BitArray::ZERO)
    }

    pub fn set(&mut self, tile: Tile) {
        self.0.set(tile as usize, true);
    }

    pub fn unset(&mut self, tile: Tile) {
        self.0.set(tile as usize, false);
    }

    pub fn get(&self, tile: Tile) -> bool {
        self.0[tile as usize]
    }
}
