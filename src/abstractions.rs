use crate::tile::{adjacent, Tile, GRID_SIZE};

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

    // NOTE: a combined `test_and_set` (to replace the `if s.get(t) { .. }
    // s.set(t)` pairing with one load + one store) was tried and MEASURED
    // WORSE: +2.2% cycles, 1/8 wins. It always stores, whereas the get/set
    // pair skips the store when the bit is already set - which is the common
    // case for overlapping neighbourhoods. Don't re-add it.

    /// OR in all six neighbours of `tile` at once.
    ///
    /// In this transposed layout (word = tile & 31, bit = tile >> 5) the six
    /// neighbours - offsets -33,-32,+1,+33,+32,-1 - land in only THREE
    /// distinct words, two bits each, because the offsets differ by 0 or +-1
    /// modulo 32. So this does three read-modify-writes instead of six, and
    /// to three *different* words, so they pipeline instead of serialising.
    #[inline]
    pub(crate) fn set_adjacent(&mut self, tile: Tile) {
        let n = adjacent(tile);
        // adjacent() yields [NW, NE, E, SE, SW, W] = tile + [-33,-32,+1,+33,+32,-1],
        // pairing up as (0,5), (1,4), (2,3) by word. Asserted so that
        // reordering Direction::all() can't silently break this.
        debug_assert_eq!(n[0] as usize & TILESET_MASK, n[5] as usize & TILESET_MASK);
        debug_assert_eq!(n[1] as usize & TILESET_MASK, n[4] as usize & TILESET_MASK);
        debug_assert_eq!(n[2] as usize & TILESET_MASK, n[3] as usize & TILESET_MASK);
        let bit = |t: Tile| 1u32 << (t as u32 >> TILESET_SHIFT);
        self.table[n[0] as usize & TILESET_MASK] |= bit(n[0]) | bit(n[5]);
        self.table[n[1] as usize & TILESET_MASK] |= bit(n[1]) | bit(n[4]);
        self.table[n[2] as usize & TILESET_MASK] |= bit(n[2]) | bit(n[3]);
    }

    pub fn copy(&self) -> TileSet {
        TileSet {table: self.table}
    }
}


// Keyed only by zobrist_hash: the cached values stored here (e.g. cut
// vertexes) are structural properties of the piece arrangement and do not
// depend on which color is to move, so board_color must never be part of
// the key (it used to be, which forced a cache miss - and a full
// recomputation - every time the "other side"'s mobility was probed for
// evaluation features, even though the board had not changed).
#[derive(Debug, Clone)]
pub struct CachedValue<T>(Option<(u64, T)>);

impl<T> CachedValue<T>
where
    T: Clone,
{
    pub fn new() -> Self {
        Self(None)
    }

    pub fn get_if_valid(&self, current_hash: u64) -> Option<&T> {
        if let Some((cached_hash, cached_result)) = &self.0 {
            if *cached_hash == current_hash {
                return Some(cached_result);
            }
        }
        None
    }

    pub fn update(&mut self, current_hash: u64, value: T) {
        self.0 = Some((current_hash, value));
    }

    pub fn is_valid(&self, current_hash: u64) -> bool {
        self.0.as_ref().map_or(false, |x| x.0 == current_hash)
    }

}

// No accompanying bitset for O(1) `.contains()` (unlike an earlier version
// of this code): every tile is removed from its previous color's set
// before being added to its new color's set (see Board::add_piece /
// remove_piece), so a tile can never be pushed here while already present
// - the duplicate-guard this used to carry (a `.contains()` check backed by
// a 128-byte-per-color TileSet, set/cleared on every single push/remove)
// was provably dead weight. Matches nokamute's plain `Vec<Hex>` here.
#[derive(Clone)]
pub struct OccupancyVec {
    pub occupants: Vec<Tile>,
}

impl OccupancyVec {
    pub fn new() -> Self {
        OccupancyVec {
            occupants: Vec::new(),
        }
    }

    pub fn push(&mut self, tile: Tile) {
        debug_assert!(!self.occupants.contains(&tile));
        self.occupants.push(tile);
    }

    pub fn remove(&mut self, tile: Tile) {
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

