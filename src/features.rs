use crate::{abstractions::TileSet, board::{Action, Board}, piece::Color, piece_type::PCT_COUNT, tile::adjacent};

impl Board {
    pub const FN: usize = 4 + PCT_COUNT*4;
    pub const FN2: usize = Self::FN * 2;
    pub const ADJ_SIZE: usize = 128;
    pub const TOTAL_FN: usize = Self::FN2 + Self::ADJ_SIZE;

    fn features_for(&self, color: Color, move_n: [u16; PCT_COUNT], immovable: &TileSet, out: &mut [i16]) {
        let mut liberties = [[0; 2]; 2];
        let qtile = self.queens[color.index()];
        if let Some(tile) = qtile {
            for &t in adjacent(tile).iter() {
                let pc = self.tile(t);
                if pc.is_none() { continue; }
                let ally = pc.color() == color;
                let fixed = immovable.get(t);
                liberties[ally as usize][fixed as usize] += 1;
            }
        }
        let mut movable = [0; PCT_COUNT];
        let mut fixed = [0; PCT_COUNT];
        let mut buried = [0; PCT_COUNT];
        for &tile in self.occupied_tiles[color.index()].iter() {
            let pct = self.tile(tile).ptype();
            let walks = self.height(tile) > 1;
            if immovable.get(tile) && !walks {
                fixed[pct.index()] += 1;
            } else {
                movable[pct.index()] += 1;
            }
        }
        for &tile in self.occupied_tiles[color.other().index()].iter() {
            if self.height(tile) > 1 {
                for pc in self.underworld.get(&tile).unwrap_or(&vec![]).iter() {
                    if pc.color() == color {
                        buried[pc.ptype().index()] += 1;
                    }
                }
            }
        }
        out[0] = liberties[0][0];
        out[1] = liberties[0][1];
        out[2] = liberties[1][0];
        out[3] = liberties[1][1];
        for i in 0..PCT_COUNT {
            out[4 + i*4 + 0] = movable[i] as i16;
            out[4 + i*4 + 1] = fixed[i] as i16;
            out[4 + i*4 + 2] = buried[i] as i16;
            out[4 + i*4 + 3] = move_n[i] as i16;
        }
    }

    const fn adj_idx(mut c1: usize, mut c2: usize) -> Option<usize> {
        if c1 > c2 {
            let tmp = c1; c1 = c2; c2 = tmp;
        }
        if c1 == c2 && (c1 % 8 == 0 || c1 % 8 >= 5) {
            return None;
        }
        let mut idx = 0;
        let mut i = 0;
        while i < 16 {
            let mut j = i;
            while j < 16 {
                if !(i == j && (i % 8 == 0 || i % 8 >= 5)) {
                    if i == c1 && j == c2 { return Some(idx); }
                    idx += 1;
                }
                j += 1;
            }
            i += 1;
        }
        None
    }

    fn other_n_by_pct(&mut self) -> [u16; PCT_COUNT] {
        self.turn_num += 1;
        let n_moves = self.generate_movements_by_pct();
        self.turn_num -= 1;
        n_moves
    }

    fn features_for_both(&mut self, my_n: [u16; PCT_COUNT]) -> [i16; Self::TOTAL_FN] {
        let immovable = self.find_cut_vertexes();
        let other_n = self.other_n_by_pct();
        let mut out = [0; Self::TOTAL_FN];
        self.features_for(self.color(), my_n, &immovable, &mut out[0..Self::FN]);
        self.features_for(self.color().other(), other_n, &immovable, &mut out[Self::FN..Self::FN2]);
        
        let mut seen = [false; crate::tile::GRID_SIZE];
        for tile in self.all_occupied_tiles() {
            let pc1 = self.tile(tile);
            let cat1 = pc1.color().index() * PCT_COUNT + pc1.ptype().index();
            seen[tile as usize] = true;
            for &adj in adjacent(tile).iter() {
                if self.occupied(adj) && !seen[adj as usize] {
                    let pc2 = self.tile(adj);
                    let cat2 = pc2.color().index() * PCT_COUNT + pc2.ptype().index();
                    if let Some(idx) = Self::adj_idx(cat1, cat2) {
                        out[Self::FN2 + idx] += 1;
                    }
                }
            }
        }

        out
    }

    pub fn features_slow(&mut self) -> [i16; Self::TOTAL_FN] {
        let my_n = self.generate_movements_by_pct();
        self.features_for_both(my_n)
    }

    pub fn features_fast(&mut self, moves: &Vec<Action>) -> [i16; Self::TOTAL_FN] {
        let mut my_n = [0; PCT_COUNT];
        for action in moves {
            if let Action::Move(from, _) = action {
                my_n[self.tile(*from).ptype().index()] += 1;
            }
        }
        self.features_for_both(my_n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adj_idx() {
        let mut used = [false; Board::ADJ_SIZE];
        let mut count = 0;
        for i in 0..16 {
            for j in i..16 {
                if let Some(idx) = Board::adj_idx(i, j) {
                    assert!(idx < Board::ADJ_SIZE, "Index {} out of bounds", idx);
                    assert!(!used[idx], "Index {} used more than once", idx);
                    used[idx] = true;
                    count += 1;
                    // Symmetry check
                    assert_eq!(Board::adj_idx(j, i), Some(idx));
                } else {
                    // Diagonal entries for pieces with count 1
                    assert_eq!(i, j);
                    assert!(i % 8 == 0 || i % 8 >= 5, "Diagonal for type {} should not be excluded", i);
                }
            }
        }
        assert_eq!(count, Board::ADJ_SIZE, "Not all indices were used");
        for (i, &u) in used.iter().enumerate() {
            assert!(u, "Index {} was never used", i);
        }
    }
}
