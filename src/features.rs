use crate::{
    abstractions::TileSet,
    board::{Action, ActionList, Board},
    piece::Color,
    piece_type::PCT_COUNT,
    tile::{adjacent, tile_to_loc, Tile, GRID_SIZE},
};

impl Board {
    pub const FN: usize = 4 + PCT_COUNT * 6; // 4 + 48 = 52
    pub const FN2: usize = Self::FN * 2; // 104
    pub const ADJ_SIZE: usize = 128;
    pub const OLD_TOTAL_FN: usize = Self::FN2 + Self::ADJ_SIZE; // 104 + 128 = 232

    pub const MAX_RADIUS: usize = 4;
    pub const HEXES_PER_QUEEN: usize = 60; // Rings 1..4 = 6 + 12 + 18 + 24 = 60
    pub const TOTAL_PCTS: usize = PCT_COUNT * 2; // 8 White + 8 Black = 16
    pub const PCT_SQUARE_PER_QUEEN: usize = Self::HEXES_PER_QUEEN * Self::TOTAL_PCTS; // 60 * 16 = 960
    pub const PCT_SQUARE_TOTAL: usize = 2 * Self::PCT_SQUARE_PER_QUEEN; // 2 * 960 = 1920

    pub const TOTAL_FN: usize = Self::OLD_TOTAL_FN + Self::PCT_SQUARE_TOTAL; // 232 + 1920 = 2152

    /// Direct O(1) lookup table from cube coordinates (q, r) with q in [-4..4], r in [-4..4]
    /// to Red Blob Games spiral ring index (0..59). Center (0,0,0) and out-of-bounds are -1.
    pub const CUBE_TO_SPIRAL_LUT: [[i8; 9]; 9] = [
        [-1, -1, -1, -1, 56, 57, 58, 59, 36],
        [-1, -1, -1, 55, 33, 34, 35, 18, 37],
        [-1, -1, 54, 32, 16, 17,  6, 19, 38],
        [-1, 53, 31, 15,  5,  0,  7, 20, 39],
        [52, 30, 14,  4, -1,  1,  8, 21, 40],
        [51, 29, 13,  3,  2,  9, 22, 41, -1],
        [50, 28, 12, 11, 10, 23, 42, -1, -1],
        [49, 27, 26, 25, 24, 43, -1, -1, -1],
        [48, 47, 46, 45, 44, -1, -1, -1, -1],
    ];

    #[inline(always)]
    pub fn cube_to_spiral(q: i16, r: i16, s: i16) -> Option<usize> {
        if q < -4 || q > 4 || r < -4 || r > 4 || s < -4 || s > 4 {
            return None;
        }
        let idx = Self::CUBE_TO_SPIRAL_LUT[(q + 4) as usize][(r + 4) as usize];
        if idx >= 0 {
            Some(idx as usize)
        } else {
            None
        }
    }

    /// Computes hex cube coordinate delta (dq, dr, ds) from `from_tile` to `to_tile`.
    /// In hex cube coordinates, dq + dr + ds = 0.
    pub fn cube_delta(from_tile: Tile, to_tile: Tile) -> (i16, i16, i16) {
        let (y1, x1) = tile_to_loc(from_tile);
        let (y2, x2) = tile_to_loc(to_tile);

        let mut dx = (x2 - x1) as i16;
        if dx > 16 {
            dx -= 32;
        } else if dx < -16 {
            dx += 32;
        }

        let mut dy = (y2 - y1) as i16;
        if dy > 16 {
            dy -= 32;
        } else if dy < -16 {
            dy += 32;
        }

        let dq = dy - dx;
        let dr = dx;
        let ds = -dy;
        (dq, dr, ds)
    }

    pub const fn adj_idx(mut c1: usize, mut c2: usize) -> Option<usize> {
        if c1 > c2 {
            let tmp = c1;
            c1 = c2;
            c2 = tmp;
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
                    if i == c1 && j == c2 {
                        return Some(idx);
                    }
                    idx += 1;
                }
                j += 1;
            }
            i += 1;
        }
        None
    }

    fn features_for(
        &self,
        color: Color,
        move_n: [u16; PCT_COUNT],
        immovable: &TileSet,
        out: &mut [i16],
    ) {
        let mut liberties = [[0; 2]; 2];
        let mut near_pct = [[0; PCT_COUNT]; 2];
        let qtile = self.queens[color.index()];
        if let Some(tile) = qtile {
            for &t in adjacent(tile).iter() {
                let pc = self.tile(t);
                if pc.is_none() {
                    continue;
                }
                let ally = pc.color() == color;
                let fixed = immovable.get(t);
                liberties[ally as usize][fixed as usize] += 1;
                near_pct[ally as usize][pc.ptype().index()] += 1;
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
                for pc in self.underworld_at(tile).iter() {
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
            out[4 + i * 6 + 0] = movable[i] as i16;
            out[4 + i * 6 + 1] = fixed[i] as i16;
            out[4 + i * 6 + 2] = buried[i] as i16;
            out[4 + i * 6 + 3] = move_n[i] as i16;
            out[4 + i * 6 + 4] = near_pct[0][i] as i16;
            out[4 + i * 6 + 5] = near_pct[1][i] as i16;
        }
    }

    pub const D6_TRANSFORMS: [fn(i16, i16, i16) -> (i16, i16, i16); 12] = [
        // 6 Rotations:
        |q, r, s| (q, r, s),           // 0 deg
        |q, r, s| (-s, -q, -r),        // 60 deg
        |q, r, s| (r, s, q),           // 120 deg
        |q, r, s| (-q, -r, -s),        // 180 deg
        |q, r, s| (s, q, r),           // 240 deg
        |q, r, s| (-r, -s, -q),        // 300 deg
        // 6 Reflections:
        |q, r, s| (-q, -s, -r),        // reflection along NW-SE axis
        |q, r, s| (r, q, s),           // reflection + rot 60
        |q, r, s| (-s, -r, -q),        // reflection + rot 120
        |q, r, s| (q, s, r),           // reflection + rot 180
        |q, r, s| (-r, -q, -s),        // reflection + rot 240
        |q, r, s| (s, r, q),           // reflection + rot 300
    ];

    fn populate_pct_square_features(
        &self,
        t: fn(i16, i16, i16) -> (i16, i16, i16),
        out: &mut [i16; Self::TOTAL_FN],
    ) {
        let queens = [
            self.queens[Color::White.index()],
            self.queens[Color::Black.index()],
        ];

        for (q_idx, &queen_opt) in queens.iter().enumerate() {
            if let Some(q_tile) = queen_opt {
                let base_offset = Self::OLD_TOTAL_FN + q_idx * Self::PCT_SQUARE_PER_QUEEN;

                for &tile in self.occupied_tiles[0].iter().chain(self.occupied_tiles[1].iter()) {
                    let (dq, dr, ds) = Self::cube_delta(q_tile, tile);
                    let (dq_c, dr_c, ds_c) = t(dq, dr, ds);

                    if let Some(spiral_idx) = Self::cube_to_spiral(dq_c, dr_c, ds_c) {
                        let top_pc = self.tile(tile);
                        let top_cat = top_pc.color().index() * PCT_COUNT + top_pc.ptype().index();
                        out[base_offset + spiral_idx * Self::TOTAL_PCTS + top_cat] = 1;

                        if self.height(tile) > 1 {
                            for under_pc in self.underworld_at(tile) {
                                let under_cat =
                                    under_pc.color().index() * PCT_COUNT + under_pc.ptype().index();
                                out[base_offset + spiral_idx * Self::TOTAL_PCTS + under_cat] = 1;
                            }
                        }
                    }
                }
            }
        }
    }

    fn features_for_both(
        &mut self,
        my_n: [u16; PCT_COUNT],
        other_n: [u16; PCT_COUNT],
    ) -> [i16; Self::TOTAL_FN] {
        let immovable = self.find_cut_vertexes();
        let mut out = [0i16; Self::TOTAL_FN];

        // 1. Restored tactical features (232 features from main)
        self.features_for(self.color(), my_n, &immovable, &mut out[0..Self::FN]);
        self.features_for(
            self.color().other(),
            other_n,
            &immovable,
            &mut out[Self::FN..Self::FN2],
        );

        let mut seen = [false; GRID_SIZE];
        for &tile in self.occupied_tiles[0].iter().chain(self.occupied_tiles[1].iter()) {
            let pc1 = self.tile(tile);
            let cat1 = (pc1.color() != self.color()) as usize * PCT_COUNT + pc1.ptype().index();
            seen[tile as usize] = true;
            for &adj in adjacent(tile).iter() {
                if self.occupied(adj) && !seen[adj as usize] {
                    let pc2 = self.tile(adj);
                    let cat2 =
                        (pc2.color() != self.color()) as usize * PCT_COUNT + pc2.ptype().index();
                    if let Some(idx) = Self::adj_idx(cat1, cat2) {
                        out[Self::FN2 + idx] += 1;
                    }
                }
            }
        }

        // 2. Canonical D6 candidate transform selection for pct-square features
        let w_queen = self.queens[Color::White.index()];
        let b_queen = self.queens[Color::Black.index()];

        let candidates: Vec<usize> = match (w_queen, b_queen) {
            (Some(wq), Some(bq)) => {
                let vq = Self::cube_delta(bq, wq);
                let mut min_v = (i16::MAX, i16::MAX, i16::MAX);
                let mut cands = Vec::new();
                for k in 0..12 {
                    let v_k = Self::D6_TRANSFORMS[k](vq.0, vq.1, vq.2);
                    if v_k < min_v {
                        min_v = v_k;
                        cands.clear();
                        cands.push(k);
                    } else if v_k == min_v {
                        cands.push(k);
                    }
                }
                cands
            }
            _ => (0..12).collect(),
        };

        if candidates.len() == 1 {
            self.populate_pct_square_features(Self::D6_TRANSFORMS[candidates[0]], &mut out);
            out
        } else {
            let mut best_out = out;
            self.populate_pct_square_features(Self::D6_TRANSFORMS[candidates[0]], &mut best_out);

            for &k in &candidates[1..] {
                let mut cand_out = out;
                self.populate_pct_square_features(Self::D6_TRANSFORMS[k], &mut cand_out);
                if cand_out < best_out {
                    best_out = cand_out;
                }
            }
            best_out
        }
    }

    pub const fn compute_adj_perm() -> [usize; Self::ADJ_SIZE] {
        let mut perm = [0usize; Self::ADJ_SIZE];
        let mut i = 0;
        while i < 16 {
            let mut j = i;
            while j < 16 {
                if let Some(idx1) = Self::adj_idx(i, j) {
                    let inv_i = (i + 8) % 16;
                    let inv_j = (j + 8) % 16;
                    if let Some(idx2) = Self::adj_idx(inv_i, inv_j) {
                        perm[idx1] = idx2;
                    }
                }
                j += 1;
            }
            i += 1;
        }
        perm
    }

    pub const ADJ_PERM: [usize; Self::ADJ_SIZE] = Self::compute_adj_perm();

    /// Swaps White and Black perspective:
    /// - Tactical features:
    ///   - STM block (0..52) <-> Opponent block (52..104)
    ///   - Inside each block: liberties ally <-> enemy, near_pct ally <-> enemy
    ///   - Adjacency histogram (104..232): permuted via ADJ_PERM
    /// - Canonical Pct-Square features:
    ///   - White Queen block (232..1192) <-> Black Queen block (1192..2152)
    ///   - Inside each 16-pct slot: White pieces (0..7) <-> Black pieces (8..15)
    pub fn swap_features_color(f: &[i16; Self::TOTAL_FN]) -> [i16; Self::TOTAL_FN] {
        let mut f_new = [0i16; Self::TOTAL_FN];

        // 1. Swap STM and Other 52-feature blocks
        for block in 0..2 {
            let src_base = block * Self::FN;
            let dst_base = (1 - block) * Self::FN;

            // liberties[ally][fixed] <-> liberties[enemy][fixed]
            f_new[dst_base + 0] = f[src_base + 2];
            f_new[dst_base + 1] = f[src_base + 3];
            f_new[dst_base + 2] = f[src_base + 0];
            f_new[dst_base + 3] = f[src_base + 1];

            // 8 piece types
            for i in 0..PCT_COUNT {
                f_new[dst_base + 4 + i * 6 + 0] = f[src_base + 4 + i * 6 + 0]; // movable
                f_new[dst_base + 4 + i * 6 + 1] = f[src_base + 4 + i * 6 + 1]; // fixed
                f_new[dst_base + 4 + i * 6 + 2] = f[src_base + 4 + i * 6 + 2]; // buried
                f_new[dst_base + 4 + i * 6 + 3] = f[src_base + 4 + i * 6 + 3]; // move_n
                f_new[dst_base + 4 + i * 6 + 4] = f[src_base + 4 + i * 6 + 5]; // near enemy is old near ally
                f_new[dst_base + 4 + i * 6 + 5] = f[src_base + 4 + i * 6 + 4]; // near ally is old near enemy
            }
        }

        // 2. Adjacency histogram permutation
        for (idx1, &idx2) in Self::ADJ_PERM.iter().enumerate() {
            f_new[Self::FN2 + idx1] = f[Self::FN2 + idx2];
        }

        // 3. Swap White Queen and Black Queen pct-square blocks
        // White Queen block is 232..1192, Black Queen block is 1192..2152
        for q_src in 0..2 {
            let q_dst = 1 - q_src;
            let src_q_base = Self::OLD_TOTAL_FN + q_src * Self::PCT_SQUARE_PER_QUEEN;
            let dst_q_base = Self::OLD_TOTAL_FN + q_dst * Self::PCT_SQUARE_PER_QUEEN;

            for hex in 0..Self::HEXES_PER_QUEEN {
                let src_hex_base = src_q_base + hex * Self::TOTAL_PCTS;
                let dst_hex_base = dst_q_base + hex * Self::TOTAL_PCTS;

                // Swap White pcts (0..7) and Black pcts (8..15)
                for pct in 0..PCT_COUNT {
                    f_new[dst_hex_base + pct] = f[src_hex_base + PCT_COUNT + pct];
                    f_new[dst_hex_base + PCT_COUNT + pct] = f[src_hex_base + pct];
                }
            }
        }

        f_new
    }

    fn other_n_by_pct(&mut self) -> [u16; PCT_COUNT] {
        self.turn_num += 1;
        let n_moves = self.generate_movements_by_pct();
        self.turn_num -= 1;
        n_moves
    }

    pub fn features_slow(&mut self) -> [i16; Self::TOTAL_FN] {
        let my_n = self.generate_movements_by_pct();
        let other_n = self.other_n_by_pct();
        self.features_for_both(my_n, other_n)
    }

    pub fn features_fast(&mut self, moves: &ActionList) -> [i16; Self::TOTAL_FN] {
        let mut my_n = [0; PCT_COUNT];
        for action in moves {
            if let Action::Move(from, _) = action {
                my_n[self.tile(*from).ptype().index()] += 1;
            }
        }
        let other_n = self.other_n_by_pct();
        self.features_for_both(my_n, other_n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece_type::PieceType;
    use crate::tile::Direction;

    #[test]
    fn test_cube_to_spiral_bijection() {
        let mut seen = [false; Board::HEXES_PER_QUEEN];
        let mut count = 0;

        for q in -4..=4 {
            for r in -4..=4 {
                let s = -q - r;
                if s >= -4 && s <= 4 {
                    if (q, r, s) == (0, 0, 0) {
                        assert_eq!(Board::cube_to_spiral(q, r, s), None);
                    } else {
                        let idx = Board::cube_to_spiral(q, r, s).expect("Must map within radius 4");
                        assert!(idx < Board::HEXES_PER_QUEEN);
                        assert!(!seen[idx], "Duplicate spiral index: {}", idx);
                        seen[idx] = true;
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(count, Board::HEXES_PER_QUEEN);
        assert!(seen.iter().all(|&x| x));
    }

    #[test]
    fn test_feature_vector_generation() {
        let mut board = Board::new();
        let feats = board.features_slow();
        assert_eq!(feats.len(), Board::TOTAL_FN);
        assert_eq!(Board::TOTAL_FN, 2152);

        // Place White Queen and Black Queen
        board.do_action(Action::Place(crate::tile::TILE_ZERO, PieceType::Queen));
        let adj_tile = crate::tile::TILE_ZERO + Direction::E;
        board.do_action(Action::Place(adj_tile, PieceType::Queen));

        let feats2 = board.features_slow();
        // Queen liberties should be updated
        assert!(feats2[..Board::OLD_TOTAL_FN].iter().any(|&v| v > 0));
        // Pct-square features should have entries for Queens
        assert!(feats2[Board::OLD_TOTAL_FN..].iter().any(|&v| v > 0));
    }

    #[test]
    fn test_adj_perm_involution() {
        let perm = Board::ADJ_PERM;
        let mut seen = [false; Board::ADJ_SIZE];
        for (i, &p) in perm.iter().enumerate() {
            assert!(p < Board::ADJ_SIZE);
            assert!(!seen[p], "Duplicate entry in ADJ_PERM: {}", p);
            seen[p] = true;
            assert_eq!(perm[p], i, "ADJ_PERM must be an involution: perm[perm[{i}]] == {i}");
        }
    }

    #[test]
    fn test_swap_features_color_involution() {
        let s = "Base+MLP;InProgress;White[9];wP;bA1 wP\\;wQ -wP;bA2 /bA1;wA1 \\wP;bL bA1-;wA2 wP/;bQ bL-;wA1 bQ\\;bA3 bA1\\;wA2 wA1/;bM bA3\\;wM wA2/;bM wQ\\;wP \\bL;bP /bM";
        let mut board = Board::parse_game_string(s).unwrap();
        let f = board.features_slow();
        let f_swapped = Board::swap_features_color(&f);
        let f_double_swapped = Board::swap_features_color(&f_swapped);

        assert_ne!(f, f_swapped, "Swapped features should be different for non-symmetric position");
        assert_eq!(f, f_double_swapped, "Double swap must restore the original features");
    }
}
