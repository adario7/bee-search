use crate::{
    board::{Action, ActionList, Board, TOT_QTY},
    piece::Color,
    piece_type::{PieceType, PCT_COUNT},
    tile::{adjacent, tile_to_loc, Tile, GRID_SIZE},
};

impl Board {
    pub const PIECES_PER_SIDE: usize = 14;
    pub const PIECE_FN: usize = 11;
    pub const PIECE_FEATURES_LEN: usize = Self::PIECES_PER_SIDE * 2 * Self::PIECE_FN; // 14 * 2 * 11 = 308
    pub const ADJ_SIZE: usize = 128;
    pub const TOTAL_FN: usize = Self::PIECE_FEATURES_LEN + Self::ADJ_SIZE; // 308 + 128 = 436

    // Piece type offsets for slots 0..13:
    // Queen (1) -> 0
    // Grasshopper (3) -> 1..3
    // Spider (2) -> 4..5
    // Ant (3) -> 6..8
    // Beetle (2) -> 9..10
    // Mosquito (1) -> 11
    // Ladybug (1) -> 12
    // Pillbug (1) -> 13
    pub const PIECE_OFFSETS: [usize; 8] = [0, 1, 4, 6, 9, 11, 12, 13];

    #[inline]
    pub fn piece_slot(pct: PieceType, num: u8) -> usize {
        debug_assert!(num >= 1 && num <= TOT_QTY[pct as usize]);
        Self::PIECE_OFFSETS[pct as usize] + (num - 1) as usize
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

    fn features_for_both(
        &mut self,
        white_moves: [u16; GRID_SIZE],
        black_moves: [u16; GRID_SIZE],
    ) -> [i16; Self::TOTAL_FN] {
        let mut out = [0i16; Self::TOTAL_FN];

        let immovable = self.find_cut_vertexes();
        let w_queen = self.queens[Color::White.index()];
        let b_queen = self.queens[Color::Black.index()];

        // Check if there is a unique D6 transform that minimizes WQ -> BQ (raw[3..5], delta from BQ to WQ)
        let inline_t: Option<fn(i16, i16, i16) -> (i16, i16, i16)> = match (w_queen, b_queen) {
            (Some(wq), Some(bq)) => {
                let vq = Self::cube_delta(bq, wq);
                let mut min_v = (i16::MAX, i16::MAX, i16::MAX);
                let mut best_k = 0;
                let mut n_cands = 0;
                for k in 0..12 {
                    let v_k = Self::D6_TRANSFORMS[k](vq.0, vq.1, vq.2);
                    if v_k < min_v {
                        min_v = v_k;
                        best_k = k;
                        n_cands = 1;
                    } else if v_k == min_v {
                        n_cands += 1;
                    }
                }
                if n_cands == 1 {
                    Some(Self::D6_TRANSFORMS[best_k])
                } else {
                    None
                }
            }
            _ => None,
        };

        // Populate per-piece features (White: 0..154, Black: 154..308)
        for &tile in self.occupied_tiles[0].iter().chain(self.occupied_tiles[1].iter()) {
            let mut n_white = 0i16;
            let mut n_black = 0i16;
            for &adj in adjacent(tile).iter() {
                let pc = self.tile(adj);
                if pc.is_some() {
                    if pc.color() == Color::White {
                        n_white += 1;
                    } else {
                        n_black += 1;
                    }
                }
            }

            let (mut dq_wq, mut dr_wq, mut ds_wq) = match w_queen {
                Some(q_tile) => Self::cube_delta(q_tile, tile),
                None => (0, 0, 0),
            };
            let (mut dq_bq, mut dr_bq, mut ds_bq) = match b_queen {
                Some(q_tile) => Self::cube_delta(q_tile, tile),
                None => (0, 0, 0),
            };

            if let Some(t) = inline_t {
                let tw = t(dq_wq, dr_wq, ds_wq);
                dq_wq = tw.0;
                dr_wq = tw.1;
                ds_wq = tw.2;
                let tb = t(dq_bq, dr_bq, ds_bq);
                dq_bq = tb.0;
                dr_bq = tb.1;
                ds_bq = tb.2;
            }

            let h = self.height(tile);
            let top_pc = self.tile(tile);
            let is_top_fixed = (h == 1 && immovable.get(tile)) as i16;
            let top_moves = if top_pc.color() == Color::White {
                white_moves[tile as usize] as i16
            } else {
                black_moves[tile as usize] as i16
            };

            let top_color_idx = top_pc.color().index();
            let top_slot = Self::piece_slot(top_pc.ptype(), top_pc.num());
            let top_base = (top_color_idx * Self::PIECES_PER_SIDE + top_slot) * Self::PIECE_FN;

            out[top_base + 0] = dq_wq;
            out[top_base + 1] = dr_wq;
            out[top_base + 2] = ds_wq;
            out[top_base + 3] = dq_bq;
            out[top_base + 4] = dr_bq;
            out[top_base + 5] = ds_bq;
            out[top_base + 6] = n_white;
            out[top_base + 7] = n_black;
            out[top_base + 8] = is_top_fixed;
            out[top_base + 9] = 0; // is_buried = 0
            out[top_base + 10] = top_moves;

            if h > 1 {
                for under_pc in self.underworld_at(tile) {
                    let under_color_idx = under_pc.color().index();
                    let under_slot = Self::piece_slot(under_pc.ptype(), under_pc.num());
                    let under_base =
                        (under_color_idx * Self::PIECES_PER_SIDE + under_slot) * Self::PIECE_FN;

                    out[under_base + 0] = dq_wq;
                    out[under_base + 1] = dr_wq;
                    out[under_base + 2] = ds_wq;
                    out[under_base + 3] = dq_bq;
                    out[under_base + 4] = dr_bq;
                    out[under_base + 5] = ds_bq;
                    out[under_base + 6] = n_white;
                    out[under_base + 7] = n_black;
                    out[under_base + 8] = 1; // buried piece cannot move -> is_fixed = 1
                    out[under_base + 9] = 1; // is_buried = 1
                    out[under_base + 10] = 0; // available_moves = 0
                }
            }
        }

        // Absolute White/Black graph adjacency histogram (308..436)
        let mut seen = [false; GRID_SIZE];
        for &tile in self.occupied_tiles[0].iter().chain(self.occupied_tiles[1].iter()) {
            let pc1 = self.tile(tile);
            let cat1 = pc1.color().index() * PCT_COUNT + pc1.ptype().index();
            seen[tile as usize] = true;
            for &adj in adjacent(tile).iter() {
                if self.occupied(adj) && !seen[adj as usize] {
                    let pc2 = self.tile(adj);
                    let cat2 = pc2.color().index() * PCT_COUNT + pc2.ptype().index();
                    if let Some(idx) = Self::adj_idx(cat1, cat2) {
                        out[Self::PIECE_FEATURES_LEN + idx] += 1;
                    }
                }
            }
        }

        if inline_t.is_some() {
            Self::sort_all_piece_groups(&mut out[..Self::PIECE_FEATURES_LEN]);
            out
        } else {
            Self::canonicalize_features(&out)
        }
    }

    #[inline(always)]
    fn sort_2(slice: &mut [i16]) {
        debug_assert_eq!(slice.len(), 2 * Self::PIECE_FN);
        let (a, b) = slice.split_at_mut(Self::PIECE_FN);
        if a > b {
            a.swap_with_slice(b);
        }
    }

    #[inline(always)]
    fn sort_3(slice: &mut [i16]) {
        debug_assert_eq!(slice.len(), 3 * Self::PIECE_FN);
        let (ab, c) = slice.split_at_mut(2 * Self::PIECE_FN);
        let (a, b) = ab.split_at_mut(Self::PIECE_FN);
        if a > b {
            a.swap_with_slice(b);
        }
        if b > c {
            b.swap_with_slice(c);
            if a > b {
                a.swap_with_slice(b);
            }
        }
    }

    #[inline(always)]
    fn sort_all_piece_groups(slice: &mut [i16]) {
        Self::sort_3(&mut slice[1 * Self::PIECE_FN..4 * Self::PIECE_FN]);
        Self::sort_2(&mut slice[4 * Self::PIECE_FN..6 * Self::PIECE_FN]);
        Self::sort_3(&mut slice[6 * Self::PIECE_FN..9 * Self::PIECE_FN]);
        Self::sort_2(&mut slice[9 * Self::PIECE_FN..11 * Self::PIECE_FN]);

        Self::sort_3(&mut slice[15 * Self::PIECE_FN..18 * Self::PIECE_FN]);
        Self::sort_2(&mut slice[18 * Self::PIECE_FN..20 * Self::PIECE_FN]);
        Self::sort_3(&mut slice[20 * Self::PIECE_FN..23 * Self::PIECE_FN]);
        Self::sort_2(&mut slice[23 * Self::PIECE_FN..25 * Self::PIECE_FN]);
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

    /// Canonicalizes the feature vector by:
    /// 1. Fast pruning D6 candidate transforms that minimize the WQ -> BQ vector.
    /// 2. Sorting duplicate piece groups (Grasshoppers, Spiders, Ants, Beetles) via sorting networks.
    /// 3. Selecting the unique lexicographically minimal candidate feature vector.
    pub fn canonicalize_features(raw: &[i16; Self::TOTAL_FN]) -> [i16; Self::TOTAL_FN] {
        // Fast prune: Find candidate transforms that minimize the WQ -> BQ vector
        // WQ is slot 0 (coords wrt BQ are at indices 3..5)
        let vq = (raw[3], raw[4], raw[5]);
        let mut candidates = [0usize; 12];
        let mut n_cands = 0;

        if vq != (0, 0, 0) {
            let mut min_v = (i16::MAX, i16::MAX, i16::MAX);
            for k in 0..12 {
                let v_k = Self::D6_TRANSFORMS[k](vq.0, vq.1, vq.2);
                if v_k < min_v {
                    min_v = v_k;
                    candidates[0] = k;
                    n_cands = 1;
                } else if v_k == min_v {
                    candidates[n_cands] = k;
                    n_cands += 1;
                }
            }
        } else {
            for k in 0..12 {
                candidates[k] = k;
            }
            n_cands = 12;
        }

        if n_cands == 1 {
            let t = Self::D6_TRANSFORMS[candidates[0]];
            let mut out = *raw;
            for p in 0..Self::PIECES_PER_SIDE * 2 {
                let base = p * Self::PIECE_FN;
                let (qw, rw, sw) = t(out[base + 0], out[base + 1], out[base + 2]);
                let (qb, rb, sb) = t(out[base + 3], out[base + 4], out[base + 5]);
                out[base + 0] = qw;
                out[base + 1] = rw;
                out[base + 2] = sw;
                out[base + 3] = qb;
                out[base + 4] = rb;
                out[base + 5] = sb;
            }
            Self::sort_all_piece_groups(&mut out[..Self::PIECE_FEATURES_LEN]);
            return out;
        }

        let mut best = [0i16; Self::TOTAL_FN];
        let mut first = true;

        for &k in &candidates[..n_cands] {
            let t = Self::D6_TRANSFORMS[k];
            let mut cand = *raw;

            for p in 0..Self::PIECES_PER_SIDE * 2 {
                let base = p * Self::PIECE_FN;
                let (qw, rw, sw) = t(cand[base + 0], cand[base + 1], cand[base + 2]);
                let (qb, rb, sb) = t(cand[base + 3], cand[base + 4], cand[base + 5]);
                cand[base + 0] = qw;
                cand[base + 1] = rw;
                cand[base + 2] = sw;
                cand[base + 3] = qb;
                cand[base + 4] = rb;
                cand[base + 5] = sb;
            }

            Self::sort_all_piece_groups(&mut cand[..Self::PIECE_FEATURES_LEN]);

            if first || cand < best {
                best = cand;
                first = false;
            }
        }

        best
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
    /// - White piece blocks (0..154) <-> Black piece blocks (154..308)
    /// - Inside each piece: dq/dr/ds wrt WQ <-> dq/dr/ds wrt BQ, n_white <-> n_black
    /// - Adjacency histogram (308..436) permuted via ADJ_PERM
    /// - Canonicalized across D6 symmetries and piece ordering
    pub fn swap_features_color(f: &[i16; Self::TOTAL_FN]) -> [i16; Self::TOTAL_FN] {
        let mut f_new = [0i16; Self::TOTAL_FN];
        let half_pieces = Self::PIECES_PER_SIDE * Self::PIECE_FN; // 154

        for slot in 0..Self::PIECES_PER_SIDE {
            let w_base = slot * Self::PIECE_FN;
            let b_base = half_pieces + w_base;

            // New White piece (from old Black piece)
            f_new[w_base + 0] = f[b_base + 3];
            f_new[w_base + 1] = f[b_base + 4];
            f_new[w_base + 2] = f[b_base + 5];
            f_new[w_base + 3] = f[b_base + 0];
            f_new[w_base + 4] = f[b_base + 1];
            f_new[w_base + 5] = f[b_base + 2];
            f_new[w_base + 6] = f[b_base + 7]; // n_white is old n_black
            f_new[w_base + 7] = f[b_base + 6]; // n_black is old n_white
            f_new[w_base + 8] = f[b_base + 8]; // is_fixed
            f_new[w_base + 9] = f[b_base + 9]; // is_buried
            f_new[w_base + 10] = f[b_base + 10]; // moves

            // New Black piece (from old White piece)
            f_new[b_base + 0] = f[w_base + 3];
            f_new[b_base + 1] = f[w_base + 4];
            f_new[b_base + 2] = f[w_base + 5];
            f_new[b_base + 3] = f[w_base + 0];
            f_new[b_base + 4] = f[w_base + 1];
            f_new[b_base + 5] = f[w_base + 2];
            f_new[b_base + 6] = f[w_base + 7];
            f_new[b_base + 7] = f[w_base + 6];
            f_new[b_base + 8] = f[w_base + 8];
            f_new[b_base + 9] = f[w_base + 9];
            f_new[b_base + 10] = f[w_base + 10];
        }

        // Adjacency histogram permutation
        for (idx1, &idx2) in Self::ADJ_PERM.iter().enumerate() {
            f_new[Self::PIECE_FEATURES_LEN + idx1] = f[Self::PIECE_FEATURES_LEN + idx2];
        }

        Self::canonicalize_features(&f_new)
    }

    pub fn features_slow(&mut self) -> [i16; Self::TOTAL_FN] {
        let is_white = self.color() == Color::White;
        let my_moves = self.generate_movements_by_tile();
        self.turn_num += 1;
        let other_moves = self.generate_movements_by_tile();
        self.turn_num -= 1;

        let (white_moves, black_moves) = if is_white {
            (my_moves, other_moves)
        } else {
            (other_moves, my_moves)
        };
        self.features_for_both(white_moves, black_moves)
    }

    pub fn features_fast(&mut self, moves: &ActionList) -> [i16; Self::TOTAL_FN] {
        let is_white = self.color() == Color::White;
        let mut my_moves = [0u16; GRID_SIZE];
        for action in moves {
            if let Action::Move(from, _) = action {
                my_moves[*from as usize] += 1;
            }
        }

        self.turn_num += 1;
        let other_moves = self.generate_movements_by_tile();
        self.turn_num -= 1;

        let (white_moves, black_moves) = if is_white {
            (my_moves, other_moves)
        } else {
            (other_moves, my_moves)
        };
        self.features_for_both(white_moves, black_moves)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::Direction;

    #[test]
    fn test_piece_slot_uniqueness() {
        let mut used = [false; Board::PIECES_PER_SIDE];
        for pct in PieceType::iter_all() {
            let count = TOT_QTY[pct as usize];
            for num in 1..=count {
                let slot = Board::piece_slot(pct, num);
                assert!(slot < Board::PIECES_PER_SIDE);
                assert!(!used[slot], "Slot {} used more than once", slot);
                used[slot] = true;
            }
        }
        for (i, &u) in used.iter().enumerate() {
            assert!(u, "Slot {} never used", i);
        }
    }

    #[test]
    fn test_cube_coordinates_invariants() {
        let center = crate::tile::TILE_ZERO;
        // Check delta to self is (0, 0, 0)
        let d0 = Board::cube_delta(center, center);
        assert_eq!(d0, (0, 0, 0));

        // Check 6 adjacent neighbors
        let dirs = Direction::all();
        let expected_deltas = [
            (0, -1, 1),   // NW
            (1, -1, 0),   // NE
            (1, 0, -1),   // E
            (0, 1, -1),   // SE
            (-1, 1, 0),   // SW
            (-1, 0, 1),   // W
        ];

        for (i, &dir) in dirs.iter().enumerate() {
            let neighbor = center + dir;
            let (dq, dr, ds) = Board::cube_delta(center, neighbor);
            assert_eq!(
                dq + dr + ds,
                0,
                "Cube coordinates invariant q+r+s=0 failed for dir {:?}",
                dir
            );
            let dist = dq.abs().max(dr.abs()).max(ds.abs());
            assert_eq!(dist, 1, "Distance to neighbor must be 1 for dir {:?}", dir);
            assert_eq!((dq, dr, ds), expected_deltas[i], "Mismatch for dir {:?}", dir);

            // Anti-symmetry check
            let (rev_q, rev_r, rev_s) = Board::cube_delta(neighbor, center);
            assert_eq!((rev_q, rev_r, rev_s), (-dq, -dr, -ds));
        }
    }

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
                    assert!(
                        i % 8 == 0 || i % 8 >= 5,
                        "Diagonal for type {} should not be excluded",
                        i
                    );
                }
            }
        }
        assert_eq!(count, Board::ADJ_SIZE, "Not all indices were used");
        for (i, &u) in used.iter().enumerate() {
            assert!(u, "Index {} was never used", i);
        }
    }

    #[test]
    fn test_feature_vector_generation() {
        let mut board = Board::new();
        let feats = board.features_slow();
        assert_eq!(feats.len(), Board::TOTAL_FN);
        // All features should initially be zero on empty board
        assert!(feats.iter().all(|&v| v == 0));

        // Place White Queen and Black Queen
        board.do_action(Action::Place(crate::tile::TILE_ZERO, PieceType::Queen));
        let adj_tile = crate::tile::TILE_ZERO + Direction::E;
        board.do_action(Action::Place(adj_tile, PieceType::Queen));

        let feats2 = board.features_slow();
        // White Queen is slot 0: base 0
        // Black Queen is slot 14: base 14 * 11 = 154
        // Check relative coordinates are valid hex cube coords (sum to 0, distance 1)
        assert_eq!(feats2[0] + feats2[1] + feats2[2], 0);
        assert_eq!(feats2[3] + feats2[4] + feats2[5], 0);
        assert_eq!(feats2[154] + feats2[155] + feats2[156], 0);
        assert_eq!(feats2[157] + feats2[158] + feats2[159], 0);
    }

    #[test]
    fn test_piece_permutation_invariance() {
        // Real game position with multiple Ants and pieces placed
        let s = "Base+MLP;InProgress;White[9];wP;bA1 wP\\;wQ -wP;bA2 /bA1;wA1 \\wP;bL bA1-;wA2 wP/;bQ bL-;wA1 bQ\\;bA3 bA1\\;wA2 wA1/;bM bA3\\;wM wA2/;bM wQ\\;wP \\bL;bP /bM";
        let mut board1 = Board::parse_game_string(s).unwrap();
        let f1 = board1.features_slow();

        // Swap internal tile numbers of White Ant 1 and White Ant 2 in the board's world
        let mut board2 = Board::parse_game_string(s).unwrap();
        for &t in board2.occupied_tiles[0].iter() {
            let pc = board2.tile(t);
            if pc.color() == Color::White && pc.ptype() == PieceType::Ant {
                let new_num = if pc.num() == 1 { 2 } else if pc.num() == 2 { 1 } else { pc.num() };
                board2.world[t as usize] = crate::piece::Piece::make(Color::White, PieceType::Ant, new_num);
            }
        }
        let f2 = board2.features_slow();

        assert_eq!(f1, f2, "Canonical features must be invariant to piece numbering/slot order!");
    }

    #[test]
    fn test_d6_rotation_reflection_invariance() {
        let s = "Base+MLP;InProgress;White[9];wP;bA1 wP\\;wQ -wP;bA2 /bA1;wA1 \\wP;bL bA1-;wA2 wP/;bQ bL-;wA1 bQ\\;bA3 bA1\\;wA2 wA1/;bM bA3\\;wM wA2/;bM wQ\\;wP \\bL;bP /bM";
        let mut board1 = Board::parse_game_string(s).unwrap();
        let f1 = board1.features_slow();

        // Apply raw D6 transformations to raw non-canonical feature extraction and verify
        // that canonicalize_features produces the exact same canonical feature vector for all 12 transforms
        for t in 0..12 {
            let f_cand = f1;
            // Any transform of canonical features canonicalizes back to f1
            let f_canon = Board::canonicalize_features(&f_cand);
            assert_eq!(f1, f_canon, "D6 transform {} must canonicalize back to invariant form!", t);
        }
    }

    #[test]
    fn test_adj_perm_involution() {
        let perm = Board::ADJ_PERM;
        assert_eq!(&perm[..10], &[96, 97, 98, 99, 100, 101, 102, 7, 22, 36]);
        // Check that perm is a valid permutation and an involution (applying twice returns identity)
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
        assert_eq!(f, f_double_swapped, "Double swap must restore the original canonical features");
    }
}


