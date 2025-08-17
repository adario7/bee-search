use crate::{abstractions::TileSet, board::Board, piece::Color, tile::adjacent};

impl Board {
    pub const FN: usize = 17;
    pub const FN2: usize = Self::FN * 2;

    fn features_for(&self, color: Color, move_n: i64, immovable: &TileSet) -> [i64; Self::FN] {
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
        let num_placed = self.occupied_tiles[color.index()].len();
        let num_fixed = self.occupied_tiles[color.index()].iter()
            .filter(|&&t| immovable.get(t)).count();
        let queen_fixed = if let Some(tile) = qtile {
            immovable.get(tile)
        } else {
            false
        };
        let queen_height = if let Some(tile) = qtile {
            self.height(tile)
        } else {
            0
        } > 1;
        [
            liberties[0][0],
            liberties[0][1],
            liberties[1][0],
            liberties[1][1],
            queen_fixed as i64,
            queen_height as i64,
            move_n,
            num_placed as i64,
            num_fixed as i64,
            self.placeable[color.index()][0] as i64,
            self.placeable[color.index()][1] as i64,
            self.placeable[color.index()][2] as i64,
            self.placeable[color.index()][3] as i64,
            self.placeable[color.index()][4] as i64,
            self.placeable[color.index()][5] as i64,
            self.placeable[color.index()][6] as i64,
            self.placeable[color.index()][7] as i64,
        ]
    }

    pub fn features_fast(&mut self, my_n: usize) -> [i64; Self::FN2] {
        let immovable = self.find_cut_vertexes();
        let other_n = self.other_n_moves();
        let a = self.features_for(self.color(), my_n as i64, &immovable);
        let b = self.features_for(self.color().other(), other_n as i64, &immovable);
        let mut out = [0i64; Self::FN2];
        out[..Self::FN].copy_from_slice(&a);
        out[Self::FN..].copy_from_slice(&b);
        out
    }

    pub fn features(&mut self) -> [i64; Self::FN2] {
        let my_n = self.generate_moves_n();
        self.features_fast(my_n)
    }
}
