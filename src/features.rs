use crate::{abstractions::TileSet, board::{Action, Board}, piece::Color, piece_type::PCT_COUNT, tile::adjacent};

impl Board {
    pub const FN: usize = 5 + PCT_COUNT*3;
    pub const FN2: usize = Self::FN * 2;

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
        let queen_height = if let Some(tile) = qtile {
            self.height(tile)
        } else {
            0
        } > 1;
        let mut placed = [0; PCT_COUNT];
        let mut fixed = [0; PCT_COUNT];
        for &tile in self.occupied_tiles[color.index()].iter() {
            let pct = self.tile(tile).ptype();
            placed[pct.index()] += 1;
            if immovable.get(tile) {
                fixed[pct.index()] += 1;
            }
        }
        out[0] = liberties[0][0];
        out[1] = liberties[0][1];
        out[2] = liberties[1][0];
        out[3] = liberties[1][1];
        out[4] = queen_height as i16;
        for i in 0..PCT_COUNT {
            out[5 + i*3 + 0] = placed[i] as i16;
            out[5 + i*3 + 1] = fixed[i] as i16;
            out[5 + i*3 + 2] = move_n[i] as i16;
        }
    }

    fn other_n_by_pct(&mut self) -> [u16; PCT_COUNT] {
        self.turn_num += 1;
        let n_moves = self.generate_movements_by_pct();
        self.turn_num -= 1;
        n_moves
    }

    fn features_for_both(&mut self, my_n: [u16; PCT_COUNT]) -> [i16; Self::FN2] {
        let immovable = self.find_cut_vertexes();
        let other_n = self.other_n_by_pct();
        let mut out = [0; Self::FN2];
        self.features_for(self.color(), my_n, &immovable, &mut out[0..Self::FN]);
        self.features_for(self.color().other(), other_n, &immovable, &mut out[Self::FN..]);
        out
    }

    pub fn features_slow(&mut self) -> [i16; Self::FN2] {
        let my_n = self.generate_movements_by_pct();
        self.features_for_both(my_n)
    }

    pub fn features_fast(&mut self, moves: &Vec<Action>) -> [i16; Self::FN2] {
        let mut my_n = [0; PCT_COUNT];
        for action in moves {
            if let Action::Move(from, _) = action {
                my_n[self.tile(*from).ptype().index()] += 1;
            }
        }
        self.features_for_both(my_n)
    }
}
