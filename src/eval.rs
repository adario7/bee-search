use crate::{board::Board, tile::adjacent};

pub type Eval = i16;
pub type Value = Eval;

pub const FEATURES_EVAL: bool = true;

impl Board {
    fn queen_score(&self) -> Eval {
        let color = self.color();
        match self.queens[color.index()] {
            Some(tile) => adjacent(tile).iter().filter(|&&t| self.tile(t).is_none()).count() as Eval,
            None => 5
        }
    }

    fn score(&self, my_moves_n: usize) -> Eval {
        1000 * self.queen_score() + my_moves_n as Eval
    }

    pub fn other_n_moves(&mut self) -> usize {
        self.turn_num += 1;
        let n_moves = self.generate_moves().len();
        self.turn_num -= 1;
        n_moves
    }
    
    pub fn static_eval(&mut self) -> Eval {
        #[cfg(feature = "gnn")]
        {
            return self.gnn_eval();
        }
        let my_moves_n = self.generate_moves().len();
        self.static_eval_fast(my_moves_n)
    }

    pub fn static_eval_fast(&mut self, my_moves_n: usize) -> Eval {
        #[cfg(feature = "gnn")]
        {
            return self.gnn_eval();
        }
        if FEATURES_EVAL {
            const W0: f64 = 0.0;
            const W: [f64; 34] = [-1054.40654305728, -1196.5900998213406, -506.37151595160645, -1085.1452142778787, 723.8616987250335, -853.6640933241488, 4.292676886977574, 138.161684065775, -129.5084706020009, -1377.0735314394146, -63.29651709100909, 140.54577125265865, -126.28281419271696, -102.65066138166875, -164.03745751315873, -143.8012948157078, -243.95784489972712, 1054.4065430574867, 1196.5900998216512, 506.3715159514984, 1085.14521427672, -723.861698724765, 853.6640933243536, -4.292676886981326, -138.16168406580442, 129.50847060198888, 1377.0735314393064, 63.29651709102073, -140.54577125267193, 126.28281419272327, 102.65066138167778, 164.03745751315546, 143.80129481570467, 243.95784489973593];
            let f = self.features_fast(my_moves_n);
            (W0 + W.iter().zip(f.iter()).map(|(&a, &b)| a * b as f64).sum::<f64>()).round() as Eval
        } else {
            let my_score = self.score(my_moves_n);
            self.turn_num += 1;
            let their_move_n = self.generate_moves().len();
            let their_score = self.score(their_move_n);
            self.turn_num -= 1;
            my_score - their_score
        }
    }

    #[cfg(feature = "gnn")]
    pub fn gnn_eval(&mut self) -> Eval {
        let graph = self.get_graph();
        self.gnn.evaluate(&graph).unwrap_or(0.0) as Eval
    }
}
