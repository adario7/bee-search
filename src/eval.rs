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
        let n_moves = self.generate_moves_n();
        self.turn_num -= 1;
        n_moves
    }
    
    pub fn static_eval(&mut self) -> Eval {
        #[cfg(feature = "gnn")]
        {
            return self.gnn_eval();
        }
        let my_moves_n = self.generate_moves_n();
        self.static_eval_fast(my_moves_n)
    }

    pub fn static_eval_fast(&mut self, my_moves_n: usize) -> Eval {
        #[cfg(feature = "gnn")]
        {
            return self.gnn_eval();
        }
        if FEATURES_EVAL {
            // eval with 06cd5e7 after fixing move_n @ depth 6
            // MSE: 1593901.4749, MAE: 810.2136, R2:  0.7115
            const W0: f64 = 0.0;
            const W: [f64; 34] = [-1015.9385495890275, -1258.8641032119087, -230.7337553197135, -871.5195086839251, 631.681704849666, -1334.0513483719926, 12.645273847124372, 294.557017407922, -46.55545383870309, -834.3379773220444, -62.78093885553172, 380.09345808559357, -13.509943921515106, -130.5575471341165, -135.403450841371, -190.9241310643996, -427.17564807483205, 1015.9385495897571, 1258.864103211562, 230.73375532018252, 871.5195086843385, -631.6817048497805, 1334.05134837225, -12.64527384712494, -294.5570174079467, 46.55545383875793, 834.3379773221393, 62.780938855530025, -380.09345808559937, 13.509943921515037, 130.55754713411417, 135.40345084138434, 190.92413106439415, 427.17564807479783];
            let f = self.features_fast(my_moves_n);
            (W0 + W.iter().zip(f.iter()).map(|(&a, &b)| a * b as f64).sum::<f64>()).round() as Eval
        } else {
            let my_score = self.score(my_moves_n);
            self.turn_num += 1;
            let their_move_n = self.generate_moves_n();
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
