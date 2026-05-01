use crate::{board::{Action, Board}, eval_mlp::mlp_inference, tile::adjacent};

pub type Eval = i16;
pub type Value = Eval;

pub const MLP_EVAL: bool = true;
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
        let my_moves_n = self.generate_moves();
        self.static_eval_fast(&my_moves_n)
    }

    pub fn static_eval_fast(&mut self, my_moves: &Vec<Action>) -> Eval {
        if MLP_EVAL {
            let f = self.features_fast(my_moves);
            mlp_inference(&f)
        } else if FEATURES_EVAL {
            let f = self.features_fast(my_moves);
            Self::lr_inference(&f)
        } else {
            let my_score = self.score(my_moves.len());
            self.turn_num += 1;
            let their_move_n = self.generate_moves_n();
            let their_score = self.score(their_move_n);
            self.turn_num -= 1;
            my_score - their_score
        }
    }

    fn lr_inference(f: &[i16]) -> Eval {
        // eval with e45ab18 @ depth 6
        // MSE: 1471231.5289, MAE: 818.2225, R2:  0.7846
        const W: [f32; Board::FN] = [-565.7567307332503, -915.2212142778455, -109.51756127949724, -650.4413527922027, 404.04926465374035, 1256.8194668382557, -468.66413566098487, 335.64626124318903, 320.64424120593884, -3.495870259939693e-12, 265.1330186261176, 239.83073949200974, -55.41967437889457, 95.50024700848343, -144.47262123277443, -185.54668810312194, -388.6210887649594, -398.6504905017605, -424.631145944288, 130.7295628353084, -304.9813263304336, -122.37930097467401, 160.04436928622596, 109.54766294535688, -52.730048296975156, 14.46390951568867, -306.2466887794543, -195.44733842387302, -82.11381610357137, 232.24417003857235, 146.27994381239506, 160.89220023883644, -384.29142798085854, -173.1007015778787, 229.26180412233347, 191.3388482060837, -129.42810557489457, 12.758808280238327, -329.82812921458606, 7.49814803233491, 191.8828928857512, 266.2216770165069, 129.50709754003455, 45.797379410683334, -108.8342419886686, -147.58988332606833, 537.1163050988263, 535.6718267074967, -150.01837374148897, -15.704836598323482, -222.96775069143212, 56.606850301611374];
        ((0..Board::FN).map(|i| W[i] * (f[i] - f[i + Board::FN]) as f32).sum::<f32>()).round() as Eval
    }
}
