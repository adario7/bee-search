use crate::{board::{Action, Board}, tile::adjacent};

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
        let my_moves_n = self.generate_moves();
        self.static_eval_fast(&my_moves_n)
    }

    pub fn static_eval_fast(&mut self, my_moves: &Vec<Action>) -> Eval {
        #[cfg(feature = "gnn")]
        {
            return self.gnn_eval();
        }
        if FEATURES_EVAL {
            // eval with bbbb3cf @ depth 6
            // MSE: 1501344.0842, MAE: 799.7939, R2:  0.7578
            const W0: f64 = 0.0;
            const W: [f64; Board::FN2] = [-840.6598149658546, -1160.2584105230026, -202.20750909748017, -797.0567590885644, -240.9527483088732, 654.532727970969, 849.6971352503681, 367.3581538339563, 284.79764316346774, -31.43509841481221, 67.64146532051845, -293.1308156453455, 47.59602550381271, 115.21272968276139, 218.42807547836105, -55.69961732759956, 15.03234475417105, 127.19067769308327, 151.74488763308318, 98.95364813074835, 267.4670478566127, 18.95092147552657, 15.800555816379259, 197.4456314488952, 71.1392202736896, 36.44901958300005, 510.2030738204322, 73.74597683287516, 23.337241068620287, 840.6598149654401, 1160.2584105253197, 202.20750909784675, 797.0567590885186, 240.95274830886592, -654.5327279716182, -849.6971352503402, -367.3581538340715, -284.7976431638169, 31.435098415075117, -67.64146532045442, 293.13081564542864, -47.596025503870294, -115.21272968273776, -218.4280754783682, 55.69961732753342, -15.032344754171731, -127.19067769302501, -151.74488763311933, -98.953648130759, -267.4670478565905, -18.950921475584437, -15.800555816380097, -197.44563144883506, -71.13922027373565, -36.449019583002666, -510.20307382045047, -73.74597683288195, -23.33724106861621];
            let f = self.features_fast(my_moves);
            (W0 + W.iter().zip(f.iter()).map(|(&a, &b)| a * b as f64).sum::<f64>()).round() as Eval
        } else {
            let my_score = self.score(my_moves.len());
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
