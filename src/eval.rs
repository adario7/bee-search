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
            // eval with e45ab18 @ depth 6
            // MSE: 1548246.0074, MAE: 831.1957, R2:  0.7780
            const W: [f32; Board::FN] = [-553.575247964289, -916.8807898439247, -69.37878013128994, -609.9678298319595, -1579.2266855173816, 221.88787469438748, 931.4173704142598, 436.08300662537914, 353.0235122552682, -2.1032064978498966e-12, 255.17529061468792, -6.795684862899158, 104.27451398226486, -116.90032018144794, -202.2645872513572, -448.1483317436962, 66.46275161311678, 166.88596515603967, -315.3381070902843, -132.16725987778753, 68.61255129006145, 63.15214258353592, 18.082777982176083, -292.83999056258665, -196.01235473139226, -300.95027239089814, 517.1520545185759, 222.76804734790812, -336.93514583039115, -156.40703671623726, 206.4418092876126, -18.887343940677198, 13.925403578327916, -328.78234164631135, 18.052130661896474, 151.74848817612684, 119.38328627760868, 49.74854064516035, -78.57194264185044, -163.34298146735182, 359.87671131747163, 44.54891043101526, 39.915407918705625, -354.1117021107434, 152.79547941878604];
            let f = self.features_fast(my_moves);
            ((0..Board::FN).map(|i| W[i] * (f[i] - f[i + Board::FN]) as f32).sum::<f32>()).round() as Eval
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
