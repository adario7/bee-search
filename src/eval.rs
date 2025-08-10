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

    fn other_queen_score(&self) -> Eval {
        let color = self.color().other();
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
            const W0: f64 = -932.3573973472558;
            const W: [f64; 34] = [-994.54127159537, -1165.1300436183544, -402.2331546006423, -1074.0264610856254, 650.6029947832957, -801.1057472370619, 5.5035769856667685, 154.8350332064152, -137.36272556164636, -1506.5084924627242, -64.31659299384276, 119.25673695518428, -118.920076926504, -108.87926900177106, -149.31833615625533, -160.8432544096951, -221.31828545440177, 1190.3241702596383, 1288.2665418923157, 664.1304023582576, 1127.7994767762377, -821.6408144224184, 942.154570244006, -4.6752381573087405, -101.9347825855653, 138.89178055240978, 1547.529279615086, 117.52693726407541, -81.19370217381257, 246.79683867038577, 175.75536291860794, 286.7175984747997, 224.26320991967197, 359.09682529161574];
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
