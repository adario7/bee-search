use crate::{board::Board, tile::adjacent};

pub type Eval = i16;
pub type Value = Eval;

impl Board {
    pub fn queen_score(&self) -> Eval {
        let color = self.color();
        match self.queens[color.index()] {
            Some(tile) => adjacent(tile).iter().filter(|&&t| self.tile(t).is_none()).count() as Eval,
            None => 5
        }
    }

    pub fn other_queen_score(&self) -> Eval {
        let color = self.color().other();
        match self.queens[color.index()] {
            Some(tile) => adjacent(tile).iter().filter(|&&t| self.tile(t).is_none()).count() as Eval,
            None => 5
        }
    }

    fn score(&self, my_moves_n: usize) -> Eval {
        1000 * self.queen_score() + my_moves_n as Eval
    }

    pub fn n_moves(&self) -> usize {
        self.generate_moves_n()
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
        let my_score = self.score(my_moves_n);
        self.turn_num += 1;
        let their_move_n = self.generate_moves_n();
        let their_score = self.score(their_move_n);
        self.turn_num -= 1;
        my_score - their_score
    }

    #[cfg(feature = "gnn")]
    pub fn gnn_eval(&mut self) -> Eval {
        let graph = self.get_graph();
        self.gnn.evaluate(&graph).unwrap_or(0.0) as Eval
    }
}
