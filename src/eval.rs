use crate::{board::Board, tile::adjacent};

pub type Eval = i16;
pub type Value = Eval;

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

    pub fn static_eval(&mut self) -> Eval {
        let my_moves_n = self.generate_moves().len();
        self.static_eval_fast(my_moves_n)
    }

    pub fn static_eval_fast(&mut self, my_moves_n: usize) -> Eval {
        let my_score = self.score(my_moves_n);
        self.turn_num += 1;
        let their_move_n = self.generate_moves().len();
        let their_score = self.score(their_move_n);
        self.turn_num -= 1;
        my_score - their_score
    }

    pub fn GNN_eval(&mut self) -> Eval {
        let graph = self.get_graph();
        self.gnn.evaluate(&graph).unwrap_or(0.0) as Eval
    }
}
