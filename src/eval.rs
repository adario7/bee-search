use crate::{board::{ActionList, Board}, eval_mlp::mlp_inference, piece::Color, tile::adjacent};

pub type Eval = i16;
pub type Value = Eval;

pub const MLP_EVAL: bool = true;
pub const FEATURES_EVAL: bool = false;

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

    pub fn static_eval_fast(&mut self, my_moves: &ActionList) -> Eval {
        if MLP_EVAL {
            let f = self.features_fast(my_moves);
            let score = mlp_inference(&f);
            match self.color() {
                Color::White => score,
                Color::Black => -score,
            }
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

    fn lr_inference(_f: &[i16]) -> Eval {
        0
    }
}

