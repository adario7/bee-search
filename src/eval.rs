use crate::{board::Board, tile::adjacent};

pub type Eval = i16;

impl Board {
    fn queen_score(&self) -> Eval {
        let color = self.color();
        match self.queens[color.index()] {
            Some(tile) => adjacent(tile).iter().filter(|&&t| self.tile(t).is_none()).count() as Eval,
            None => 5
        }
    }

    fn moves_score(&mut self) -> Eval {
        self.generate_moves().len() as Eval
    }

    fn score(&mut self) -> Eval {
        1000 * self.queen_score() + self.moves_score()
    }

    pub fn static_eval(&mut self) -> Eval {
        let my_score = self.score();
        self.turn_num += 1;
        let their_score = self.score();
        self.turn_num -= 1;
        my_score - their_score
    }
}
