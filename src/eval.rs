use crate::{board::Board, piece::Color, tile::adjacent};

pub type Eval = i16;

impl Board {
    fn queen_score(&self, color: Color) -> Eval {
        match self.queens[color.index()] {
            Some(tile) => adjacent(tile).iter().filter(|&&t| self.tile(t).is_none()).count() as Eval,
            None => 5
        }
    }

    pub fn static_eval(&self) -> Eval {
        let me = self.color();
        let them = me.other();
        self.queen_score(me) - self.queen_score(them)
    }
}
