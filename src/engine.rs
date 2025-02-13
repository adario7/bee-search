use crate::board::{Action, Board};
use std::time::Duration;

pub type Depth = u8;

pub struct Engine {
}

impl Engine {
    pub fn new() -> Self {
        Self { }
    }

    pub fn best_move(&mut self, board: &mut Board, _max_depth: Depth, _max_time: Duration) -> Action {
        board.generate_moves().into_iter()
            .map(|mv| {
                board.do_action(mv);
                let score = -board.static_eval(); // after playing the move the eval is from the POV of the opponent, since it's his turn
                board.undo_action();
                (mv, score)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap().0
    }
}
