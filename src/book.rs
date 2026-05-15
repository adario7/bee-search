use lazy_static::lazy_static;
use std::collections::HashMap;
use crate::board::{Board, Action};

lazy_static! {
    pub static ref OPENING_BOOK: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        m.insert("", "wL");
        m.insert("wL", "bL /wL");
        m.insert("wL;bL /wL", "wM wL/");
        m.insert("wL;bL /wL;wM wL/", "bM /bL");
        m.insert("wP", "bL /wP");
        m.insert("wP;bL /wP", "wQ wP/");
        m.insert("wL;bL wL-", "wM -wL");
        m.insert("wL;bL wL-;wA1 -wL", "bM bL-");
        m.insert("wP;bM /wP", "wQ wP/");
        m.insert("wL;bG1 /wL", "wA1 wL/");
        m.insert("wL;bL /wL;wQ wL-", "bQ -bL");
        m.insert("wL;bL /wL;wA1 wL/", "bM /bL");
        m.insert("wL;bL /wL;wA1 wL/;bQ bL\\", "wQ \\wL");
        m.insert("wP;bL wP-", "wQ -wP");
        m.insert("wG1", "bL /wG1");
        m.insert("wL;bP /wL", "wA1 wL/");
        m.insert("wL;bL /wL;wM wL/;bQ bL\\", "wQ \\wL");
        m.insert("wL;bL wL-;wM -wL", "bM bL-");
        m.insert("wL;bL /wL;wA1 wL/;bM /bL", "wQ \\wL");
        m.insert("wL;bL /wL;wA1 wL/;bM /bL;wQ \\wL", "bQ bL\\");
        m.insert("wL;bL wL\\", "wM \\wL");
        m.insert("wM", "bL /wM");
        m.insert("wL;bL wL\\;wM \\wL", "bM bL\\");
        m.insert("wL;bL /wL;wQ \\wL", "bQ bL\\");
        m.insert("wP;bL /wP;wQ wP/", "bM /bL");
        m.insert("wG1;bL /wG1", "wA1 wG1/");
        m.insert("wL;bL /wL;wQ \\wL;bQ bL\\", "wA1 wL-");
        m
    };
}

pub fn get_book_move(board: &Board) -> Option<Action> {
    let log = board.game_log_string();
    let move_str = OPENING_BOOK.get(log.as_str())?;
    board.parse_action(move_str).ok()
}
