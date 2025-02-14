use std::option;

use crate::uhp::{UhpError, UhpResult};
use crate::tile::{Direction, Tile, GRID_SIZE, TILE_ZERO};
use crate::piece_type::{Pct, PieceType};
use crate::piece::{Color, Piece};
use crate::board::{Action, Board, GameResult};

impl PieceType {
    fn is_numbered(self) -> bool {
        matches!(self, Pct::Ant | Pct::Grasshopper | Pct::Beetle | Pct::Spider)
    }
}

impl Direction {
    fn suffix_name(self) -> &'static str {
        match self {
            Self::SE => "\\",
            Self::E => "-",
            Self::NE => "/",
            _ => "",
        }
    }
    fn prefix_name(self) -> &'static str {
        match self {
            Self::SW => "/",
            Self::W => "-",
            Self::NW => "\\",
            _ => "",
        }
    }
    // returns direction and remaining string
    fn parse(s: &str) -> (&str, Option<Self>) {
        if s.is_empty() {
            return (s, None);
        }
        for &dir in Self::all() {
            if dir.prefix_name() == &s[0..1] {
                return (&s[1..], Some(dir));
            }
            if dir.suffix_name() == &s[s.len() - 1..] {
                return (&s[..s.len() - 1], Some(dir));
            }
        }
        (s, None)
    }
}

impl Color {
    fn parse(c: char) -> Option<Self> {
        match c {
            'w' => Some(Self::White),
            'b' => Some(Self::Black),
            _ => None,
        }
    }
}

// board output
impl Board {
    fn piece_name(&self, p: Piece, out: &mut String) {
        if !p.is_some() {
            out.push_str("??");
            return;
        }
        out.push(p.color().to_char());
        out.push(p.ptype().to_char().to_ascii_uppercase());
        if p.ptype().is_numbered() {
            out.push(char::from_digit(p.num() as u32, 10).unwrap());
        }
    }

    fn new_piece_name(&self, pct: Pct, out: &mut String) {
        self.piece_name(self.make_next_piece(pct), out)
    }

    fn tile_name(&self, origin: Tile, tile: Tile, out: &mut String) {
        let piece = self.tile(tile);    
        if piece.is_some() {
            self.piece_name(piece, out);
            return;
        }
        for dir in Direction::all() {
            let adj = self.tile(tile + *dir);
            if adj.is_some() && tile + *dir != origin {
                out.push_str(dir.opposite().prefix_name());
                self.piece_name(adj, out);
                out.push_str(dir.opposite().suffix_name());
                return;
            }
        }
        out.push_str("??");
    }

    pub fn action_to_string(&self, m: Action) -> String {
        let mut out = String::new();
        match m {
            Action::Move(start, _) => self.piece_name(self.tile(start), &mut out),
            Action::Place(_, pct) => self.new_piece_name(pct, &mut out),
            Action::Pass => return "pass".to_string(),
        }

        if self.turn_num == 0 {
            return out;
        }
        out.push(' ');

        match m {
            Action::Move(start, end) => self.tile_name(start, end, &mut out),
            Action::Place(tile, _) => self.tile_name((GRID_SIZE + 1) as Tile, tile, &mut out), //TODO obrobrioso
            Action::Pass => unreachable!(),
        }
        out
    }

    fn game_result_string(&self) -> &'static str {
        if self.turn_history.is_empty() {
            return "NotStarted";
        }
        match self.game_result() {
            GameResult::InProgress => "InProgress",
            GameResult::WhiteWins => "WhiteWins",
            GameResult::BlackWins => "BlackWins",
            GameResult::Draw => "Draw",
        }
    }

    fn turn_string(&self) -> String {
        format!("{:?}[{}]", self.color().name(), self.turn_history.len() / 2 + 1)
    }


    fn game_log_string(&self) -> String {
        let mut board = Board::parse_game_type(&self.gametype).unwrap();
        let mut log = String::new();
        for &m in &self.turn_history {
            log.push_str(&board.action_to_string(m));
            log.push(';');
            board.do_action(m);
        }
        if log.ends_with(';') {
            log.pop();
        }
        log
    }

    pub fn game_string(&self) -> String {
        let mut out = self.gametype.clone();
        out.push(';');
        out.push_str(self.game_result_string());
        out.push(';');
        out.push_str(&self.turn_string());
        if !self.turn_history.is_empty() {
            out.push(';');
            out.push_str(&self.game_log_string());
        }
        out
    }

    pub fn legal_moves_string(&mut self) -> String {
        self.generate_moves()
            .iter().map(|&m| self.action_to_string(m))
            .collect::<Vec<_>>().join(";")
    }
}

// board input parsing
impl Board {
    fn parse_game_type(game_type: &str) -> UhpResult<Self> {
        let mut options = game_type.split('+');
        
        if options.next() != Some("Base") { // TODO: add support for other game types
            return Err(UhpError::InvalidGameType(game_type.to_owned()));
        }

        let str = options.next().unwrap_or("");
        let mut M = 0u8;
        let mut L = 0u8;
        let mut P = 0u8;

        if str.contains('M') {
            M = 1u8;
        }
        if str.contains('L') {
            L = 1u8;
        }
        if str.contains('P') {
            P = 1u8;
        }

        Ok(Board::new_mlp(M, L, P))
    }

    // parses piece descrption -> (piece, direction), e.g. "wB2-" -> ((White, Beetle, 2), NW)
    fn parse_piece_direction(&self, piece_string: &str) -> Option<(Piece, Option<Direction>)> {
        let (piece_string, dir) = Direction::parse(piece_string);
        let mut chars = piece_string.chars();
        let color = Color::parse(chars.next()?)?;
        let pct = Pct::from_char(chars.next()?)?;
        let num = if pct.is_numbered() {
            char::to_digit(chars.next()?, 10)? as u8
        } else if chars.next().is_some() {
            return None;
        } else {
            1
        };
        Some((Piece::make(color, pct, num), dir))
    }

    fn find_piece(&self, piece: Piece) -> Option<Tile> {
        self.occupied_tiles.iter()
            .flat_map(|v| v.iter()).copied()
            .find(|&t| self.tile(t) == piece)
    }

    fn parse_tile(&self, s: &str) -> Option<Tile> {
        let (end_piece, dir) = self.parse_piece_direction(s)?;
        let tile = self.find_piece(end_piece)?;
        Some(match dir {
            Some(dir) => tile + dir,
            None => tile,
        })
    }

    pub fn parse_action(&self, move_string: &str) -> UhpResult<Action> {
        let err = || UhpError::InvalidMove(move_string.to_owned());
        if move_string == "pass" {
            return Ok(Action::Pass);
        }
        let tokens = move_string.split(' ').collect::<Vec<_>>();
        let (piece, _) = self.parse_piece_direction(tokens[0]).ok_or_else(err)?;
        // first turn alwyas places on staritng tile
        if self.turn_history.is_empty() {
            if tokens.len() != 1 {
                return Err(err());
            }
            return Ok(Action::Place(TILE_ZERO, piece.ptype()));
        }
        // other turns have two operands
        if tokens.len() != 2 {
            return Err(err());
        }
        let start = self.find_piece(piece);
        let end = self.parse_tile(tokens[1]).ok_or_else(err)?;
        Ok(match start {
            Some(start) => Action::Move(start, end),
            None => Action::Place(end, piece.ptype()),
        })
    }

    pub fn parse_game_string(s: &str) -> UhpResult<Self> {
        let mut toks = s.split(';');
        let game_type = toks.next().ok_or_else(|| UhpError::InvalidGameString("missing game type".to_owned()))?;
        let mut board = Board::parse_game_type(game_type)?;
        // game state
        if toks.next().is_none() {
            return Ok(board);
        }
        // turn string
        toks.next().ok_or_else(|| UhpError::InvalidGameString("missing turn string".to_owned()))?;
        // sequence of moves
        for move_string in toks {
            let m = board.parse_action(move_string)?;
            board.do_action(m); // TODO: check move legality
        }
        Ok(board)
    }
}


