#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Piece {
    Queen = 0,
    Grasshopper = 1,
    Spider = 2,
    Ant = 3,
    Beetle = 4,
    Mosquito = 5,
    Ladybug = 6,
    Pillbug = 7,
}

impl Piece {
    pub fn iter_all() -> impl Iterator<Item = Self> {
        [
            Piece::Queen,
            Piece::Grasshopper,
            Piece::Spider,
            Piece::Ant,
            Piece::Beetle,
            Piece::Mosquito,
            Piece::Ladybug,
            Piece::Pillbug,
        ]
        .iter()
        .copied()
    }

    pub fn name(&self) -> &'static str {
        match *self {
            Piece::Queen => "queen",
            Piece::Grasshopper => "grasshopper",
            Piece::Spider => "spider",
            Piece::Ant => "ant",
            Piece::Beetle => "beetle",
            Piece::Mosquito => "mosquito",
            Piece::Ladybug => "ladybug",
            Piece::Pillbug => "pillbug",
        }
    }

    pub fn from_char(c: char) -> Option<Piece> {
        match c.to_ascii_lowercase() {
            'q' => Some(Piece::Queen),
            'g' => Some(Piece::Grasshopper),
            's' => Some(Piece::Spider),
            'a' => Some(Piece::Ant),
            'b' => Some(Piece::Beetle),
            'm' => Some(Piece::Mosquito),
            'l' => Some(Piece::Ladybug),
            'p' => Some(Piece::Pillbug),
            _ => None,
        }
    }

    pub fn to_char(&self) -> char {
        self.name().chars().next().unwrap()
    }
}
