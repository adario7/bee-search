use modular_bitfield::prelude::*;

pub const PCT_COUNT: usize = 8;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[derive(BitfieldSpecifier)]
pub enum PieceType {
    Queen = 0,
    Grasshopper = 1,
    Spider = 2,
    Ant = 3,
    Beetle = 4,
    Mosquito = 5,
    Ladybug = 6,
    Pillbug = 7,
}

pub type Pct = PieceType;

impl PieceType {
    pub fn index(self) -> usize {
        self as usize
    }

    pub fn iter_all() -> impl Iterator<Item = Self> {
        [
            PieceType::Queen,
            PieceType::Grasshopper,
            PieceType::Spider,
            PieceType::Ant,
            PieceType::Beetle,
            PieceType::Mosquito,
            PieceType::Ladybug,
            PieceType::Pillbug,
        ]
        .iter()
        .copied()
    }

    pub fn name(&self) -> &'static str {
        match *self {
            PieceType::Queen => "queen",
            PieceType::Grasshopper => "grasshopper",
            PieceType::Spider => "spider",
            PieceType::Ant => "ant",
            PieceType::Beetle => "beetle",
            PieceType::Mosquito => "mosquito",
            PieceType::Ladybug => "ladybug",
            PieceType::Pillbug => "pillbug",
        }
    }

    pub fn from_char(c: char) -> Option<PieceType> {
        match c.to_ascii_lowercase() {
            'q' => Some(PieceType::Queen),
            'g' => Some(PieceType::Grasshopper),
            's' => Some(PieceType::Spider),
            'a' => Some(PieceType::Ant),
            'b' => Some(PieceType::Beetle),
            'm' => Some(PieceType::Mosquito),
            'l' => Some(PieceType::Ladybug),
            'p' => Some(PieceType::Pillbug),
            _ => None,
        }
    }

    pub fn to_char(&self) -> char {
        self.name().chars().next().unwrap()
    }
}
