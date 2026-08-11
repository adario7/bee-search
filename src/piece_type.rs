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

    /// Inverse of `index()`. A dense match rather than a `transmute`: this
    /// enum carries no `#[repr]`, so transmuting an integer into it would be
    /// relying on an unspecified layout. LLVM folds this to the identity.
    #[inline]
    pub fn from_index(i: u8) -> Self {
        match i {
            0 => PieceType::Queen,
            1 => PieceType::Grasshopper,
            2 => PieceType::Spider,
            3 => PieceType::Ant,
            4 => PieceType::Beetle,
            5 => PieceType::Mosquito,
            6 => PieceType::Ladybug,
            _ => {
                debug_assert_eq!(i, 7, "piece type index out of range");
                PieceType::Pillbug
            }
        }
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
