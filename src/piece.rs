use modular_bitfield::prelude::*;
use crate::piece_type::Pct;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[derive(BitfieldSpecifier)]
pub enum Color {
    Black = 1,
    White = 0,
}

impl Color {
    pub fn other(self) -> Color {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
        }
    }

    pub fn from_index(i: u8) -> Self {
        debug_assert!(i < 2);
        unsafe { std::mem::transmute(i) }
    }

    pub fn index(self) -> usize {
        self as usize
    }
}

/* bit 0 = color
 * bits 1-3 = bug
 * bits 4-5 = number
*/
#[bitfield]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Piece {
    #[bits = 1] pub color: Color,
    #[bits = 3] pub ptype: Pct,
    num: B2,
    #[skip] _unused: B2,
}

impl Piece {
    pub fn make(color: Color, pct: Pct, num: u8) -> Self {
        let p_num = num & 0x3; // 2 bits
        Self::new()
            .with_color(color)
            .with_ptype(pct)
            .with_num(p_num)
    }

    pub fn empty() -> Self {
        Self::new()
    }

    pub fn is_some(&self) -> bool {
        self.num() != 0
    }
}

#[test]
fn test_piece_bitset() {
    let p = Piece::make(Color::White, Pct::Ant, 2);
    assert_eq!(p.is_some(), true);
    assert_eq!(p.color(), Color::White);
    assert_eq!(p.ptype(), Pct::Ant);
    assert_eq!(p.num(), 2);
    let p = Piece::empty();
    assert_eq!(p.is_some(), false);
}
