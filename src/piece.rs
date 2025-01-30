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
}

/* bit 0 = color
 * bits 1-3 = bug
 * bits 4-5 = number
*/
#[bitfield]
#[derive(Clone, Copy)]
pub struct Piece {
    #[bits = 1] is_some: bool,
    #[bits = 1] color: Color,
    #[bits = 3] ptype: Pct,
    num: B2,
    #[skip] _unused: B1,
}

impl Piece {
    fn make(color: Color, pct: Pct, num: u8) -> Self {
        let p_num = num & 0x3; // 2 bits
        Self::new()
            .with_is_some(true)
            .with_color(color)
            .with_ptype(pct)
            .with_num(p_num)
    }

    fn empty() -> Self {
        Self::new().with_is_some(false)
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
