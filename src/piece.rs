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

    pub fn to_char(self) -> char {
        match self {
            Color::White => 'w',
            Color::Black => 'b',
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Color::White => "White",
            Color::Black => "Black",
        }
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
    pub num: B2,
    #[skip] _unused: B2,
}

impl Piece {
    pub fn make(color: Color, pct: Pct, num: u8) -> Self {
        debug_assert_ne!(num, 0);
        debug_assert!(num <= 3);
        Self::new()
            .with_color(color)
            .with_ptype(pct)
            .with_num(num)
    }

    pub fn empty() -> Self {
        Self::new()
    }

    pub fn is_some(&self) -> bool {
        // Equivalent to `self.num() != 0`: `num` is never zero for a real
        // piece (see debug_assert in `make`), and every other bit is zero
        // for an empty piece, so a raw whole-byte compare - no bitfield
        // shift/mask needed - decides it. `occupied()`/`is_some()` are the
        // single most frequently called check in movegen (every adjacent-
        // tile test in every slide), so avoiding that extraction there adds
        // up.
        self.into_bytes() != [0]
    }

    pub fn is_none(&self) -> bool {
        self.into_bytes() == [0]
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
