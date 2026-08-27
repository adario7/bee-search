use crate::board::{Action, ActionList, Board};
use crate::movegen::OtherMoves;
use crate::piece::Color;
use crate::tile::{adjacent, tile_to_loc, GRID_SIZE};
use crate::piece_type::{Pct, PieceType, PCT_COUNT};

pub const NUM_TOKENS: usize = 28;
pub const TOKENS_PER_SIDE: usize = 14;
pub const TOKEN_FEAT_DIM: usize = 8;
pub const ADJ_MATRIX_SIZE: usize = NUM_TOKENS * NUM_TOKENS;

pub const TOT_QTY: [u8; 8] = [1, 3, 2, 3, 2, 1, 1, 1];

pub const PCT_OFFSETS: [usize; 8] = [
    0,  // Queen (1)
    1,  // Grasshopper (3)
    4,  // Spider (2)
    6,  // Ant (3)
    9,  // Beetle (2)
    11, // Mosquito (1)
    12, // Ladybug (1)
    13, // Pillbug (1)
];

#[inline]
pub fn piece_to_token_idx(color: Color, pct: Pct, num: u8) -> usize {
    debug_assert!(num >= 1 && num <= TOT_QTY[pct.index()]);
    let base = if color == Color::Black { TOKENS_PER_SIDE } else { 0 };
    base + PCT_OFFSETS[pct.index()] + (num as usize - 1)
}

#[inline]
pub fn token_idx_to_piece(idx: usize) -> (Color, Pct, u8) {
    debug_assert!(idx < NUM_TOKENS);
    let color = if idx >= TOKENS_PER_SIDE { Color::Black } else { Color::White };
    let local = idx % TOKENS_PER_SIDE;
    for (pct_idx, &offset) in PCT_OFFSETS.iter().enumerate().rev() {
        if local >= offset {
            let pct = PieceType::iter_all().nth(pct_idx).unwrap();
            let num = (local - offset + 1) as u8;
            return (color, pct, num);
        }
    }
    unreachable!()
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeType {
    None = 0,
    NW = 1,
    NE = 2,
    E = 3,
    SE = 4,
    SW = 5,
    W = 6,
    StackUp = 7,
    StackDown = 8,
    SelfLoop = 9,
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
pub struct TokenGraph {
    /// 28 tokens x 8 features
    /// Tokens 0..13: White pieces, Tokens 14..27: Black pieces
    /// [0]: on_board (1.0 if on board, 0.0 if in hand)
    /// [1]: x_centered (x - mean_queen_x, 0.0 if in hand)
    /// [2]: y_centered (y - mean_queen_y, 0.0 if in hand)
    /// [3]: height (0.0 for ground/hand, 1.0 for level 1, etc.)
    /// [4]: movable (1.0 if can move/place, 0.0 otherwise)
    /// [5]: legal_moves (count of legal move destinations / placements)
    /// [6]: is_pinned (1.0 if cut vertex on ground, 0.0 otherwise)
    /// [7]: is_buried (1.0 if covered in stack, 0.0 otherwise)
    pub features: [[f32; TOKEN_FEAT_DIM]; NUM_TOKENS],
    /// 28 x 28 adjacency matrix
    pub adj: [[u8; NUM_TOKENS]; NUM_TOKENS],
}

impl TokenGraph {
    pub fn new() -> Self {
        Self {
            features: [[0.0; TOKEN_FEAT_DIM]; NUM_TOKENS],
            adj: [[EdgeType::None as u8; NUM_TOKENS]; NUM_TOKENS],
        }
    }
}

// Backward compatibility for UHP `graph` command
pub struct GameGraph {
    pub features: Vec<[i8; 5]>, // [color + ptype, height, fixed, buried, move_count]
    pub edges: Vec<[u8; 3]>,    // [src, dst, type]
}

impl GameGraph {
    pub fn new() -> Self {
        GameGraph {
            features: Vec::new(),
            edges: Vec::new(),
        }
    }
}

#[derive(Clone, Copy)]
struct PlacedPieceInfo {
    token_idx: usize,
    tile: u16,
    loc_x: f32,
    loc_y: f32,
    height: usize,
    is_top: bool,
    is_buried: bool,
}

impl Board {
    pub fn get_token_graph(&mut self) -> TokenGraph {
        let my_moves_by_tile = self.generate_movements_by_tile();
        let my_place_count = self.generate_placements_n();
        let other = self.other_moves();
        self.get_token_graph_with_moves(my_moves_by_tile, my_place_count, &other)
    }

    /// Optimized version of `get_token_graph` that directly utilizes `my_moves` and `other` already computed,
    /// extracting `my_moves_by_tile` in O(|moves|) without regenerating.
    pub fn get_token_graph_fast(
        &mut self,
        my_moves: &ActionList,
        other: &OtherMoves,
    ) -> TokenGraph {
        let mut my_moves_by_tile = [0u16; GRID_SIZE];
        let mut my_place_count = 0usize;

        for mv in my_moves.iter() {
            match mv {
                Action::Move(from, _) => {
                    my_moves_by_tile[*from as usize] += 1;
                }
                Action::Place(_, _) => {
                    my_place_count += 1;
                }
                Action::Pass => {}
            }
        }

        self.get_token_graph_with_moves(my_moves_by_tile, my_place_count, other)
    }

    fn get_token_graph_with_moves(
        &mut self,
        my_moves_by_tile: [u16; GRID_SIZE],
        my_place_count: usize,
        other: &OtherMoves,
    ) -> TokenGraph {
        let mut tg = TokenGraph::new();

        let cut_vertexes = self.find_cut_vertexes();
        let white_queen_placed = self.queens[Color::White.index()].is_some();
        let black_queen_placed = self.queens[Color::Black.index()].is_some();

        let (
            white_moves_by_tile,
            black_moves_by_tile,
            white_place_count,
            black_place_count,
            white_queen_required,
            black_queen_required,
        ) = match self.color() {
            Color::White => (
                my_moves_by_tile,
                other.moves_by_tile,
                my_place_count,
                other.place_count,
                self.queen_required(),
                other.queen_required,
            ),
            Color::Black => (
                other.moves_by_tile,
                my_moves_by_tile,
                other.place_count,
                my_place_count,
                other.queen_required,
                self.queen_required(),
            ),
        };

        // 1. Scan board and underworld to find all placed pieces
        let mut placed = Vec::with_capacity(NUM_TOKENS);
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;

        for hex in self.all_occupied_tiles() {
            let h = self.height(hex) as usize;
            let (loc_y, loc_x) = tile_to_loc(hex);
            let fx = loc_x as f32;
            let fy = loc_y as f32;

            if h > 1 {
                for (i, p) in self.underworld_at(hex).iter().enumerate() {
                    let token_idx = piece_to_token_idx(p.color(), p.ptype(), p.num());
                    placed.push(PlacedPieceInfo {
                        token_idx,
                        tile: hex,
                        loc_x: fx,
                        loc_y: fy,
                        height: i,
                        is_top: false,
                        is_buried: true,
                    });
                    sum_x += fx;
                    sum_y += fy;
                }
            }

            let top = self.tile(hex);
            let token_idx = piece_to_token_idx(top.color(), top.ptype(), top.num());
            placed.push(PlacedPieceInfo {
                token_idx,
                tile: hex,
                loc_x: fx,
                loc_y: fy,
                height: h - 1,
                is_top: true,
                is_buried: false,
            });
            sum_x += fx;
            sum_y += fy;
        }

        let n_placed = placed.len();
        // Shift coordinates with respect to the average (x, y) of the two Queens
        let white_q = self.queens[Color::White.index()];
        let black_q = self.queens[Color::Black.index()];

        let (mean_x, mean_y) = match (white_q, black_q) {
            (Some(q1), Some(q2)) => {
                let (y1, x1) = tile_to_loc(q1);
                let (y2, x2) = tile_to_loc(q2);
                ((x1 as f32 + x2 as f32) * 0.5, (y1 as f32 + y2 as f32) * 0.5)
            }
            (Some(q1), None) => {
                let (y1, x1) = tile_to_loc(q1);
                (x1 as f32, y1 as f32)
            }
            (None, Some(q2)) => {
                let (y2, x2) = tile_to_loc(q2);
                (x2 as f32, y2 as f32)
            }
            (None, None) => {
                if n_placed > 0 {
                    (sum_x / n_placed as f32, sum_y / n_placed as f32)
                } else {
                    (0.0, 0.0)
                }
            }
        };

        // 2. Populate features for placed pieces
        let mut is_placed = [false; NUM_TOKENS];

        for p in &placed {
            is_placed[p.token_idx] = true;
            let is_black = p.token_idx >= TOKENS_PER_SIDE;

            let is_pinned = p.height == 0 && cut_vertexes.get(p.tile) && self.height(p.tile) == 1;

            let (movable, legal_moves) = if p.is_buried {
                (0.0, 0.0)
            } else if !is_black {
                if white_queen_placed {
                    let moves = white_moves_by_tile[p.tile as usize] as f32;
                    (if moves > 0.0 { 1.0 } else { 0.0 }, moves)
                } else {
                    (0.0, 0.0)
                }
            } else {
                if black_queen_placed {
                    let moves = black_moves_by_tile[p.tile as usize] as f32;
                    (if moves > 0.0 { 1.0 } else { 0.0 }, moves)
                } else {
                    (0.0, 0.0)
                }
            };

            tg.features[p.token_idx][0] = 1.0; // on_board
            tg.features[p.token_idx][1] = p.loc_x - mean_x; // x_centered
            tg.features[p.token_idx][2] = p.loc_y - mean_y; // y_centered
            tg.features[p.token_idx][3] = p.height as f32; // height
            tg.features[p.token_idx][4] = movable; // movable
            tg.features[p.token_idx][5] = legal_moves; // legal_moves
            tg.features[p.token_idx][6] = if is_pinned { 1.0 } else { 0.0 }; // is_pinned
            tg.features[p.token_idx][7] = if p.is_buried { 1.0 } else { 0.0 }; // is_buried
        }

        // 3. Populate features for pieces in hand
        for idx in 0..NUM_TOKENS {
            if is_placed[idx] {
                continue;
            }
            let (color, pct, _num) = token_idx_to_piece(idx);
            let (movable, legal_moves) = if color == Color::White {
                if self.placeable[Color::White.index()][pct.index()] > 0 {
                    if white_queen_required && pct != Pct::Queen {
                        (0.0, 0.0)
                    } else {
                        (if white_place_count > 0 { 1.0 } else { 0.0 }, white_place_count as f32)
                    }
                } else {
                    (0.0, 0.0)
                }
            } else {
                if self.placeable[Color::Black.index()][pct.index()] > 0 {
                    if black_queen_required && pct != Pct::Queen {
                        (0.0, 0.0)
                    } else {
                        (if black_place_count > 0 { 1.0 } else { 0.0 }, black_place_count as f32)
                    }
                } else {
                    (0.0, 0.0)
                }
            };

            tg.features[idx][0] = 0.0; // on_board
            tg.features[idx][1] = 0.0; // x_centered
            tg.features[idx][2] = 0.0; // y_centered
            tg.features[idx][3] = 0.0; // height
            tg.features[idx][4] = movable;
            tg.features[idx][5] = legal_moves;
            tg.features[idx][6] = 0.0;
            tg.features[idx][7] = 0.0;
        }

        // 4. Populate 28x28 Adjacency Matrix
        // Self-loops for all placed tokens
        for p in &placed {
            tg.adj[p.token_idx][p.token_idx] = 9; // Self
        }

        // Pairwise edges between placed tokens
        for i in 0..n_placed {
            let pi = placed[i];
            for j in 0..n_placed {
                if i == j {
                    continue;
                }
                let pj = placed[j];

                if pi.tile == pj.tile {
                    // Stacking edge
                    if pj.height == pi.height + 1 {
                        tg.adj[pi.token_idx][pj.token_idx] = 7; // Stack-Up
                    } else if pi.height == pj.height + 1 {
                        tg.adj[pi.token_idx][pj.token_idx] = 8; // Stack-Down
                    }
                } else if pi.is_top && pj.is_top {
                    // Directional hex adjacency between top pieces
                    for (dir_idx, &adj_tile) in adjacent(pi.tile).iter().enumerate() {
                        if adj_tile == pj.tile {
                            tg.adj[pi.token_idx][pj.token_idx] = 1 + dir_idx as u8;
                            break;
                        }
                    }
                }
            }
        }

        tg
    }

    // Converts the board state into legacy graph representation for UHP
    pub fn get_graph(&mut self) -> GameGraph {
        let mut features = Vec::new();
        let mut edges = Vec::new();

        let num_moves_by_tile = self.generate_movements_by_tile();
        let cut_vertexes = self.find_cut_vertexes();
        let mut node_indices = [0u8; GRID_SIZE];
        let mut next_node_idx = 0u8;

        for hex in self.all_occupied_tiles() {
            node_indices[hex as usize] = next_node_idx;
            let h = self.height(hex) as usize;
            next_node_idx += h as u8;
        }

        let mut current_node_idx = 0u8;

        for hex in self.all_occupied_tiles() {
            let h = self.height(hex) as usize;
            let top_piece = self.world[hex as usize];
            let is_fixed = cut_vertexes.get(hex) && h == 1;

            if h > 1 {
                for (i, piece) in self.underworld_at(hex).iter().enumerate() {
                    let color_bin = (piece.color() != self.color()) as i8;
                    let r_ptype = piece.ptype().index() as i8;
                    let is_buried = 1;
                    let mut fixed_val = 0;
                    if i == 0 && cut_vertexes.get(hex) {
                        fixed_val = 1;
                    }
                    features.push([color_bin * (PCT_COUNT as i8) + r_ptype, i as i8, fixed_val, is_buried, 0]);
                    edges.push([current_node_idx, current_node_idx + 1, 6]);
                    current_node_idx += 1;
                }
            }

            let color_bin = (top_piece.color() != self.color()) as i8;
            let r_ptype = top_piece.ptype().index() as i8;
            features.push([
                color_bin * (PCT_COUNT as i8) + r_ptype,
                (h - 1) as i8,
                is_fixed as i8,
                0,
                num_moves_by_tile[hex as usize] as i8,
            ]);

            for (dir, &adj) in adjacent(hex).iter().enumerate() {
                if self.occupied(adj) {
                    let adj_top_idx = node_indices[adj as usize] + (self.height(adj) as u8) - 1;
                    edges.push([current_node_idx, adj_top_idx, dir as u8]);
                }
            }

            current_node_idx += 1;
        }

        GameGraph { features, edges }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piece_token_mapping() {
        for color in [Color::White, Color::Black] {
            for pct in PieceType::iter_all() {
                let max_num = TOT_QTY[pct.index()];
                for num in 1..=max_num {
                    let idx = piece_to_token_idx(color, pct, num);
                    assert!(idx < NUM_TOKENS);
                    let (c, p, n) = token_idx_to_piece(idx);
                    assert_eq!(c, color);
                    assert_eq!(p, pct);
                    assert_eq!(n, num);
                }
            }
        }
    }

    #[test]
    fn test_token_graph_initial_board() {
        let mut board = Board::new();
        let tg = board.get_token_graph();
        for i in 0..NUM_TOKENS {
            assert_eq!(tg.features[i][0], 0.0); // all in hand
            assert_eq!(tg.features[i][1], 0.0);
            assert_eq!(tg.features[i][2], 0.0);
        }
        for i in 0..NUM_TOKENS {
            for j in 0..NUM_TOKENS {
                assert_eq!(tg.adj[i][j], 0);
            }
        }
    }

    #[test]
    fn test_token_graph_game_moves() {
        let pos = "Base+MLP;InProgress;White[3];wS1;bA1 wS1-;wA1 -wS1;bA2 bA1\\";
        let mut board = Board::parse_game_string(pos).unwrap();
        let tg = board.get_token_graph();

        let mut on_board_count = 0;
        for i in 0..NUM_TOKENS {
            if tg.features[i][0] == 1.0 {
                on_board_count += 1;
            }
        }
        assert_eq!(on_board_count, 4);

        // Check self loops
        for i in 0..NUM_TOKENS {
            if tg.features[i][0] == 1.0 {
                assert_eq!(tg.adj[i][i], 9);
            }
        }
    }
}
