use crate::board::Board;
use crate::tile::{GRID_SIZE, adjacent};
use crate::piece_type::PCT_COUNT;

pub struct GameGraph{
    pub features: Vec<[i8; 5]>, // [color + ptype, height, fixed, buried, move_count]
    pub edges: Vec<[u8; 3]>, // [src, dst, type]
}

impl GameGraph {
    pub fn new() -> Self {
        GameGraph {
            features: Vec::new(),
            edges: Vec::new(),
        }
    }
}

impl Board {
    // Converts the board state into a graph representation
    // Each piece (on the ground or in a stack) is a node.
    pub fn get_graph(&mut self) -> GameGraph {
        let mut features = Vec::new();
        let mut edges = Vec::new();

        let num_moves_by_tile = self.generate_movements_by_tile();
        let cut_vertexes = self.find_cut_vertexes();
        let mut node_indices = [0u8; GRID_SIZE]; // Store start node index for each tile index
        
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
                if let Some(underworld) = self.underworld.get(&hex) {
                    for (i, piece) in underworld.iter().enumerate() {
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
            }

            let color_bin = (top_piece.color() != self.color()) as i8;
            let r_ptype = top_piece.ptype().index() as i8;
            features.push([
                color_bin * (PCT_COUNT as i8) + r_ptype, 
                (h - 1) as i8, 
                is_fixed as i8, 
                0, 
                num_moves_by_tile[hex as usize] as i8
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
