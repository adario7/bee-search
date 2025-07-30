use crate::board::Board;
use crate::tile::{adjacent, Tile, GRID_SIZE, TILE_ZERO};
use crate::piece_type::{PCT_COUNT};

impl Board {

    // Converts a tile to a feature vector for graph neural networks
    fn tile_to_feature(&self, tile: Tile) -> Vec<f32> {
        let piece = self.tile(tile);
        let mut feat = vec![0.0; 2 + PCT_COUNT + 1];
        // Color one-hot
        feat[piece.color().index()] = 1.0;
        // Piece type one-hot
        feat[2 + piece.ptype().index()] = 1.0;
        // Stack height (normalized)
        feat[2 + PCT_COUNT] = (self.height(tile) as f32) / 4.0;
        feat
    }

    fn empty_tile_feature() -> Vec<f32> {
        vec![0.0; 2 + PCT_COUNT + 1] // Empty tile feature
    }

    // Converts the board state into a graph representation
    pub fn get_graph(&self) -> (Vec<Tile>, Vec<Vec<bool>>, Vec<Vec<f32>>) {

        let mut ids: [i16; GRID_SIZE] = [-1; GRID_SIZE];

        let start: Tile = *self.occupied_tiles[self.color().index()].first().unwrap_or_else(
            || self.occupied_tiles[self.color().index()].first().unwrap_or(&TILE_ZERO));

        let mut nodes = Vec::new();
        let mut features = Vec::new();
        nodes.push(start);
        ids[start as usize] = 0;
        if self.tile(start).is_some() {
            features.push(self.tile_to_feature(start));
        } else {
            features.push(Board::empty_tile_feature());
        }

        let mut i = 0;

        // BFS to find all occupied tiles and tiles adjacent to them
        while i < nodes.len() {
            let tile = nodes[i];

            if self.tile(tile).is_none() {
                i += 1;
                continue;
            }

            for adj in adjacent(tile) {
                if ids[adj as usize] == -1 {
                    if self.tile(adj).is_some() {
                        ids[adj as usize] = nodes.len() as i16;
                        nodes.push(adj);
                        features.push(self.tile_to_feature(adj));
                    } else {
                        ids[adj as usize] = nodes.len() as i16;
                        nodes.push(adj);
                        features.push(Board::empty_tile_feature());
                    }
                }
            }
            i += 1;
        }
        
        let mut edges = vec![vec![false; nodes.len()]; nodes.len()];
        for i in 0..nodes.len() {
            for adj in adjacent(nodes[i]) {
                if ids[adj as usize] != -1 {
                    edges[i][ids[adj as usize] as usize] = true;
                }
            }
        }

        (nodes, edges, features)
    }
}





