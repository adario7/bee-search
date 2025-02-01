use crate::abstractions::TileBitmask;
use crate::board::{Action, Board};
use crate::piece::Color;
use crate::piece_type::PieceType;
use crate::tile::{Tile, adjacent};
use crate::movegen::Action::{Place, Move};

impl Board {
    
    // The functions sets the correct value of tiles_placeable[player][tile] according to the current state of the board
    fn set_placeable(&mut self, tile : Tile, player : Color) {
        // there is already a piece the tile cannot be placeable
        if self.world[tile as usize].is_some() {
            self.tiles_placeable[player.index()].set_bit(tile, false);
            return;
        }

        // flg checks if there is at least one tile with the same color adjacent to the current one
        let mut flg = false;
        for adj in adjacent(tile){
            let adj_piece = self.world[adj as usize];

            if adj_piece.is_some() {
                if adj_piece.color() == player {
                    flg = true;
                }
                else {
                    self.tiles_placeable[player.index()].set_bit(tile, false);
                    return;
                }
            }
        }

        if flg {
            self.tiles_placeable[player.index()].set_bit(tile, true);
        }
        else {
            self.tiles_placeable[player.index()].set_bit(tile, false);
        }
    }



    fn generate_placements(&self, moves : &mut Vec<Action>) {
        let player = self.color();
        
        let mut vis = TileBitmask::new(false);
        for &tile in self.occupied_hexes[player.index()].iter() {
            for adj in adjacent(tile) {
                if vis.is_on(adj) {
                    continue;
                }
                vis.set_bit(adj, true);
                
                if self.tiles_placeable[player.index()].is_on(adj) {

                    if self.queen_required() {
                        moves.push(Place(adj, PieceType::Queen));
                    }
                    else {
                    for bug in PieceType::iter_all(){
                            if self.placeable[player.index()][bug as usize] > 0 {
                                moves.push(Place(adj, bug));
                            }
                        }
                    }
                }
            }
        }
    } 

    fn generate_slidable_tiles(){
        todo!()
    }

    pub fn generate_moves(&self) -> Vec<Action> {
        todo!()
    }
}
