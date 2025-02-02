use std::cmp::{min, max};
use lazy_static::lazy_static;

use crate::abstractions::TileBitmask;
use crate::board::{Action, Board};
use crate::piece::{Color, Piece};
use crate::piece_type::PieceType;
use crate::tile::{adjacent, Tile, Direction, GRID_SIZE};
use crate::movegen::Action::{Place, Move};

// precomputation of slidable directions for every possible neighborhood configuration


lazy_static! {

    static ref SLIDABLE_DIRECTIONS: [Vec<Direction>; 1<<6] = {
        let mut directions: [Vec<Direction>; 1<<6] = std::array::from_fn(|_| Vec::new());

        for bitmask in 0..(1<<6) {
            let mut occupied = [false; 6];

            for i in 0..6 {
                occupied[i] = ((bitmask >> i) & 1) > 0;
            }

            for dir in Direction::all() {
                let i = *dir as usize;
                let prev = (i + 5)%6;
                let next = (i + 1)%6;
                
                if !occupied[i] && (occupied[prev] ^ occupied[next]) {
                    directions[bitmask].push(*dir);
                }
            }
        }

        directions
    };

    static ref SLIDABLE_DIRECTIONS_BEETLE: [Vec<Direction>; 1<<6] = {
        let mut directions: [Vec<Direction>; 1<<6] = std::array::from_fn(|_| Vec::new());

        for bitmask in 0..(1<<6) {
            let mut occupied = [false; 6];

            for i in 0..6 {
                occupied[i] = ((bitmask >> i) & 1) > 0;
            }

            todo!()
        }

        directions
    };
}


impl Board {
    
    // The function sets the correct value of tiles_placeable[player][tile] according to the current state of the board
    // tiles_placeable[player][tile] is true iff the player can place a bug on the tile
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

    fn set_all_placeables(&mut self, player : Color) {
        let siz = self.occupied_hexes[player as usize].len();
        for i in 0..siz {
            let tile = self.occupied_hexes[player as usize][i];
            for adj in adjacent(tile) {
                self.set_placeable(tile, player);
            }
        }
    }

    fn generate_placements(&mut self, moves : &mut Vec<Action>) {

        let player = self.color();

        // right now it is slow, ideally we want to call set_placeable every time we modify the board only on the cells adjacent to the modification
        self.set_all_placeables(player);
        
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

    pub fn find_cut_vertices(&self) -> TileBitmask {

        let mut dis = [0; GRID_SIZE];
        let mut cut = TileBitmask::new(false);

        fn dfs(world: &[Piece; GRID_SIZE], dis: &mut [i32; GRID_SIZE], cut: &mut TileBitmask, node: Tile, p: i32) -> i32{

            dis[node as usize] = p;
            let mut min_dis = GRID_SIZE as i32;
            let mut num_children = 0;

            cut.set_bit(node, true);

            for adj in adjacent(node) {
                if world[adj as usize].is_none() {
                    continue;
                }

                if dis[adj as usize] == 0 {
                    num_children += 1;
                    if dfs(world, dis, cut, adj, p+1) < dis[node as usize] {
                        cut.set_bit(node, false);
                    }
                }else {
                    min_dis = min(min_dis, dis[adj as usize]);
                }
            }
            
            if p == 1 {
                cut.set_bit(node, num_children > 1);
            }

            return min_dis;

        }


        

        let start = self.occupied_hexes[0][0];
        dfs(&self.world, &mut dis, &mut cut, start, 1);
        cut

    }


    // probably we need to make another function for the beetle
    fn slidable_adjacent(&self, tile: Tile, origin: Tile) -> std::slice::Iter<'_, Direction> {
        let neighbors = adjacent(tile);
        let mut bitmask = 0;

        for i in 0..6 {
            if self.world[neighbors[i] as usize].is_some() && neighbors[i] != origin{
                bitmask |= 1 << i;
            }
        }

        if self.world[origin as usize].ptype() == PieceType::Beetle {
            SLIDABLE_DIRECTIONS_BEETLE[bitmask].iter()
        }
        else{
            SLIDABLE_DIRECTIONS[bitmask].iter()
        }

    }

    fn generate_slidable_tiles(){
        todo!()
    }

    pub fn generate_moves(&self) -> Vec<Action> {
        todo!()
    }
}
