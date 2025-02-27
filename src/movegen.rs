use std::cmp::{min, max};
use lazy_static::lazy_static;
use std::collections::VecDeque;

use crate::abstractions::{TileBitmask, ActionContainer};
use crate::board::{Action, Board};
use crate::piece::{Color, Piece};
use crate::piece_type::PieceType;
use crate::tile::{adjacent, Direction, Tile, GRID_SIZE, TILE_ZERO};
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
                let i = dir.index();
                let prev = (i + 5)%6;
                let next = (i + 1)%6;
                
                if !occupied[i] && (occupied[prev] ^ occupied[next]) {
                    directions[bitmask].push(*dir);
                }
            }
        }

        directions
    };
/*
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
    */
}


impl Board {
    
    // The function sets the correct value of tiles_placeable[player][tile] according to the current state of the board
    // tiles_placeable[player][tile] is true iff the player can place a bug on the tile
    fn set_placeable(&mut self, tile : Tile, player : Color) {
        // there is already a piece the tile cannot be placeable
        if self.tile(tile).is_some() {
            self.tiles_placeable[player.index()].set_bit(tile, false);
            return;
        }

        // flg checks if there is at least one tile with the same color adjacent to the current one
        let mut flg = false;
        for adj in adjacent(tile){
            let adj_piece = self.tile(adj);

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

        self.tiles_placeable[player.index()].set_bit(tile, flg);
    }

    fn set_all_placeables(&mut self, player : Color) {
        let siz = self.occupied_tiles[player.index()].len();
        for i in 0..siz {
            let tile = self.occupied_tiles[player.index()][i];
            for adj in adjacent(tile) {
                self.set_placeable(adj, player);
            }
        }
    }

    fn generate_placements(&mut self, moves : &mut ActionContainer) {

        let player = self.color();

        if self.turn_history.len() < 2 {

            if self.turn_history.len() == 0 {
                for bug in PieceType::iter_all() {
                    if self.placeable[player.index()][bug.index()] > 0 && bug != PieceType::Queen {
                        moves.push(Place(TILE_ZERO as u16, bug));
                    }
                }
            }else {
                for tile in adjacent(TILE_ZERO) {
                    for bug in PieceType::iter_all() {
                        if self.placeable[player.index()][bug.index()] > 0 && bug != PieceType::Queen {
                            moves.push(Place(tile as u16, bug));
                        }
                    }
                }
            }
            return;
        }

        // right now it is slow, ideally we want to call set_placeable every time we modify the board only on the cells adjacent to the modification
        self.set_all_placeables(player);
        
        let mut vis = TileBitmask::new(false);
        for &tile in self.occupied_tiles[player.index()].iter() {
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
                            if self.placeable[player.index()][bug.index()] > 0 {
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

            for adj in adjacent(node) {
                if world[adj as usize].is_none() {
                    continue;
                }
                
                if dis[adj as usize] == 0 {
                    num_children += 1;
                    let adj_d = dfs(world, dis, cut, adj, p+1);
                    if adj_d >= dis[node as usize] {
                        cut.set_bit(node, true);
                    }
                    min_dis = min(min_dis, adj_d);
                }else {
                    min_dis = min(min_dis, dis[adj as usize]);
                }
            }

            
            if p == 1 {
                cut.set_bit(node, num_children > 1);
            }

            return min_dis;

        }


        

        let start = self.occupied_tiles[0][0];
        dfs(&self.world, &mut dis, &mut cut, start, 1);
        cut

    }


    // probably we need to make another function for the beetle
    fn slidable_adjacent(&self, tile: Tile, origin: Tile) -> std::slice::Iter<'_, Direction> {
        let neighbors = adjacent(tile);
        let mut bitmask = 0;

        for i in 0..6 {
            if self.tile(neighbors[i]).is_some() && neighbors[i] != origin{
                bitmask |= 1 << i;
            }
        }

        SLIDABLE_DIRECTIONS[bitmask].iter()

    }

    fn slidable_adjacent_beetle(&self, tile: Tile, origin: Tile) -> Vec<Direction> {
        let directions = Direction::all();
        let neighbors = adjacent(tile);
        let mut moves: Vec<Direction> = Vec::new();

        let mut curr_height = self.height(tile);

        if tile == origin {
            curr_height -= 1;
        }

        let mut heights = [0i32;6];
        for i in 0..6{
            heights[i] = self.height(neighbors[i]);
            if neighbors[i] == origin {
                heights[i] -= 1;
            }
        }

        for i in 0..6 {
            let prev = (i + 5)%6;
            let next = (i + 1)%6;

            let max_height = max(curr_height, heights[i]);
            let min_height = min(heights[prev], heights[next]);

            if max_height >= min_height && (curr_height > 0 || heights[prev] + heights[i] + heights[next] > 0) {
                moves.push(directions[i]);
            }

        }

        moves
    }

    fn generate_walk1(&self, origin: Tile, moves : &mut ActionContainer) {

        for dir in self.slidable_adjacent(origin , origin) {
            moves.push(Move(origin, origin + *dir));
        }

    }

    fn generate_beetle(&self, origin: Tile, moves : &mut ActionContainer) {
        for dir in self.slidable_adjacent_beetle(origin, origin) {
            moves.push(Move(origin, origin + dir));
        }
    }

    fn generate_spider(&self, origin: Tile, moves : &mut ActionContainer) {

        let mut vis = TileBitmask::new(false);
        vis.set_bit(origin, true);

        for d1 in self.slidable_adjacent(origin, origin) {
            let t1 = origin + *d1;
            for d2 in self.slidable_adjacent(t1, origin) {
                let t2 = t1 + *d2;
                if t2 != origin {
                    for d3 in self.slidable_adjacent(t2, origin) {
                        let t3 = t2 + *d3;
                        if t3 != t1 && !vis.is_on(t3) {
                            moves.push(Move(origin, t3));
                            vis.set_bit(t3, true);
                        }
                    }
                }
            }
        }
    }

    fn generate_grasshopper(&self, origin : Tile, moves : &mut ActionContainer) {
        
        for dir in Direction::all() {
            let mut next = origin + *dir;

            if self.tile(next).is_some() {

                while self.tile(next).is_some() {
                    next = next + *dir;
                }

                moves.push(Move(origin, next));
            }
        }
    }

    fn generate_ladybug(&self, origin: Tile, moves : &mut ActionContainer) {
        let mut vis = TileBitmask::new(false);
        vis.set_bit(origin, true);

        for d1 in self.slidable_adjacent_beetle(origin, origin) {
            let t1 = origin + d1;
            if self.tile(t1).is_some() {
                for d2 in self.slidable_adjacent_beetle(t1, origin) {
                    let t2 = t1 + d2;
                    if t2 != origin && self.tile(t2).is_some() {
                        for d3 in self.slidable_adjacent_beetle(t2, origin) {
                            let t3 = t2 + d3;
                            if t3 != t1 && self.tile(t3).is_none() && !vis.is_on(t3) {
                                moves.push(Move(origin, t3));
                                vis.set_bit(t3, true);
                            }
                        }
                    }
                }
            }
        }
    }
    
    fn generate_ant(&self, origin: Tile, moves : &mut ActionContainer) {

        let mut vis = TileBitmask::new(false);
        vis.set_bit(origin, true);

        let mut queue: VecDeque<Tile> = VecDeque::new();
        queue.push_back(origin);


        while let Some(tile) = queue.pop_front() {
            
            for dir in self.slidable_adjacent(tile, origin) {
                if !vis.is_on(tile + *dir) {
                    moves.push(Move(origin, tile + *dir));
                    vis.set_bit(tile + *dir, true);
                    queue.push_back(tile + *dir);
                }
            }

        }

    }
    
    fn generate_pillbug(&self, cut_vertices: &TileBitmask, origin: Tile, moves : &mut ActionContainer){
        if !cut_vertices.is_on(origin) {
            self.generate_walk1(origin, moves);
        }

        for start in adjacent(origin) {
            for end in adjacent(origin) {
                if start != end && !cut_vertices.is_on(start) && self.height(start) == 1 && self.height(end) == 0{
                    moves.push(Move(start, end));
                }
            }
        }
    }

    fn generate_mosquito(&self, cut_vertices: &TileBitmask, origin: Tile, moves : &mut ActionContainer) {

        if self.height(origin) > 1 {
            self.generate_beetle(origin, moves);
            return;
        }

        let mut adjacent_pieces = [false; 8];
        for adj in adjacent(origin) {
            if self.tile(adj).is_some() {
                adjacent_pieces[self.tile(adj).ptype().index()] = true;
            }
        }

        if adjacent_pieces[PieceType::Pillbug.index()] {
            self.generate_pillbug(cut_vertices, origin, moves);
        }

        if cut_vertices.is_on(origin) {
            return;
        }

        if adjacent_pieces[PieceType::Ant.index()] {
            self.generate_ant(origin, moves);
        }else {
            if adjacent_pieces[PieceType::Queen.index()] || adjacent_pieces[PieceType::Pillbug.index()] {
                self.generate_walk1(origin, moves);
            }
            if adjacent_pieces[PieceType::Spider.index()] {
                self.generate_spider(origin, moves);
            }
        }

        if adjacent_pieces[PieceType::Beetle.index()] {
            self.generate_beetle(origin, moves);
        }

        if adjacent_pieces[PieceType::Grasshopper.index()] {
            self.generate_grasshopper(origin, moves);   
        }

        if adjacent_pieces[PieceType::Ladybug.index()] {
            self.generate_ladybug(origin, moves);
        }

    }

    pub fn generate_moves(&mut self) -> Vec<Action> {

        let mut moves = ActionContainer::new();
        self.generate_placements(&mut moves);

        if self.turn_history.len() < 2 || self.placeable[self.color().index()][PieceType::Queen as usize] > 0 {
            return moves.moves;
        }

        let mut cut_vertices = self.find_cut_vertices();

        let stunned = match self.turn_history.last() {
            Some(Action::Move(_, dest)) => Some(dest),
            _ => None,
        };
        if let Some(moved) = stunned {
            cut_vertices.set_bit(*moved, true);
        }

        for tile in self.occupied_tiles[self.color().index()].iter() {
            if Some(tile) == stunned {
                continue;
            }
            
            if !cut_vertices.is_on(*tile){
                match self.tile(*tile).ptype() {
                    PieceType::Queen => self.generate_walk1(*tile, &mut moves),
                    PieceType::Grasshopper => self.generate_grasshopper(*tile, &mut moves),
                    PieceType::Spider => self.generate_spider(*tile, &mut moves),
                    PieceType::Ant => self.generate_ant(*tile, &mut moves),
                    PieceType::Beetle => self.generate_beetle(*tile, &mut moves),
                    PieceType::Mosquito => self.generate_mosquito(&cut_vertices, *tile, &mut moves),
                    PieceType::Ladybug => self.generate_ladybug(*tile, &mut moves),
                    PieceType::Pillbug => self.generate_pillbug(&cut_vertices, *tile, &mut moves)
                };
            }
            else if self.tile(*tile).ptype() == PieceType::Mosquito {
                self.generate_mosquito(&cut_vertices, *tile, &mut moves);
            }
            else if self.tile(*tile).ptype() == PieceType::Pillbug {
                self.generate_pillbug(&cut_vertices, *tile, &mut moves);
            }
            else if self.tile(*tile).ptype() == PieceType::Beetle && self.height(*tile) > 1 {
                self.generate_beetle(*tile, &mut moves);
            }
        }

        moves.moves
    }
}

#[test]
fn test_first_move(){
    let mut board = Board::new();
    
    board.do_action(Place(TILE_ZERO, PieceType::Ant));
    board.do_action(Place(TILE_ZERO + Direction::E, PieceType::Ant));

    for action in board.generate_moves() {
        println!("{:?}", action);
    }

}
