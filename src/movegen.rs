use std::cmp::{min, max};

use crate::{abstractions::TileSet, board::{Action, Board}, piece_type::Pct, tile::{adjacent, Direction, Tile, GRID_SIZE, TILE_ZERO}};

impl Board {
    fn generate_placements(&self, turns: &mut Vec<Action>) {
        let mut no_placement = TileSet::new();
        for &enemy in self.occupied_tiles[self.color().other().index()].iter() {
            for adj in adjacent(enemy) {
                no_placement.set(adj);
            }
        }

        for &friend in self.occupied_tiles[self.color() as usize].iter() {
            for hex in adjacent(friend) {
                if no_placement.get(hex) {
                    continue;
                }
                no_placement.set(hex);
                if self.occupied(hex) {
                    continue;
                }
                let remaining = self.placeable[self.color().index()];
                for bug in Pct::iter_all() {
                    let num_left = remaining[bug as usize];
                    if self.queen_required() && bug != Pct::Queen {
                        continue;
                    }
                    if num_left > 0 {
                        turns.push(Action::Place(hex, bug));
                    }
                }
            }
        }
    }

    // Linear algorithm to find all cut vertexes.
    // Algorithm explanation: https://web.archive.org/web/20180830110222/https://www.eecs.wsu.edu/~holder/courses/CptS223/spr08/slides/graphapps.pdf
    // Example code: https://cp-algorithms.com/graph/cutpoints.html
    pub(crate) fn find_cut_vertexes(&self) -> TileSet {
        struct State<'a> {
            board: &'a Board,
            visited: TileSet,
            immovable: TileSet,
            // Visitation number in DFS traversal.
            num: [u8; GRID_SIZE],
            // Lowest-numbered node reachable using DFS edges and then at most
            // one back edge.
            low: [u8; GRID_SIZE],
            visit_num: u8,
        }
        let mut state = State {
            board: self,
            visited: TileSet::new(),
            immovable: TileSet::new(),
            num: [0; GRID_SIZE],
            low: [0; GRID_SIZE],
            visit_num: 1,
        };
        fn dfs(state: &mut State, hex: Tile, parent: Tile) {
            state.visited.set(hex);
            state.num[hex as usize] = state.visit_num;
            state.low[hex as usize] = state.visit_num;
            state.visit_num += 1;
            let root = hex == parent;
            let mut children = 0;
            for adj in adjacent(hex) {
                if !state.board.occupied(adj) {
                    continue;
                }
                if adj == parent {
                    continue;
                }
                if state.visited.get(adj) {
                    state.low[hex as usize] = min(state.low[hex as usize], state.num[adj as usize]);
                } else {
                    dfs(state, adj, hex);
                    state.low[hex as usize] = min(state.low[hex as usize], state.low[adj as usize]);
                    if state.low[adj as usize] >= state.num[hex as usize] && !root {
                        state.immovable.set(hex);
                    }
                    children += 1;
                }
            }
            if root && children > 1 {
                state.immovable.set(hex);
            }
        }

        let start = *self.occupied_tiles[self.color().index()].first().unwrap_or_else(
            || self.occupied_tiles[self.color().index()].first().unwrap_or(&TILE_ZERO));
        dfs(&mut state, start, start);
        state.immovable
    }

    // For a position on the outside (whether occupied or not), find all
    // adjacent locations still connected to the hive that are slidable.
    // A slidable position has 2 empty slots next to an occupied slot.
    // For all 2^6 possibilities, there can be 0, 2, or 4 slidable neighbors.
    pub(crate) fn slidable_adjacent<'a>(
        &self, neighbors: &'a mut [Tile; 6], origin: Tile, hex: Tile,
    ) -> impl Iterator<Item = Tile> + 'a {
        *neighbors = adjacent(hex);
        // Each bit is whether neighbor is occupied.
        let mut occupied = 0;
        for neighbor in neighbors.iter().rev() {
            occupied <<= 1;
            // Since the origin bug is moving, we can't crawl around it.
            if self.occupied(*neighbor) && *neighbor != origin {
                occupied |= 1;
            }
        }
        // Wrap around in each direction
        occupied |= (occupied << 6) | (occupied << 12);
        let slidable = (!occupied & ((occupied << 1) ^ (occupied >> 1))) >> 6;

        neighbors.iter().enumerate().filter_map(move |(i, &hex)| {
            if (slidable >> i) & 1 != 0 {
                Some(hex)
            } else {
                None
            }
        })
    }

    // Find all walkable tiles where either the source or the dest is on the hive.
    // Unlike what the original rules say (where a beetle on the hive is
    // unrestricted), climbing bugs need to slide into/out of the higher of
    // source or dest heights.
    // https://www.boardgamegeek.com/thread/332467
    fn slidable_adjacent_beetle<'a>(
        &self, out: &'a mut [Tile; 6], orig: Tile, hex: Tile,
    ) -> impl Iterator<Item = Tile> + 'a {
        let mut self_height = self.height(hex);
        if orig == hex {
            self_height -= 1;
        }
        let mut heights = [0; 6];
        let neighbors = adjacent(hex);
        for i in 0..6 {
            heights[i] = self.height(neighbors[i]);
        }

        let mut n = 0;
        for i in 0..6 {
            let barrier = max(self_height, heights[i]);
            if barrier == 0 {
                // Walking at height zero uses regular sliding rules.
                continue;
            }
            if heights[(i + 1) % 6] > barrier && heights[(i + 5) % 6] > barrier {
                // Piles on both sides are too high and we cannot pass through.
                continue;
            }
            out[n] = neighbors[i];
            n += 1;
        }

        out.iter().take(n).copied()
    }

    // From any bug on top of a stack.
    fn generate_stack_walking(&self, hex: Tile, turns: &mut Vec<Action>) {
        let mut buf = [0; 6];
        for adj in self.slidable_adjacent_beetle(&mut buf, hex, hex) {
            turns.push(Action::Move(hex, adj));
        }
    }

    // Jumping over contiguous linear lines of tiles.
    fn generate_jumps(&self, hex: Tile, turns: &mut Vec<Action>) {
        for dir in Direction::all() {
            let mut jump = hex + *dir;
            let mut dist = 1;
            while self.occupied(jump) {
                jump = jump + *dir;
                dist += 1;
                if jump == hex {
                    // Exit out if we'd infinitey loop.
                    dist = 0;
                    break;
                }
            }
            if dist > 1 {
                turns.push(Action::Move(hex, jump));
            }
        }
    }

    fn generate_walk1(&self, hex: Tile, turns: &mut Vec<Action>) {
        let mut buf = [0; 6];
        for adj in self.slidable_adjacent(&mut buf, hex, hex) {
            turns.push(Action::Move(hex, adj));
        }
    }

    fn generate_walk3(&self, orig: Tile, turns: &mut Vec<Action>) {
        let mut buf1 = [0; 6];
        let mut buf2 = [0; 6];
        let mut buf3 = [0; 6];
        let mut visited = TileSet::new();
        visited.set(orig);

        for s1 in self.slidable_adjacent(&mut buf1, orig, orig) {
            for s2 in self.slidable_adjacent(&mut buf2, orig, s1) {
                if s2 != orig {
                    for s3 in self.slidable_adjacent(&mut buf3, orig, s2) {
                        if s3 != s1 && !visited.get(s3) {
                            turns.push(Action::Move(orig, s3));
                            visited.set(s3);
                        }
                    }
                }
            }
        }
    }

    fn generate_walk_all(&self, orig: Tile, turns: &mut Vec<Action>) {
        let mut visited = TileSet::new();
        let mut queue = [0; 16];
        queue[0] = orig;
        let mut qsize = 1;
        let mut buf = [0; 6];
        while qsize > 0 {
            qsize -= 1;
            let node = queue[qsize];
            if visited.get(node) {
                continue;
            }
            visited.set(node);
            if node != orig {
                turns.push(Action::Move(orig, node));
            }
            for adj in self.slidable_adjacent(&mut buf, orig, node) {
                if !visited.get(adj) {
                    queue[qsize] = adj;
                    qsize += 1;
                }
            }
        }
    }

    fn generate_ladybug(&self, hex: Tile, turns: &mut Vec<Action>) {
        let mut buf1 = [0; 6];
        let mut buf2 = [0; 6];
        let mut buf3 = [0; 6];
        let mut step2 = TileSet::new();
        let mut step3 = TileSet::new();
        for s1 in self.slidable_adjacent_beetle(&mut buf1, hex, hex) {
            if self.occupied(s1) {
                for s2 in self.slidable_adjacent_beetle(&mut buf2, hex, s1) {
                    if self.occupied(s2) && !step2.get(s2) {
                        step2.set(s2);
                        for s3 in self.slidable_adjacent_beetle(&mut buf3, hex, s2) {
                            if !self.occupied(s3) && !step3.get(s3) {
                                step3.set(s3);
                                turns.push(Action::Move(hex, s3));
                            }
                        }
                    }
                }
            }
        }
    }

    fn generate_throws(
        &self, immovable: &TileSet, hex: Tile, turns: &mut Vec<Action>, throw_starts: &mut TileSet,
        throw_ends: &mut TileSet,
    ) {
        let mut starts = [0; 6];
        let mut num_starts = 0;
        let mut ends = [0; 6];
        let mut num_ends = 0;
        let mut buf = [0; 6];
        let origin = hex + Direction::NW + Direction::NW; // something not adjacent
        for adj in self.slidable_adjacent_beetle(&mut buf, origin, hex) {
            match self.height(adj) {
                0 => {
                    ends[num_ends] = adj;
                    num_ends += 1;
                }
                1 => {
                    if !immovable.get(adj) {
                        starts[num_starts] = adj;
                        num_starts += 1;
                    }
                }
                _ => {}
            }
        }
        for &start in starts[..num_starts].iter() {
            for &end in ends[..num_ends].iter() {
                turns.push(Action::Move(start, end));
                throw_starts.set(start);
                throw_ends.set(end);
            }
        }
    }

    fn generate_mosquito(&self, hex: Tile, turns: &mut Vec<Action>) {
        let mut targets = [false; 8];
        for adj in adjacent(hex) {
            let node = self.tile(adj);
            if node.is_some() {
                targets[node.ptype() as usize] = true;
            }
        }

        let mut i = turns.len();
        if targets[Pct::Ant as usize] {
            self.generate_walk_all(hex, turns);
        } else {
            // Avoid adding strictly duplicative moves to the ant.
            if targets[Pct::Queen as usize]
                || targets[Pct::Beetle as usize]
                || targets[Pct::Pillbug as usize]
            {
                self.generate_walk1(hex, turns);
            }
            if targets[Pct::Spider as usize] {
                self.generate_walk3(hex, turns);
            }
        }
        if targets[Pct::Grasshopper as usize] {
            self.generate_jumps(hex, turns);
        }
        if targets[Pct::Beetle as usize] {
            self.generate_stack_walking(hex, turns);
        }
        if targets[Pct::Ladybug as usize] {
            self.generate_ladybug(hex, turns);
        }

        // Remove duplicates.
        let mut dests = TileSet::new();
        while i < turns.len() {
            if let Action::Move(_, dest) = turns[i] {
                if dests.get(dest) {
                    turns.swap_remove(i);
                } else {
                    dests.set(dest);
                    i += 1;
                }
            }
        }
    }

    pub(crate) fn generate_movements(&self, turns: &mut Vec<Action>) {
        let mut immovable = self.find_cut_vertexes();
        let stunned = match self.turn_history.last() {
            Some(Action::Move(_, dest)) => Some(dest),
            _ => None,
        };
        if let Some(moved) = stunned {
            // Can't move pieces that were moved on the opponent's turn.
            immovable.set(*moved);
        }

        // Pillbug throws need to be deduped against organic movements, so generate them first.
        let mut throw_starts = TileSet::new();
        let mut throw_ends = TileSet::new();
        let first_move = turns.len();
        let mut marker;
        for &hex in self.occupied_tiles[self.color() as usize].iter() {
            marker = turns.len();
            let node = self.tile(hex);
            if stunned == Some(&hex) {
                continue;
            }
            if node.ptype() == Pct::Pillbug
                || (node.ptype() == Pct::Mosquito
                    && !self.is_stacked(hex)
                    && adjacent(hex).iter().any(|&adj| {
                        let n = self.tile(adj);
                        n.is_some() && n.ptype() == Pct::Pillbug
                    }))
            {
                self.generate_throws(&immovable, hex, turns, &mut throw_starts, &mut throw_ends);
                // Dedup throws from pillbug and mosquito
                if marker > 0 {
                    let mut i = marker;
                    while i < turns.len() {
                        if turns[first_move..marker].contains(&turns[i]) {
                            turns.swap_remove(i);
                        } else {
                            i += 1;
                        }
                    }
                }
            }
        }
        let num_throws = turns.len();

        for &hex in self.occupied_tiles[self.color() as usize].iter() {
            marker = turns.len();
            let node = self.tile(hex);
            if self.is_stacked(hex) {
                self.generate_stack_walking(hex, turns);
                continue;
            }
            if immovable.get(hex) {
                continue;
            }
            match node.ptype() {
                Pct::Queen => self.generate_walk1(hex, turns),
                Pct::Grasshopper => self.generate_jumps(hex, turns),
                Pct::Spider => self.generate_walk3(hex, turns),
                Pct::Ant => self.generate_walk_all(hex, turns),
                Pct::Beetle => {
                    self.generate_walk1(hex, turns);
                    self.generate_stack_walking(hex, turns);
                }
                Pct::Mosquito => self.generate_mosquito(hex, turns),
                Pct::Ladybug => self.generate_ladybug(hex, turns),
                Pct::Pillbug => self.generate_walk1(hex, turns),
            }

            // Dedup against pillbug throws.
            if throw_starts.get(hex) {
                let mut i = marker;
                while i < turns.len() {
                    let turn = turns[i];
                    let end = match turn {
                        Action::Move(_, end) => end,
                        _ => {
                            i += 1;
                            continue;
                        }
                    };
                    if throw_ends.get(end) && turns[first_move..num_throws].contains(&turn) {
                        turns.swap_remove(i);
                    } else {
                        i += 1;
                    }
                }
            }
        }
    }


    pub fn generate_moves(self: &Board) -> Vec<Action> {
        let mut turns: Vec<Action> = Vec::new();
        let remaining = self.placeable[self.color().index()];
        if self.turn_num < 2 {
            // Special case for the first 2 turns:
            for bug in Pct::iter_all() {
                let num_left = remaining[bug as usize];
                if bug == Pct::Queen {
                    // To reduce draws, implement tournament rule where
                    // you can't place your queen first.
                    continue;
                }
                if num_left > 0 {
                    if self.turn_num == 0 {
                        turns.push(Action::Place(TILE_ZERO, bug));
                    } else {
                        for &hex in adjacent(TILE_ZERO).iter() {
                            turns.push(Action::Place(hex, bug));
                        }
                    }
                }
            }
            return turns;
        }
        // Once queen has been placed, pieces may move.
        if remaining[Pct::Queen as usize] == 0 {
            // For movable pieces, generate all legal moves.
            self.generate_movements(&mut turns);
        }
        if remaining.iter().any(|&num| num > 0) {
            // Find placeable positions.
            self.generate_placements(&mut turns);
        }
        if turns.is_empty() {
            turns.push(Action::Pass);
        }

        turns
    }
}

#[test]
fn test_first_move(){
    let mut board = Board::new();
    
    board.do_action(Action::Place(TILE_ZERO, Pct::Ant));
    board.do_action(Action::Place(TILE_ZERO + Direction::E, Pct::Ant));

    for action in board.generate_moves() {
        println!("{:?}", action);
    }

}
