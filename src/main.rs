mod tile;
mod piece_type;
mod piece;
mod board;
mod movegen;
mod uhp;
mod board_io;
mod abstractions;
mod perft;
mod eval;
mod engine;

fn main() {
    let mut uhp = uhp::Uhp::new();
    uhp.io_loop();
}
