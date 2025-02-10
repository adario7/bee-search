mod tile;
mod piece_type;
mod piece;
mod board;
mod movegen;
mod uhp;
mod board_io;
mod abstractions;

fn main() {
    let mut uhp = uhp::Uhp::new();
    uhp.io_loop();
}
