use std::io::stdin;
use std::sync::Arc;
use std::time::Duration;

use crate::board::Board;
use crate::engine::{Depth, Engine};
use crate::eval::FEATURES_EVAL;
use crate::movegen::CutVertexes;
use crate::perft;

const FORCE_ST: bool = false;
const MAX_THREADS: usize = if FORCE_ST { 1 } else { 32 };

pub struct Uhp {
    board: Board,
    engine: Arc<Engine>,
    num_threads: usize,
}

#[derive(Debug)]
pub enum UhpError {
    IoError(std::io::Error),
    UnknownPiece(String),
    InvalidGameString(String),
    InvalidGameType(String),
    InvalidMove(String),
    InvalidOption(String),
    UnrecognizedCommand(String),
    SyntaxError(String),
    EngineError(String),
    TooManyUndos,
}

impl From<std::io::Error> for UhpError {
    fn from(error: std::io::Error) -> Self {
        UhpError::IoError(error)
    }
}

pub type UhpResult<T> = std::result::Result<T, UhpError>;

impl Uhp {
    pub fn new() -> Uhp {
        let n = num_cpus::get();
        Uhp {
            board: Board::new(),
            engine: Arc::new(Engine::new()),
            num_threads: n.min(MAX_THREADS)
        }
    }

    fn info(&mut self) -> UhpResult<()> {
        let version = env!("CARGO_PKG_VERSION");
        let hash = env!("GIT_HASH");
        print!("id bee-search {}-{}", version, hash);
        #[cfg(feature = "gnn")]
        {
            print!("-GNN");
        }
        if FEATURES_EVAL { print!("-F"); }
        if FORCE_ST { print!("-ST"); }
        if cfg!(debug_assertions) { print!("-DEBUG"); }
        println!();
        println!("Mosquito;Ladybug;Pillbug");
        Ok(())
    }

    fn new_game(&mut self, args: &str) -> UhpResult<()> {
        self.board = Board::parse_game_string(args)?;
        println!("{}", self.board.game_string());
        Ok(())
    }

    fn play(&mut self, args: &str) -> UhpResult<()> {
        let m = self.board.parse_action(args)?;
        if !self.board.is_legal(m, &mut immovable_vertexes) {
            return Err(UhpError::InvalidMove(args.to_string()));
        }
        self.board.do_action(m); // TODO: check for illegal moves
        println!("{}", self.board.game_string());
        Ok(())
    }

    fn valid_moves(&mut self) -> UhpResult<()> {
        println!("{}", self.board.legal_moves_string());
        Ok(())
    }

    fn parse_hhmmss(time: &str) -> Option<Duration> {
        let mut toks = time.split(':');
        let hours = toks.next().unwrap_or("").parse::<u64>().ok()?;
        let minutes = toks.next().unwrap_or("").parse::<u64>().ok()?;
        let seconds = toks.next().unwrap_or("").parse::<u64>().ok()?;
        Some(Duration::from_secs(hours * 3600 + minutes * 60 + seconds))
    }

    fn best_move(&mut self, args: &str) -> UhpResult<()> {
        let (depth, time) = if let Some(arg) = args.strip_prefix("depth ") {
            let depth = arg.parse::<Depth>().map_err(|_| UhpError::SyntaxError(args.to_string()))?;
            (depth, Duration::from_secs(99999))
        } else if let Some(arg) = args.strip_prefix("time ") {
            let time = Self::parse_hhmmss(arg).ok_or_else(|| UhpError::SyntaxError(args.to_string()))?;
            // safety margin
            let time = time - Duration::from_millis((2.0 + 2.7*(self.num_threads as f64).sqrt()).round() as u64);
            (30, time)
        } else {
            return Err(UhpError::SyntaxError(args.to_string()));
        };
        let (_, m) = self.engine.clone().best_move(&self.board, depth, time, self.num_threads, &mut immovable_vertexes);
        println!("{}", self.board.action_to_string(m));
        Ok(())
    }

    fn undo(&mut self, args: &str) -> UhpResult<()> {
        let num_undo = if args.is_empty() {
            1
        } else {
            args.parse::<usize>().map_err(|_| UhpError::SyntaxError(args.to_string()))?
        };
        if num_undo > self.board.turn_history.len() {
            return Err(UhpError::TooManyUndos);
        }
        for _ in 0..num_undo {
            self.board.undo_action();
        }
        println!("{}", self.board.game_string());
        Ok(())
    }

    fn print_options(&mut self) {
        println!("NumThreads;int;{};{};1;{}", self.num_threads, num_cpus::get(), MAX_THREADS);
    }

    fn get_option(&mut self, option: &str) -> UhpResult<()> {
        if option == "NumThreads" {
            println!("{}", self.num_threads);
            return Ok(());
        }
        Err(UhpError::InvalidOption(option.into()))
    }

    fn set_option(&mut self, option: &str, value: &str) -> UhpResult<()> {
        if option == "NumThreads" {
            let value = value.parse::<usize>().map_err(|_| UhpError::SyntaxError(value.into()))?;
            if value > 0 && value <= MAX_THREADS {
                self.num_threads = value;
                self.print_options();
                return Ok(());
            }
        }
        Err(UhpError::InvalidOption(option.into()))
    }

    fn options(&mut self, args: &str) -> UhpResult<()> {
        let tokens = args.split(' ').collect::<Vec<_>>();
        if args.is_empty() {
            self.print_options();
        } else if tokens.len() == 2 && tokens[0] == "get" {
            self.get_option(tokens[1])?;
        } else if tokens.len() == 3 && tokens[0] == "set" {
            self.set_option(tokens[1], tokens[2])?;
        } else {
            return Err(UhpError::SyntaxError(args.into()));
        }
        Ok(())
    }

    fn perft(&mut self, args: &str) -> UhpResult<()> {
        let depth = args.parse::<usize>().unwrap_or(8);
        perft::display_perft(&mut self.board, depth);
        Ok(())
    }

    fn eval(&mut self, args: &str) -> UhpResult<()> {
        let depth = args.parse::<u8>().unwrap_or(0);
        let score = if depth == 0 {
            self.board.static_eval(&mut immovable_vertexes)
        } else {
            self.engine.clone().best_move(&mut self.board, depth, Duration::from_secs(99999), self.num_threads, &mut immovable_vertexes).0
        };
        println!("{}", score);
        Ok(())
    }

    #[cfg(feature = "gnn")]
    fn print_graph(&self) -> UhpResult<()> {
        use crate::piece::Piece;

        let graph = self.board.get_graph();
        println!("{} ", graph.nodes.len());
        for node in &graph.nodes {
            println!("{} ", node);
        }
        for feature in &graph.features {
            for f in feature {
                print!("{} ", f);
            }
            println!();
        }
        println!("{}", graph.edges.len());
        for i in 0..graph.edges.len() {
            for j in 0..graph.edges[i].len() {
                print!("{} ", graph.edges[i][j] as u8);
            }
            println!()
        }

        // Additional features

        for node in &graph.nodes {
            println!("{} ", self.board.height(*node));
            if self.board.tile(*node) == Piece::empty() {
                continue;
            }

            println!("{} {} ",self.board.tile(*node).color().index() ^ self.board.color().index(), self.board.tile(*node).ptype().index());

            if let Some(underworld) = self.board.underworld.get(node) {
                for piece in underworld {
                    println!("{} {} ", piece.color().index() ^ self.board.color().index(), piece.ptype().index());
                }
            }
        }
        Ok(())
    }

    fn features(&mut self) -> UhpResult<()> {
        let f = self.board.features(&mut immovable_vertexes);
        println!("{}", f.iter().map(|&x| x.to_string()).collect::<Vec<_>>().join(";"));
        Ok(())
    }

    fn static_eval(&mut self) -> UhpResult<()> {
        let score = self.board.static_eval(&mut immovable_vertexes);
        println!("{}", score);
        Ok(())
    }

    // https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#engine-commands
    fn command(&mut self, line: &str) {
        let line = line.trim();
        let space = line.find(' ');
        let command = if let Some(i) = space { &line[..i] } else { line };
        let args = if let Some(i) = space { &line[i + 1..] } else { "" };
        let result = match command {
            "info" => self.info(),
            "newgame" => self.new_game(args),
            "play" => self.play(args),
            "pass" => self.play("pass"),
            "validmoves" => self.valid_moves(),
            "bestmove" => self.best_move(args),
            "undo" => self.undo(args),
            "options" => self.options(args),
            // secret commands
            "perft" => self.perft(args),
            "eval" => self.eval(args),
            #[cfg(feature = "gnn")]
            "graph" => self.print_graph(),
            "features" => self.features(),
            "static_eval" => self.static_eval(),
            _ => Err(UhpError::UnrecognizedCommand(command.to_string())),
        };
        if let Err(err) = result {
            if let UhpError::InvalidMove(invalid) = err {
                println!("invalidmove {}", invalid);
            } else {
                println!("err {:?}", err);
            }
        }
    }

    pub fn io_loop(&mut self) {
        self.info().unwrap();
        println!("ok");
        loop {
            let mut line = String::new();
            match stdin().read_line(&mut line) {
                Ok(size) => {
                    if size == 0 {
                        return;
                    }
                }
                Err(err) => {
                    eprintln!("{}", err);
                    return;
                }
            };
            self.command(&line);
            println!("ok");
        }
    }
}

