use std::io::stdin;
use std::time::Duration;

use crate::board::Board;
use crate::engine::{Depth, Engine};
use crate::perft;
use crate::graph_nn::GameGraph;

pub struct Uhp {
    board: Board,
    engine: Engine
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
        Uhp {
            board: Board::new(),
            engine: Engine::new()
        }
    }

    fn info(&mut self) -> UhpResult<()> {
        let version = env!("CARGO_PKG_VERSION");
        let hash = env!("GIT_HASH");
        let profile = if cfg!(debug_assertions) {
            "-DEBUG"
        } else {
            ""
        };
        println!("id bee-search {}-{}{}", version, hash, profile);
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
        if !self.board.is_legal(m) {
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
            (99, time)
        } else {
            return Err(UhpError::SyntaxError(args.to_string()));
        };
        let m = self.engine.best_move(&mut self.board, depth, time);
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
        // TODO: print options
    }

    fn get_option(&mut self, option: &str) -> UhpResult<()> {
        Err(UhpError::InvalidOption(option.into()))
    }

    fn set_option(&mut self, option: &str, _value: &str) -> UhpResult<()> {
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
        let depth = args.parse::<u8>().unwrap_or(5);
        if let Some(eval) = self.engine.eval(&mut self.board, depth){
            println!("{}", eval);
            Ok(())
        }else{
            Err(UhpError::EngineError("Engine unable to find evaluation".to_owned()))
        }
        
    }

    fn print_graph(&self) -> UhpResult<()> {
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
            "graph" => self.print_graph(),
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

