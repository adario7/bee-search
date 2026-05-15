use std::io::stdin;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::time::Duration;

use crate::board::Board;
use crate::engine::{Depth, Engine};
use crate::eval::{FEATURES_EVAL, MLP_EVAL};
use crate::perft;

const FORCE_ST: bool = false;
const MAX_THREADS: usize = if FORCE_ST { 1 } else { 32 };
const PONDER_RESP: bool = true;

pub struct Uhp {
    board: Board,
    engine: Arc<Engine>,
    num_threads: usize,
    ponder_enabled: bool,
    ponder_abort: Option<Arc<AtomicBool>>,
    time_total: f64,
    time_increment: f64,
    time_available: f64,
    otb_relay_time: f64,
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
            num_threads: n.min(MAX_THREADS),
            ponder_enabled: true,
            ponder_abort: None,
            time_total: 0.0,
            time_increment: 0.0,
            time_available: 0.0,
            otb_relay_time: 2.0,
        }
    }

    fn stop_ponder(&mut self) {
        if let Some(abort) = self.ponder_abort.take() {
            abort.store(true, Ordering::Relaxed);
        }
    }

    fn start_ponder(&mut self) {
        self.stop_ponder();
        if self.ponder_enabled && self.ponder_abort.is_none() {
            let mut board = self.board.clone();

            if PONDER_RESP {
                if let Some(pv_move) = self.engine.get_pv(&board) {
                    eprintln!("pondering on reponse {}...", board.action_to_string(pv_move));
                    board.do_action(pv_move);
                } else {
                    eprintln!("pondering on the opponent's board (no pv available)...");
                }
            } else {
                eprintln!("pondering on the opponent's board...");
            }

            let abort = Arc::new(AtomicBool::new(false));
            self.ponder_abort = Some(abort.clone());
            let engine = self.engine.clone();
            let num_threads = self.num_threads;
            std::thread::spawn(move || {
                engine.best_move(&mut board, 30, Duration::from_secs(3600), num_threads, false, Some(abort));
            });
        }
    }

    fn info(&mut self) -> UhpResult<()> {
        let version = env!("CARGO_PKG_VERSION");
        let hash = env!("GIT_HASH");
        print!("id bee-search {}-{}", version, hash);
        if MLP_EVAL { print!("-MLP"); }
        else if FEATURES_EVAL { print!("-F"); }
        if FORCE_ST { print!("-ST"); }
        if cfg!(debug_assertions) { print!("-DEBUG"); }
        println!();
        println!("Mosquito;Ladybug;Pillbug");
        Ok(())
    }

    fn new_game(&mut self, args: &str) -> UhpResult<()> {
        self.stop_ponder();
        self.board = Board::parse_game_string(args)?;
        if self.time_total > 0.0 {
            self.time_available = self.time_total;
        }
        println!("{}", self.board.game_string());
        Ok(())
    }

    fn play(&mut self, args: &str) -> UhpResult<()> {
        let m = self.board.parse_action(args)?;
        if !self.board.is_legal(m) {
            return Err(UhpError::InvalidMove(args.to_string()));
        }
        self.board.do_action(m);
        println!("{}", self.board.game_string());
        self.start_ponder();
        Ok(())
    }

    fn valid_moves(&mut self) -> UhpResult<()> {
        println!("{}", self.board.legal_moves_string());
        Ok(())
    }

    fn parse_hhmmss(time: &str) -> Option<Duration> {
        let toks: Vec<&str> = time.split(':').collect();
        if toks.len() == 2 {
            let m = toks[0].parse::<u64>().ok()?;
            let s = toks[1].parse::<u64>().ok()?;
            Some(Duration::from_secs(m * 60 + s))
        } else if toks.len() == 3 {
            let h = toks[0].parse::<u64>().ok()?;
            let m = toks[1].parse::<u64>().ok()?;
            let s = toks[2].parse::<u64>().ok()?;
            Some(Duration::from_secs(h * 3600 + m * 60 + s))
        } else {
            None
        }
    }

    fn best_move(&mut self, args: &str) -> UhpResult<()> {
        self.stop_ponder();
        let mut manage_time = false;
        let (depth, time) = if let Some(arg) = args.strip_prefix("depth ") {
            let depth = arg.parse::<Depth>().map_err(|_| UhpError::SyntaxError(args.to_string()))?;
            (depth, Duration::from_secs(99999))
        } else if let Some(arg) = args.strip_prefix("time ") {
            manage_time = self.time_total > 0.0;
            let time = if manage_time {
                let tau = 30.0;
                let real_inc = (self.time_increment - self.otb_relay_time).max(0.0);
                let allocated = ((self.time_available + tau * real_inc) / tau).max(1.0);
                eprintln!("available time: {:.2}s, allocated time: {:.2}s", self.time_available, allocated);
                Duration::from_secs_f64(allocated)
            } else {
                Self::parse_hhmmss(arg).ok_or_else(|| UhpError::SyntaxError(args.to_string()))?
            };
            // safety margin
            let time = time.saturating_sub(Duration::from_millis((2.0 + 2.7*(self.num_threads as f64).sqrt()).round() as u64));
            (30, time)
        } else {
            return Err(UhpError::SyntaxError(args.to_string()));
        };
        let start_time = std::time::Instant::now();
        let (_, m) = self.engine.clone().best_move(&mut self.board, depth, time, self.num_threads, true, None);
        let used_time = start_time.elapsed().as_secs_f64();
        if manage_time {
            self.time_available += - used_time - self.otb_relay_time + self.time_increment;
        }
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
        self.stop_ponder();
        println!("{}", self.board.game_string());
        Ok(())
    }

    fn print_options(&mut self) {
        println!("NumThreads;int;{};{};1;{}", self.num_threads, num_cpus::get(), MAX_THREADS);
        println!("Ponder;bool;{};true", if self.ponder_enabled { "true" } else { "false" });
        println!("TimeTotal;int;{};0;0;100000000", self.time_total as i64);
        println!("TimeIncrement;int;{};0;0;100000000", self.time_increment as i64);
        println!("OtbRelayTime;double;{};0.0;0.0;100000000.0", self.otb_relay_time);
    }

    fn get_option(&mut self, option: &str) -> UhpResult<()> {
        if option == "NumThreads" {
            println!("{}", self.num_threads);
            return Ok(());
        } else if option.eq_ignore_ascii_case("Ponder") {
            println!("{}", if self.ponder_enabled { "true" } else { "false" });
            return Ok(());
        } else if option.eq_ignore_ascii_case("TimeTotal") {
            println!("{}", self.time_total as i64);
            return Ok(());
        } else if option.eq_ignore_ascii_case("TimeIncrement") {
            println!("{}", self.time_increment as i64);
            return Ok(());
        } else if option.eq_ignore_ascii_case("OtbRelayTime") {
            println!("{}", self.otb_relay_time);
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
        } else if option.eq_ignore_ascii_case("Ponder") {
            if value.eq_ignore_ascii_case("true") {
                self.ponder_enabled = true;
                return Ok(());
            } else if value.eq_ignore_ascii_case("false") {
                self.ponder_enabled = false;
                self.stop_ponder();
                return Ok(());
            }
        } else if option.eq_ignore_ascii_case("TimeTotal") {
            let value = value.parse::<f64>().map_err(|_| UhpError::SyntaxError(value.into()))?;
            self.time_total = value;
            self.time_available = value;
            return Ok(());
        } else if option.eq_ignore_ascii_case("TimeIncrement") {
            let value = value.parse::<f64>().map_err(|_| UhpError::SyntaxError(value.into()))?;
            self.time_increment = value;
            return Ok(());
        } else if option.eq_ignore_ascii_case("OtbRelayTime") {
            let value = value.parse::<f64>().map_err(|_| UhpError::SyntaxError(value.into()))?;
            self.otb_relay_time = value;
            return Ok(());
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

    fn set_time(&mut self, args: &str) -> UhpResult<()> {
        if let Some(t) = Self::parse_hhmmss(args) {
            self.time_available = t.as_secs_f64();
            Ok(())
        } else {
            Err(UhpError::SyntaxError(args.to_string()))
        }
    }

    fn perft(&mut self, args: &str) -> UhpResult<()> {
        self.stop_ponder();
        let depth = args.parse::<usize>().unwrap_or(8);
        perft::display_perft(&mut self.board, depth);
        Ok(())
    }

    fn eval(&mut self, args: &str) -> UhpResult<()> {
        self.stop_ponder();
        let depth = args.parse::<u8>().unwrap_or(0);
        let score = if depth == 0 {
            self.board.static_eval()
        } else {
            self.engine.clone().best_move(&mut self.board, depth, Duration::from_secs(99999), self.num_threads, true, None).0
        };
        println!("{}", score);
        Ok(())
    }

    fn print_graph(&self) -> UhpResult<()> {
        let mut board = self.board.clone();
        let graph = board.get_graph();
        println!("{} ", graph.features.len());
        for feature in &graph.features {
            for f in feature {
                print!("{} ", f);
            }
            println!();
        }
        println!("{}", graph.edges.len());
        for i in 0..graph.edges.len() {
            println!("{} {} {}", graph.edges[i][0], graph.edges[i][1], graph.edges[i][2]);
        }
        Ok(())
    }

    fn features(&mut self) -> UhpResult<()> {
        let mv = self.board.generate_moves();
        let f = self.board.features_fast(&mv);
        println!("{}", f.iter().map(|&x| x.to_string()).collect::<Vec<_>>().join(";"));
        Ok(())
    }

    fn static_eval(&mut self) -> UhpResult<()> {
        let score = self.board.static_eval();
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
            "time" => self.set_time(args),
            "perft" => self.perft(args),
            "eval" => self.eval(args),
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

