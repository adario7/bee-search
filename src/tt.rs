use crate::{board::Action, engine::Depth, eval::{Eval, Value}};

#[derive(Default, Copy, Clone, Debug, Eq, PartialEq)]
pub enum TTFlag {
    #[default]
    Null = 0,
    LowerBound = 1,
    UpperBound = 2,
    Exact = 3,
}

//#[repr(packed)] TODO: test consequences of this
#[derive(Default, Copy, Clone, Debug)]
pub struct TEntry {
    pub hash: u64,
    pub pv: Action,
    pub value: Value,
    pub eval: Eval,
    pub depth: Depth,
    pub flag: TTFlag,
}

pub struct TTable {
    buf: Vec<TEntry>,
}

impl TTable {
    pub fn new(size: usize) -> Self {
        Self {
            buf: vec![Default::default(); size],
        }
    }

    pub fn get(&self, hash: u64) -> Option<TEntry> {
        let idx = (hash % self.buf.len() as u64) as usize;
        let entry = self.buf[idx];
        if entry.flag == TTFlag::Null || entry.hash != hash {
            return None;
        }
        Some(entry)
    }

    pub fn put(&mut self, hash: u64, alpha: Value, beta: Value, pv: Action, value: Value, eval: Eval, depth: Depth) {
        let flag = if value >= beta {
            TTFlag::LowerBound
        } else if value <= alpha {
            TTFlag::UpperBound
        } else {
            TTFlag::Exact
        };
        self.put_entry(TEntry {
            hash,
            pv,
            value,
            eval,
            depth,
            flag,
        });
    }

    pub fn put_entry(&mut self, entry: TEntry) {
        let idx = (entry.hash % self.buf.len() as u64) as usize;
        // on collision always keep the new data, to prevent stale entries from living too long
        // if there is no collising keep the deepest entry
        // use >= instead of > to prefer frasher entries in case of a reasearch
        if self.buf[idx].hash != entry.hash || entry.depth >= self.buf[idx].depth {
            self.buf[idx] = entry;
        }
    }

    pub fn clear(&mut self) {
        for entry in &mut self.buf {
            *entry = Default::default();
        }
    }
}
