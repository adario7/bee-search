use std::cell::UnsafeCell;

use crate::{board::Action, engine::Depth, eval::{Eval, Value}};

#[derive(Default, Copy, Clone, Debug, Eq, PartialEq)]
pub enum TTFlag {
    #[default]
    Null = 0,
    LowerBound = 1,
    UpperBound = 2,
    Exact = 3,
    OnlyEval = 4,
}

//#[repr(packed)] TODO: test consequences of this
#[derive(Default, Copy, Clone, Debug)]
pub struct TEntry {
    pub hash: u32,
    pub pv: Action,
    pub value: Value,
    pub eval: Option<Eval>,
    pub depth: Depth,
    pub flag: TTFlag,
}

pub struct TTable {
    buf: Vec<UnsafeCell<TEntry>>,
    mask: usize,
}

unsafe impl Send for TTable {}
unsafe impl Sync for TTable {}

fn check_bits(hash: u64) -> u32 {
    (hash >> 32) as u32
}

impl TTable {
    pub fn new(byte_size: usize) -> Self {
        let size = (byte_size / std::mem::size_of::<TEntry>()).next_power_of_two();
        Self {
            buf: (0..size)
                .map(|_| UnsafeCell::new(TEntry::default()))
                .collect(),
            mask: size - 1,
        }
    }

    fn index(&self, hash: u64) -> usize {
        hash as usize & self.mask
    }

    pub fn get(&self, hash: u64) -> Option<TEntry> {
        let idx = self.index(hash);
        let entry = unsafe { *self.buf[idx].get() };
        if entry.flag == TTFlag::Null || entry.hash != check_bits(hash) {
            return None;
        }
        Some(entry)
    }

    pub fn put(&self, hash: u64, alpha: Value, beta: Value, pv: Action, value: Value, eval: Option<Eval>, depth: Depth) {
        let flag = if value >= beta {
            TTFlag::LowerBound
        } else if value <= alpha {
            TTFlag::UpperBound
        } else {
            TTFlag::Exact
        };
        self.put_entry(self.index(hash), TEntry {
            hash: check_bits(hash),
            pv,
            value,
            eval,
            depth,
            flag,
        });
    }

    pub fn put_entry(&self, idx: usize, entry: TEntry) {
        let slot = unsafe { &mut *self.buf[idx].get() };
        // on collision always keep the new data, to prevent stale entries from living too long
        // if there is no collising keep the deepest entry
        // use >= instead of > to prefer frasher entries in case of a reasearch
        let collision = slot.hash != entry.hash;
        // overwrite on collisions, at the root node, or if we got a deeper entry
        if collision || entry.depth >= slot.depth {
            let prev_eval = slot.eval;
            *slot = entry;
            if !collision && entry.eval.is_none() && prev_eval.is_some() { // don't forget the eval!
                slot.eval = prev_eval;
            }
        }
    }

    pub fn put_eval(&self, hash: u64, eval: Eval) {
        let idx = self.index(hash);
        let slot = unsafe { &mut *self.buf[idx].get() };
        if slot.flag == TTFlag::Null { // free spot
            slot.hash = check_bits(hash);
            slot.flag = TTFlag::OnlyEval;
            slot.eval = Some(eval);
        } else if slot.hash == check_bits(hash) && slot.eval.is_none() { // existing entry missing eval
            slot.eval = Some(eval);
        }
    }

    pub fn clear_one(&self, hash: u64) {
        let idx = self.index(hash);
        let slot = unsafe { &mut *self.buf[idx].get() };
        *slot = Default::default();
    }

    pub fn clear(&mut self) {
        for entry in &mut self.buf {
            *entry = Default::default();
        }
    }
}
