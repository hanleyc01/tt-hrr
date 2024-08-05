//! Implementation of memory storage systems.
use crate::holo::*;
use ndarray::concatenate;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use std::collections::HashMap;

/// Trait which implements all of the important functions that a memory
/// system ought to: take in some `T`, and be able to store it recall
/// it in a reasonable manner.
trait Memory<T> {
    type Output;

    /// Memorize some data.
    fn memorize(&mut self, data: &T);

    /// Recall some data.
    fn recall(&self, data: &T) -> Self::Output;
}

/// Simple cleanup memory.
#[derive(Debug)]
struct SimpleCleanup {
    memory: Array2<f32>,
    trace_dimensionality: usize,
    init_max_traces: usize,
    trace_increment: usize,
    current_traces: usize,
}

impl SimpleCleanup {
    /// Initialize a new [SimpleCleanup], where the memory matrix is
    /// an `n x m` matrix, `n` is the trace dimensionality, and
    /// the initial maximum amount of traces and the increment for
    /// adding more traces is `m`.
    pub fn new(n: usize, m: usize) -> Self {
        let memory = Array2::zeros((n, m));
        let trace_dimensionality = n;
        let init_max_traces = m;
        let trace_increment = m;
        let current_traces = 0;
        Self {
            memory,
            trace_dimensionality,
            init_max_traces,
            trace_increment,
            current_traces,
        }
    }

    /// Initialize a new [SimpleCleanup] with default settings of `n = 512`,
    /// and `m = 100`. To initialize with separate arguments, see [Self::new].
    pub fn init() -> Self {
        Self::new(512, 100)
    }
}

impl Memory<Holo> for SimpleCleanup {
    type Output = Holo;

    fn memorize(&mut self, data: &Holo) {
        if self.current_traces > self.init_max_traces {
            self.memory = concatenate![
                Axis(0),
                self.memory,
                Array2::zeros((self.trace_dimensionality, self.init_max_traces))
            ];
        }
        self.memory
            .slice_mut(s![self.current_traces, ..])
            .assign(&data.v);
        self.current_traces += 1;
    }

    fn recall(&self, data: &Holo) -> Self::Output {
        let activations = self.memory.dot(&data.v);
        let argmax = activations.argmax().unwrap();
        let arr = self.memory.slice(s![argmax, ..]).to_owned();
        Holo::from_data(arr, data.large, data.small)
    }
}

/// Associated list mapping strings to HRR's.
pub struct Lexicon {
    mapping: Vec<(String, Holo)>,
    reverse: Vec<(Holo, String)>,
}

impl Lexicon {
    /// Initialize a new associative array; note that this struct 
    /// keeps a reverse of itself in memory, in order to make 
    /// reverse lookups easier.
    pub fn new() -> Self {
        let mapping = Vec::new();
        let reverse = Vec::new();
        Self { mapping, reverse }
    }

    /// Retrieve an item from the lexicon given a key.
    pub fn get(&mut self, key: String) -> Option<&Holo> {
        for (x, y) in self.mapping.iter() {
            if x == &key {
                return Some(y)
            }
        }

        None
    }

    /// Add an item to the internal associative array.
    pub fn set(&mut self, key: String, item: Holo) {
        let mut mappings: Vec<(String, Holo)> = self.mapping.iter()
            .filter(|(x, y)| x != &key)
            .map(|(x, y)| (x.to_owned(), y.to_owned()))
            .collect();
        let mut reverse: Vec<(Holo, String)> = self.reverse.iter()
            .filter(|(x, y)| x != &item)
            .map(|(x, y)| (x.to_owned(), y.to_owned()))
            .collect();

        mappings.push((key.clone(), item.clone()));
        reverse.push((item, key));
        self.mapping = mappings;
        self.reverse = reverse;
    }
}

impl Default for Lexicon {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Iterator for &'a Lexicon {
    type Item = &'a (String, Holo);

    fn next(&mut self) -> Option<Self::Item> {
        let mut map_iter = self.mapping.iter();
        map_iter.next()
    }
}

/// Reverse of a Lexicon.
type ReverseLexicon = Vec<(Holo, String)>;

struct SimpleAssoc {
    a: Lexicon,
    lexicon: Lexicon,
    reverse: ReverseLexicon,
    theta: f32,
}

impl SimpleAssoc {
    /// Initialize a new [SimpleAssoc] with the provided parameters.
    pub fn new(
        a: Lexicon, 
        lexicon: Lexicon, 
        reverse: ReverseLexicon, 
        theta: f32
    ) -> Self {
        Self {
            a,
            lexicon,
            reverse,
            theta,
        }
    }

    /// Initialize a new [SimpleAssoc] with default parameters.
    pub fn init(a: Lexicon) -> Self {
        let lexicon = Lexicon::new();
        let reverse = lexicon.into_iter()
            .map(|(x, y)| (y.to_owned(), x.to_owned()))
            .collect();
        let theta = 0.2;
        Self::new(a, lexicon, reverse, theta)
    }
}

impl Memory<Holo> for SimpleAssoc {
    type Output = Holo;

    fn memorize(&mut self, data: &Holo) {
        todo!()
    }

    fn recall(&self, data: &Holo) -> Self::Output {
        todo!()
    }
}
