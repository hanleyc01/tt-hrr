#![allow(unused, dead_code)]

use ndarray::{concatenate, linalg::Dot, prelude::*, IxDynImpl, Shape};
use ndarray_linalg::{normalize, Norm};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
// Extra library that imlements discrete fourier transforms that
// we need for our HRR multiplication.
use ndrustfft::{ndfft_r2c, ndifft_r2c, Complex, R2cFftHandler};

use std::{collections::HashMap, ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub}};

/// Holographic reduced representation.
#[derive(Debug)]
struct Holo {
    small: f32,
    large: f32,
    v: Array1<f32>,
}

impl Holo {
    /// Initialize a zero'd out vector of size `size`.
    pub fn zeros(size: usize, large: f32, small: f32) -> Self {
        let v = Array1::zeros(size);
        Self { v, large, small }
    }

    /// Initialize a normally distributed vector from size `size`.
    pub fn normal(size: usize, large: f32, small: f32) -> Self {
        let sd = 1.0 / ((size as f32).sqrt());
        let v = Array1::random((size), Normal::new(0.0, sd).unwrap());
        Self { v, large, small }
    }

    /// Initialize an Holo from a provided vector.
    pub fn from_data(data: Array1<f32>, large: f32, small: f32) -> Self {
        Self {
            v: data,
            large,
            small,
        }
    }

    /// Initialize an Holo from another Holo.
    pub fn from_hrr(data: Holo, large: f32, small: f32) -> Self {
        Self {
            v: data.v,
            large,
            small,
        }
    }

    /// Return the length of an axis in the internal array.
    pub fn shape_of(&self, axis: usize) -> usize {
        self.v.len_of(Axis(axis))
    }

    /// Get the shape of the array, in general.
    pub fn dim(&self) -> usize {
        self.v.dim()
    }

    /// The Euclidian and length/magnitude of the vector.
    pub fn magnitude(&self) -> f32 {
        self.v.dot(&self.v).sqrt()
    }

    /// Compare two vectors using vector cosine.
    pub fn cmp(&self, other: &Holo) -> f32 {
        let scale = self.dim() * other.dim();
        if scale == 0 {
            0.0
        } else {
            (self.v.dot(&other.v)) / (scale as f32)
        }
    }
}

impl Dot<Holo> for Holo {
    type Output = f32;

    /// Dot product between two holographic reduced representations.
    fn dot(&self, rhs: &Holo) -> Self::Output {
        self.v.dot(&rhs.v)
    }
}

impl Mul for Holo {
    type Output = Self;

    /// Associate or bind two vectors together.
    /// Python:
    /// ```py
    /// def __mul__(self, other):
    ///     return HRR(ifft(fft(self.v) * fft(other.v)).real)
    /// ```
    fn mul(self, rhs: Self) -> Self::Output {
        // fft(self.v)
        let self_dim = self.dim();
        let self_out_dim = if self_dim % 2 == 0 {
            self_dim / 2 + 1
        } else {
            (self_dim + 1) / 2
        };
        let mut self_vhat = Array1::<Complex<f32>>::zeros(self_out_dim);
        let self_fft_handler = R2cFftHandler::<f32>::new(self_dim);
        ndfft_r2c(&self.v, &mut self_vhat, &self_fft_handler, 0);

        // fft(other.v)
        let rhs_dim = rhs.dim();
        let rhs_out_dim = if rhs_dim & 2 == 0 {
            rhs_dim / 2 + 1
        } else {
            (rhs_dim + 1) / 2
        };
        let mut rhs_vhat = Array1::<Complex<f32>>::zeros(rhs_out_dim);
        let rhs_fft_handler = R2cFftHandler::<f32>::new(rhs_dim);
        ndfft_r2c(&rhs.v, &mut rhs_vhat, &rhs_fft_handler, 0);

        // ifft(fft(self.v) * fft(other.v))
        let res = self_vhat * rhs_vhat;
        let res_dim = res.dim();
        let res_out_dim = if res_dim % 2 == 0 {
            res_dim / 2 + 1
        } else {
            (res_dim + 1) / 2
        };
        let mut out_vhat = Array1::<f32>::zeros(res_out_dim);
        let res_fft_handler = R2cFftHandler::<f32>::new(res_dim);
        ndifft_r2c(&res, &mut out_vhat, &res_fft_handler, 0);

        Self::from_data(out_vhat, 0.8, 0.2)
    }
}

impl Mul<ArrayD<f32>> for Holo {
    type Output = ArrayD<f32>;

    fn mul(self, rhs: ArrayD<f32>) -> Self::Output {
        self.v * rhs
    }
}

impl Add for Holo {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::from_data(self.v + rhs.v, self.large, self.small)
    }
}

impl Sub for Holo {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_data(self.v - rhs.v, self.large, self.small)
    }
}

impl Neg for Holo {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::from_data(-self.v, self.large, self.small)
    }
}

impl Div for Holo {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self::from_data(self.v * rhs.v.t(), self.large, self.small)
    }
}

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
        self.memory.slice_mut(s![self.current_traces, ..]).assign(&data.v);
        self.current_traces += 1;
    }

    fn recall(&self, data: &Holo) -> Self::Output {
        let activations = self.memory.dot(&data.v);
        let argmax = activations.argmax().unwrap();
        let arr = self.memory.slice(s![argmax, ..]).to_owned();
        Holo::from_data(arr, data.large, data.small)
    }
}

/// [HashMap] which contains all the base symbols and their holographic 
/// vector representations.
type Lexicon = HashMap<String, Holo>;
type ReverseLexicon = HashMap<Holo, String>;

struct SimpleAssoc {
    a: Lexicon,
    lexicon: Lexicon,
    reverse: ReverseLexicon,
    theta: f32,
}

impl SimpleAssoc {
}

fn main() {}
