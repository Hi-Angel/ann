#![allow(dead_code)]

extern crate nalgebra as na;
extern crate typenum;

// use std::ops::Mul;
use na::{Matrix, MatrixArray, U100};

const LAYERS_NUM: i32 = 3; // hidden layers, the in/out two are excluded

type Matrix100x100f = Matrix<f64, U100, U100, MatrixArray<f64, U100, U100>>;

enum Either<T1, T2> {
  Left(T1),
  Right(T2)
}

// a standard logistic function. Values (0..1), good for output layer.
fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + 1.0 / std::f64::consts::E.powf(x) )
}

struct LayerIn {
    next_layer: Option<Box<LayerHid>>,
    weights: Matrix100x100f
}

struct LayerHid {
    next_layer: Either<Box<LayerHid>, Box<LayerOut>>,
    weights: Matrix100x100f,
    activation: fn(f64) -> f64
}

struct LayerOut {
    activation: fn(f64) -> f64
}

fn main() {
}
