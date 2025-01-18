//! Utilities for generating examples, benchmarks, and test cases.

use crate::{num::Float, Point};

type Contours<F> = Vec<Vec<Point<F>>>;

/// Generate a bunch of squares, arranged in a grid.
///
/// The top-left of the first square is at (x0, y0). Each square has size `size
/// x size`, and the distance between squares (both horizontally and vertically)
/// is `offset`.
///
/// If `slant` is non-zero, generates parallelograms instead of squares: the
/// right-hand side of each square gets translated down by `slant`.
fn squares<F: Float>((x0, y0): (F, F), size: F, offset: F, slant: F, count: usize) -> Contours<F> {
    let mut ret = Vec::new();
    for i in 0..count {
        let x = x0.clone() + F::from_f32(i as f32) * offset.clone();
        for j in 0..count {
            let y = y0.clone() + F::from_f32(j as f32) * offset.clone();
            ret.push(vec![
                Point::new(x.clone(), y.clone()),
                Point::new(x.clone(), y.clone() + size.clone()),
                Point::new(
                    x.clone() + size.clone(),
                    y.clone() + size.clone() + slant.clone(),
                ),
                Point::new(x.clone() + size.clone(), y.clone() + slant.clone()),
            ]);
        }
    }

    ret
}

/// Generate an `n` by `n` checkerboard-like pattern with overlapping squares.
/// For `n = 3`, it looks like:
///
/// ```text
/// ┌────┐ ┌────┐ ┌────┐
/// │    │ │    │ │    │
/// │  ┌─┼─┼─┐┌─┼─┼─┐  │
/// └──┼─┘ └─┼┼─┘ └─┼──┘
/// ┌──┼─┐ ┌─┼┼─┐ ┌─┼──┐
/// │  └─┼─┼─┘└─┼─┼─┘  │
/// │  ┌─┼─┼─┐┌─┼─┼─┐  │
/// └──┼─┘ └─┼┼─┘ └─┼──┘
/// ┌──┼─┐ ┌─┼┼─┐ ┌─┼──┐
/// │  └─┼─┼─┘└─┼─┼─┘  │
/// │    │ │    │ │    │
/// └────┘ └────┘ └────┘
/// ```
///
/// We return the pattern in two parts: the outer collection of `n x n`
/// non-overlapping squares, and the inner collection of `(n - 1) x (n - 1)`
/// non-overlapping squares.
pub fn checkerboard<F: Float>(n: usize) -> (Contours<F>, Contours<F>) {
    let f = F::from_f32;
    (
        squares((f(0.0), f(0.0)), f(30.0), f(40.0), f(0.0), n),
        squares((f(20.0), f(20.0)), f(30.0), f(40.0), f(0.0), n - 1),
    )
}

/// Like `checkerboard`, but with no exactly-horizontal lines.
///
/// Horizontal lines have special handling in the sweep-line algorithm, so
/// their presence or absence can affect performance.
pub fn slanted_checkerboard<F: Float>(n: usize) -> (Contours<F>, Contours<F>) {
    let f = F::from_f32;
    (
        squares((f(0.0), f(0.0)), f(30.0), f(40.0), f(1.0), n),
        squares((f(20.0), f(20.0)), f(30.0), f(40.0), f(1.0), n - 1),
    )
}
