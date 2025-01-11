#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

mod geom;
mod num;
mod segments;
pub mod sweep;
pub mod topology;

pub use geom::{Point, Segment};
pub use num::{CheapOrderedFloat, Float};
pub use segments::{SegIdx, Segments};

use ordered_float::NotNan;
use topology::{Topology, WindingNumber};

#[cfg(test)]
pub mod perturbation;

/// A fill rule tells us how to decide whether a point is "inside" a polyline.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FillRule {
    /// The point is "inside" if its winding number is odd.
    EvenOdd,
    /// The point is "inside" if its winding number is non-zero.
    NonZero,
}

/// Binary operations between sets.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BooleanOp {
    /// A point is in the union of two sets if it is in either one.
    Union,
    /// A point is in the intersection of two sets if it is in both.
    Intersection,
    /// A point is in the difference of two sets if it is in the first but not the second.
    Difference,
    /// A point is in the exclusive-or of two sets if it is in one or the other, but not both.
    Xor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
/// The input points were faulty.
pub enum Error {
    /// At least one of the inputs was infinite.
    Infinity,
    /// At least one of the inputs was not a number.
    NaN,
}

fn extrema(mut xs: impl Iterator<Item = f64>) -> Result<(f64, f64), Error> {
    xs.try_fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), x| {
        if x.is_nan() {
            Err(Error::NaN)
        } else {
            Ok((x.min(min), x.max(max)))
        }
    })
}

/// Computes a boolean operation between two sets, each of which is described as a collection of closed polylines.
pub fn boolean_op(
    set_a: &[Vec<(f64, f64)>],
    set_b: &[Vec<(f64, f64)>],
    fill_rule: FillRule,
    op: BooleanOp,
) -> Result<topology::Contours<NotNan<f64>>, Error> {
    // Find the extremal values, to figure out how much precision we can support.
    let (min, max) = extrema(
        set_a
            .iter()
            .flatten()
            .chain(set_b.iter().flatten())
            .flat_map(|p| [p.0, p.1]),
    )?;
    if min.is_infinite() || max.is_infinite() {
        return Err(Error::Infinity);
    }

    // This is M/2 in the write-up.
    let m_2 = min.abs().max(max.abs());

    // The write-up says we can take eps = 64 eps_0 M, where eps_0 is the basic relative
    // error of addition and subtraction, which is EPSILON / 2.
    // unwrap: we already checked that min and max are non-NaN. This could overflow to infinity,
    // but it can't be NaN.
    let eps = NotNan::try_from(m_2 * (f64::EPSILON * 64.0)).unwrap();

    debug_assert!(eps.is_finite());

    // unwrap: the conversions only fail for NaN, and we already checked that our points
    // don't have any of those.
    let top = Topology::new(
        set_a
            .iter()
            .map(|ps| ps.iter().map(|p| Point::try_from(*p).ok().unwrap())),
        set_b
            .iter()
            .map(|ps| ps.iter().map(|p| Point::try_from(*p).ok().unwrap())),
        &eps,
    );

    let inside = |windings: WindingNumber| {
        let inside_one = |winding| match fill_rule {
            FillRule::EvenOdd => winding % 2 != 0,
            FillRule::NonZero => winding != 0,
        };

        match op {
            BooleanOp::Union => inside_one(windings.shape_a) || inside_one(windings.shape_b),
            BooleanOp::Intersection => inside_one(windings.shape_a) && inside_one(windings.shape_b),
            BooleanOp::Xor => inside_one(windings.shape_a) != inside_one(windings.shape_b),
            BooleanOp::Difference => inside_one(windings.shape_a) && !inside_one(windings.shape_b),
        }
    };

    Ok(top.contours(inside))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_squares() {
        let a = [vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]];
        let b = [vec![(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]];
        let output = boolean_op(&a, &b, FillRule::EvenOdd, BooleanOp::Intersection).unwrap();

        insta::assert_ron_snapshot!(output);
    }
}
