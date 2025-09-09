#![deny(missing_docs)]
#![doc = include_str!("../README.md")]

#[macro_use]
mod typed_vec;

#[cfg(any(test, feature = "arbitrary"))]
pub mod arbitrary;
pub mod curve;
mod geom;
mod num;
pub mod order;
mod position;
mod segments;
pub mod sweep;
pub mod topology;

#[cfg(feature = "generators")]
pub mod generators;

pub use geom::{Point, Segment};
use kurbo::Shape;
pub use segments::{SegIdx, Segments};

// pub so that we can use it in fuzz tests, but it's really private
#[doc(hidden)]
pub mod treevec;

use topology::{BinaryWindingNumber, Topology};

use crate::segments::NonClosedPath;

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
pub enum BinaryOp {
    /// A point is in the union of two sets if it is in either one.
    Union,
    /// A point is in the intersection of two sets if it is in both.
    Intersection,
    /// A point is in the difference of two sets if it is in the first but not the second.
    Difference,
    /// A point is in the exclusive-or of two sets if it is in one or the other, but not both.
    Xor,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// The input points were faulty.
pub enum Error {
    /// At least one of the inputs was infinite.
    Infinity,
    /// At least one of the inputs was not a number.
    NaN,
    /// One of the inputs had a non-closed path.
    NonClosedPath(NonClosedPath),
}

impl From<NonClosedPath> for Error {
    fn from(ncp: NonClosedPath) -> Self {
        Error::NonClosedPath(ncp)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Infinity => write!(f, "one of the inputs was infinite"),
            Error::NaN => write!(f, "one of the inputs had a NaN"),
            Error::NonClosedPath(_) => write!(f, "one of the inputs had a non-closed path"),
        }
    }
}

impl std::error::Error for Error {}

/// Computes a boolean operation between two sets, each of which is described as a collection of closed polylines.
pub fn binary_op(
    set_a: &kurbo::BezPath,
    set_b: &kurbo::BezPath,
    fill_rule: FillRule,
    op: BinaryOp,
) -> Result<topology::Contours, Error> {
    // Find the extremal values, to figure out how much precision we can support.
    let bbox = set_a.bounding_box().union(set_b.bounding_box());
    let min = bbox.min_x().min(bbox.min_y());
    let max = bbox.max_x().max(bbox.max_y());
    if min.is_infinite() || max.is_infinite() {
        return Err(Error::Infinity);
    }
    // If there was any NaN in the input, it should have propagated to the min and max.
    if min.is_nan() || max.is_nan() {
        return Err(Error::NaN);
    }

    // TODO: we did some analysis for error bounds in the case of polylines.
    // Think more about what makes sense for curves.
    let m_2 = min.abs().max(max.abs());
    let eps = m_2 * (f64::EPSILON * 64.0);
    let eps = eps.max(1e-6);

    debug_assert!(eps.is_finite());

    let top = Topology::from_paths_binary(set_a, set_b, eps)?;
    #[cfg(feature = "debug-svg")]
    {
        svg::save(
            "out.svg",
            &top.dump_svg(|tag| {
                if tag {
                    "red".to_owned()
                } else {
                    "blue".to_owned()
                }
            }),
        )
        .unwrap();
    }

    let inside = |windings: BinaryWindingNumber| {
        let inside_one = |winding| match fill_rule {
            FillRule::EvenOdd => winding % 2 != 0,
            FillRule::NonZero => winding != 0,
        };

        match op {
            BinaryOp::Union => inside_one(windings.shape_a) || inside_one(windings.shape_b),
            BinaryOp::Intersection => inside_one(windings.shape_a) && inside_one(windings.shape_b),
            BinaryOp::Xor => inside_one(windings.shape_a) != inside_one(windings.shape_b),
            BinaryOp::Difference => inside_one(windings.shape_a) && !inside_one(windings.shape_b),
        }
    };

    Ok(top.contours(inside))
}

#[cfg(test)]
mod tests {
    use kurbo::BezPath;

    use super::*;

    #[test]
    fn two_squares() {
        fn to_bez(mut points: impl Iterator<Item = (f64, f64)>) -> BezPath {
            let p = points.next().unwrap();
            let mut ret = BezPath::default();
            ret.move_to(p);
            for q in points {
                ret.line_to(q);
            }
            ret.line_to(p);
            ret
        }
        let a = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let b = vec![(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)];
        let output = binary_op(
            &to_bez(a.into_iter()),
            &to_bez(b.into_iter()),
            FillRule::EvenOdd,
            BinaryOp::Intersection,
        )
        .unwrap();

        insta::assert_ron_snapshot!(output);
    }

    // This example has a union of two not-quite-axis-aligned crosses. It
    // suffers from some extra quadratic segments near the intersection points,
    // but it used to be worse.
    #[test]
    fn path_blowup() {
        let path1 = "M-90.03662872314453,-212 L-90.03662872314453,212 L90.03565216064453,212 L90.03565216064453,-212 L-90.03662872314453,-212 Z";
        let path2 = "M211.99964904785156,-90.03582000732422 L-212.00035095214844,-90.03646087646484 L-212.00062561035156,90.03582000732422 L211.99937438964844,90.03646087646484 L211.99964904785156,-90.03582000732422 Z";
        if let Ok(output) = binary_op(
            &BezPath::from_svg(path1).unwrap(),
            &BezPath::from_svg(path2).unwrap(),
            FillRule::NonZero,
            BinaryOp::Union,
        ) {
            let output = output.contours().next().unwrap();
            let path_length = output.path.elements().len();
            // This should ideally be more like 12, but most
            // intersection points have like 2 spurious quadratics.
            // It used to be 77, though...
            assert!(path_length < 30);
        }
    }
}
