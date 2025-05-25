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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
/// The input points were faulty.
pub enum Error {
    /// At least one of the inputs was infinite.
    Infinity,
    /// At least one of the inputs was not a number.
    NaN,
}

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

    let top = Topology::from_paths_binary(set_a, set_b, eps);

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

    // This example came from fiddling with graphite, caused by the positioning not making progress.
    #[test]
    fn graphite_hang() {
        let p1 = BezPath::from_svg("M27.30779709,-257.85907878 C27.30779665,-257.85907835 27.307796200000002,-257.85907792 27.30779576,-257.8590775 C27.292172620000002,-257.8439677 -109.98458240000001,-125.07781586 -233.88593633,-5.24757973 C-233.88593684,-5.24757923 -233.88593736,-5.24757873 -233.88593787,-5.24757824 C-233.88593736,-5.2475777400000005 -233.88593684,-5.24757724 -233.88593632,-5.24757675 C-174.52579263,51.894883570000005 -115.13821436,109.22164426 -77.72402204000001,145.36173051 C-77.72402157,145.36173091 -77.72402116,145.3617313 -77.7240208,145.3617317 C-77.7240208,145.3617317 -77.72402077,145.3617317 -77.72402077,145.3617317 C-77.72401908,145.3617313 -77.72401740000001,145.36173091 -77.72401572,145.36173051 C-1.60745976,127.42335802000001 72.89114167,108.36966443 101.12619937000001,101.0791851 C101.12620127,101.07918461 101.12620318,101.07918412000001 101.12620509,101.07918362000001 C101.1262059,101.07918313 101.12620674,101.07918264 101.1262076,101.07918215000001 C105.56968257,98.47134695 110.06808693,95.83127419 114.61338829,93.16367801 C114.61338927,93.16367744 114.61339025000001,93.16367686 114.61339123,93.16367628 C114.61339123,93.16367628 114.61339123,93.16367628 114.61339123,93.16367628 C114.61339142,93.16367571 114.61339161000001,93.16367513 114.6133918,93.16367456 C123.52785808,66.12754849 166.90764887,-69.59356743000001 182.19132273,-195.37502561 C182.19132278,-195.37502607 182.19132284,-195.37502653 182.19132290000002,-195.37502699 C182.19132175000001,-195.37502746 182.19132061,-195.37502792 182.19131946000002,-195.37502838 C99.03601942,-228.9220441 27.32536748,-257.85199043 27.307800320000002,-257.8590775 C27.30779923,-257.85907792 27.30779816,-257.85907835 27.30779709,-257.85907878 L27.30779709,-257.85907878 Z M-408.62463824,-172.46325538 C-408.62463827,-172.46325472 -408.62463831,-172.46325406 -408.62463834,-172.4632534 C-414.02520486000003,-63.40231834 -377.28160484,34.20037327 -342.02221595000003,99.33558422 C-342.02221565,99.33558477 -342.02221536,99.33558531 -342.02221506,99.33558586000001 C-342.02221506,99.33558586000001 -342.02221506,99.33558586000001 -342.02221506,99.33558586000001 C-342.0222145,99.33558531 -342.02221393,99.33558477 -342.02221337000003,99.33558422 C-310.40978596,68.76186957 -272.80781456,32.395413760000004 -233.88593941,-5.24757675 C-233.8859389,-5.24757724 -233.88593838,-5.2475777400000005 -233.88593787,-5.24757824 C-233.88593787,-5.24757824 -233.88593787,-5.24757824 -233.88593787,-5.24757824 C-233.88593839,-5.24757873 -233.88593891,-5.24757923 -233.88593942,-5.24757973 C-304.53464918000003,-73.25686748 -375.14449713,-141.00509201 -408.62463617000003,-172.4632534 C-408.62463687,-172.46325406 -408.62463756,-172.46325472 -408.62463824,-172.46325538 L-408.62463824,-172.46325538 Z").unwrap();
        let p2 = BezPath::from_svg("M27.3077958,-257.85907753 C27.2923816,-257.84416980000003 -109.98448809,-125.07790706 -233.88593634,-5.24757971 C-233.88593634,-5.24757971 -233.88593825,-5.24757971 -233.88593825,-5.24757971 C-233.88593941,-5.24757971 -233.88593942,-5.24757972 -233.88593942,-5.24757972 C-304.53459173,-73.25681217 -375.1443823,-141.00498183 -408.62463995,-172.46317667 C-408.62464066,-172.46317733 -408.62464135,-172.46317798 -408.62464203,-172.46317863000002 C-408.62464208,-172.46317798 -408.62464211,-172.46317733 -408.62464214,-172.46317667 C-414.02518763,-63.40227258 -377.28159657000003,34.20038853 -342.02221596,99.3355842 C-342.02221566000003,99.33558476 -342.02221536,99.33558531 -342.02221506,99.33558586000001 C-342.02221449,99.33558531 -342.02221392,99.33558476 -342.02221335,99.3355842 C-310.40978594,68.76186955 -272.80781455,32.39541375 -233.88593941,-5.24757675 C-233.88593941,-5.24757675 -233.88593941,-5.24757675 -233.88593941,-5.24757675 C-233.88593889,-5.24757724 -233.88593838,-5.2475777400000005 -233.88593787,-5.24757823 C-233.88593735,-5.2475777400000005 -233.88593684,-5.24757724 -233.88593632,-5.24757675 C-233.88593632,-5.24757675 -233.88593632,-5.24757675 -233.88593632,-5.24757675 C-174.52579259,51.89488361 -115.13821427,109.22164435 -77.72402196,145.36173056 C-77.72402157,145.36173094 -77.72402117,145.36173132 -77.72402078,145.3617317 C-77.72401917,145.36173132 -77.72401755,145.36173094 -77.72401594,145.36173056 C-1.60745982,127.42335803 72.89114182,108.36966439 101.12619946,101.07918507000001 C101.12620136,101.07918458 101.12620325,101.07918409 101.12620514,101.07918360000001 C101.12620595,101.07918311 101.12620679,101.07918263 101.12620764,101.07918214 C105.56968257,98.47134695 110.0680869,95.83127422 114.61338822,93.16367806 C114.61338923,93.16367747 114.61339023000001,93.16367689 114.61339123,93.1636763 C114.61339142,93.16367571 114.61339161000001,93.16367512000001 114.61339181,93.16367453000001 C123.52785811,66.12754843 166.90764887,-69.59356747 182.19132273,-195.37502563 C182.19132279000002,-195.37502608 182.19132285,-195.37502654 182.19132289,-195.37502699 C182.19132177,-195.37502745 182.19132064000001,-195.3750279 182.1913195,-195.37502836000002 C99.03588674,-228.92209763 27.32513858,-257.85208278 27.30780024,-257.85907753 C27.30780024,-257.85907753 27.30779791,-257.85907753 27.30779791,-257.85907753 C27.30779791,-257.85907753 27.3077958,-257.85907753 27.3077958,-257.85907753 L27.3077958,-257.85907753 Z").unwrap();
        let output = binary_op(&p1, &p2, FillRule::EvenOdd, BinaryOp::Union).unwrap();

        insta::assert_ron_snapshot!(output);
    }
}
