//! Geometric primitives, like points and lines.

use arrayvec::ArrayVec;
use kurbo::{CubicBez, ParamCurve, ParamCurveExtrema};

use crate::curve::{monic_quadratic_roots, solve_t_for_y, solve_x_for_y};
use crate::num::CheapOrderedFloat;

/// A two-dimensional point.
///
/// Points are sorted by `y` and then by `x`, for the convenience of our sweep-line
/// algorithm (which moves in increasing `y`).
#[cfg_attr(test, derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq)]
pub struct Point {
    /// Vertical coordinate.
    ///
    /// Although it isn't important for functionality, the documentation and method naming
    /// assumes that larger values are down.
    pub y: f64,
    /// Horizontal component.
    ///
    /// Although it isn't important for functionality, the documentation and method naming
    /// assumes that larger values are to the right.
    pub x: f64,
}

impl Ord for Point {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (
            CheapOrderedFloat::from(self.y),
            CheapOrderedFloat::from(self.x),
        )
            .cmp(&(
                CheapOrderedFloat::from(other.y),
                CheapOrderedFloat::from(other.x),
            ))
    }
}

impl PartialOrd for Point {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Point {}

impl std::fmt::Debug for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:?}, {:?})", self.x, self.y)
    }
}

impl Point {
    /// Create a new point.
    ///
    /// Note that the `x` coordinate comes first. This might be a tiny bit
    /// confusing because we're sorting by `y` coordinate first, but `(x, y)` is
    /// the only sane order (prove me wrong).
    pub fn new(x: f64, y: f64) -> Self {
        debug_assert!(x.is_finite());
        debug_assert!(y.is_finite());
        Point { x, y }
    }

    /// Compute an affine combination between `self` and `other`; that is, `(1 - t) * self + t * other`.
    ///
    /// Panics if a NaN is encountered, which might happen if some input is infinite.
    pub fn affine(&self, other: &Self, t: f64) -> Self {
        Point {
            x: (1.0 - t) * self.x + t * other.x,
            y: (1.0 - t) * self.y + t * other.y,
        }
    }

    /// Compatibility with `kurbo` points.
    pub fn to_kurbo(self) -> kurbo::Point {
        kurbo::Point::new(self.x, self.y)
    }

    /// Compatibility with `kurbo` points.
    pub fn from_kurbo(c: kurbo::Point) -> Self {
        Self::new(c.x, c.y)
    }
}

impl From<(f64, f64)> for Point {
    fn from((x, y): (f64, f64)) -> Self {
        Self { x, y }
    }
}

impl From<Point> for kurbo::Point {
    fn from(p: Point) -> Self {
        kurbo::Point::new(p.x, p.y)
    }
}

impl From<kurbo::Point> for Point {
    fn from(p: kurbo::Point) -> Self {
        Point::new(p.x, p.y)
    }
}

/// A contour segment.
#[derive(Clone, PartialEq, Eq)]
pub struct Segment {
    inner: SegmentInner,
}

#[derive(Clone, PartialEq, Eq)]
enum SegmentInner {
    Line {
        p0: Point,
        p1: Point,
    },
    Cubic {
        p0: Point,
        p1: Point,
        p2: Point,
        p3: Point,
    },
}

impl std::fmt::Debug for Segment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.inner {
            SegmentInner::Line { p0, p1 } => write!(f, "{p0:?} -- {p1:?}"),
            SegmentInner::Cubic { p0, p1, p2, p3 } => {
                write!(f, "{p0:?} -- {p1:?} -- {p2:?} -- {p3:?}")
            }
        }
    }
}

// Checks whether the cubic Bezier with these control points is monotonic increasing
// in y.
//
// This is subject to some numerical error, and doesn't guarantee that the error is
// one-sided.
fn monotonic_cubic(p0: &Point, p1: &Point, p2: &Point, p3: &Point) -> bool {
    // The tangent curve has control points 3(p1 - p0), 3(p2 - p1), and 3(p3 - p2),
    // but we only care about the y coordinate and the 3s don't affect the sign.
    //
    // Note that these are control points, not the usual coefficients of the quadratic.
    // In particular, we want to know whether (1-t)^2 q0 + 2 t (1 - t) q1 + t^2 q2
    // goes negative on [0, 1].
    let q0 = p1.y - p0.y;
    let q1 = p2.y - p1.y;
    let q2 = p3.y - p2.y;

    // If q0 or q2 is negative, the quadratic is negative at one of the
    // endpoints.
    if q0 < 0.0 || q2 < 0.0 {
        return false;
    }

    // The extremum of the quadratic is at t = (q0 - q1) / (q0 - q1 + q2 - q1),
    // so consider the four possibilities for the signs of q0 - q2 and q2 - q1.
    // If they're both negative, then so is the coefficient of t^2 and so there's
    // no minimum between the endpoints. If one is negative and the other positive,
    // then the extremum is either less than zero or bigger than one, and so again
    // there's no extremum between the endpoints.
    if q0 <= q1 || q2 <= q1 {
        return true;
    }

    // There's a minimum between 0 and 1, and its value turns out to be
    // (q2 q0 - q1^2) / (q0 - q1 + q2 - q1). We've already checked that the
    // denominator is positive.
    q2 * q0 >= q1 * q1
}

pub(crate) fn monotonic_pieces(cub: CubicBez) -> ArrayVec<CubicBez, 3> {
    let mut ret = ArrayVec::new();
    let q0 = cub.p1.y - cub.p0.y;
    let q1 = cub.p2.y - cub.p1.y;
    let q2 = cub.p3.y - cub.p2.y;

    // Convert to the representation a t^2 + b t + c.
    let c = q0;
    let b = (q1 - q0) * 2.0;
    let a = q0 - q1 * 2.0 + q2;

    // Convert to the monic representation t^2 + b t + c.
    let scaled_c = c * a.recip();
    let scaled_b = b * a.recip();

    let mut roots: ArrayVec<f64, 3> = ArrayVec::new();
    if scaled_c.is_infinite() || scaled_b.is_infinite() {
        // A was small; treat it as a linear equation.
        // We could use scaled_c here, but the originals should be more accurate.
        roots.push(-c / b)
    } else {
        let (r0, r1) = monic_quadratic_roots(scaled_b, scaled_c);
        roots.push(r0);
        if r1 != r0 {
            roots.push(r1);
        }
    }

    let mut last_r = 0.0;
    //dbg!(&roots);
    // TODO: better handling for roots that are very close to 0.0 or 1.0
    for r in roots {
        if r > 0.0 && r < 1.0 {
            let mut piece_before = cub.subsegment(last_r..r);
            // Thanks to numerical errors, we could end up with tangents that
            // are just barely pointing in the wrong direction. Fix them up.
            if piece_before.p0.y < piece_before.p3.y {
                piece_before.p1.y = piece_before.p0.y.max(piece_before.p1.y);
                piece_before.p2.y = piece_before.p3.y.min(piece_before.p2.y);
                ret.push(piece_before);
            } else if piece_before.p3.y < piece_before.p0.y {
                piece_before.p1.y = piece_before.p0.y.min(piece_before.p1.y);
                piece_before.p2.y = piece_before.p3.y.max(piece_before.p2.y);
                ret.push(piece_before);
            } else if piece_before.p0 != piece_before.p3 {
                ret.push(piece_before);
            }
            last_r = r;
        }
    }

    // TODO: c/p
    let mut piece_before = cub.subsegment(last_r..1.0);
    // Thanks to numerical errors, we could end up with tangents that
    // are just barely pointing in the wrong direction. Fix them up.
    if piece_before.p0.y < piece_before.p3.y {
        piece_before.p1.y = piece_before.p0.y.max(piece_before.p1.y);
        piece_before.p2.y = piece_before.p3.y.min(piece_before.p2.y);
        ret.push(piece_before);
    } else if piece_before.p3.y < piece_before.p0.y {
        piece_before.p1.y = piece_before.p0.y.min(piece_before.p1.y);
        piece_before.p2.y = piece_before.p3.y.max(piece_before.p2.y);
        ret.push(piece_before);
    } else if piece_before.p0 != piece_before.p3 {
        ret.push(piece_before);
    }

    ret
}

impl Segment {
    /// Create a new cubic segment that must be increasing in `y`.
    pub fn monotonic_cubic(p0: Point, p1: Point, p2: Point, p3: Point) -> Self {
        debug_assert!(monotonic_cubic(&p0, &p1, &p2, &p3));
        Self {
            inner: SegmentInner::Cubic { p0, p1, p2, p3 },
        }
    }

    /// Create a new segment that's just a straight line.
    ///
    /// `start` must be less than `end`.
    pub fn straight(start: Point, end: Point) -> Self {
        debug_assert!(start <= end);
        Self {
            inner: SegmentInner::Line { p0: start, p1: end },
        }
    }

    /// The starting point (smallest in the sweep-line order) of this segment.
    pub fn start(&self) -> Point {
        match self.inner {
            SegmentInner::Line { p0, .. } | SegmentInner::Cubic { p0, .. } => p0,
        }
    }

    /// The ending point (largest in the sweep-line order) of this segment.
    pub fn end(&self) -> Point {
        match self.inner {
            SegmentInner::Line { p1, .. } => p1,
            SegmentInner::Cubic { p3, .. } => p3,
        }
    }

    /// Compatibility with `kurbo`'s cubics.
    pub fn to_kurbo_cubic(&self) -> kurbo::CubicBez {
        match self.inner {
            SegmentInner::Line { p0, p1 } => {
                let p0 = p0.to_kurbo();
                let p3 = p1.to_kurbo();
                let p1 = p0 + (1. / 3.) * (p3 - p0);
                let p2 = p0 + (2. / 3.) * (p3 - p0);
                kurbo::CubicBez { p0, p1, p2, p3 }
            }
            SegmentInner::Cubic { p0, p1, p2, p3 } => kurbo::CubicBez {
                p0: p0.to_kurbo(),
                p1: p1.to_kurbo(),
                p2: p2.to_kurbo(),
                p3: p3.to_kurbo(),
            },
        }
    }

    /// TODO: maybe From instead?
    pub fn from_kurbo_cubic(c: kurbo::CubicBez) -> Self {
        Self::monotonic_cubic(
            Point::from_kurbo(c.p0),
            Point::from_kurbo(c.p1),
            Point::from_kurbo(c.p2),
            Point::from_kurbo(c.p3),
        )
    }

    /// A crude lower bound on our minimum horizontal position.
    pub fn min_x(&self) -> f64 {
        match self.inner {
            SegmentInner::Line { p0, p1 } => p0.x.min(p1.x),
            SegmentInner::Cubic { p0, p1, p2, p3 } => p0.x.min(p1.x).min(p2.x).min(p3.x),
        }
    }

    /// A crude upper bound on our maximum horizontal position.
    pub fn max_x(&self) -> f64 {
        match self.inner {
            SegmentInner::Line { p0, p1 } => p0.x.max(p1.x),
            SegmentInner::Cubic { p0, p1, p2, p3 } => p0.x.max(p1.x).max(p2.x).max(p3.x),
        }
    }

    /// Our `x` coordinate at the given `y` coordinate.
    ///
    /// Horizontal segments will return their largest `x` coordinate.
    ///
    /// # Panics
    ///
    /// Panics if `y` is outside the `y` range of this segment.
    pub fn at_y(&self, y: f64) -> f64 {
        debug_assert!(
            (self.start().y..=self.end().y).contains(&y),
            "segment {self:?}, y={y:?}"
        );

        if self.is_horizontal() {
            self.end().x
        } else {
            // TODO: special-case lines
            solve_x_for_y(self.to_kurbo_cubic(), y)
        }
    }

    fn local_bbox(&self, y: f64, eps: f64) -> kurbo::Rect {
        let start_y = (y - eps).max(self.start().y);
        let end_y = (y + eps).min(self.end().y);

        // TODO: special-case lines
        let c = self.to_kurbo_cubic();
        let t_min = solve_t_for_y(c, start_y);
        let t_max = solve_t_for_y(c, end_y);

        c.subsegment(t_min..t_max).bounding_box()
    }

    /// Returns a lower bound on the `x` coordinate near `y`.
    ///
    /// More precisely, take a Minkowski sum between this curve
    /// and a small square. This returns a lower bound on the `x`
    /// coordinate of the resulting set at height `y`.
    pub fn lower(&self, y: f64, eps: f64) -> f64 {
        self.local_bbox(y, eps).min_x() - eps
    }

    /// Returns an upper bound on the `x` coordinate near `y`.
    ///
    /// More precisely, take a Minkowski sum between this curve
    /// and a small square. This returns a upper bound on the `x`
    /// coordinate of the resulting set at height `y`.
    pub fn upper(&self, y: f64, eps: f64) -> f64 {
        self.local_bbox(y, eps).max_x() + eps
    }

    /// Returns true if this segment is exactly horizontal.
    pub fn is_horizontal(&self) -> bool {
        self.start().y == self.end().y
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::num::tests::Reasonable;
    use proptest::prelude::*;

    impl Reasonable for Point {
        type Strategy = BoxedStrategy<Point>;

        fn reasonable() -> Self::Strategy {
            (f64::reasonable(), f64::reasonable())
                .prop_map(|(x, y)| Point::new(x, y))
                .boxed()
        }
    }

    // impl Reasonable for Segment {
    //     type Strategy = BoxedStrategy<Segment>;

    //     fn reasonable() -> Self::Strategy {
    //         (Point::reasonable(), Point::reasonable())
    //             .prop_map(|(start, end)| {
    //                 if start <= end {
    //                     Segment::new(start, end)
    //                 } else {
    //                     Segment::new(end, start)
    //                 }
    //             })
    //             .boxed()
    //     }
    // }

    // fn segment_and_y() -> BoxedStrategy<(Segment, f64)> {
    //     Segment::reasonable()
    //         .prop_flat_map(|s| {
    //             let y0 = s.start.y;
    //             let y1 = s.end.y;

    //             // proptest's float sampler doesn't like a range like x..=x
    //             // https://github.com/proptest-rs/proptest/issues/343
    //             if y0 < y1 {
    //                 (Just(s), (y0..=y1).boxed())
    //             } else {
    //                 (Just(s), Just(y0).boxed())
    //             }
    //         })
    //         .boxed()
    // }
}
