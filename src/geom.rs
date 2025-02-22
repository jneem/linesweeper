//! Geometric primitives, like points and lines.

use crate::curve::{solve_t_for_y, solve_x_for_y};
use crate::num::CheapOrderedFloat;

/// A two-dimensional point.
///
/// Points are sorted by `y` and then by `x`, for the convenience of our sweep-line
/// algorithm (which moves in increasing `y`).
#[derive(Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
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

    pub fn to_kurbo(self) -> kurbo::Point {
        kurbo::Point::new(self.x, self.y)
    }
}

impl From<(f64, f64)> for Point {
    fn from((x, y): (f64, f64)) -> Self {
        Self { x, y }
    }
}

/// A contour segment, in sweep-line order.
#[derive(Clone, PartialEq, Eq)]
pub struct Segment {
    pub p0: Point,
    pub p1: Point,
    pub p2: Point,
    pub p3: Point,
}

impl std::fmt::Debug for Segment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Segment { p0, p1, p2, p3 } = self;
        write!(f, "{p0:?} -- {p1:?} -- {p2:?} -- {p3:?}",)
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
    // so consider the four possiblities for the signs of q0 - q2 and q2 - q1.
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

impl Segment {
    /// Create a new segment.
    ///
    /// `start` must be less than `end`.
    pub fn new(p0: Point, p1: Point, p2: Point, p3: Point) -> Self {
        debug_assert!(monotonic_cubic(&p0, &p1, &p2, &p3));
        Self { p0, p1, p2, p3 }
    }

    pub fn straight(start: Point, end: Point) -> Self {
        let p0 = start;
        let p1 = Point::new(
            start.x + (end.x - start.x) / 3.0,
            start.y + (end.y - start.y) / 3.0,
        );
        let p2 = Point::new(
            start.x + 2.0 * (end.x - start.x) / 3.0,
            start.y + 2.0 * (end.y - start.y) / 3.0,
        );
        let p3 = end;
        Self::new(p0, p1, p2, p3)
    }

    pub fn to_kurbo(&self) -> kurbo::CubicBez {
        kurbo::CubicBez {
            p0: self.p0.to_kurbo(),
            p1: self.p1.to_kurbo(),
            p2: self.p2.to_kurbo(),
            p3: self.p3.to_kurbo(),
        }
    }

    /// A crude lower bound on our minimum horizontal position.
    pub fn min_x(&self) -> f64 {
        self.p0.x.min(self.p1.x).min(self.p2.x).min(self.p3.x)
    }

    /// A crude upper bound on our maximum horizontal position.
    pub fn max_x(&self) -> f64 {
        self.p0.x.max(self.p1.x).max(self.p2.x).max(self.p3.x)
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
            (self.p0.y..=self.p3.y).contains(&y),
            "segment {self:?}, y={y:?}"
        );

        if self.is_horizontal() {
            self.p1.x
        } else {
            solve_x_for_y(self.to_kurbo(), y)
        }
    }

    pub fn lower(&self, y: f64, eps: f64) -> f64 {
        // FIXME: this allows for some slack in y, but it's super
        // hacky. Basically, we want a lower bound on the smallest
        // x position in a small y-neigborhood
        self.at_y(y)
            .min(self.at_y((y - eps).max(self.p0.y)))
            .min(self.at_y((y + eps).min(self.p3.y)))
            - eps
    }

    pub fn upper(&self, y: f64, eps: f64) -> f64 {
        // FIXME: as above
        self.at_y(y)
            .max(self.at_y((y - eps).max(self.p0.y)))
            .max(self.at_y((y + eps).min(self.p3.y)))
            + eps
    }

    /// Returns true if this segment is exactly horizontal.
    pub fn is_horizontal(&self) -> bool {
        self.p0.y == self.p3.y
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
