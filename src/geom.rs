//! Geometric primitives, like points and lines.

use malachite::num::arithmetic::traits::Abs;
use malachite::Rational;

use crate::num::CheapOrderedFloat;

/// A two-dimensional point.
///
/// Points are sorted by `y` and then by `x`, for the convenience of our sweep-line
/// algorithm (which moves in increasing `y`).
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
}

impl From<(f64, f64)> for Point {
    fn from((x, y): (f64, f64)) -> Self {
        Self { x, y }
    }
}

/// A contour segment, in sweep-line order.
#[derive(Clone, PartialEq, Eq)]
pub struct Segment {
    /// The starting point of this segment, strictly less than `end`.
    pub start: Point,
    /// The ending point of this segment, strictly greater than `start`.
    pub end: Point,
}

impl std::fmt::Debug for Segment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} -- {:?}", self.start, self.end)
    }
}

impl Segment {
    /// Create a new segment.
    ///
    /// `start` must be less than `end`.
    pub fn new(start: Point, end: Point) -> Self {
        Self { start, end }
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
            (self.start.y..=self.end.y).contains(&y),
            "segment {self:?}, y={y:?}"
        );

        if self.start.y == self.end.y {
            self.end.x
        } else {
            // Even if the segment is *almost* horizontal, t is guaranteed
            // to be in [0.0, 1.0].
            let t = (y - self.start.y) / (self.end.y - self.start.y);
            self.start.x + t * (self.end.x - self.start.x)
        }
    }

    /// Returns the x difference between segments at the last y position that they share.
    /// Returns a positive value if `other` is to the right.
    ///
    /// (Returns nonsense if they don't share a y position.)
    pub fn end_offset(&self, other: &Self) -> f64 {
        if self.end.y < other.end.y {
            other.at_y(self.end.y) - self.end.x
        } else {
            other.end.x - self.at_y(other.end.y)
        }
    }

    /// Returns the x difference between segments at the first y position that they share.
    /// Returns a positive value if `other` is to the right.
    ///
    /// (Returns nonsense if they don't share a y position.)
    pub fn start_offset(&self, other: &Self) -> f64 {
        if self.start.y >= other.start.y {
            other.at_y(self.start.y) - self.start.x
        } else {
            other.start.x - self.at_y(other.start.y)
        }
    }

    /// Checks if `self` crosses `other`, and returns a valid crossing height if so.
    ///
    /// The notion of "crossing" is special to our sweep-line purposes; it
    /// isn't a generic line segment intersection. This should only be called
    /// when `self` starts of "to the left" (with some wiggle room allowed) of
    /// `other`. If we find (numerically, approximately) that `self` starts to
    /// the right and ends more to the right, we'll return the smallest shared
    /// height as the intersection height.
    ///
    /// This is guaranteed to find a crossing height if `self` ends at least
    /// `eps` to the right of `other`. If the ending horizontal positions are
    /// very close, we might just declare that there's no crossing.
    pub fn crossing_y(&self, other: &Self, eps: &f64) -> Option<f64> {
        let y0 = self.start.y.max(other.start.y);
        let y1 = self.end.y.min(other.end.y);

        assert!(y1 >= y0);

        let dx0 = self.at_y(y0) - other.at_y(y0);
        let dx1 = self.at_y(y1) - other.at_y(y1);

        // According the the analysis, dx1 is accurate to with eps / 8, and the analysis also
        // requires a threshold or 3 eps / 4. So we compare to 7 eps / 8.
        if dx1 < eps * 0.875 {
            return None;
        }

        if dx0 >= 0.0 {
            // If we're here, we've already compared the endpoint and decided
            // that there's a crossing. Since we think they started in the wrong
            // order, declare y0 as the crossing.
            return Some(y0);
        }

        let denom = dx1 - dx0;
        let t = -dx0 / denom;
        debug_assert!(t >= 0.0);
        debug_assert!(t <= 1.0);

        // It should be impossible to have y0 + t * (y1 - y0) < y0, but I think with
        // rounding it's possible to have y0 + t * (y1 - y0) > y1. To be sure, truncate
        // the upper bound.
        Some((y0 + t * (y1 - y0)).min(y1))
    }

    // Convert this segment to an exact segment using rational arithmetic.
    // pub fn to_exact(&self) -> Segment<Rational> {
    //     Segment::new(self.start.to_exact(), self.end.to_exact())
    // }
    // FIXME

    /// Returns true if this segment is exactly horizontal.
    pub fn is_horizontal(&self) -> bool {
        self.start.y == self.end.y
    }

    /// Scale eps based on the slope of this line.
    ///
    /// The write-up used 1/(cos theta) for scaling. Here we use
    /// the smaller (and therefore stricter) max(1, 1/|slope|) scaling,
    /// because it's possible to compute exactly when doing rational
    /// arithmetic.
    pub fn scaled_eps(&self, eps: f64) -> f64 {
        assert!(self.start.y <= self.end.y);
        if self.start.y == self.end.y {
            // See `scaled_eps_bound`
            return eps;
        }

        let dx = (self.end.x - self.start.x).abs();
        let dy = self.end.y - self.start.y;

        if dx <= dy {
            eps
        } else {
            (dx * eps) / dy
        }
    }

    /// The lower envelope of this segment at the given height.
    ///
    /// In the write-up this was called `alpha^-_(y,epsilon)`.
    pub fn lower(&self, y: f64, eps: f64) -> f64 {
        let min_x = self.end.x.min(self.start.x);

        if self.is_horizontal() {
            // Special case for horizontal lines, because their
            // `at_y` function returns the larger x position, and
            // we want the smaller one here.
            self.start.x - eps
        } else {
            (self.at_y(y) - self.scaled_eps(eps)).max(min_x - eps)
        }
    }

    /// The lower envelope of this segment at the given height.
    ///
    /// In the write-up this was called `alpha^-_(y,epsilon)`.
    pub fn lower_with_scaled_eps(&self, y: f64, eps: f64, scaled_eps: f64) -> f64 {
        let min_x = self.end.x.min(self.start.x);

        if self.is_horizontal() {
            // Special case for horizontal lines, because their
            // `at_y` function returns the larger x position, and
            // we want the smaller one here.
            self.start.x - eps
        } else {
            (self.at_y(y) - scaled_eps).max(min_x - eps)
        }
    }

    /// The upper envelope of this segment at the given height.
    ///
    /// In the write-up this was called `alpha^+_(y,epsilon)`.
    pub fn upper(&self, y: f64, eps: f64) -> f64 {
        let max_x = self.end.x.max(self.start.x);

        (self.at_y(y) + self.scaled_eps(eps)).min(max_x + eps)
    }

    /// The upper envelope of this segment at the given height.
    ///
    /// In the write-up this was called `alpha^+_(y,epsilon)`.
    pub fn upper_with_scaled_eps(&self, y: f64, eps: f64, scaled_eps: f64) -> f64 {
        let max_x = self.end.x.max(self.start.x);

        (self.at_y(y) + scaled_eps).min(max_x + eps)
    }

    pub(crate) fn quick_left_of(&self, other: &Self, eps: f64) -> bool {
        let my_max = (self.start.x).max(self.end.x);
        let other_min = (other.start.x).min(other.end.x);
        other_min - my_max > eps
    }

    pub(crate) fn to_exact(&self) -> RationalSegment {
        RationalSegment {
            start: RationalPoint {
                x: self.start.x.try_into().unwrap(),
                y: self.start.y.try_into().unwrap(),
            },
            end: RationalPoint {
                x: self.end.x.try_into().unwrap(),
                y: self.end.y.try_into().unwrap(),
            },
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct RationalPoint {
    pub x: Rational,
    pub y: Rational,
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct RationalSegment {
    pub start: RationalPoint,
    pub end: RationalPoint,
}

impl RationalSegment {
    pub fn at_y(&self, y: &Rational) -> Rational {
        debug_assert!((&self.start.y..=&self.end.y).contains(&y),);

        if self.start.y == self.end.y {
            self.end.x.clone()
        } else {
            // Even if the segment is *almost* horizontal, t is guaranteed
            // to be in [0.0, 1.0].
            let t = (y - &self.start.y) / (&self.end.y.clone() - &self.start.y);
            &self.start.x + &t * (&self.end.x - &self.start.x)
        }
    }

    pub fn scaled_eps(&self, eps: &Rational) -> Rational {
        assert!(self.start.y <= self.end.y);
        if self.start.y == self.end.y {
            return eps.clone();
        }

        let dx = (&self.end.x - &self.start.x).abs();
        let dy = &self.end.y - &self.start.y;

        if dx <= dy {
            eps.clone()
        } else {
            (&dx * eps) / &dy
        }
    }

    pub fn lower(&self, y: &Rational, eps: &Rational) -> Rational {
        let min_x = self.end.x.clone().min(self.start.x.clone());

        if self.start.y == self.end.y {
            // Special case for horizontal lines, because their
            // `at_y` function returns the larger x position, and
            // we want the smaller one here.
            &self.start.x - eps
        } else {
            (self.at_y(y) - self.scaled_eps(eps)).max(min_x - eps)
        }
    }

    pub fn upper(&self, y: &Rational, eps: &Rational) -> Rational {
        let max_x = self.end.x.clone().max(self.start.x.clone());

        (self.at_y(y) + self.scaled_eps(eps)).min(max_x + eps)
    }

    // The moment our lower bound crosses to the right of `other`'s upper bound.
    // (Actually, it could give too large a value right now, because it doesn't take the
    // "chamfers" into account.)
    #[cfg(feature = "slow-asserts")]
    pub(crate) fn exact_eps_crossing(&self, other: &Self, eps: &Rational) -> Option<Rational> {
        let y0 = self.start.y.clone().max(other.start.y.clone());
        let y1 = self.end.y.clone().min(other.end.y.clone());
        let scaled_eps = self.scaled_eps(eps);

        assert!(y1 >= y0);

        let dx0 = other.at_y(&y0) - self.at_y(&y0) + scaled_eps.clone() * Rational::from(2);
        let dx1 = other.at_y(&y1) - self.at_y(&y1) + scaled_eps * Rational::from(2);
        if dx0 == dx1 {
            // They're parallel.
            (dx0 == 0).then_some(y0)
        } else if dx1 < 0 {
            let t = &dx0 / (&dx0 - dx1);
            (0..=1).contains(&t).then(|| &y0 + t * (y1 - &y0))
        } else {
            None
        }
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

    impl Reasonable for Segment {
        type Strategy = BoxedStrategy<Segment>;

        fn reasonable() -> Self::Strategy {
            (Point::reasonable(), Point::reasonable())
                .prop_map(|(start, end)| {
                    if start <= end {
                        Segment::new(start, end)
                    } else {
                        Segment::new(end, start)
                    }
                })
                .boxed()
        }
    }

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
