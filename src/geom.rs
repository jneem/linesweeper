//! Geometric primitives, like points and lines.

use malachite::Rational;
use ordered_float::NotNan;

use crate::num::Float;

/// A two-dimensional point.
///
/// Points are sorted by `y` and then by `x`, for the convenience of our sweep-line
/// algorithm (which moves in increasing `y`).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct Point<F: Float> {
    /// Vertical coordinate.
    ///
    /// Although it isn't important for functionality, the documentation and method naming
    /// assumes that larger values are down.
    pub y: F,
    /// Horizontal component.
    ///
    /// Although it isn't important for functionality, the documentation and method naming
    /// assumes that larger values are to the right.
    pub x: F,
}

impl<F: Float> std::fmt::Debug for Point<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:?}, {:?})", self.x, self.y)
    }
}

impl<F: Float> Point<F> {
    /// Create a new point.
    ///
    /// Note that the `x` coordinate comes first. This might be a tiny bit
    /// confusing because we're sorting by `y` coordinate first, but `(x, y)` is
    /// the only sane order (prove me wrong).
    pub fn new(x: F, y: F) -> Self {
        Point { x, y }
    }

    /// Convert this point to an exact rational representation.
    pub fn to_exact(&self) -> Point<Rational> {
        Point {
            x: self.x.to_exact(),
            y: self.y.to_exact(),
        }
    }

    /// Compute and affine combination between `self` and `other`; that is, `(1 - t) * self + t * other`.
    ///
    /// Panics if a NaN is encountered, which might happen if some input is infinite.
    pub fn affine(&self, other: &Self, t: &F) -> Self {
        let one = F::from_f32(1.0);
        Point {
            x: (one.clone() - t) * &self.x + t.clone() * &other.x,
            y: (one - t) * &self.y + t.clone() * &other.y,
        }
    }
}

impl<F: Float> From<(F, F)> for Point<F> {
    fn from((x, y): (F, F)) -> Self {
        Self { x, y }
    }
}

impl TryFrom<(f64, f64)> for Point<NotNan<f64>> {
    type Error = crate::Error;

    fn try_from((x, y): (f64, f64)) -> Result<Self, Self::Error> {
        let x = NotNan::try_from(x).map_err(|_| crate::Error::NaN)?;
        let y = NotNan::try_from(y).map_err(|_| crate::Error::NaN)?;
        Ok((x, y).into())
    }
}

/// A line segment, in sweep-line order.
///
/// This has a start point and an end point, and the start point must always
/// be strictly less than the end point.
#[derive(Clone, PartialEq, Eq)]
pub struct Segment<F: Float> {
    /// The starting point of this segment, strictly less than `end`.
    pub start: Point<F>,
    /// The ending point of this segment, strictly greater than `start`.
    pub end: Point<F>,
}

impl<F: Float> std::fmt::Debug for Segment<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} -- {:?}", self.start, self.end)
    }
}

impl<F: Float> Segment<F> {
    /// Our `x` coordinate at the given `y` coordinate.
    ///
    /// Horizontal segments will return their largest `x` coordinate.
    ///
    /// # Panics
    ///
    /// Panics if `y` is outside the `y` range of this segment.
    pub fn at_y(&self, y: &F) -> F {
        debug_assert!(
            (&self.start.y..=&self.end.y).contains(&y),
            "segment {self:?}, y={y:?}"
        );

        if self.start.y == self.end.y {
            self.end.x.clone()
        } else {
            // Even if the segment is *almost* horizontal, t is guaranteed
            // to be in [0.0, 1.0].
            let t = (y.clone() - &self.start.y) / (self.end.y.clone() - &self.start.y);
            self.start.x.clone() + t * (self.end.x.clone() - &self.start.x)
        }
    }

    /// Returns the x difference between segments at the last y position that they share.
    /// Returns a positive value if `other` is to the right.
    ///
    /// (Returns nonsense if they don't share a y position.)
    pub fn end_offset(&self, other: &Self) -> F {
        if self.end.y < other.end.y {
            other.at_y(&self.end.y) - &self.end.x
        } else {
            other.end.x.clone() - self.at_y(&other.end.y)
        }
    }

    /// Returns the x difference between segments at the first y position that they share.
    /// Returns a positive value if `other` is to the right.
    ///
    /// (Returns nonsense if they don't share a y position.)
    pub fn start_offset(&self, other: &Self) -> F {
        if self.start.y >= other.start.y {
            other.at_y(&self.start.y) - &self.start.x
        } else {
            other.start.x.clone() - self.at_y(&other.start.y)
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
    pub fn crossing_y(&self, other: &Self, eps: &F) -> Option<F> {
        let y0 = self.start.y.clone().max(other.start.y.clone());
        let y1 = self.end.y.clone().min(other.end.y.clone());

        assert!(y1 >= y0);

        let dx0 = self.at_y(&y0) - other.at_y(&y0);
        let dx1 = self.at_y(&y1) - other.at_y(&y1);

        // According the the analysis, dx1 is accurate to with eps / 8, and the analysis also
        // requires a threshold or 3 eps / 4. So we compare to 7 eps / 8.
        if dx1 < eps.clone() * F::from_f32(0.875) {
            return None;
        }

        if dx0 >= F::from_f32(0.0) {
            // If we're here, we've already compared the endpoint and decided
            // that there's a crossing. Since we think they started in the wrong
            // order, declare y0 as the crossing.
            return Some(y0);
        }

        let denom = dx1 - dx0.clone();
        let t = -dx0 / denom;
        debug_assert!(t.ge(&F::from_f32(0.0)));
        debug_assert!(t.le(&F::from_f32(1.0)));

        // It should be impossible to have y0 + t * (y1 - y0) < y0, but I think with
        // rounding it's possible to have y0 + t * (y1 - y0) > y1. To be sure, truncate
        // the upper bound.
        Some((y0.clone() + t * (y1.clone() - y0)).min(y1))
    }

    /// Convert this segment to an exact segment using rational arithmetic.
    pub fn to_exact(&self) -> Segment<Rational> {
        Segment {
            start: self.start.to_exact(),
            end: self.end.to_exact(),
        }
    }

    /// Returns true if this segment is exactly horizontal.
    pub fn is_horizontal(&self) -> bool {
        self.start.y == self.end.y
    }
}

impl Segment<Rational> {
    /// The height of the (first) intersection between `self` and `other`.
    ///
    /// This differs from `crossing_y` in that it's exact, whereas `crossing_y`
    /// (even when used with `Segment<Rational>`) has some slop near the
    /// beginning and ending heights.
    pub fn exact_intersection_y(&self, other: &Self) -> Option<Rational> {
        let y0 = self.start.y.clone().max(other.start.y.clone());
        let y1 = self.end.y.clone().min(other.end.y.clone());

        assert!(y1 >= y0);

        let dx0 = other.at_y(&y0) - self.at_y(&y0);
        let dx1 = other.at_y(&y1) - self.at_y(&y1);
        if dx0 == dx1 {
            // They're parallel.
            (dx0 == 0).then_some(y0)
        } else {
            let t = &dx0 / (&dx0 - dx1);
            (0..=1).contains(&t).then(|| &y0 + t * (y1 - &y0))
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::num::tests::Reasonable;
    use ordered_float::NotNan;
    use proptest::prelude::*;

    impl<F: Reasonable + Float> Reasonable for Point<F>
    where
        F::Strategy: 'static,
    {
        type Strategy = BoxedStrategy<Point<F>>;

        fn reasonable() -> Self::Strategy {
            (F::reasonable(), F::reasonable())
                .prop_map(|(x, y)| Point::new(x, y))
                .boxed()
        }
    }

    impl<F: Reasonable + Float> Reasonable for Segment<F>
    where
        F: 'static,
        F::Strategy: 'static,
    {
        type Strategy = BoxedStrategy<Segment<F>>;

        fn reasonable() -> Self::Strategy {
            (Point::reasonable(), Point::reasonable())
                .prop_map(|(start, end)| {
                    if start <= end {
                        Segment { start, end }
                    } else {
                        Segment {
                            start: end,
                            end: start,
                        }
                    }
                })
                .boxed()
        }
    }

    fn f32_segment_and_y() -> BoxedStrategy<(Segment<NotNan<f32>>, NotNan<f32>)> {
        Segment::<NotNan<f32>>::reasonable()
            .prop_flat_map(|s| {
                let y0 = s.start.y.into_inner();
                let y1 = s.end.y.into_inner();

                // proptest's float sampler doesn't like a range like x..=x
                // https://github.com/proptest-rs/proptest/issues/343
                if y0 < y1 {
                    (
                        Just(s),
                        (y0..=y1).prop_map(|y| NotNan::try_from(y).unwrap()).boxed(),
                    )
                } else {
                    (Just(s), Just(NotNan::try_from(y0).unwrap()).boxed())
                }
            })
            .boxed()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100000))]

        // A quick sanity check that f64 addition is as accurate as I think it is. Changing the -24 to
        // a -25 makes this test fail.
        #[test]
        fn addition_accuracy(x in NotNan::<f32>::reasonable(), y in NotNan::<f32>::reasonable()) {
            let error_bound = Rational::try_from(2e6f64 * (-24.0f64).exp2()).unwrap();
            let a = x.to_exact();
            let b = y.to_exact();
            let c = a + b;

            let z = (x + y).to_exact();
            assert!((c - z).abs() <= error_bound);
        }

        // According to our analysis, the error in calculating a horizontal position should be
        // at most 8 times the base accuracy.
        // Empirically, changing the 21 to 22 doesn't break the test so either our bound is loose
        // or proptest isn't finding the worst case. Changing it to 23 finds a failure, though.
        #[test]
        fn horizontal_accuracy((s, y) in f32_segment_and_y()) {
            let error_bound = Rational::try_from(2e6f64 * (-21.0f64).exp2()).unwrap();
            let s1 = s.to_exact();
            let y1 = y.to_exact();

            assert!((s1.at_y(&y1) - s.at_y(&y).to_exact()).abs() <= error_bound);
        }

        // Check the analysis of y intersection heights. This one seems even looser than the horizontal position one:
        // our bound has a 64, but I had to reduce it to 2 before proptest found a counter-example.
        #[test]
        fn vertical_accuracy(s0 in Segment::<NotNan<f32>>::reasonable(), s1 in Segment::<NotNan<f32>>::reasonable()) {
            if s0.start.y > s1.end.y || s1.start.y > s0.end.y {
                return Ok(());
            }
            let t0 = s0.to_exact();
            let t1 = s1.to_exact();
            if t0.is_horizontal() || t1.is_horizontal() {
                return Ok(());
            }

            let inv_slope0 = ((t0.end.x.clone() - t0.start.x.clone()) / (t0.end.y.clone() - t0.start.y.clone())).abs();
            let inv_slope1 = ((t1.end.x.clone() - t1.start.x.clone()) / (t1.end.y .clone()- t1.start.y.clone())).abs();
            let inv_slope = inv_slope0.max(inv_slope1);

            let delta = Rational::try_from(2e6f64 * (-24.0f64).exp2() * 64.0).unwrap();
            let error_bound = delta.clone() * Rational::from(9i32) / Rational::from(16i32) * (Rational::from(1i32) + inv_slope);

            let y0 = t0.start.y.clone().max(t1.start.y.clone());
            let y1 = t0.end.y.clone().min(t1.end.y.clone());
            if t0.at_y(&y1) < t1.at_y(&y1) + (delta * Rational::from(3i32)) / Rational::from(4i32) {
                return Ok(());
            }
            if t0.at_y(&y0) > t1.at_y(&y0) {
                return Ok(());
            }

            let eps = NotNan::try_from(2e6f32 * (-19.0f32).exp2()).unwrap();
            let y = s0.crossing_y(&s1, &eps).unwrap().to_exact();
            let x0 = t0.at_y(&y);
            let x1 = t1.at_y(&y);
            assert!((x0 - x1).abs() <= error_bound);
        }
    }
}
