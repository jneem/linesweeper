//! Curve order comparisons, with caching.

use std::collections::HashMap;

use crate::{
    curve::{self, CurveOrder},
    SegIdx, Segments,
};

/// A cache for curve comparisons, so that each pair of curves needs to be compared at most once.
#[derive(Clone, Debug)]
pub struct ComparisonCache {
    inner: HashMap<(SegIdx, SegIdx), CurveOrder>,
    accuracy: f64,
    tolerance: f64,
}

impl ComparisonCache {
    /// Creates a new comparison cache.
    ///
    /// `tolerance` tells us how close two curves can be to be declared "ish", and
    /// `accuracy` tells us how closely we need to evaluate the tolerance. For
    /// example, if `accuracy` is `tolerance / 2.0` then we'll guarantee (up to
    /// some floating-point error) that if the two curves are further than
    /// `1.5 * tolerance` apart then we'll give them a strict order, and if they're
    /// less than `tolerance / 2.0` apart then we'll give then an "ish" order.
    pub fn new(tolerance: f64, accuracy: f64) -> Self {
        ComparisonCache {
            inner: HashMap::new(),
            accuracy,
            tolerance,
        }
    }

    /// Compares two segments, returning their order.
    ///
    /// TODO: this API will change to something with less cloning
    pub fn compare_segments(&mut self, segments: &Segments, i: SegIdx, j: SegIdx) -> CurveOrder {
        if let Some(order) = self.inner.get(&(i, j)) {
            return order.clone();
        }

        let segi = &segments[i];
        let segj = &segments[j];

        let c0 = segi.to_kurbo();
        let c1 = segj.to_kurbo();
        let forward = curve::intersect_cubics(c0, c1, self.tolerance, self.accuracy)
            .with_y_slop(self.tolerance);
        let reverse = forward.flip();
        self.inner.insert((j, i), reverse);
        self.inner.entry((i, j)).insert_entry(forward).get().clone()
    }
}

#[cfg(test)]
mod tests {
    use crate::{curve::Order, SegIdx, Segments};

    use super::ComparisonCache;

    #[test]
    fn slop_regression() {
        let mut segments = Segments::default();
        let eps = 0.1;
        segments.add_points([(-0.5, -0.5), (-0.5, 0.5)]);
        segments.add_points([(0.0, 0.0), (0.0, 1.0)]);
        let mut cmp_cache = ComparisonCache::new(eps, eps / 2.0);
        assert_eq!(
            cmp_cache
                .compare_segments(&segments, SegIdx(0), SegIdx(1))
                .iter()
                .next()
                .unwrap()
                .2,
            Order::Left
        );
    }
}
