use std::collections::HashMap;

use kurbo::CubicBez;

use crate::{
    curve::{self, CurveOrder, Order},
    SegIdx, Segments,
};

fn close_start_y(c0: CubicBez, c1: CubicBez, tolerance: f64, accuracy: f64) -> f64 {
    if c0.p0.y == c1.p0.y {
        return f64::NEG_INFINITY;
    } else if c1.p0.y < c0.p0.y {
        return close_start_y(c1, c0, tolerance, accuracy);
    }

    let y0 = c0.p0.y.max(c1.p0.y - tolerance);
    let y1 = c1.p0.y;
    let x = c1.p0.x;
    let p0 = kurbo::Point::new(x, y0);
    let p3 = kurbo::Point::new(x, y1);
    let c1_extension = CubicBez {
        p0,
        p1: p0.lerp(p3, 1.0 / 3.0),
        p2: p0.lerp(p3, 2.0 / 3.0),
        p3,
    };

    let order = curve::intersect_cubics(c0, c1_extension, tolerance, accuracy);
    order
        .iter()
        .filter(|(_, _, order)| *order == Order::Ish)
        .last()
        .map(|(_y0, y1, _order)| y1)
        .unwrap_or(f64::NEG_INFINITY)
}

fn close_end_y(c0: CubicBez, c1: CubicBez, tolerance: f64, accuracy: f64) -> f64 {
    if c0.p3.y == c1.p3.y {
        return f64::INFINITY;
    } else if c1.p3.y > c0.p3.y {
        return close_end_y(c1, c0, tolerance, accuracy);
    }

    let y0 = c1.p3.y;
    let y1 = c0.p3.y.min(c1.p3.y + tolerance);
    let x = c1.p3.x;
    let p0 = kurbo::Point::new(x, y0);
    let p3 = kurbo::Point::new(x, y1);
    let c1_extension = CubicBez {
        p0,
        p1: p0.lerp(p3, 1.0 / 3.0),
        p2: p0.lerp(p3, 2.0 / 3.0),
        p3,
    };

    let order = curve::intersect_cubics(c0, c1_extension, tolerance, accuracy);
    order
        .iter()
        .find(|(_, _, order)| *order == Order::Ish)
        .map(|(y0, _y1, _order)| y0)
        .unwrap_or(f64::INFINITY)
}

#[derive(Clone, Debug)]
pub struct ComparisonCache {
    inner: HashMap<(SegIdx, SegIdx), CurveOrder>,
    accuracy: f64,
    tolerance: f64,
}

impl ComparisonCache {
    pub fn new(tolerance: f64, accuracy: f64) -> Self {
        ComparisonCache {
            inner: HashMap::new(),
            accuracy,
            tolerance,
        }
    }

    // FIXME: less cloning
    pub fn compare_segments(&mut self, segments: &Segments, i: SegIdx, j: SegIdx) -> CurveOrder {
        if let Some(order) = self.inner.get(&(i, j)) {
            return order.clone();
        }

        let segi = &segments[i];
        let segj = &segments[j];

        let c0 = segi.to_kurbo();
        let c1 = segj.to_kurbo();
        let forward = curve::intersect_cubics(c0, c1, self.tolerance, self.accuracy).with_y_slop(
            self.tolerance,
            close_start_y(c0, c1, self.tolerance, self.accuracy),
            close_end_y(c0, c1, self.tolerance, self.accuracy),
        );
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
        segments.add_points([(1.0, -0.8788), (1.0, 1.0)]);
        segments.add_points([(1.0, -1.0), (0.0, 0.0)]);
        let mut cmp_cache = ComparisonCache::new(eps, eps / 2.0);
        assert_eq!(
            cmp_cache
                .compare_segments(&segments, SegIdx(0), SegIdx(1))
                .iter()
                .next()
                .unwrap()
                .2,
            Order::Ish
        );
    }

    #[test]
    fn slop_regression_2() {
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
