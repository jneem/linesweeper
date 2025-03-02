use kurbo::CubicBez;

use crate::curve::{self, CurveOrder};

#[derive(Clone, Debug)]
pub struct Comparison {
    /// The basic order between two curves.
    pub order: CurveOrder,
    /// A "sloppier" order between the curves.
    ///
    /// If `c0` is strictly less than `c1` according to this bound,
    /// then
    ///
    /// - `c0` is strictly less than `c1` according to `order`, but also
    /// - `c0` is strictly less (according to `order`) than every `c2` that's
    ///   bigger-than-or-ish-to `c1` (according to `order`).
    ///
    /// As far as the sweep-line is concerned, this means that whenever we
    /// see encounter a strict comparison with this `bound`, we can stop
    /// scanning for any intersections.
    pub bound: CurveOrder,
}

impl Comparison {
    pub fn new(c0: CubicBez, c1: CubicBez, eps: f64) -> Self {
        let order = curve::intersect_cubics(c0, c1, eps, eps / 2.0).with_y_slop(eps);
        let bound = curve::intersect_cubics(c0, c1, 4.0 * eps, eps / 2.0).with_y_slop(4.0 * eps);
        Self { order, bound }
    }
}
