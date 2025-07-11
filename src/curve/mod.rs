//! Curve comparison utilities.
use arrayvec::ArrayVec;
use kurbo::{
    common::solve_cubic, Affine, CubicBez, Line, ParamCurve, PathSeg, QuadBez, Shape, Vec2,
};

mod split_quad;

#[derive(Clone, Debug, PartialEq)]
struct CurveOrderEntry {
    end: f64,
    order: Order,
}

impl CurveOrderEntry {
    fn flip(&self) -> Self {
        Self {
            end: self.end,
            order: match self.order {
                Order::Right => Order::Left,
                Order::Ish => Order::Ish,
                Order::Left => Order::Right,
            },
        }
    }
}

/// The outcome of analyzing two curves for horizontal ordering.
///
/// We assume that the two curves are monotonic in y, and we partition their
/// common y extent into sub-intervals, each of which gets assigned an order.
///
/// For example, consider the curves c0 and c1 below.
///
/// ```text
///       c0     c1
/// y=0    \      /
///         \    /
///          |  /
/// y=1     ─┼─
///        / │
///       /  │
/// y=2  /   │
/// ```
///
/// They start out with c0 to the left, cross over, and end up with c0 to the
/// right. We might represent this situation with a `CurveOrder` that says
///
/// - `Order::Left` on the interval `[0, 0.99]`
/// - `Order::Ish` in the interval `[0.99, 1.01]`, and
/// - `Order::Right` on the interval `[1.01, 2]`.
///
/// Note that this representation has some slack around the intersection point,
/// because we do everything numerically and there might be some errors.
#[derive(Clone, Debug)]
pub struct CurveOrder {
    start: f64,
    cmps: Vec<CurveOrderEntry>,
}

/// An iterator over intervals in a [`CurveOrder`].
pub struct CurveOrderIter<'a> {
    next: f64,
    iter: std::slice::Iter<'a, CurveOrderEntry>,
}

/// The different ways in which two curves can interact.
#[derive(Clone, Copy, Debug)]
pub enum CurveInteraction {
    /// The curves cross at a given height.
    Cross(f64),
    /// The curves touch at a given height, but don't actually cross.
    Touch(f64),
}

impl Iterator for CurveOrderIter<'_> {
    type Item = (f64, f64, Order);

    fn next(&mut self) -> Option<Self::Item> {
        let next_entry = self.iter.next()?;
        let start = self.next;
        self.next = next_entry.end;
        Some((start, next_entry.end, next_entry.order))
    }
}

// TODO: we've grown a weird set of convenience methods. Design them better.
impl CurveOrder {
    fn new(start: f64) -> Self {
        CurveOrder {
            start,
            cmps: Vec::new(),
        }
    }

    fn push(&mut self, end: f64, order: Order) {
        if let Some(last) = self.cmps.last_mut() {
            match (last.order, order) {
                (Order::Right, Order::Right)
                | (Order::Left, Order::Left)
                | (Order::Ish, Order::Ish) => {
                    debug_assert!(end >= last.end);
                    last.end = end;
                }
                (Order::Right, Order::Left) | (Order::Left, Order::Right) => {
                    // It would be nice if we could forbid this case, but the
                    // combination of almost-horizontal segments and errors in
                    // y position make it hard to avoid.
                    //
                    // What we do is insert a zero-height "ish" interval. Then
                    // adding y-slop will turn it into a taller interval.
                    debug_assert!(end >= last.end);
                    self.cmps.push(CurveOrderEntry {
                        end,
                        order: Order::Ish,
                    });
                    self.cmps.push(CurveOrderEntry { end, order });
                }
                _ => {
                    debug_assert!(end >= last.end);
                    if end > last.end {
                        self.cmps.push(CurveOrderEntry { end, order });
                    }
                }
            }
        } else {
            self.cmps.push(CurveOrderEntry { end, order });
        }
    }

    /// Returns an iterator over the ordering intervals.
    pub fn iter(&self) -> CurveOrderIter<'_> {
        CurveOrderIter {
            next: self.start,
            iter: self.cmps.iter(),
        }
    }

    /// Returns a `CurveOrder` with all the orderings flipped.
    pub fn flip(&self) -> Self {
        Self {
            start: self.start,
            cmps: self.cmps.iter().map(CurveOrderEntry::flip).collect(),
        }
    }

    /// Imagine that our curve is to the left of the other curve at height `y`.
    /// What's the next height at which they touch?
    ///
    /// We "imagine" that our curve is to the left, but we don't actually insist
    /// on it: if our curve is actually to the right at `y` we just say that
    /// they cross immediately.
    pub fn next_touch_after(&self, y: f64) -> Option<CurveInteraction> {
        let mut iter = self
            .iter()
            .skip_while(|(_start, end, _order)| end <= &y)
            .skip_while(|(_start, _end, order)| *order == Order::Left);

        let (y0, y1, order) = iter.next()?;
        if order == Order::Right {
            return Some(CurveInteraction::Cross(y));
        }

        // If this interval is the last one, we'll say there's no touch.
        // It will get handled at the endpoint anyway.
        let (_, next_y1, next_order) = iter.next()?;

        let cross_y = (y0 + y1) / 2.0;
        match next_order {
            Order::Right => Some(CurveInteraction::Cross(cross_y)),
            Order::Left => {
                if y < y0 {
                    Some(CurveInteraction::Touch(cross_y))
                } else {
                    self.next_touch_after(next_y1)
                }
            }
            Order::Ish => {
                // The current order is Ish, and adjacent orders get merged
                // so the next one isn't Ish.
                unreachable!();
            }
        }
    }

    /// Adds some vertical imprecision to the order comparison.
    ///
    /// To understand why we need this, note that
    /// - All our computations are approximate, and so the computed `y` values where orders
    ///   change will not be exact. When curves are almost horizontal, this error in `y`
    ///   can lead to a large error in `x`.
    /// - For the sweep-line algorithm to work, we need the strong order to have no loops:
    ///   if at some height `y`, `c0` is left of `c1` and `c1` is left of `c2` then `c0`
    ///   is not allowed to be left of `c2`.
    ///
    /// Together, these two properties present a problem: the approximation
    /// error in `y` can lead to ordering loops for some specific heights `y`
    /// where there's a lot of crossing action. (To be honest, I'm not sure that this ever
    /// actually happens. Fuzzing didn't find an example, and I didn't try too hard to
    /// find one manually. But the possibility of ordering loops keeps me up at night,
    /// so let's just assume that they might happen.)
    ///
    /// Our solution to this issue is to admit that all of our `y` values are imprecise,
    /// by expanding the "ish" regions. This method expands all the "ish" regions by `slop`
    /// in both directions. Geometrically, after applying y-slop, you end up with a comparison
    /// where `c0` is declared "left" of `c1` at `y` if a small square around the point on `c0`
    /// at height `y` stays to the left of `c1`. (If one segment is shorter than the other,
    /// this is not quite true near the endpoints of the shorter segment. But close enough.)
    pub fn with_y_slop(self, slop: f64) -> CurveOrder {
        if slop == 0.0 {
            return self;
        }

        let mut ret = Vec::new();

        // unwrap: cmps is always non-empty
        let last_end = self.cmps.last().unwrap().end;

        if self.start == last_end {
            return self;
        }

        for (start, end, order) in self.iter() {
            let new_end = if end == last_end { end } else { end - slop };
            if order != Order::Ish {
                let new_start = if start == self.start {
                    start
                } else {
                    start + slop
                };
                if new_start < new_end {
                    if new_start != self.start {
                        ret.push(CurveOrderEntry {
                            end: new_start,
                            order: Order::Ish,
                        });
                    }
                    ret.push(CurveOrderEntry {
                        end: new_end,
                        order,
                    });
                }
            }
        }

        if ret.last().is_none_or(|last| last.end != last_end) {
            ret.push(CurveOrderEntry {
                end: last_end,
                order: Order::Ish,
            });
        }

        CurveOrder {
            start: self.start,
            cmps: ret,
        }
    }

    /// Asserts that we satisfy our internal invariants. For testing only.
    pub fn check_invariants(&self) {
        let mut cmps = self.cmps.iter();
        let mut last = cmps.next().unwrap();
        for cmp in cmps {
            assert!(last.end <= cmp.end);
            assert!(last.order != cmp.order);
            assert!(last.order == Order::Ish || cmp.order == Order::Ish);
            last = cmp;
        }
    }

    /// What's the order at `y`?
    ///
    /// If `y` is at the boundary of two intervals, takes the first one.
    ///
    /// TODO: maybe it would be more consistent with the other methods if
    /// we took the second one. But then we need to be careful about what
    /// happens at the endpoint...
    ///
    /// # Panics
    ///
    /// Panics if `y` is outside the range of our ordering.
    pub fn order_at(&self, y: f64) -> Order {
        self.iter()
            .find(|(_start, end, _order)| end >= &y)
            .unwrap()
            .2
    }

    /// Returns the first order entry ending after `y`.:w
    ///
    /// # Panics
    ///
    /// Panics if our comparison range ends at or before `y`.
    pub fn entry_at(&self, y: f64) -> (f64, f64, Order) {
        self.iter().find(|(_start, end, _order)| *end > y).unwrap()
    }

    /// What's the next definite (`Left` or `Right`) ordering after `y`?
    ///
    /// If there is no definite ordering (everything after `y` is just `Ish`),
    /// returns `Ish`.
    ///
    /// As a corner case, if this comparison ends exactly at `y` then we return
    /// the ordering exactly at `y`.
    pub fn order_after(&self, y: f64) -> Order {
        if y == self.cmps.last().unwrap().end {
            self.cmps.last().unwrap().order
        } else {
            self.iter()
                .skip_while(|(_start, end, _order)| end <= &y)
                .find(|(_start, _end, order)| *order != Order::Ish)
                .map_or(Order::Ish, |(_start, _end, order)| order)
        }
    }
}

/// Find the parameter `t` at which `c` crosses height `y`.
pub fn solve_t_for_y(c: CubicBez, y: f64) -> f64 {
    debug_assert!(c.p0.y <= y && y <= c.p3.y && c.p0.y < c.p3.y);

    if y == c.p0.y {
        return 0.0;
    }
    if y == c.p3.y {
        return 1.0;
    }
    let c3 = c.p3.y - 3.0 * c.p2.y + 3.0 * c.p1.y - c.p0.y;
    let c2 = 3.0 * (c.p2.y - 2.0 * c.p1.y + c.p0.y);
    let c1 = 3.0 * (c.p1.y - c.p0.y);
    let c0 = c.p0.y - y;

    let cubic = Cubic { c0, c1, c2, c3 };

    let eps = 1e-10 * cubic.max_coeff().max(1.0);
    let roots = cubic.roots_between(0.0, 1.0, eps);
    if !roots.is_empty() {
        return roots[0];
    }

    // There are situations (discovered by fuzzing) where because of rounding
    // the cubic doesn't actually change signs. (Mathematically, it does change
    // signs because it's `c.p0.y - y` at zero and `c.p3.y - y` at one, and we
    // checked that those have opposite signs.)
    //
    // In this situation, it must be that the cubic was very close to zero
    // at some endpoint, but after rounding the sign flipped.
    debug_assert_eq!(cubic.eval(0.0).signum(), cubic.eval(1.0).signum());
    if (y - c.p0.y).abs() <= (y - c.p3.y).abs() {
        0.0
    } else {
        1.0
    }
}

/// Finds the x coordinate at which `c` crosses through `y`.
pub fn solve_x_for_y(c: CubicBez, y: f64) -> f64 {
    c.eval(solve_t_for_y(c, y)).x
}

/// Restricts a Bézier curve to a vertical range.
///
/// The input curve should be monotonic in `y`, and its range should include
/// `y0` and `y1` (which should be ordered).
pub fn y_subsegment(c: CubicBez, y0: f64, y1: f64) -> CubicBez {
    debug_assert!(y0 < y1);
    debug_assert!(c.p0.y <= y0 && y1 <= c.p3.y);
    let t0 = solve_t_for_y(c, y0);
    let t1 = solve_t_for_y(c, y1);
    let mut ret = c.subsegment(t0..t1);
    ret.p0.y = y0;
    ret.p3.y = y1;
    ret
}

// Tries to solve a cubic, but only looks for accurate solutions in the interval [0.0, 1.0].
//
// This doesn't actually filter out solutions outside that interval, it only
// makes some tweaks for better numerical stability inside it.
fn solve_cubic_in_unit_interval(c0: f64, c1: f64, c2: f64, c3: f64) -> ArrayVec<f64, 3> {
    // Since we're only interested in small values of t, we can ignore c3 if it's
    // much smaller than the other coefficients.
    //
    // To explain where the 1e7 comes from, suppose we take a threshold of T.
    // By zeroing out c3, we're introducing error of order 1/T by modifying the
    // cubic. (For our applications, we care less about numerical stability of
    // the roots and more about the *value* at the roots being about zero.)
    // On the other hand, if c2 / c3 is of order T, when we use it to find roots
    // we'll have a relative error of about 1e-15, and so an absolute error of
    // about T * 1e-15 (because that's how accurate f64s are). Balancing out these
    // sources of error suggests we take T around 1e7.
    let mut new_c3 = c3;
    let mut new_c2 = c2;
    if c3.abs() < c2.abs().max(c1.abs()).max(c2.abs()) / 1e7 {
        new_c3 = 0.0;
        if c2.abs() < c1.abs().max(c0.abs()) / 1e7 {
            new_c2 = 0.0;
        }
    }
    let mut roots = solve_cubic(c0, c1, new_c2, new_c3);

    // Do a few Newton steps to increase accuracy. Also, we do this with the
    // original parameters, which helps reduce the error that we may have
    // introduced.
    //dbg!(c0, c1, c2, c3);
    for x in &mut roots {
        let mut val = c3 * *x * *x * *x + c2 * *x * *x + c1 * *x + c0;
        //dbg!(*x, val);
        let mut deriv = 3.0 * c3 * *x * *x + 2.0 * c2 * *x + c1;
        for _ in 0..3 {
            if val.abs() <= 1e-14 {
                break;
            }

            let step = val / deriv;
            // Truncate the step size, because of an annoying case. If the original
            // equation was (x - 1)^2 + eps * x^3, We'll perturb it and find that
            // perfect double-root at x = 1. But when we add back in eps * x^3, the
            // Newton step will be giant (independent of eps). We should restrict
            // it to more like sqrt(eps).
            //
            // Is there a more principled way to handle this?
            let step = step.abs().min(val.abs().sqrt()).copysign(step);
            //dbg!(c0, c1, c2, c3, *x);
            *x -= step;
            // let val_after = c3 * *x * *x * *x + c2 * *x * *x + c1 * *x + c0;
            // let deriv_after = 3.0 * c3 * *x * *x + 2.0 * c2 * *x + c1;
            // dbg!(*x, step, val, deriv, val_after, deriv_after);

            val = c3 * *x * *x * *x + c2 * *x * *x + c1 * *x + c0;
            deriv = 3.0 * c3 * *x * *x + 2.0 * c2 * *x + c1;
        }
    }
    roots
}

// Return two roots of this monic quadratic, the smaller one first.
//
// For weird cases, the precise meaning of the "roots" we return is that the
// quadratic is positive before the first root, negative between them, and
// positive after the second root.
pub(crate) fn monic_quadratic_roots(b: f64, c: f64) -> (f64, f64) {
    let disc = b * b - 4.0 * c;
    let root1 = if disc.is_finite() {
        if disc <= 0.0 {
            return (f64::NEG_INFINITY, f64::NEG_INFINITY);
        } else {
            // There are two choices for the sign here. This choice gives the
            // root with larger magnitude, which is better for numerical stability
            // because we're going to divide by it to find the other root.
            //
            // See https://math.stackexchange.com/questions/866331 for details.
            -0.5 * (b + disc.sqrt().copysign(b))
        }
    } else {
        // I don't understand the implementation at https://github.com/toastedcrumpets/stator/blob/f68de3ea091f21bbe6e36feab4e53bdf2ace868d/stator/symbolic/polynomial.hpp#L1053
        // Surely if b is about sqrt(f64::MAX) and c is around f64::MAX then the
        // approximation they take isn't valid? Here is my attempt:
        //
        // We're trying to compute
        //
        // b + sqrt(b^2 - 4 c)
        //
        // but the discriminant overflowed, so rewrite it as
        //
        // b (1 + sqrt(1 - 4 c / b^2))
        //
        // If the discriminant here overflows it must be because c dominates,
        // especially if we do the computation in this weird order:
        let scaled_disc = 1.0 - (c / b / b) * 4.0;
        if !scaled_disc.is_finite() {
            if c > 0.0 {
                // disc must have overflowed to -infinity. There are no real roots;
                // the polynomial is always positive.
                return (f64::NEG_INFINITY, f64::NEG_INFINITY);
            } else {
                // disc must have overflowed to +infinity, meaning that
                // there are two roots but they're so far out that the
                // equation is effectively constant. The constant term
                // is negative, so the quadratic is basically always
                // negative.
                return (f64::NEG_INFINITY, f64::INFINITY);
            }
        }
        -0.5 * b * (1.0 + scaled_disc.sqrt())
    };

    // root1 was the one with larger magnitude, so if it didn't overflow then neither does this.
    let root2 = c / root1;
    if root2 > root1 {
        (root1, root2)
    } else {
        (root2, root1)
    }
}

/// Specifies the roots of a quadratic, along with the signs taken before, after, and between the roots.
#[derive(Clone, Copy, Debug)]
pub struct QuadraticSigns {
    /// The smaller root.
    pub smaller_root: f64,
    /// The bigger root.
    pub bigger_root: f64,
    /// The sign of the quadratic in the limit at `f64::NEG_INFINITY`.
    ///
    /// If this is zero, the quadratic is (approximately) zero everywhere.
    /// Otherwise, the quadratic takes this sign until the first root, then the
    /// opposite sign until the second root, then this sign again.
    pub initial_sign: f64,
}

/// A quadratic function in one variable, represented as `c2 * x^2 + c1 * x + c0`.
#[derive(Clone, Copy, Debug)]
pub struct Quadratic {
    /// The quadratic coefficient.
    pub c2: f64,
    /// The linear coefficient.
    pub c1: f64,
    /// The constant coefficient.
    pub c0: f64,
}

impl Quadratic {
    /// Finds the roots of this quadratic.
    pub fn signs(&self) -> QuadraticSigns {
        let Quadratic { c2, c1, c0 } = *self;
        debug_assert!(c2.is_finite() && c1.is_finite() && c0.is_finite());

        let disc = c1 * c1 - 4.0 * c2 * c0;
        if !disc.is_finite() {
            // If one of the coefficients is larger than about square root of
            // the biggest float, the discriminant could have overflowed. In
            // that case, scaling down won't change the location of the roots.
            //
            // The exponent maxes out at 1023, so scaling down by 2^{-512} is
            // enough to ensure that squaring doesn't overflow. We do an extra
            // factor of 2^{-3} for some wiggle room.
            let scale = 2.0f64.powi(-515);
            Quadratic {
                c2: c2 * scale,
                c1: c1 * scale,
                c0: c0 * scale,
            }
            .signs()
        } else if disc < 0.0 {
            QuadraticSigns {
                initial_sign: c2,
                smaller_root: f64::NEG_INFINITY,
                bigger_root: f64::NEG_INFINITY,
            }
        } else {
            let z = c1 + disc.sqrt().copysign(c1);
            let root1 = -2.0 * c0 / z;
            let root2 = -z / (2.0 * c2);

            // If z is zero, it means that c1 and disc are both (exactly!) zero,
            // and so everything is zero.
            if z == 0.0 {
                QuadraticSigns {
                    initial_sign: 0.0,
                    smaller_root: f64::NEG_INFINITY,
                    bigger_root: f64::NEG_INFINITY,
                }
            } else {
                // Here z is non-zero, meaning that root1 and root2 are defined
                // (although possibly infinite).
                debug_assert!(!root1.is_nan() && !root2.is_nan());

                let initial_sign = if root2.is_infinite() { c1 } else { c2 };

                QuadraticSigns {
                    initial_sign,
                    smaller_root: root1.min(root2),
                    bigger_root: root1.max(root2),
                }
            }
        }
    }

    /// Returns the largest coefficient (in absolute value).
    pub fn max_coeff(&self) -> f64 {
        self.c2.abs().max(self.c1.abs()).max(self.c0.abs())
    }

    /// Evaluates this quadratic at a point.
    pub fn eval(&self, t: f64) -> f64 {
        self.c2 * t * t + self.c1 * t + self.c0
    }

    /// Shift this quadratic "sideways".
    ///
    /// If this quadratic represents the function `q(x)`, returns a quadratic representing `q(x - shift)`.
    pub fn shift(self, shift: f64) -> Self {
        Self {
            c2: self.c2,
            c1: self.c1 - 2.0 * self.c2 * shift,
            c0: self.c0 + self.c2 * shift * shift - self.c1 * shift,
        }
    }
}

impl std::ops::Add<f64> for Quadratic {
    type Output = Quadratic;

    fn add(mut self, rhs: f64) -> Self::Output {
        self.c0 += rhs;
        self
    }
}

/// A cubic in one variable, represented as `c3 * x^3 + c2 * x^2 + c1 * x + c0`.
#[derive(Clone, Copy, Debug)]
pub struct Cubic {
    /// The cubic coefficient.
    pub c3: f64,
    /// The quadratic coefficient.
    pub c2: f64,
    /// The linear coefficient.
    pub c1: f64,
    /// The constant coefficient.
    pub c0: f64,
}

impl Cubic {
    /// Returns the cubic function representing the `y` coordinate of a cubic Bézier.
    pub fn from_bez_y(c: CubicBez) -> Self {
        let c3 = c.p3.y - 3.0 * c.p2.y + 3.0 * c.p1.y - c.p0.y;
        let c2 = 3.0 * (c.p2.y - 2.0 * c.p1.y + c.p0.y);
        let c1 = 3.0 * (c.p1.y - c.p0.y);
        let c0 = c.p0.y;
        Self { c3, c2, c1, c0 }
    }

    /// Returns the cubic function representing the `x` coordinate of a cubic Bézier.
    pub fn from_bez_x(c: CubicBez) -> Self {
        let c3 = c.p3.x - 3.0 * c.p2.x + 3.0 * c.p1.x - c.p0.x;
        let c2 = 3.0 * (c.p2.x - 2.0 * c.p1.x + c.p0.x);
        let c1 = 3.0 * (c.p1.x - c.p0.x);
        let c0 = c.p0.x;
        Self { c3, c2, c1, c0 }
    }

    /// Evaluates this cubic at a point.
    pub fn eval(&self, t: f64) -> f64 {
        self.c3 * t * t * t + self.c2 * t * t + self.c1 * t + self.c0
    }

    /// Returns the derivative of this cubic.
    pub fn deriv(&self) -> Quadratic {
        Quadratic {
            c2: 3.0 * self.c3,
            c1: 2.0 * self.c2,
            c0: self.c1,
        }
    }

    fn one_root(&self, mut lower: f64, mut upper: f64, accuracy: f64) -> f64 {
        let val_lower = self.eval(lower);
        let val_upper = self.eval(upper);
        debug_assert_ne!(val_lower.signum(), val_upper.signum());

        // We do one "binary search" step before the truncated Newton
        // iterations. If the range `(lower, upper)` doesn't include any
        // critical points, this guarantees that the Newton method converges (as
        // pointed out by Yuksel): Newton can only oscillate if it crosses an
        // inflection point, and chopping the inter-critical-point range in half
        // guarantees that it doesn't contain an inflection point.
        let mut x = (upper + lower) / 2.0;
        let mut val_x = self.eval(x);
        if val_x.signum() == val_lower.signum() {
            lower = x;
        } else {
            upper = x;
        }

        while val_x.abs() >= accuracy {
            let deriv_x = self.deriv().eval(x);

            let step = -val_x / deriv_x;
            x = (x + step).clamp(lower, upper);
            val_x = self.eval(x);
            //dbg!(val_x, deriv_x, x, step);
        }
        x
    }

    /// Computes all roots between `lower` and `upper`, to the desired accuracy.
    ///
    /// "Accuracy" is measured with respect to the cubic's value: if this cubic
    /// is called `f` and we find some `x` with `|f(x)| < accuracy` (and `x` is
    /// contained between two endpoints where `f` has opposite signs) then we'll
    /// call `x` a root.
    ///
    /// We make no guarantees about multiplicity. In fact, if there's a
    /// double-root that isn't a triple-root (and therefore has no sign change
    /// nearby) then there's a good chance we miss it altogether. This is
    /// fine if you're using this root-finding to optimize a quartic, because
    /// double-roots of the derivative aren't local extrema.
    pub fn roots_between(&self, lower: f64, upper: f64, accuracy: f64) -> ArrayVec<f64, 3> {
        let q = self.deriv();
        let q_roots = q.signs();

        let possible_endpoints = [q_roots.smaller_root, q_roots.bigger_root, upper];

        let mut last = lower;
        let mut last_sign = self.eval(last).signum();
        let mut ret = ArrayVec::new();

        for x in possible_endpoints {
            if x > last && x <= upper {
                let sign = self.eval(x).signum();
                if sign != last_sign {
                    ret.push(self.one_root(last, x, accuracy));
                }

                last = x;
                last_sign = sign;
            }
        }
        ret
    }

    /// Returns the largest absolute value of any coefficient.
    pub fn max_coeff(&self) -> f64 {
        self.c3
            .abs()
            .max(self.c2.abs())
            .max(self.c1.abs())
            .max(self.c0.abs())
    }
}

/// Analyze the sign of a quadratic.
///
/// Consider the quadratic equation `a y^2 + b y + c` for `y` between `y0` and `y1`, and some
/// thresholds `lower < upper`. We consider the quadratic "less" if it's smaller than `lower`,
/// "greater" if it's bigger than `upper`, and "ish" if it's between the two thresholds.
/// We push the results of this sign analysis to `out`.
#[allow(clippy::too_many_arguments)]
fn push_quadratic_signs(
    a: f64,
    b: f64,
    c: f64,
    lower: f64,
    upper: f64,
    y0: f64,
    y1: f64,
    out: &mut CurveOrder,
) {
    debug_assert!(lower < upper);

    let mut push = |end: f64, order: Order| {
        //dbg!(end, order);
        if end > y0
            && out
                .cmps
                .last()
                .is_none_or(|last| last.end < y1 && last.end < end)
        {
            out.push(end.min(y1), order);
        }
    };

    let scaled_c_lower = (c - lower) * a.recip();
    let scaled_c_upper = (c - upper) * a.recip();
    let scaled_b = b * a.recip();
    if !scaled_c_lower.is_finite() || !scaled_c_upper.is_finite() || !scaled_b.is_finite() {
        // a is zero or very small, treat as linear eqn
        let root0 = -(c - lower) / b;
        let root1 = -(c - upper) / b;
        if root0.is_finite() && root1.is_finite() {
            if b > 0.0 {
                push(root0, Order::Right);
                push(root1, Order::Ish);
                push(y1, Order::Left);
            } else {
                push(root1, Order::Left);
                push(root0, Order::Ish);
                push(y1, Order::Right);
            }
        } else if c < lower {
            // It's basically a constant, so we just need to check where
            // the constant is in comparison to our targets.
            push(y1, Order::Right);
        } else if c > upper {
            push(y1, Order::Left);
        } else {
            push(y1, Order::Ish);
        }
        return;
    }

    let (r_lower, s_lower) = monic_quadratic_roots(scaled_b, scaled_c_lower);
    let (r_upper, s_upper) = monic_quadratic_roots(scaled_b, scaled_c_upper);
    //dbg!(r_lower, s_lower, r_upper, s_upper);

    if a > 0.0 {
        debug_assert!(r_upper <= r_lower || r_lower.is_infinite());
        debug_assert!(s_lower <= s_upper);

        push(r_upper, Order::Left);
        push(r_lower, Order::Ish);
        push(s_lower, Order::Right);
        push(s_upper, Order::Ish);
        push(y1, Order::Left);
    } else {
        debug_assert!(r_upper >= r_lower || r_upper.is_infinite());
        debug_assert!(s_lower >= s_upper);

        push(r_lower, Order::Right);
        push(r_upper, Order::Ish);
        push(s_upper, Order::Left);
        push(s_lower, Order::Ish);
        push(y1, Order::Right);
    }
}

/// An approximate horizontal ordering.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Order {
    /// The first thing is to the right of the second thing.
    Right,
    /// The two things are close.
    Ish,
    /// The first thing is to the left of the second thing.
    Left,
}

impl Order {
    /// Returns the opposite ordering.
    pub fn flip(self) -> Order {
        match self {
            Order::Right => Order::Left,
            Order::Ish => Order::Ish,
            Order::Left => Order::Right,
        }
    }
}

/// An axis-aligned quadratic approximation to a cubic Bézier.
///
/// The quadratic here gives `y` as a function of `x`.
#[derive(Clone, Copy, Debug)]
pub struct EstParab {
    /// The constant coefficient of the quadratic approximation.
    /// (TODO: consider re-using [`Quadratic`] here)
    pub c0: f64,
    /// The linear coefficient of the quadratic approximation.
    pub c1: f64,
    /// The quadratic coefficient of the quadratic approximation.
    pub c2: f64,
    /// The amount by which the approximation undershoots the real
    /// value. Always non-positive.
    ///
    /// To be precise, the true `x` coordinate of the curve we're
    /// approximating will be at least `<quadratic approx> + dmin`.
    ///
    /// Note that while this value is intended to bound the error,
    /// this isn't strictly a guarantee: we use floating-point to
    /// compute it and we aren't careful with rounding direction.
    /// But because the basic algorithm has some slack built in,
    /// it's very likely that the slack dominates the floating-point
    /// error. So this *probably* is a bound, and our sweep-line
    /// algorithm treats it as one.
    pub dmin: f64,
    /// The amount by which the approximation overshoots the real
    /// value. Always non-negative.
    ///
    /// To be precise, the true `x` coordinate of the curve we're
    /// approximating will be at most `<quadratic approx> + dmax`.
    pub dmax: f64,
}

impl EstParab {
    /// Compute an approximation for a cubic Bézier, which we assume to be
    /// monotonically increasing in `y`.
    ///
    /// Note that this can produce very large coefficients if
    ///
    /// - the Bézier's y range is small, or
    /// - any of the y coordinates is large
    ///
    /// The second effect can be mitigated by translation (especially if
    /// the y range is small *and* the y coordinates are large, because then
    /// translation would make them all small), and the first effect can be
    /// mitigated by rescaling.
    pub fn from_cubic(c: CubicBez) -> Self {
        let seg = PathSeg::Cubic(c);
        let close_seg = PathSeg::Line(Line::new(c.p3, c.p0));
        let area = seg.area() + close_seg.area();
        let dy = c.p3.y - c.p0.y;
        // Note: this solution gives 0 error at endpoints. Arguably
        // a better solution would be based on mean / moments.
        let c2 = -6. * area / dy.powi(3);
        let c1 = (c.p3.x - c.p0.x - (c.p3.y.powi(2) - c.p0.y.powi(2)) * c2) / dy;
        let c0 = c.p0.x - c1 * c.p0.y - c2 * c.p0.y.powi(2);
        /*
        println!("{} {} {}", c0, c1, c2);
        let a = Affine::new([500., 0., 0., 500., 550., 10.]);
        print_svg(c);
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            let y = c.p0.lerp(c.p3, t).y;
            let x = c0 + c1 * y + c2 * y * y;
            let p = a * Point::new(x, y);
            println!("  <circle cx=\"{}\" cy=\"{}\" r=\"3\" />", p.x, p.y);
        }
        */
        // Hybrid bezier concept from North Masters thesis
        let q0 = QuadBez::new(c.p0, c.p0.lerp(c.p1, 1.5), c.p3);
        let q1 = QuadBez::new(c.p0, c.p3.lerp(c.p2, 1.5), c.p3);
        //print_svg(q0.raise());
        //print_svg(q1.raise());
        let mut dmin = 0.0f64;
        let mut dmax = 0.0f64;
        for q in [&q0, &q1] {
            // Solve tangency with estimated parabola
            // Maybe this should be a separate function?
            let params = quad_parameters(*q);
            let dparams = (params.1, 2. * params.2);
            // d.qy/dt * dpara.x/dy - dq.x/dt = 0
            // para.x = c0 + c1 * x + c2 * x^2
            // dpara.x/dy = c1 + 2 * c2 * x = d0 + d1 * x
            let d0 = c1;
            let d1 = 2. * c2;
            let dxdt0 = d0 + d1 * params.0.y;
            let dxdt1 = d1 * params.1.y;
            let dxdt2 = d1 * params.2.y;
            let f0 = dparams.0.y * dxdt0 - dparams.0.x;
            let f1 = dparams.0.y * dxdt1 + dparams.1.y * dxdt0 - dparams.1.x;
            let f2 = dparams.0.y * dxdt2 + dparams.1.y * dxdt1;
            let f3 = dparams.1.y * dxdt2;

            // I think this is the same as what's above, just re-derived because I didn't follow it...
            // let f0 = 2.0 * params.0.y * params.1.y * c2 + params.1.y * c1 - params.1.x;
            // let f1 = 2.0 * params.1.y * params.1.y * c2
            //     + 4.0 * params.0.y * params.2.y * c2
            //     + 2.0 * params.2.y * c1
            //     - 2.0 * params.2.x;
            // let f2 = 2.0 * params.1.y * params.2.y * c2 + 4.0 * params.1.y * params.2.y * c2;
            // let f3 = 4.0 * params.2.y * params.2.y * c2;
            for t in solve_cubic_in_unit_interval(f0, f1, f2, f3) {
                if (0.0..=1.0).contains(&t) {
                    let p = q.eval(t);
                    let x = p.x - (c0 + c1 * p.y + c2 * p.y.powi(2));
                    dmin = dmin.min(x);
                    dmax = dmax.max(x);
                }
                //println!("t = {}, pt = {:?}", t, q.eval(t));
            }
        }
        //println!("dmin = {}, dmax = {}", dmin, dmax);
        EstParab {
            c0,
            c1,
            c2,
            dmin,
            dmax,
        }
    }

    #[cfg(test)]
    fn eval(&self, y: f64) -> f64 {
        self.c0 + self.c1 * y + self.c2 * y * y
    }

    fn max_param(&self) -> f64 {
        self.c0.abs().max(self.c1.abs()).max(self.c2.abs())
    }
}

impl std::ops::Sub for EstParab {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        EstParab {
            c0: self.c0 - other.c0,
            c1: self.c1 - other.c1,
            c2: self.c2 - other.c2,
            dmin: self.dmin - other.dmax,
            dmax: self.dmax - other.dmin,
        }
    }
}

/// Get the parameters such that the curve can be represented by the following formula:
///     B(t) = c0 + c1 * t + c2 * t^2
pub fn quad_parameters(q: QuadBez) -> (Vec2, Vec2, Vec2) {
    let c0 = q.p0.to_vec2();
    let c1 = (q.p1 - q.p0) * 2.0;
    let c2 = c0 - q.p1.to_vec2() * 2.0 + q.p2.to_vec2();
    (c0, c1, c2)
}

fn with_rotation(
    c0: CubicBez,
    c1: CubicBez,
    y0: f64,
    y1: f64,
    tolerance: f64,
    accuracy: f64,
    out: &mut CurveOrder,
) -> bool {
    let dx0 = c0.p3.x - c0.p0.x;
    let dx1 = c1.p3.x - c1.p0.x;
    let dy = y1 - y0;

    if dx0.signum() != dx1.signum() {
        return false;
    }
    if dx0.abs() < dy * 16.0 || dx1.abs() < dy * 16.0 {
        return false;
    }

    let center = kurbo::Point::new((c0.p0.x + c0.p3.x) / 2.0, (y0 + y1) / 2.0);
    let theta = if dx0 > 0.0 {
        std::f64::consts::FRAC_PI_4
    } else {
        -std::f64::consts::FRAC_PI_4
    };
    let transform = Affine::rotate_about(theta, center);
    let c0_rot = transform * c0;
    let c1_rot = transform * c1;

    // Check that they're still y-monotonic after rotation (false negatives are allowed).
    if c0_rot.p1.y < c0_rot.p0.y || c0_rot.p2.y < c0_rot.p1.y || c0_rot.p3.y < c0_rot.p2.y {
        return false;
    }
    if c1_rot.p1.y < c1_rot.p0.y || c1_rot.p2.y < c1_rot.p1.y || c1_rot.p3.y < c1_rot.p2.y {
        return false;
    }

    let y0_rot = c0_rot.p0.y.max(c1_rot.p0.y);
    let y1_rot = c0_rot.p3.y.min(c1_rot.p3.y);

    if y0_rot >= y1_rot {
        let order = if c0.p0.x < c1.p0.x - tolerance {
            Order::Left
        } else if c0.p0.x > c1.p0.x + tolerance {
            Order::Right
        } else {
            Order::Ish
        };
        out.push(y1, order);
        return true;
    }

    let mut order_rot = CurveOrder::new(y0_rot);
    //dbg!(c0_rot, c1_rot, y0_rot, y1_rot);
    intersect_cubics_rec(
        c0_rot,
        c1_rot,
        y0_rot,
        y1_rot,
        tolerance,
        accuracy,
        &mut order_rot,
    );

    //dbg!(&order_rot);
    for (_new_y0, new_y1, order) in order_rot.iter() {
        let x1 = (solve_x_for_y(c0_rot, new_y1) + solve_x_for_y(c1_rot, new_y1)) / 2.0;
        let p1 = transform.inverse() * kurbo::Point::new(x1, new_y1);
        out.push(p1.y.clamp(y0, y1), order);
    }
    //dbg!(&out);
    out.cmps.last_mut().unwrap().end = y1;

    true
}

fn intersect_cubics_rec(
    orig_c0: CubicBez,
    orig_c1: CubicBez,
    y0: f64,
    y1: f64,
    tolerance: f64,
    accuracy: f64,
    out: &mut CurveOrder,
) {
    // eprintln!("recursing to {y0}..{y1}");
    let mut c0 = y_subsegment(orig_c0, y0, y1);
    let mut c1 = y_subsegment(orig_c1, y0, y1);
    // dbg!(c0, orig_c0);
    // dbg!(c1, orig_c1);

    if y1 - y0 < accuracy {
        // For very short intervals there's some numerical instability in constructing the
        // approximating quadratics, so we just do a coarser comparison based on bounding
        // boxes.
        let b0 = Shape::bounding_box(&c0);
        let b1 = Shape::bounding_box(&c1);
        let order = if b1.min_x() >= b0.max_x() + tolerance {
            Order::Left
        } else if b0.min_x() >= b1.max_x() + tolerance {
            Order::Right
        } else {
            Order::Ish
        };
        out.push(y1, order);
        return;
    }
    if with_rotation(c0, c1, y0, y1, tolerance, accuracy, out) {
        debug_assert_eq!(out.cmps.last().unwrap().end, y1);
        return;
    }

    // If the y coordinates are very off-center, the quadratic coefficients become
    // large and cause instability. So we re-center and then compensate afterwards.
    let y_mid = (y0 + y1) / 2.0;
    let y_unscale = (y1 - y0).max(1.0);
    //let y_unscale = 1.0f64;
    let y_scale = y_unscale.recip();
    c0.p0.y = (c0.p0.y - y_mid) * y_scale;
    c0.p1.y = (c0.p1.y - y_mid) * y_scale;
    c0.p2.y = (c0.p2.y - y_mid) * y_scale;
    c0.p3.y = (c0.p3.y - y_mid) * y_scale;
    c1.p0.y = (c1.p0.y - y_mid) * y_scale;
    c1.p1.y = (c1.p1.y - y_mid) * y_scale;
    c1.p2.y = (c1.p2.y - y_mid) * y_scale;
    c1.p3.y = (c1.p3.y - y_mid) * y_scale;

    let ep0 = EstParab::from_cubic(c0);
    // println!("making ep1");
    let ep1 = EstParab::from_cubic(c1);
    // println!("done with ep1");
    let mut dep = ep1 - ep0;

    // If the quadratic coefficient is too large (which happens for
    // almost-horizontal pieces), we can't really trust the accuracy estimates,
    // so pad them a little.
    let max_coeff = ep0.max_param().max(ep1.max_param());
    let err = max_coeff * 1e-12;
    dep.dmax += err;
    dep.dmin -= err;
    // dbg!(ep0);
    // dbg!(ep1);
    // dbg!(dep);
    // ep1.brute_force_d(c1);
    let mut scratch = CurveOrder::new(y0 - y_mid);
    push_quadratic_signs(
        dep.c2,
        dep.c1,
        dep.c0,
        -dep.dmax - tolerance,
        -dep.dmin + tolerance,
        (y0 - y_mid) * y_scale,
        (y1 - y_mid) * y_scale,
        &mut scratch,
    );

    // Re-center the roots. It's important that the starting and ending positions
    // have no rounding error, so deal with them separately.
    //dbg!(&scratch);
    scratch.start = y0;
    for entry in &mut scratch.cmps {
        entry.end = (entry.end * y_unscale + y_mid).clamp(y0, y1);
    }
    scratch.cmps.last_mut().unwrap().end = y1;
    //dbg!(&scratch);

    // As an extra debug check, we do some point evaluations of our curves and
    // check that they agree with the orders we've assigned. First, add some error
    // bars in the y direction.
    for (new_y0, new_y1, order) in scratch.clone().with_y_slop(accuracy).iter() {
        // dbg!(new_y0, new_y1, order);
        // dbg!(ep0.eval(new_y0 - y_mid));
        // dbg!(ep1.eval(new_y0 - y_mid));
        // dbg!(solve_x_for_y(c0, new_y0 - y_mid));
        // dbg!(solve_x_for_y(orig_c0, new_y0));
        // dbg!(solve_x_for_y(orig_c1, new_y0));
        // dbg!(solve_t_for_y(orig_c0, new_y1));
        // let t = solve_t_for_y(orig_c1, new_y1);
        // dbg!(ep0.eval(dbg!(orig_c1.eval(t).y)));
        // dbg!(orig_c1);
        // let t = solve_t_for_y(c0, new_y0);
        // dbg!(t);
        if order == Order::Left {
            debug_assert!(solve_x_for_y(orig_c0, new_y0) < solve_x_for_y(orig_c1, new_y0));
            debug_assert!(solve_x_for_y(orig_c0, new_y1) < solve_x_for_y(orig_c1, new_y1));
        } else if order == Order::Right {
            debug_assert!(solve_x_for_y(orig_c0, new_y0) > solve_x_for_y(orig_c1, new_y0));
            debug_assert!(solve_x_for_y(orig_c0, new_y1) > solve_x_for_y(orig_c1, new_y1));
        }
    }

    //println!("ep1 - ep0 = {:?}", dep);
    for (new_y0, new_y1, order) in scratch.iter() {
        // dbg!(new_y0, new_y1, order);
        // dbg!(
        //     solve_x_for_y(orig_c0, new_y0),
        //     solve_x_for_y(orig_c1, new_y0)
        // );
        // dbg!(
        //     solve_x_for_y(orig_c0, new_y1),
        //     solve_x_for_y(orig_c1, new_y1)
        // );
        if order == Order::Ish {
            let mid = 0.5 * (new_y0 + new_y1);

            // We test the difference between y1 and y0, not new_y1 and new_y0.
            // This help reduce false positives where the error in dep is substantial
            // but it just barely had a root and so we picked up a very small
            // "ish" interval that's just an artifact. In this case, new_y0 and new_y1
            // will be very close, but we'll recurse one more time to get a better
            // quadratic approximation.
            //
            // TODO: investigate fuzz/artifacts/curve_order/crash-bee7187920a9fe9e38bbf40ce6ff8cd80774c7a7
            // more closely. It seems to recurse more than it should...
            if y1 - y0 <= accuracy || dep.dmax - dep.dmin <= accuracy {
                out.push(new_y1, order);
            } else if new_y1 - new_y0 < 0.5 * (y1 - y0) {
                intersect_cubics_rec(orig_c0, orig_c1, new_y0, new_y1, tolerance, accuracy, out);
            } else {
                // eprintln!(
                //     "recursing because interval didn't shrink: {y0} - {new_y0} - {new_y1} - {y1}, error = {}, scratch {:?}, x slope {}",
                //     dep.dmax - dep.dmin, scratch,
                //     (c0.p3.x - c0.p0.x) / (y1 - y0)
                // );
                intersect_cubics_rec(orig_c0, orig_c1, new_y0, mid, tolerance, accuracy, out);
                intersect_cubics_rec(orig_c0, orig_c1, mid, new_y1, tolerance, accuracy, out);
            }
        } else {
            out.push(new_y1, order);
        }
    }
}

/// Compute the horizontal order between two cubics, for the vertical range that they have in common.
///
/// The returned order breaks the vertical range into regions, and in each region either specifies
/// a definite order or says that the curves are close. For example, for the following two curves
/// we might identify five vertical regions.
///
/// ```text
/// c0      c1
///  \      /
///   \    /    left
///    \  /     ______
///     \/      close
///     /\      ______
///    /  \
///   (    \    right
///    \    )
///     \  /    ______
///      \/     close
///      /\     ______
///     /  \    left
/// ```
///
/// There are two "closeness" parameters, `tolerance` and `accuracy`. The
/// `tolerance` parameter determines (more-or-less) what "close" means. If `c0`
/// is more than `tolerance` to the left of `c1` then we return "left", if `c0`
/// is more than `tolerance` to the right of `c1` then we return "right", and
/// otherwise we return "close".
///
/// But because our algorithm is iterative and approximate, we also take
/// an `accuracy` parameter that determines how accurately we honor the
/// `tolerance`. So more precisely, if `c0` is more than `tolerance + accuracy`
/// on one side of `c1` then we definitely return that ordering, if `c0` and
/// `c1` are within `tolerance - accuracy` of one another then we definitely
/// return "close", and in the other cases we're allowed to return anything.
///
/// In practice, our calculations are also subject to floating point error,
/// which we don't account for. You probably shouldn't ask for tolerance and
/// accuracy that are too close to the limits of `f64` accuracy. We test with a
/// relative (to the magnitude of the cubic parameters) error of `1e-8`, so that
/// should be fine.
///
/// Finally, bear in mind that the returned vertical coordinates are
/// subject to floating-point errors as well, and so we may not pinpoint the
/// exact vertical coordinate where "left" transitions to "close". This is
/// particularly important when the curves are almost horizontal, because then
/// a small vertical error means a big horizontal error. You may want to apply
/// [`CurveOrder::with_y_slop`] to the returned orders.
pub fn intersect_cubics(c0: CubicBez, c1: CubicBez, tolerance: f64, accuracy: f64) -> CurveOrder {
    debug_assert!(tolerance > 0.0 && accuracy > 0.0 && accuracy <= tolerance);
    // dbg!(c0, c1);

    let y0 = c0.p0.y.max(c1.p0.y);
    let y1 = c0.p3.y.min(c1.p3.y);

    let mut ret = CurveOrder::new(y0);
    if y0 < y1 {
        intersect_cubics_rec(c0, c1, y0, y1, tolerance, accuracy, &mut ret);
    } else if y0 == y1 {
        // Neither of the curves should be purely horizontal, so it must be
        // that they just overlap at a single point.
        debug_assert!(c0.p0.y == c1.p3.y || c0.p3.y == c1.p0.y);

        let (x0, x1) = if c0.p0.y == c1.p3.y {
            (c0.p0.x, c1.p3.x)
        } else {
            (c0.p3.x, c1.p0.x)
        };
        let order = if x0 < x1 - tolerance {
            Order::Left
        } else if x0 > x1 + tolerance {
            Order::Right
        } else {
            Order::Ish
        };

        ret.push(y1, order);
    }
    //fix_up_endpoints(&mut ret, c0, c1, eps);
    debug_assert_eq!(ret.cmps.last().unwrap().end, y1);
    ret
}

#[cfg(any(test, feature = "arbitrary"))]
#[doc(hidden)]
pub mod arbtests {
    use arbitrary::Unstructured;
    use kurbo::ParamCurve as _;

    use super::{solve_t_for_y, Cubic};

    pub fn cubic_roots(u: &mut Unstructured) -> Result<(), arbitrary::Error> {
        let c = crate::arbitrary::cubic(1e8, u)?;

        // How much relative accuracy do we expect?
        let accuracy = 1e-11;
        let range = crate::arbitrary::float_in_range(1.0, 1e4, u)?;

        let size = c.c3.abs() * range.powi(3)
            + c.c2.abs() * range.powi(2)
            + c.c1.abs() * range
            + c.c0.abs();
        let threshold = accuracy * size.max(1.0);
        let roots = c.roots_between(-range, range, threshold);
        if c.eval(-range).signum() != c.eval(range).signum() {
            assert!(!roots.is_empty());
        }
        for r in roots {
            assert!(c.eval(r).abs() <= threshold);
        }

        Ok(())
    }

    pub fn solve_for_t(u: &mut Unstructured) -> Result<(), arbitrary::Error> {
        let c = crate::arbitrary::monotonic_bezier(u)?;
        if c.p0.y == c.p3.y {
            return Ok(());
        }

        // How much relative accuracy do we expect?
        // This was determined empirically: 1e-10 fails fuzz tests.
        let accuracy = 1e-9;
        let max_coeff = Cubic::from_bez_x(c)
            .max_coeff()
            .max(Cubic::from_bez_y(c).max_coeff());
        let threshold = accuracy * max_coeff.max(1.0);

        let y = crate::arbitrary::float_in_range(c.p0.y, c.p3.y, u)?;
        let t = solve_t_for_y(c, y);
        assert!((c.eval(t).y - y).abs() <= threshold);

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::Segment;

    use super::*;

    #[test]
    fn solve_for_t_accuracy() {
        let c = CubicBez {
            p0: (0.5, 0.0).into(),
            p1: (0.19531249655301508, 0.6666666864572713).into(),
            p2: (0.00781250000181899, 1.0).into(),
            p3: (0.0, 1.0).into(),
        };
        let y0 = 0.0;
        let y1 = 0.9999999406281859;
        let c0 = c.subsegment(solve_t_for_y(c, y0)..solve_t_for_y(c, y1));
        assert!((dbg!(c0.p3.y) - dbg!(y1)).abs() < 1e-8);
        let y1 = 0.9999999;
        let c0 = c.subsegment(solve_t_for_y(c, y0)..solve_t_for_y(c, y1));
        assert!((dbg!(c0.p3.y) - dbg!(y1)).abs() < 1e-8);
    }

    #[test]
    fn solve_for_t_accuracy2() {
        let c = CubicBez {
            p0: (0.9999999406281859, 0.0).into(),
            p1: (0.011764705882352941, 0.011764705882352941).into(),
            p2: (0.011764705882352941, 0.023391003460207612).into(),
            p3: (0.011764705882352941, 0.03488052106655811).into(),
        };
        let y = 0.012468608207852864;
        let t = solve_t_for_y(c, y);
        assert!(dbg!(c.eval(t).y - y).abs() < 1e-8);
    }

    #[test]
    fn solve_for_t_accuracy3() {
        let c = CubicBez {
            p0: (0.1443023801753, -0.00034678801186988073).into(),
            p1: (0.3760345290602, -0.00011491438194505266).into(),
            p2: (0.6070068772783, 0.00011680717653411721).into(),
            p3: (0.8372203215247, 0.0003483767633017387).into(),
        };

        let y = 0.00016929232880729566;
        let t = solve_t_for_y(c, y);
        assert!(dbg!(c.eval(t).y - y).abs() < 1e-8);
    }

    #[test]
    fn cubic_roots_arbtest() {
        arbtest::arbtest(arbtests::cubic_roots);
    }

    #[test]
    fn solve_for_t_arbtest() {
        arbtest::arbtest(arbtests::solve_for_t);
    }

    #[test]
    fn non_touching() {
        let eps = 0.01;
        let tol = 0.02;
        let a = Line::new((-1e6, 0.0), (0.0, 1.0));
        let b = Line::new((2.5 * eps, 0.0), (2.5 * eps, 1.0));
        let a = PathSeg::Line(a).to_cubic();
        let b = PathSeg::Line(b).to_cubic();

        let cmp = intersect_cubics(a, b, tol, eps);
        assert_eq!(cmp.order_at(1.0), Order::Left);
    }

    #[test]
    fn quad_approximation() {
        let c = CubicBez {
            p0: (0.1443023801753, -0.00034678801186988073).into(),
            p1: (0.3760345290602, -0.00011491438194505266).into(),
            p2: (0.6070068772783, 0.00011680717653411721).into(),
            p3: (0.8372203215247, 0.0003483767633017387).into(),
        };

        let y = 0.00016929232880729566;
        let t = solve_t_for_y(c, y);
        dbg!(t, c.eval(t));

        let ep = EstParab::from_cubic(c);
        dbg!(y);
        dbg!(ep);
        dbg!(ep.eval(y));
        dbg!(solve_x_for_y(c, y));

        let diff = solve_x_for_y(c, y) - ep.eval(y);
        assert!(diff <= ep.dmax);
        assert!(diff >= ep.dmin);
    }

    // At some point, this test case looped infinitely.
    #[test]
    fn test_loop() {
        let c0 = CubicBez {
            p0: (0.5, 0.0).into(),
            p1: (0.0014551791831513819, 0.9999847770668531).into(),
            p2: (0.9999999850988388, 1.0).into(),
            p3: (0.0, 1.0).into(),
        };
        let c1 = CubicBez {
            p0: (0.5, 0.0).into(),
            p1: (0.0014551791831513819, 0.9999847770668531).into(),
            p2: (1.0, 0.9999999999999717).into(),
            p3: (0.0, 0.9999999999999717).into(),
        };

        dbg!(intersect_cubics(c0, c1, 1e-6, 1e-6));
    }

    #[test]
    fn y_slop_single_seg() {
        let order = CurveOrder {
            start: 0.0,
            cmps: vec![CurveOrderEntry {
                end: 1.0,
                order: Order::Right,
            }],
        };

        assert_eq!(order.clone().with_y_slop(0.5).cmps, order.cmps);
    }

    #[test]
    fn y_slop_no_height() {
        let order = CurveOrder {
            start: 1.0,
            cmps: vec![CurveOrderEntry {
                end: 1.0,
                order: Order::Right,
            }],
        };

        assert_eq!(order.clone().with_y_slop(0.5).cmps, order.cmps);
    }

    // Cases where two curves share just a single y coordinate.
    #[test]
    fn curve_order_no_overlap() {
        let c0 = CubicBez {
            p0: (0.0, 0.0).into(),
            p1: (0.0, 0.0).into(),
            p2: (0.0, 1.0).into(),
            p3: (0.0, 1.0).into(),
        };
        let c1 = CubicBez {
            p0: (1.0, 1.0).into(),
            p1: (1.0, 1.0).into(),
            p2: (1.0, 2.0).into(),
            p3: (1.0, 2.0).into(),
        };
        let order = intersect_cubics(c0, c1, 0.25, 0.125);
        assert_eq!(order.start, 1.0);
        assert_eq!(
            order.cmps,
            vec![CurveOrderEntry {
                end: 1.0,
                order: Order::Left
            }]
        );

        let c1 = CubicBez {
            p0: (1.0, 1.0).into(),
            p1: (1.0, 1.0).into(),
            p2: (0.0, 2.0).into(),
            p3: (0.0, 2.0).into(),
        };
        let order = intersect_cubics(c0, c1, 0.25, 0.125);
        assert_eq!(order.start, 1.0);
        assert_eq!(
            order.cmps,
            vec![CurveOrderEntry {
                end: 1.0,
                order: Order::Left
            }]
        );

        let c1 = CubicBez {
            p0: (0.1, 1.0).into(),
            p1: (0.1, 1.0).into(),
            p2: (0.0, 2.0).into(),
            p3: (0.0, 2.0).into(),
        };
        let order = intersect_cubics(c0, c1, 0.25, 0.125);
        assert_eq!(order.start, 1.0);
        assert_eq!(
            order.cmps,
            vec![CurveOrderEntry {
                end: 1.0,
                order: Order::Ish
            }]
        );
    }

    #[test]
    fn graphite_example() {
        // Fiddling with graphite turned up this example, in which c0 and c1 compare "ish",
        // c1 compares left of c2, and c0 compares right of c2. That's all fine: c1, c2, c0
        // would be a valid sweep-line order.
        //
        // But suppose that we start with c0, c1 and then try to insert c2. Since it's to
        // the right of c1 even with the bigger thresholds, we think it's ok to put to the
        // right of c1, and then the bigger threshold comparison stops us from seeing c0
        // when scanning to the left.
        //
        // I think the solution has to be to re-introduce the close_before and close_after
        // stuff for y-slop.
        let eps = 1e-6;
        let tolerance = eps;
        let accuracy = eps / 2.0;
        let y = -227.53699416;
        let c0 = CubicBez::new(
            (-4.04445106, -227.53699448),
            (-4.0443963, -227.53699414000002),
            (-4.0443415400000005, -227.5369938),
            (-4.04428678, -227.53699347),
        );
        let c1 = CubicBez::new(
            (-4.04445106, -227.53699448),
            (-4.04445141, -227.53699414000002),
            (-4.04445176, -227.5369938),
            (-4.0444521, -227.53699347),
        );
        let c2 = CubicBez::new(
            (-4.04442515, -227.53699416),
            (-4.04443617, -227.53699411),
            (-4.0444375500000005, -227.53699405),
            (-4.04442929, -227.536994),
        );
        let cmp01 = intersect_cubics(c0, c1, tolerance, accuracy).with_y_slop(tolerance);
        let cmp02 = intersect_cubics(c0, c2, tolerance, accuracy).with_y_slop(tolerance);
        let cmp12 = intersect_cubics(c1, c2, tolerance, accuracy).with_y_slop(tolerance);

        dbg!(&cmp01, &cmp12, &cmp02);
        dbg!(solve_x_for_y(c0, y));
        dbg!(solve_x_for_y(c1, y));
        assert!(c2.p0.x >= dbg!(c1.p0.x.max(c1.p1.x).max(c1.p2.x).max(c1.p3.x)) + 2.0 * eps);

        let c0_seg = Segment::from_kurbo(c0);
        let c1_seg = Segment::from_kurbo(c1);
        let c2_seg = Segment::from_kurbo(c2);

        dbg!(c0_seg.lower(y, eps));
        dbg!(c1_seg.lower(y, eps));
        dbg!(c2_seg.lower(y, eps));
    }
}
