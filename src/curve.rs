#![allow(missing_docs)]

use arrayvec::ArrayVec;
use kurbo::{common::solve_cubic, CubicBez, Line, ParamCurve as _, PathSeg, QuadBez, Shape, Vec2};

#[derive(Clone, Debug)]
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

pub struct CurveOrderIter<'a> {
    next: f64,
    iter: std::slice::Iter<'a, CurveOrderEntry>,
}

#[derive(Clone, Copy, Debug)]
pub enum NextTouch {
    Cross(f64),
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
                    dbg!(self);
                    panic!("no no no");
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
    pub fn next_touch_after(&self, y: f64) -> Option<NextTouch> {
        let mut iter = self
            .iter()
            .skip_while(|(_start, end, _order)| end <= &y)
            .skip_while(|(_start, _end, order)| *order == Order::Left);
        let (before_y, after_y, order_at_y) = iter.next()?;

        match order_at_y {
            Order::Right => Some(NextTouch::Cross(y)),
            Order::Ish => {
                // If this interval is the last one, we'll say there's no touch.
                // It will get handled at the endpoint anyway.
                let (_, _, next_order) = iter.next()?;
                let cross_y = (after_y + before_y) / 2.0;
                if next_order == Order::Right {
                    Some(NextTouch::Cross(cross_y))
                } else {
                    debug_assert!(next_order == Order::Left);
                    Some(NextTouch::Touch(cross_y))
                }
            }

            Order::Left => {
                // We skipped over these already.
                unreachable!()
            }
        }
    }

    // TODO: add unit tests
    // (and docs)
    pub fn with_y_slop(self, slop: f64) -> CurveOrder {
        let mut ret = Vec::new();

        if self.cmps.len() <= 1 {
            // TODO: think more carefully about corner cases at endpoints.
            return self;
        }

        // unwrap: cmps is always non-empty
        let last_end = self.cmps.last().unwrap().end;

        for (start, end, order) in self.iter() {
            let new_end = if end == last_end { end } else { end - slop };
            if order != Order::Ish && start + slop < new_end {
                if start == self.start {
                    ret.push(CurveOrderEntry {
                        end: new_end,
                        order,
                    });
                } else {
                    ret.push(CurveOrderEntry {
                        end: start + slop,
                        order: Order::Ish,
                    });
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

    pub fn check_invariants(&self) {
        let mut cmps = self.cmps.iter();
        let mut last = cmps.next().unwrap();
        for cmp in cmps {
            assert!(last.end < cmp.end);
            assert!(last.order != cmp.order);
            assert!(last.order == Order::Ish || cmp.order == Order::Ish);
            last = cmp;
        }
    }

    /// What's the order at `y`?
    ///
    /// If `y` is at the boundary of two intervals, takes the first one.
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

    /// What's the next definite (`Left` or `Right`) ordering after `y`?
    ///
    /// If there is no definite ordering (everything after `y` is just `Ish`),
    /// returns `Ish`.
    pub fn order_after(&self, y: f64) -> Order {
        self.iter()
            .skip_while(|(_start, end, _order)| end <= &y)
            .find(|(_start, _end, order)| *order != Order::Ish)
            .map_or(Order::Ish, |(_start, _end, order)| order)
    }
}

pub fn solve_t_for_y(c: CubicBez, y: f64) -> f64 {
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

    for t in solve_cubic_in_unit_interval(c0, c1, c2, c3) {
        if (0.0..=1.0).contains(&t) {
            return t;
        }
    }

    // The sharp cutoff at the endpoint can miss some legitimate internal
    // points. The 1e-4 threshold is a bit arbitrary, though.
    if (y - c.p3.y).abs() < 1e-4 {
        return 1.0;
    } else if (y - c.p0.y).abs() < 1e-4 {
        return 0.0;
    }
    println!("{:?}", c);
    println!("{} {} {} {}", c0, c1, c2, c3);
    panic!("no solution found, y = {}", y);
}

pub fn solve_x_for_y(c: CubicBez, y: f64) -> f64 {
    c.eval(solve_t_for_y(c, y)).x
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
// positve after the second root.
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Order {
    Right,
    Ish,
    Left,
}

impl Order {
    pub fn flip(self) -> Order {
        match self {
            Order::Right => Order::Left,
            Order::Ish => Order::Ish,
            Order::Left => Order::Right,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct EstParab {
    c0: f64,
    c1: f64,
    c2: f64,
    dmin: f64,
    dmax: f64,
}

impl EstParab {
    // c0 + c1 * y + c2 * y^2 is approx equal to cubic bez
    //
    // Note that this can produce very large coefficients if
    //
    // - dy is small
    // - any of the y coordinates is large
    //
    // Maybe the second effect could be mitigated by translation?
    // (Especially if dy is small *and* the y coordinates are large,
    // because then translation would make them all small.)
    fn from_cubic(mut c: CubicBez) -> Self {
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
                //dbg!(t);
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

    fn eval(&self, y: f64) -> f64 {
        self.c0 + self.c1 * y + self.c2 * y * y
    }

    fn brute_force_d(&self, c: CubicBez) {
        let mut dmin = 0.0f64;
        let mut dmax = 0.0f64;
        let mut which_min = 0.0f64;
        for i in 0..10001 {
            let t = i as f64 / 10000.0;
            let p = c.eval(t);
            let x = p.x - self.eval(p.y);
            if x < dmin {
                which_min = t;
            }
            dmin = dmin.min(x);
            dmax = dmax.max(x);
        }

        // let t = 0.9392882704660391;
        // let p = c.eval(t);
        // let x = dbg!(p.x) - dbg!(self.eval(dbg!(p.y)));
        // dbg!(x);

        eprintln!("dmin {dmin:.7}, dmax {dmax:.7}, min at {which_min:.7}");
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

fn intersect_cubics_rec(
    orig_c0: CubicBez,
    orig_c1: CubicBez,
    y0: f64,
    y1: f64,
    tolerance: f64,
    accuracy: f64,
    out: &mut CurveOrder,
) {
    let mut c0 = orig_c0.subsegment(solve_t_for_y(orig_c0, y0)..solve_t_for_y(orig_c0, y1));
    let mut c1 = orig_c1.subsegment(solve_t_for_y(orig_c1, y0)..solve_t_for_y(orig_c1, y1));

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

    // If the y coordinates are very off-center, the quadratic coefficients become
    // large and cause instability. So we re-center and then compensate afterwards.
    let y_mid = (y0 + y1) / 2.0;
    c0.p0.y -= y_mid;
    c0.p1.y -= y_mid;
    c0.p2.y -= y_mid;
    c0.p3.y -= y_mid;
    c1.p0.y -= y_mid;
    c1.p1.y -= y_mid;
    c1.p2.y -= y_mid;
    c1.p3.y -= y_mid;

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
        y0 - y_mid,
        y1 - y_mid,
        &mut scratch,
    );

    // Re-center the roots. It's important that the starting and ending positions
    // have no rounding error, so deal with them separately.
    scratch.start = y0;
    for entry in &mut scratch.cmps {
        entry.end += y_mid;
    }
    scratch.cmps.last_mut().unwrap().end = y1;

    // As an extra debug check, we do some point evaluations of our curves and
    // check that they agree with the orders we've assigned. First, add some error
    // bars in the y direction.
    for (new_y0, new_y1, order) in scratch.clone().with_y_slop(accuracy).iter() {
        //dbg!(new_y0, new_y1, order);
        // dbg!(ep0.eval(new_y1));
        // dbg!(ep1.eval(new_y1));
        // dbg!(solve_x_for_y(orig_c0, new_y1));
        // dbg!(solve_x_for_y(orig_c1, new_y1));
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
        //dbg!(new_y0, new_y1, order);
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
                intersect_cubics_rec(orig_c0, orig_c1, new_y0, mid, tolerance, accuracy, out);
                intersect_cubics_rec(orig_c0, orig_c1, mid, new_y1, tolerance, accuracy, out);
            }
        } else {
            out.push(new_y1, order);
        }
    }
}

fn bboxes_disjoint(c0: CubicBez, c1: CubicBez, y0: f64, y1: f64, eps: f64) -> bool {
    let c0 = c0.subsegment(solve_t_for_y(c0, y0)..solve_t_for_y(c0, y1));
    let c1 = c1.subsegment(solve_t_for_y(c1, y0)..solve_t_for_y(c1, y1));
    let b0 = Shape::bounding_box(&c0);
    let b1 = Shape::bounding_box(&c1);
    b0.max_x() + eps < b1.min_x() || b1.min_x() + eps < b0.max_x()
}

// TODO: docme
fn fix_up_endpoints(order: &mut CurveOrder, c0: CubicBez, c1: CubicBez, eps: f64) {
    if order.cmps.len() <= 1 {
        return;
    }

    let first = order.cmps.first().unwrap();
    if first.order == Order::Ish && bboxes_disjoint(c0, c1, order.start, first.end, eps) {
        order.cmps.remove(0);
    }

    if order.cmps.len() <= 1 {
        return;
    }
    let last = order.cmps.last().unwrap();
    let prev = order.cmps.len() - 2;
    if last.order == Order::Ish && bboxes_disjoint(c0, c1, order.cmps[prev].end, last.end, eps) {
        order.cmps[prev].end = last.end;
        order.cmps.pop();
    }
}

pub fn intersect_cubics(c0: CubicBez, c1: CubicBez, tolerance: f64, accuracy: f64) -> CurveOrder {
    debug_assert!(tolerance > 0.0 && accuracy > 0.0 && accuracy <= tolerance);

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

#[cfg(test)]
mod test {
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
        assert!(dbg!(c.eval(t).y) - y.abs() < 1e-8);
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
}
