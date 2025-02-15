#![allow(missing_docs)]
use std::borrow::BorrowMut;

use kurbo::{
    common::solve_cubic, CubicBez, Line, ParamCurve as _, ParamCurveExtrema, PathSeg, QuadBez,
    Shape, Vec2,
};

#[derive(Clone, Debug)]
struct CurveOrderEntry {
    end: f64,
    order: Ternary,
}

#[derive(Clone, Debug)]
pub struct CurveOrder {
    start: f64,
    cmps: Vec<CurveOrderEntry>,
}

pub struct CurveOrderIter<'a> {
    next: f64,
    iter: std::slice::Iter<'a, CurveOrderEntry>,
}

impl Iterator for CurveOrderIter<'_> {
    type Item = (f64, f64, Ternary);

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

    fn push(&mut self, end: f64, order: Ternary) {
        if let Some(last) = self.cmps.last_mut() {
            match (last.order, order) {
                (Ternary::Less, Ternary::Less)
                | (Ternary::Greater, Ternary::Greater)
                | (Ternary::Ish, Ternary::Ish) => {
                    debug_assert!(end >= last.end);
                    last.end = end;
                }
                (Ternary::Less, Ternary::Greater) | (Ternary::Greater, Ternary::Less) => {
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

    pub fn iter(&self) -> CurveOrderIter<'_> {
        CurveOrderIter {
            next: self.start,
            iter: self.cmps.iter(),
        }
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
    for t in solve_cubic(c0, c1, c2, c3) {
        if (0.0..=1.0).contains(&t) {
            return t;
        }
    }
    println!("{:?}", c);
    println!("{} {} {} {}", c0, c1, c2, c3);
    panic!("no solution found, y = {}", y);
}

// Return two roots of this monic quadratic, the smaller one first.
//
// For weird cases, the precise meaning of the "roots" we return is that the
// quadratic is positive before the first root, negative between them, and
// positve after the second root.
fn monic_quadratic_roots(b: f64, c: f64) -> (f64, f64) {
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

/// Consider the quadratic equation `a x^2 + b x + c` for `x` between `x0` and `x1`.
#[allow(clippy::too_many_arguments)]
fn push_quadratic_signs(
    a: f64,
    b: f64,
    c: f64,
    lower: f64,
    upper: f64,
    x0: f64,
    x1: f64,
    out: &mut CurveOrder,
) {
    debug_assert!(lower < upper);

    let mut push = |end: f64, order: Ternary| {
        if end > x0 && out.cmps.last().is_none_or(|last| last.end < x1) {
            out.push(end.min(x1), order);
        }
    };

    let c_lower = (c - lower) * a.recip();
    let c_upper = (c - upper) * a.recip();
    let b = b * a.recip();
    if !c_lower.is_finite() || !c_upper.is_finite() || !b.is_finite() {
        // a is zero or very small, treat as linear eqn
        let root0 = -(c - lower) / b;
        let root1 = -(c - upper) / b;
        if root0.is_finite() && root1.is_finite() {
            if b > 0.0 {
                push(root0, Ternary::Less);
                push(root1, Ternary::Ish);
                push(x1, Ternary::Greater);
            } else {
                push(root1, Ternary::Greater);
                push(root0, Ternary::Ish);
                push(x1, Ternary::Less);
            }
        } else if c < lower {
            // It's basically a constant, so we just need to check where
            // the constant is in comparison to our targets.
            push(x1, Ternary::Less);
        } else if c > upper {
            push(x1, Ternary::Greater);
        } else {
            push(x1, Ternary::Ish);
        }
        return;
    }

    let (r_lower, s_lower) = monic_quadratic_roots(b, c_lower);
    let (r_upper, s_upper) = monic_quadratic_roots(b, c_upper);

    if a > 0.0 {
        debug_assert!(r_upper <= r_lower || r_lower.is_infinite());
        debug_assert!(s_lower <= s_upper);

        push(r_upper, Ternary::Greater);
        push(r_lower, Ternary::Ish);
        push(s_lower, Ternary::Less);
        push(s_upper, Ternary::Ish);
        push(x1, Ternary::Greater);
    } else {
        debug_assert!(r_upper >= r_lower || r_upper.is_infinite());
        debug_assert!(s_lower >= s_upper);

        push(r_lower, Ternary::Less);
        push(r_upper, Ternary::Ish);
        push(s_upper, Ternary::Greater);
        push(s_lower, Ternary::Ish);
        push(x1, Ternary::Less);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ternary {
    Less,
    Ish,
    Greater,
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
    fn from_cubic(c: CubicBez) -> Self {
        //let c = c.subsegment(0.0..0.3);
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
            for t in solve_cubic(f0, f1, f2, f3) {
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
    eps: f64,
    out: &mut CurveOrder,
) {
    let c0 = orig_c0.subsegment(solve_t_for_y(orig_c0, y0)..solve_t_for_y(orig_c0, y1));
    let c1 = orig_c1.subsegment(solve_t_for_y(orig_c1, y0)..solve_t_for_y(orig_c1, y1));

    if y1 - y0 < eps {
        // For very short intervals there's some numerical instability in constructing the
        // approximating quadratics, so we just do a coarser comparison based on bounding
        // boxes.
        let b0 = Shape::bounding_box(&c0);
        let b1 = Shape::bounding_box(&c1);
        let order = if b1.min_x() >= b0.max_x() + eps {
            Ternary::Greater
        } else if b0.min_x() >= b1.max_x() + eps {
            Ternary::Less
        } else {
            Ternary::Ish
        };
        out.push(y1, order);
        return;
    }

    let ep0 = EstParab::from_cubic(c0);
    let ep1 = EstParab::from_cubic(c1);
    //println!("ep0 = {:?}", ep0);
    //println!("ep0 = {:?}", ep1);
    let dep = ep1 - ep0;
    let mut scratch = CurveOrder::new(y0);
    push_quadratic_signs(
        dep.c2,
        dep.c1,
        dep.c0,
        dep.dmin - eps,
        dep.dmax + eps,
        y0,
        y1,
        &mut scratch,
    );

    //println!("ep1 - ep0 = {:?}", dep);
    for (new_y0, new_y1, t) in scratch.iter() {
        if t == Ternary::Ish {
            let mid = 0.5 * (new_y0 + new_y1);

            // We test the difference between y1 and y0, not new_y1 and new_y0.
            // This help reduce false positives where the error in dep is substantial
            // but it just barely had a root and so we picked up a very small
            // "ish" interval that's just an artifact. In this case, new_y0 and new_y1
            // will be very close, but we'll recurse one more time to get a better
            // quadratic approximation.
            if y1 - y0 <= eps || dep.dmax - dep.dmin <= eps {
                out.push(new_y1, t);
            } else if new_y1 - new_y0 < 0.5 * (y1 - y0) {
                intersect_cubics_rec(orig_c0, orig_c1, new_y0, new_y1, eps, out);
            } else {
                intersect_cubics_rec(orig_c0, orig_c1, new_y0, mid, eps, out);
                intersect_cubics_rec(orig_c0, orig_c1, mid, new_y1, eps, out);
            }
        } else {
            out.push(new_y1, t);
        }
    }
}

pub fn intersect_cubics(c0: CubicBez, c1: CubicBez, eps: f64) -> CurveOrder {
    let y0 = c0.p0.y.max(c1.p0.y);
    let y1 = c0.p3.y.min(c1.p3.y);

    let mut ret = CurveOrder::new(y0);
    if y0 < y1 {
        intersect_cubics_rec(c0, c1, y0, y1, eps, &mut ret);
    }
    ret
}
