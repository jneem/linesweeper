use kurbo::{BezPath, CubicBez};

use crate::{
    curve::{slice_bez, EstParab, Order},
    sweep::ComparisonCache,
    SegIdx, Segments,
};

struct PositionContext<'a> {
    segs: &'a Segments,
    cmp: &'a mut ComparisonCache,
    order: &'a [SegIdx],
    out: &'a mut [BezPath],
    y0: f64,
    y1: f64,
    accuracy: f64,
}

impl PositionContext<'_> {
    fn bump_y(&mut self, y0: f64) -> PositionContext<'_> {
        PositionContext {
            segs: self.segs,
            cmp: self.cmp,
            order: self.order,
            out: self.out,
            y0,
            y1: self.y1,
            accuracy: self.accuracy,
        }
    }

    // TODO: can we make this generic over all the ranges we want to use?
    fn slice(&mut self, range: std::ops::Range<usize>, y1: f64) -> PositionContext<'_> {
        PositionContext {
            segs: self.segs,
            cmp: self.cmp,
            order: &self.order[range.clone()],
            out: &mut self.out[range],
            y0: self.y0,
            y1,
            accuracy: self.accuracy,
        }
    }
}

pub fn ordered_curves(
    segs: &Segments,
    cmp: &mut ComparisonCache,
    order: &[SegIdx],
    y0: f64,
    y1: f64,
    accuracy: f64,
) -> Vec<kurbo::BezPath> {
    let mut dummy_out = Vec::new();
    let mut out = Vec::new();
    let mut ctx = PositionContext {
        segs,
        cmp,
        order,
        out: &mut dummy_out,
        y0,
        y1,
        accuracy,
    };

    // Find a valid initial point for each curve.
    let mut initial = Vec::new();
    horizontal_positions(ctx.order, ctx.y0, ctx.segs, ctx.accuracy, &mut initial);
    for (min_x, max_x) in initial {
        // TODO: be cleverer about the initial positions
        let mut path = BezPath::new();
        path.move_to(((min_x + max_x) / 2.0, y0));
        out.push(path);
    }
    ctx.out = &mut out;

    ordered_curves_inner(&mut ctx);
    out
}

fn ordered_curves_inner(ctx: &mut PositionContext<'_>) {
    eprintln!("recursing {:?} on {}..{}", ctx.order, ctx.y0, ctx.y1);
    let mut start_y = ctx.y0;

    'outer: while start_y < ctx.y1 {
        let mut next_y = ctx.y1;
        for i in 1..ctx.order.len() {
            let cmp =
                ctx.cmp
                    .compare_segments(ctx.segs, ctx.order[i - 1], ctx.order[i], ctx.accuracy);
            dbg!(&cmp);

            let (_start, end, order) = cmp
                .iter()
                .find(|(_start, end, _order)| end > &start_y)
                .unwrap();

            next_y = next_y.min(end);
            if order != Order::Ish {
                debug_assert_eq!(order, Order::Left);
                ordered_curves_all_close(&mut ctx.slice(0..i, next_y));
                ordered_curves_inner(&mut ctx.slice(i..ctx.order.len(), next_y));
                start_y = next_y;
                continue 'outer;
            }
        }

        // If we made it through that last loop, it means that everything is close.
        ordered_curves_all_close(&mut ctx.slice(0..ctx.order.len(), next_y));
        start_y = next_y;
    }
}

// This is copy-pasted and modified from sweep_line, because we wanted a slightly different
// interface.
fn horizontal_positions(
    entries: &[SegIdx],
    y: f64,
    segments: &Segments,
    eps: f64,
    out: &mut Vec<(f64, f64)>,
) {
    out.clear();
    let mut max_so_far = f64::NEG_INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut min = f64::INFINITY;

    for entry in entries {
        let seg = &segments[*entry];
        max_so_far = max_so_far.max(seg.lower(y, eps));
        // Fill out the minimum allowed positions, with a placeholder for the maximum.
        out.push((max_so_far, 0.0));

        let x = seg.at_y(y);
        max = max.max(x);
        min = min.min(x);
    }

    let mut min_so_far = f64::INFINITY;

    for ((min_allowed, max_allowed), seg_idx) in out.iter_mut().rev().zip(entries.iter().rev()) {
        let x = segments[*seg_idx].upper(y, eps);
        min_so_far = min_so_far.min(x);
        *max_allowed = min_so_far.min(max);
        *min_allowed = min_allowed.max(min);
    }
}

fn ordered_curves_all_close(ctx: &mut PositionContext<'_>) {
    eprintln!(
        "handling close curves {:?} from {}..{}",
        ctx.order, ctx.y0, ctx.y1
    );
    // TODO: special case for only one curve. Maybe also for two curves.

    let mut approxes = Vec::new();
    let mut y0 = ctx.y0;

    // TODO: rescaling the y interval to ensure stability
    while y0 < ctx.y1 {
        let next_y0 = approximate(&ctx.bump_y(y0), &mut approxes);

        let y_mid = (y0 + next_y0) / 2.0;
        let mut x_mid_max_so_far = f64::NEG_INFINITY;
        let mut x1_max_so_far = f64::NEG_INFINITY;
        for (quad, out_path) in approxes.iter().zip(ctx.out.iter_mut()) {
            let x_mid = quad.c0 + y_mid * quad.c1 + y0 * next_y0 * quad.c2;
            let x1 = quad.c0 + quad.c1 * next_y0 + quad.c2 * next_y0 * next_y0;

            x_mid_max_so_far = x_mid_max_so_far.max(x_mid);
            x1_max_so_far = x1_max_so_far.max(x1);

            out_path.quad_to((x_mid_max_so_far, y_mid), (x1_max_so_far, next_y0));
        }

        y0 = next_y0;
    }
}

fn horizontal_error_weight(c: CubicBez) -> f64 {
    let tangents = [c.p1 - c.p0, c.p2 - c.p1, c.p3 - c.p2];
    let mut weight = 0.0f64;

    fn different_signs(x: f64, y: f64) -> bool {
        (x > 0.0 && y < 0.0) || (x < 0.0) && (y > 0.0)
    }

    if different_signs(tangents[0].x, tangents[1].x)
        || different_signs(tangents[1].x, tangents[2].x)
    {
        return 1.0;
    }

    for v in tangents {
        let denom = v.hypot2().sqrt();
        if denom != 0.0 {
            let w = v.y.abs() / denom;
            weight = weight.max(w);
        }
    }
    weight
}

// Pushes approximating parabolas into `out`, and returns the final y position.
fn approximate(ctx: &PositionContext<'_>, out: &mut Vec<EstParab>) -> f64 {
    let mut y1 = ctx.y1;
    let orig_len = out.len();

    'retry: loop {
        for seg_idx in ctx.order {
            let cubic = ctx.segs[*seg_idx].to_kurbo();
            let cubic = slice_bez(cubic, ctx.y0, y1);
            let approx = EstParab::from_cubic(cubic);

            let factor = horizontal_error_weight(cubic);

            let too_short = y1 <= ctx.y0 + ctx.accuracy;
            let accurate = approx.dmax.max(-approx.dmin) < factor * ctx.accuracy;
            if !too_short && !accurate {
                y1 = (ctx.y0 + y1) / 2.0;
                out.truncate(orig_len);
                continue 'retry;
            }

            out.push(approx);
        }

        return y1;
    }
}

#[cfg(test)]
mod tests {
    use kurbo::{CubicBez, Line, PathSeg};

    use crate::{sweep::ComparisonCache, Point, Segments};

    use super::{horizontal_error_weight, ordered_curves};

    #[test]
    fn error_weight() {
        // Vertical lines have a weight of 1.
        let c: CubicBez = PathSeg::from(Line::new((0.0, 0.0), (0.0, 1.0))).to_cubic();
        assert_eq!(horizontal_error_weight(c), 1.0);

        // Horizontal lines have a weight of 0.
        let c: CubicBez = PathSeg::from(Line::new((1.0, 0.0), (0.0, 0.0))).to_cubic();
        assert_eq!(horizontal_error_weight(c), 0.0);

        let c: CubicBez = PathSeg::from(Line::new((0.0, 0.0), (1.0, 0.0))).to_cubic();
        assert_eq!(horizontal_error_weight(c), 0.0);

        // S-shaped curves have a weight of 1
        let c = CubicBez::new((0.0, 0.0), (1.0, 0.0), (-1.0, 1.0), (0.0, 1.0));
        assert_eq!(horizontal_error_weight(c), 1.0);

        // Diagonal lines have a weight of about 1/sqrt(2).
        let c: CubicBez = PathSeg::from(Line::new((0.0, 0.0), (1.0, 1.0))).to_cubic();
        assert_eq!(horizontal_error_weight(c), 1.0 / 2.0f64.sqrt());
    }

    fn mk_segs(xs: &[(f64, f64)]) -> Segments {
        let mut segs = Segments::default();

        for &(x0, x1) in xs {
            segs.add_points([Point::new(x0, 0.0), Point::new(x1, 1.0)]);
        }
        segs
    }

    #[test]
    fn baby_cases() {
        let segs = mk_segs(&[(0.0, 0.0), (1.0, 1.0)]);
        let order: Vec<_> = segs.indices().collect();
        let mut cmp = ComparisonCache::default();
        let out = ordered_curves(&segs, &mut cmp, &order, 0.0, 1.0, 1e-6);
        dbg!(out);

        let segs = mk_segs(&[(1e-12, 1e-12), (0.0, 0.0)]);
        let order: Vec<_> = segs.indices().collect();
        let mut cmp = ComparisonCache::default();
        let out = ordered_curves(&segs, &mut cmp, &order, 0.0, 1.0, 1e-6);
        dbg!(out);

        let segs = mk_segs(&[(0.0, 1.0), (2.0, 1.0)]);
        let order: Vec<_> = segs.indices().collect();
        let mut cmp = ComparisonCache::default();
        let out = ordered_curves(&segs, &mut cmp, &order, 0.0, 1.0, 1e-6);
        dbg!(out);
    }

    // TODO: add test for the provable ordering of outputs.
}
