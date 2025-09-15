//! An algorithm for laying out curves in a given order.
//!
//! Our boolean ops algorithms work by first deciding horizontal orderings for curves
//! (with some guarantees about being approximately correct)
//! and then laying out the curves in a way that has the horizontal orders we decided
//! on. For this second step, which is the task of this module, we really
//! guarantee correctness: the curves that we output really have the specified
//! orders.

use std::collections::BinaryHeap;

use kurbo::{BezPath, CubicBez};

use crate::{
    curve::{transverse::transversal_after, y_subsegment, EstParab, Order},
    num::CheapOrderedFloat,
    order::ComparisonCache,
    topology::{HalfOutputSegVec, OutputSegIdx, OutputSegVec, ScanLineOrder},
    SegIdx, Segment, Segments,
};

// TODO: maybe a Context type to keep the parameters in check?
#[allow(clippy::too_many_arguments)]
fn ordered_curves_all_close(
    segs: &Segments,
    order: &[SegIdx],
    output_order: &[OutputSegIdx],
    out: &mut OutputSegVec<(BezPath, Option<usize>)>,
    mut y0: f64,
    y1: f64,
    endpoints: &HalfOutputSegVec<kurbo::Point>,
    accuracy: f64,
) {
    if order.len() == 2 {
        let s0 = order[0];
        let s1 = order[1];
        let o0 = output_order[0];
        let o1 = output_order[1];

        let p0 = bez_end(&out[o0].0);
        let p1 = bez_end(&out[o1].0);

        if p0.y == y0 && p1.y == y0 {
            let c0 = next_subsegment(&segs[s0], &out[o0].0, y1, endpoints[o0.second_half()]);
            let c1 = next_subsegment(&segs[s1], &out[o1].0, y1, endpoints[o1.second_half()]);

            if transversal_after(c0, c1, y1) {
                return;
            }
        }

        // TODO: for checking transversal_before, we need the endpoint.
    }

    // Ensure everything in `out` goes up to `y0`. Anything that doesn't go up to `y0` is
    // an output where we can just copy from the input.
    for (&seg_idx, &out_idx) in order.iter().zip(output_order) {
        let out_bez = &mut out[out_idx].0;
        if bez_end_y(out_bez) < y0 {
            let c = next_subsegment(
                &segs[seg_idx],
                out_bez,
                y0,
                endpoints[out_idx.second_half()],
            );
            out[out_idx].1 = Some(out[out_idx].0.elements().len() - 1);
            out[out_idx].0.curve_to(c.p1, c.p2, c.p3);
        }
    }

    let mut approxes = Vec::new();

    // The recentering helps, but there are still some issues with approximation when almost horizontal
    while y0 < y1 {
        let next_y0 = approximate(y0, y1, order, segs, accuracy, &mut approxes);

        let y_mid = (y0 + next_y0) / 2.0;

        let mut x_mid_max_so_far = f64::NEG_INFINITY;
        let mut x1_max_so_far = f64::NEG_INFINITY;
        for (quad, out_idx) in approxes.iter().zip(output_order) {
            // These would be the control points, but since `approximate` recenters the y interval,
            // they're different...
            // let x_mid = quad.c0 + y_mid * quad.c1 + y0 * next_y0 * quad.c2;
            // let x1 = quad.c0 + quad.c1 * next_y0 + quad.c2 * next_y0 * next_y0;
            let x_mid = quad.c0 + (y0 - y_mid) * (next_y0 - y_mid) * quad.c2;
            let x1 = quad.c0
                + quad.c1 * (next_y0 - y_mid)
                + quad.c2 * (next_y0 - y_mid) * (next_y0 - y_mid);

            x_mid_max_so_far = x_mid_max_so_far.max(x_mid);
            x1_max_so_far = x1_max_so_far.max(x1);

            out[*out_idx]
                .0
                .quad_to((x_mid_max_so_far, y_mid), (x1_max_so_far, next_y0));
        }
        approxes.clear();

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
fn approximate(
    y0: f64,
    mut y1: f64,
    order: &[SegIdx],
    segs: &Segments,
    accuracy: f64,
    out: &mut Vec<EstParab>,
) -> f64 {
    let orig_len = out.len();

    'retry: loop {
        let y_mid = (y0 + y1) / 2.0;
        for seg_idx in order {
            let cubic = segs[*seg_idx].to_kurbo();
            let mut cubic = y_subsegment(cubic, y0, y1);
            cubic.p0.y -= y_mid;
            cubic.p1.y -= y_mid;
            cubic.p2.y -= y_mid;
            cubic.p3.y -= y_mid;
            let approx = EstParab::from_cubic(cubic);

            let factor = horizontal_error_weight(cubic);

            let too_short = y1 <= y0 + accuracy;
            let accurate = approx.dmax.max(-approx.dmin) * factor < accuracy;
            if !too_short && !accurate {
                // TODO: can we be smarter about this?
                y1 = (y0 + y1) / 2.0;
                out.truncate(orig_len);
                continue 'retry;
            }

            out.push(approx);
        }

        return y1;
    }
}

#[derive(Debug, PartialEq, Eq)]
struct HeapEntry {
    y: CheapOrderedFloat,
    idx: OutputSegIdx,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.y.cmp(&self.y).then(self.idx.cmp(&other.idx))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn bez_end(path: &BezPath) -> kurbo::Point {
    match path.elements().last().unwrap() {
        kurbo::PathEl::MoveTo(p)
        | kurbo::PathEl::LineTo(p)
        | kurbo::PathEl::QuadTo(_, p)
        | kurbo::PathEl::CurveTo(_, _, p) => *p,
        kurbo::PathEl::ClosePath => unreachable!(),
    }
}

fn bez_end_y(path: &BezPath) -> f64 {
    bez_end(path).y
}

fn next_subsegment(seg: &Segment, out: &BezPath, y1: f64, endpoint: kurbo::Point) -> CubicBez {
    let p0 = bez_end(out);
    let mut c = y_subsegment(seg.to_kurbo(), p0.y, y1);
    c.p0 = p0;
    if endpoint.y == y1 {
        c.p3 = endpoint;
    }
    c
}

/// Compute positions for all of the output segments.
///
/// The orders between the output segments is specified by `order`. The endpoints
/// should have been already computed (in a way that satisfies the order), and
/// are provided in `endpoints`. For each output segment, we return a Bézier
/// path.
///
/// The `usize` return value tells which segment (if any) in the returned
/// path was the one that was "far" from any other paths. This is really
/// only interesting for diagnosis/visualization so the API should probably
/// be refined somehow to make it optional. (TODO)
pub(crate) fn compute_positions(
    segs: &Segments,
    orig_seg_map: &OutputSegVec<SegIdx>,
    cmp: &mut ComparisonCache,
    endpoints: &HalfOutputSegVec<kurbo::Point>,
    scan_order: &ScanLineOrder,
    accuracy: f64,
) -> OutputSegVec<(BezPath, Option<usize>)> {
    let mut out = OutputSegVec::<(BezPath, Option<usize>)>::with_size(orig_seg_map.len());
    // We try to build `out` lazily, by avoiding copying input segments to outputs segments
    // until they're needed (by copying in one go, we avoid excess subdivisions). That means
    // we need to separately keep track of how far down we've looked at each output. If
    // we weren't lazy, that would just be the last `y` coordinate in `out`.
    let mut last_y = OutputSegVec::<f64>::with_size(orig_seg_map.len());
    let mut queue = BinaryHeap::<HeapEntry>::new();
    for idx in out.indices() {
        let p = endpoints[idx.first_half()];
        let q = endpoints[idx.second_half()];
        if p.y == q.y {
            out[idx].0.move_to(p);
            out[idx].0.line_to(q);
            out[idx].1 = Some(0);
            continue;
        }
        out[idx].0.move_to(p);
        queue.push(HeapEntry { y: p.y.into(), idx });
    }

    while let Some(entry) = queue.pop() {
        // Maybe this entry was already handled by one of its neighbors, in which case we skip it.
        let y0 = entry.y.into_inner();
        if last_y[entry.idx] > y0 {
            continue;
        }

        let mut y1 = endpoints[entry.idx.second_half()].y;
        let mut west_scan = vec![entry.idx];
        let mut cur = entry.idx;
        while let Some(nbr) = scan_order.west_neighbor_after(cur, y0) {
            let order = cmp.compare_segments(segs, orig_seg_map[nbr], orig_seg_map[cur]);
            let (_, cmp_end_y, cmp_order) = order.iter().find(|(_, end_y, _)| *end_y > y0).unwrap();

            if cmp_order == Order::Left {
                let next_close_y = scan_order
                    .close_west_neighbor_height_after(cur, y0, orig_seg_map, segs, cmp)
                    .unwrap_or(f64::INFINITY);
                y1 = y1.min(next_close_y.max(cmp_end_y));
                break;
            } else {
                y1 = y1.min(cmp_end_y).min(endpoints[nbr.second_half()].y);
                west_scan.push(nbr);
            }
            cur = nbr;
        }

        let mut east_scan = vec![];
        let mut cur = entry.idx;
        while let Some(nbr) = scan_order.east_neighbor_after(cur, y0) {
            let order = cmp.compare_segments(segs, orig_seg_map[cur], orig_seg_map[nbr]);
            let (_, cmp_end_y, cmp_order) = order.iter().find(|(_, end_y, _)| *end_y > y0).unwrap();

            if cmp_order == Order::Left {
                let next_close_y = scan_order
                    .close_east_neighbor_height_after(cur, y0, orig_seg_map, segs, cmp)
                    .unwrap_or(f64::INFINITY);
                y1 = y1.min(next_close_y.max(cmp_end_y));
                break;
            } else {
                y1 = y1.min(cmp_end_y).min(endpoints[nbr.second_half()].y);
                east_scan.push(nbr);
            }
            cur = nbr;
        }

        let mut neighbors = west_scan;
        neighbors.reverse();
        neighbors.extend(east_scan);
        if neighbors.len() == 1 {
            let idx = entry.idx;
            // We're far from everything, so we can just copy the input bezier to the output.
            // We only actually do this eagerly for horizontal segments: for other segments
            // we'll hold off on the copying because it might allow us to avoid further
            // subdivision.
            if y0 == y1 {
                out[idx].1 = Some(out[idx].0.elements().len() - 1);
                out[idx].0.line_to(endpoints[idx.second_half()]);
            }
        } else {
            let orig_neighbors = neighbors
                .iter()
                .map(|idx| orig_seg_map[*idx])
                .collect::<Vec<_>>();

            ordered_curves_all_close(
                segs,
                &orig_neighbors,
                &neighbors,
                &mut out,
                y0,
                y1,
                endpoints,
                accuracy,
            );
        }
        for idx in neighbors {
            let y_end = endpoints[idx.second_half()].y;
            if y1 < y_end {
                queue.push(HeapEntry { y: y1.into(), idx });
            }
            last_y[idx] = y1;
        }
    }

    for out_idx in out.indices() {
        let y0 = bez_end_y(&out[out_idx].0);
        let y1 = endpoints[out_idx.second_half()].y;
        if y0 != y1 {
            debug_assert!(y0 < y1);
            let c = y_subsegment(segs[orig_seg_map[out_idx]].to_kurbo(), y0, y1);
            out[out_idx].1 = Some(out[out_idx].0.elements().len() - 1);
            out[out_idx].0.curve_to(c.p1, c.p2, c.p3);
        } else {
            // The quadratic approximations don't respect the fixed endpoints, so tidy them
            // up. Since both the quadratic approximations and the endpoints satisfy
            // the ordering, this doesn't mess up the ordering.
            match out[out_idx].0.elements_mut().last_mut().unwrap() {
                kurbo::PathEl::MoveTo(p)
                | kurbo::PathEl::LineTo(p)
                | kurbo::PathEl::QuadTo(_, p)
                | kurbo::PathEl::CurveTo(_, _, p) => {
                    *p = endpoints[out_idx.second_half()];
                }
                kurbo::PathEl::ClosePath => unreachable!(),
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use kurbo::{BezPath, CubicBez, Line, PathSeg};

    use crate::Segments;

    use super::horizontal_error_weight;

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

    // Here's an example of lines where we used to have a bad
    // approximation. Check that they only require a single
    // approximation step.
    #[test]
    fn linear_approx() {
        let mut p0 = BezPath::new();
        p0.move_to((-212.00062561035156, 90.03582000732422));
        p0.line_to((211.99937438964844, 90.03646087646484));
        p0.close_path();

        let mut p1 = BezPath::new();
        p1.move_to((211.99964904785153, -90.0358200073242));
        p1.line_to((211.99937438964844, 90.03646087646484));
        p1.close_path();

        let mut segs = Segments::default();
        segs.add_bez_path(&p0).unwrap();
        segs.add_bez_path(&p1).unwrap();
        let s0 = segs.indices().next().unwrap();
        let s1 = segs.indices().nth(2).unwrap();

        let mut out = Vec::new();
        let y1 = 90.03646087646484;
        let next_y0 = super::approximate(90.03645987646233, y1, &[s0, s1], &segs, 2.5e-7, &mut out);
        assert_eq!(next_y0, y1);
    }
}
