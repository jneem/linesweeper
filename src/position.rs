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
    curve::{y_subsegment, EstParab, Order},
    num::CheapOrderedFloat,
    order::ComparisonCache,
    topology::{HalfOutputSegVec, OutputSegIdx, OutputSegVec, ScanLineOrder},
    SegIdx, Segments,
};

fn ordered_curves_all_close(
    segs: &Segments,
    order: &[SegIdx],
    output_order: &[OutputSegIdx],
    out: &mut OutputSegVec<(BezPath, Option<usize>)>,
    mut y0: f64,
    y1: f64,
    accuracy: f64,
) {
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
            let accurate = approx.dmax.max(-approx.dmin) < factor * accuracy;
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

fn bez_end(path: &BezPath) -> f64 {
    match path.elements().last().unwrap() {
        kurbo::PathEl::MoveTo(p)
        | kurbo::PathEl::LineTo(p)
        | kurbo::PathEl::QuadTo(_, p)
        | kurbo::PathEl::CurveTo(_, _, p) => p.y,
        kurbo::PathEl::ClosePath => unreachable!(),
    }
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
        if bez_end(&out[entry.idx].0) != y0 {
            debug_assert!(bez_end(&out[entry.idx].0) > y0);
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
                debug_assert_eq!(cmp_order, Order::Ish);
                y1 = y1.min(cmp_end_y).min(endpoints[cur.second_half()].y);
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
                debug_assert_eq!(cmp_order, Order::Ish);
                y1 = y1.min(cmp_end_y).min(endpoints[cur.second_half()].y);
                east_scan.push(nbr);
            }
            cur = nbr;
        }

        let mut neighbors = west_scan;
        neighbors.reverse();
        neighbors.extend(east_scan);
        if neighbors.len() == 1 {
            let idx = entry.idx;
            // We're far from everything, so just copy the input bezier to the output.
            if y0 == y1 {
                out[idx].1 = Some(out[idx].0.elements().len() - 1);
                out[idx].0.line_to(endpoints[idx.second_half()]);
            } else {
                let c = y_subsegment(segs[orig_seg_map[idx]].to_kurbo(), y0, y1);
                out[idx].1 = Some(out[idx].0.elements().len() - 1);
                out[idx].0.curve_to(c.p1, c.p2, c.p3);
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
                accuracy,
            );
        }
        for idx in neighbors {
            let y_end = endpoints[idx.second_half()].y;
            if y1 < y_end {
                queue.push(HeapEntry { y: y1.into(), idx });
            } else {
                debug_assert_eq!(y1, y_end);
                match out[idx].0.elements_mut().last_mut().unwrap() {
                    kurbo::PathEl::MoveTo(p)
                    | kurbo::PathEl::LineTo(p)
                    | kurbo::PathEl::QuadTo(_, p)
                    | kurbo::PathEl::CurveTo(_, _, p) => {
                        *p = endpoints[idx.second_half()];
                    }
                    kurbo::PathEl::ClosePath => unreachable!(),
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use kurbo::{CubicBez, Line, PathSeg};

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
}
