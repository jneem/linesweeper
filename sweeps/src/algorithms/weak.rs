//! A sweep-line implementation using weak orderings.
//!
//! This algorithm is documented in `docs/sweep.typ`.
//!
//! TODO: I think in this algorithm it makes sense to put Exit events first.

use std::{cmp::Reverse, collections::HashMap};

use crate::{
    geom::Segment,
    num::{Bounds, Float},
    sweep::{
        SegIdx, Segments, SweepEvent, SweepEventKind, SweepLine, SweepLineEntry, SweepLineSeg,
    },
};

#[derive(Clone, Debug)]
pub struct WeakSweepLine<F: Float> {
    pub y: F,
    pub segs: Vec<SegIdx>,
    // A list of segments (possibly with duplicates) that changed their relative orders
    // since the last sweep line.
    pub segs_that_changed_order: Vec<SegIdx>,
}

impl<F: Float> WeakSweepLine<F> {
    pub fn new(y: F) -> Self {
        Self {
            y,
            segs: Vec::new(),
            segs_that_changed_order: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EventQueue<F: Float> {
    inner: std::collections::BinaryHeap<std::cmp::Reverse<SweepEvent<F>>>,
}

impl<F: Float> EventQueue<F> {
    pub fn push(&mut self, ev: SweepEvent<F>) {
        self.inner.push(std::cmp::Reverse(ev));
    }

    pub fn pop(&mut self) -> Option<SweepEvent<F>> {
        self.inner.pop().map(|x| x.0)
    }

    pub fn next_y(&self) -> Option<&F> {
        self.inner.peek().map(|Reverse(ev)| &ev.y)
    }
}

#[derive(Clone, Debug)]
pub struct State<F: Float> {
    pub eps: F,
    pub line: WeakSweepLine<F>,
    pub events: EventQueue<F>,
    pub segments: Segments<F>,
}

impl<F: Float> State<F> {
    pub fn intersection_scan_right(&mut self, start_idx: usize) {
        let seg = self.segments.get(self.line.segs[start_idx]);
        let y = &self.line.y;

        // We're allowed to take a potentially-smaller height bound by taking
        // into account the current queue. A larger height bound is still ok,
        // just a little slower.
        let mut height_bound = seg.end.y.clone();

        for j in (start_idx + 1)..self.line.segs.len() {
            let other = self.segments.get(self.line.segs[j]);
            height_bound = height_bound.min(other.end.y.clone());
            // In the write-up, we check whether `seg` crosses the upper bound
            // of `other`, and we aren't allowed to have false negatives. Here
            // we'll just find the intersections of the center-lines, and assume
            // we're accurate enough that
            // - we're always epsilon-close at the computed intersection point, and
            // - if `seg` truly crosses the upper bound of `other` then we
            //   always find an intersection.

            let int_y = if seg.at_y(y) >= other.at_y(y) {
                Some(y.clone())
            } else {
                // TODO: we can probably use an approximate intersection without interval
                // arithmetic.
                seg.approx_intersection_y(other).map(|bounds| bounds.mid())
            };

            if let Some(int_y) = int_y {
                // TODO: justify the second condition
                if int_y <= height_bound
                    && seg.end_offset(other) * F::from_f32(-2.0) >= self.eps.clone()
                {
                    self.events.push(SweepEvent::intersection(
                        self.line.segs[start_idx],
                        self.line.segs[j],
                        int_y.clone(),
                    ));
                    height_bound = int_y;
                }
            }

            // For the early stopping, we need to check whether `seg` is less than `other`'s lower
            // bound on the whole interesting `y` interval. Since they're lines, it's enough to check
            // at the two interval endpoints.
            let y1 = &height_bound;
            if seg.at_y_bound(y).upper <= other.lower_bound(y, &self.eps).lower
                && seg.at_y_bound(y1).upper <= other.lower_bound(y1, &self.eps).lower
            {
                break;
            }
        }
    }

    pub fn intersection_scan_left(&mut self, start_idx: usize) {
        let seg = self.segments.get(self.line.segs[start_idx]);
        let y = &self.line.y;

        let mut height_bound = seg.end.y.clone();

        for j in (0..start_idx).rev() {
            let other = self.segments.get(self.line.segs[j]);
            height_bound = height_bound.min(other.end.y.clone());
            let int_y = if seg.at_y(y) <= other.at_y(y) {
                Some(y.clone())
            } else {
                other.approx_intersection_y(seg).map(|bounds| bounds.mid())
            };

            if let Some(int_y) = int_y {
                if int_y <= height_bound
                    && seg.end_offset(other) * F::from_f32(2.0) >= self.eps.clone()
                {
                    self.events.push(SweepEvent::intersection(
                        self.line.segs[j],
                        self.line.segs[start_idx],
                        int_y.clone(),
                    ));
                    height_bound = int_y;
                }
            }

            // For the early stopping, we need to check whether `seg` is greater than `other`'s upper
            // bound on the whole interesting `y` interval. Since they're lines, it's enough to check
            // at the two interval endpoints.
            let y1 = &height_bound;
            if seg.at_y_bound(y).lower >= other.upper_bound(y, &self.eps).upper
                && seg.at_y_bound(y1).lower >= other.upper_bound(y1, &self.eps).upper
            {
                break;
            }
        }
    }

    fn remove(&mut self, pos: usize) {
        self.line.segs.remove(pos);
        if pos > 0 {
            self.intersection_scan_right(pos - 1);
            self.intersection_scan_left(pos - 1);
        }
    }

    fn insert(&mut self, pos: usize, seg: SegIdx) {
        self.line.segs.insert(pos, seg);
        self.intersection_scan_right(pos);
        self.intersection_scan_left(pos);
    }

    pub fn step(&mut self) {
        let Some(ev) = self.events.pop() else {
            return;
        };

        self.line.y = ev.y;
        match ev.kind {
            SweepEventKind::Enter(seg_idx) => {
                let new_seg = self.segments.get(seg_idx);
                let pos = self.line.insertion_idx(&self.segments, new_seg, &self.eps);
                self.insert(pos, seg_idx);
            }
            SweepEventKind::Exit(seg_idx) => {
                let pos = self
                    .line
                    .position(seg_idx)
                    .expect("exit for a segment we don't have");
                self.remove(pos);
            }
            SweepEventKind::Intersection { left, right } => {
                let left_idx = self.line.segs.iter().position(|&x| x == left).unwrap();
                let right_idx = self.line.segs.iter().position(|&x| x == right).unwrap();
                if left_idx < right_idx {
                    self.line
                        .segs_that_changed_order
                        .extend_from_slice(&self.line.segs[left_idx..=right_idx]);
                    let left_seg = self.segments.get(left);
                    let eps = &self.eps;
                    let y = &self.line.y;

                    // We're going to put `left_seg` after `right_seg` in the
                    // sweep line, and while doing so we need to "push" along
                    // all segments that are strictly bigger than `left_seg`
                    // (slight false positives are allowed).
                    let mut to_move = vec![(left_idx, self.line.segs[left_idx])];
                    for j in (left_idx + 1)..right_idx {
                        let seg = self.segments.get(self.line.segs[j]);
                        if left_seg.upper_bound(y, eps).lower < seg.lower_bound(y, eps).upper {
                            to_move.push((j, self.line.segs[j]));
                        }
                    }

                    // Remove them in reverse to make indexing easier.
                    for &(j, _) in to_move.iter().rev() {
                        self.remove(j);
                    }

                    // We want to insert them at what was previously `right_idx + 1`, but the
                    // index changed because of the removal.
                    let insertion_pos = right_idx + 1 - to_move.len();

                    for &(_, seg) in to_move.iter().rev() {
                        self.insert(insertion_pos, seg);
                    }
                }
            }
        }
    }

    pub fn finished(&self) -> bool {
        self.events.inner.is_empty()
    }

    pub fn check_invariants(&self) {
        assert!(self
            .line
            .find_invalid_order(&self.segments, &self.eps)
            .is_none());

        for i in 0..self.line.segs.len() {
            for j in (i + 1)..self.line.segs.len() {
                let segi = self.segments.get(self.line.segs[i]).to_exact();
                let segj = self.segments.get(self.line.segs[j]).to_exact();

                if let Some(y_int) = segi.exact_intersection_y(&segj) {
                    if y_int >= self.line.y.to_exact() {
                        // Find an event between i and j.
                        let is_between = |idx: SegIdx| -> bool {
                            self.line
                                .position(idx)
                                .map_or(false, |pos| i <= pos && pos <= j)
                        };
                        let has_witness = self.events.inner.iter().any(|ev| match &ev.0.kind {
                            SweepEventKind::Enter(_) => false,
                            SweepEventKind::Intersection { left, right } => {
                                is_between(*left) && is_between(*right)
                            }
                            SweepEventKind::Exit(seg_idx) => is_between(*seg_idx),
                        });
                        assert!(has_witness);
                    }
                }
            }
        }
    }
}

impl<F: Float> Segment<F> {
    // Scale eps based on the slope of this line.
    //
    // The write-up used 1/(cos theta) for scaling. Here we use
    // the smaller (and therefore stricter) max(1, 1/|slope|) scaling,
    // because it's possible to compute exactly when doing rational
    // arithmetic.
    fn scaled_eps(&self, eps: &F) -> F {
        assert!(self.start.y < self.end.y);
        let dx = (self.end.x.clone() - &self.start.x).abs();
        let dy = self.end.y.clone() - &self.start.y;

        if dx <= dy {
            eps.clone()
        } else {
            (dx * eps) / dy
        }
    }

    // TODO: for a more efficient algorithm we can probably avoid tracking
    // intervals: we just need some analysis to bound our numerical error,
    // and then do some comparisons with epsilons.
    fn scaled_eps_bound(&self, eps: &F) -> Bounds<F> {
        assert!(self.start.y < self.end.y);
        let max_x = self.end.x.clone().max(self.start.x.clone());
        let min_x = self.end.x.clone().min(self.start.x.clone());
        let dx = Bounds::single(max_x) - Bounds::single(min_x);
        let dy = Bounds::single(self.end.y.clone()) - Bounds::single(self.start.y.clone());

        if dx.upper <= dy.lower {
            Bounds::single(eps.clone())
        } else {
            (dx * Bounds::single(eps.clone())) / dy
        }
    }

    /// The lower envelope of this segment at the given height.
    ///
    /// In the write-up this was called `alpha^-_(y,epsilon)`.
    ///
    /// Panics on a horizontal segment.
    fn lower(&self, y: &F, eps: &F) -> F {
        let min_x = self.end.x.clone().min(self.start.x.clone());

        (self.at_y(y) - self.scaled_eps(eps)).max(min_x - eps)
    }

    /// Like [`lower`], but returns an interval.
    fn lower_bound(&self, y: &F, eps: &F) -> Bounds<F> {
        let min_x = self.end.x.clone().min(self.start.x.clone());
        let scaled_eps = self.scaled_eps_bound(eps);

        (self.at_y_bound(y) - scaled_eps).max((min_x - eps).next_down())
    }

    fn upper(&self, y: &F, eps: &F) -> F {
        let max_x = self.end.x.clone().max(self.start.x.clone());

        (self.at_y(y) + self.scaled_eps(eps)).min(max_x + eps)
    }

    fn upper_bound(&self, y: &F, eps: &F) -> Bounds<F> {
        let max_x = self.end.x.clone().max(self.start.x.clone());
        let scaled_eps = self.scaled_eps_bound(eps);

        (self.at_y_bound(y) + scaled_eps).min((max_x + eps).next_up())
    }
}

impl<F: Float> WeakSweepLine<F> {
    /// If the ordering invariants fail, returns a pair of indices witnessing that failure.
    pub fn find_invalid_order(&self, segments: &Segments<F>, eps: &F) -> Option<(SegIdx, SegIdx)> {
        let eps = eps.to_exact();
        let y = self.y.to_exact();
        for i in 0..self.segs.len() {
            for j in (i + 1)..self.segs.len() {
                let segi = segments.get(self.segs[i]).to_exact();
                let segj = segments.get(self.segs[j]).to_exact();

                if segi.lower(&y, &eps) > segj.upper(&y, &eps) {
                    return Some((self.segs[i], self.segs[j]));
                }
            }
        }

        None
    }

    // Finds an index into this sweep line where it's ok to insert this new segment.
    fn insertion_idx(&self, segments: &Segments<F>, seg: &Segment<F>, eps: &F) -> usize {
        // Checks if `other` is smaller than `seg` with no false negatives: if `other` is actually smaller than `seg`
        // it will definitely return true.
        let maybe_strictly_smaller = |other: &SegIdx| -> bool {
            let other = segments.get(*other);
            other.upper_bound(&self.y, eps).lower < seg.lower_bound(&self.y, eps).upper
        };

        // The rust stdlib docs say that we're not allowed to do this, because our array isn't sorted
        // with respect to `maybe_strictly_smaller`. But for now at least, their implementation does a
        // normal binary search and so it's guaranteed to return an index where `maybe_strictly_smaller`
        // fails but the index before it succeeds.
        let search_start = self.segs.partition_point(maybe_strictly_smaller);
        let mut idx = search_start;
        for i in search_start..self.segs.len() {
            if maybe_strictly_smaller(&self.segs[i]) {
                idx = i + 1;
            }

            // Once we've found a segment whose lower bound is definitely bigger than seg's, there's no need
            // to look further.
            let other = segments.get(self.segs[i]);
            if other.lower_bound(&self.y, eps).lower >= seg.lower_bound(&self.y, eps).upper {
                break;
            }
        }
        idx
    }

    // Find the position of the given segment in our array.
    //
    // TODO: if we're large, we could use a binary search.
    fn position(&self, seg: SegIdx) -> Option<usize> {
        self.segs.iter().position(|&x| x == seg)
    }

    /// Return all the segments in this sweep-line, along with a valid x position.
    ///
    /// TODO: this returns the smallest possible valid x position, which is correct but leads
    /// to weird output, with unnecessary horizontal segments. We can probably find a heuristic
    /// to improve this. (Like maybe also calculating the largest possible valid x position,
    /// and then choosing something in between.)
    fn ordered_xs<'a>(
        &'a self,
        segments: &'a Segments<F>,
        eps: &'a F,
    ) -> impl Iterator<Item = (SegIdx, F)> + 'a {
        let mut max_so_far = self
            .segs
            .first()
            .map(|seg| segments.get(*seg).lower(&self.y, eps))
            // If `self.segs` is empty our y doesn't matter; we're going to return
            // an empty iterator.
            .unwrap_or(F::from_f32(0.0));

        self.segs.iter().map(move |seg_idx| {
            let x = segments.get(*seg_idx).lower(&self.y, eps);
            max_so_far = max_so_far.clone().max(x);
            (*seg_idx, max_so_far.clone())
        })
    }
}

/// Runs a sweep over all the segments, returning a sweep line at every `y` where
/// there was an event.
pub fn sweep<F: Float>(segments: &Segments<F>, eps: &F) -> Vec<WeakSweepLine<F>> {
    let events = EventQueue {
        inner: segments
            .indices()
            .filter(|idx| !segments.get(*idx).is_horizontal())
            .flat_map(|idx| {
                let (a, b) = SweepEvent::from_segment(idx, segments);
                [a, b]
            })
            .map(std::cmp::Reverse)
            .collect(),
    };

    let line = WeakSweepLine::new(events.next_y().unwrap().clone());

    let mut state = State {
        eps: eps.clone(),
        line,
        events,
        segments: segments.clone(),
    };
    state.check_invariants();

    let mut ret = Vec::new();

    while let Some(y) = state.events.next_y().cloned() {
        state.line.segs_that_changed_order.clear();
        // Loop over all the events at the current y.
        while Some(&y) == state.events.next_y() {
            state.step();
            state.check_invariants();
        }

        ret.push(state.line.clone());
    }

    ret
}

/// Adds to this sweep-line all horizontal segments that belong in it.
fn add_horizontals_to_sweep_line<F: Float>(
    line: &mut SweepLine<F>,
    horizontals: &[SegIdx],
    segments: &Segments<F>,
) {
    // `horizontals` are sorted by y coordinate, so find the subrange that's at our current y
    let start_idx = horizontals.partition_point(|s| segments.get(*s).start.y >= line.y);
    let end_idx = horizontals.partition_point(|s| segments.get(*s).start.y > line.y);

    if end_idx < start_idx {
        return;
    }

    for &idx in &horizontals[start_idx..end_idx] {
        let seg = segments.get(idx);

        line.segs.push(SweepLineEntry {
            x: SweepLineSeg::EnterExit(
                seg.start.x.clone().min(seg.end.x.clone()),
                seg.start.x.clone().max(seg.end.x.clone()),
            ),
            idx,
        });
    }
}

/// Converts a sequence of weakly-ordered sweep lines into a sequence
/// of actual sweep lines, in the naivest possibly way: subdividing every
/// segment at every sweep line.
///
/// We don't currently handle horizontal segments.
pub fn weaks_to_sweeps_dense<F: Float>(
    weaks: &[WeakSweepLine<F>],
    segments: &Segments<F>,
    eps: &F,
) -> Vec<SweepLine<F>> {
    // The first sweep-line just has a single entry for everything
    let mut ret = vec![SweepLine {
        y: weaks[0].y.clone(),
        segs: weaks[0]
            .ordered_xs(segments, eps)
            .map(|(idx, x)| SweepLineEntry {
                idx,
                x: SweepLineSeg::Single(x),
            })
            .collect(),
    }];

    let mut horizontals: Vec<_> = segments
        .indices()
        .filter(|&idx| segments.get(idx).is_horizontal())
        .collect();
    horizontals.sort_by(|s, t| segments.get(*s).start.y.cmp(&segments.get(*t).start.y));

    let mut prev = weaks[0].clone();
    for line in &weaks[1..] {
        prev.y = line.y.clone();
        // TODO: should be able to build things up in order by iterating over
        // both `prev` and `line` in one pass. But it isn't quite trivial
        // because we need to keep track of segments that were in one but
        // haven't yet been encountered in the other.
        let mut entries: HashMap<_, _> = prev
            .ordered_xs(segments, eps)
            .map(|(idx, x)| (idx, SweepLineSeg::Single(x)))
            .collect();

        for (idx, x) in line.ordered_xs(segments, eps) {
            match entries.entry(idx) {
                std::collections::hash_map::Entry::Occupied(mut occ) => {
                    let SweepLineSeg::Single(enter_x) = occ.get().clone() else {
                        unreachable!()
                    };
                    *occ.get_mut() = SweepLineSeg::EnterExit(enter_x, x);
                }
                std::collections::hash_map::Entry::Vacant(vac) => {
                    vac.insert(SweepLineSeg::Single(x));
                }
            }
        }

        let entries: Vec<_> = entries
            .into_iter()
            .map(|(idx, x)| SweepLineEntry { idx, x })
            .collect();

        let mut sweep_line = SweepLine {
            y: prev.y,
            segs: entries,
        };
        add_horizontals_to_sweep_line(&mut sweep_line, &horizontals, segments);
        sweep_line.segs.sort();
        ret.push(sweep_line);

        prev = line.clone();
    }

    ret
}

#[cfg(test)]
mod tests {
    use super::*;
    use ordered_float::NotNan;
    use proptest::prelude::*;

    use crate::{
        geom::{Point, Segment},
        perturbation::{
            f32_perturbation, f64_perturbation, perturbation, rational_perturbation,
            realize_perturbation, FloatPerturbation, Perturbation,
        },
        sweep::Segments,
    };

    fn mk_segs(xs: &[(f64, f64)]) -> Segments<NotNan<f64>> {
        let y0: NotNan<f64> = 0.0.try_into().unwrap();
        let y1: NotNan<f64> = 1.0.try_into().unwrap();
        let segs: Vec<_> = xs
            .iter()
            .map(|&(x0, x1)| Segment {
                start: Point::new(x0.try_into().unwrap(), y0),
                end: Point::new(x1.try_into().unwrap(), y1),
            })
            .collect();
        Segments {
            segs,
            ..Segments::default()
        }
    }

    #[test]
    fn invalid_order() {
        fn check_order(xs: &[(f64, f64)], at: f64, eps: f64) -> Option<(usize, usize)> {
            let eps: NotNan<f64> = eps.try_into().unwrap();
            let y: NotNan<f64> = at.try_into().unwrap();
            let segs = mk_segs(xs);

            let line = WeakSweepLine {
                y,
                segs: (0..xs.len()).map(SegIdx).collect(),
                segs_that_changed_order: Vec::new(),
            };

            line.find_invalid_order(&segs, &eps)
                .map(|(a, b)| (a.0, b.0))
        }

        let crossing = &[(-1.0, 1.0), (1.0, -1.0)];
        let eps = 1.0 / 128.0;
        assert!(check_order(crossing, 0.0, eps).is_none());
        assert!(check_order(crossing, 0.5, eps).is_none());
        assert_eq!(check_order(crossing, 1.0, eps), Some((0, 1)));

        let not_quite_crossing = &[(-0.75 * eps, 0.75 * eps), (0.75 * eps, -0.75 * eps)];
        assert!(check_order(not_quite_crossing, 0.0, eps).is_none());
        assert!(check_order(not_quite_crossing, 0.5, eps).is_none());
        assert!(check_order(not_quite_crossing, 1.0, eps).is_none());

        let barely_crossing = &[(-1.5 * eps, 1.5 * eps), (1.5 * eps, -1.5 * eps)];
        assert!(check_order(barely_crossing, 0.0, eps).is_none());
        assert!(check_order(barely_crossing, 0.5, eps).is_none());
        assert_eq!(check_order(barely_crossing, 1.0, eps), Some((0, 1)));

        let non_adj_crossing = &[(-1.5 * eps, 1.5 * eps), (0.0, 0.0), (1.5 * eps, -1.5 * eps)];
        assert!(check_order(non_adj_crossing, 0.0, eps).is_none());
        assert!(check_order(non_adj_crossing, 0.5, eps).is_none());
        assert_eq!(check_order(non_adj_crossing, 1.0, eps), Some((0, 2)));

        let flat_crossing = &[(-1e6, 1e6), (-10.0 * eps, -10.0 * eps)];
        assert_eq!(check_order(flat_crossing, 0.5, eps), None);

        let end_crossing_bevel = &[(2.5 * eps, 2.5 * eps), (-1e6, 0.0)];
        assert_eq!(check_order(end_crossing_bevel, 1.0, eps), Some((0, 1)));

        let start_crossing_bevel = &[(2.5 * eps, 2.5 * eps), (0.0, -1e6)];
        assert_eq!(check_order(start_crossing_bevel, 1.0, eps), Some((0, 1)));
    }

    #[test]
    fn insertion_idx() {
        fn insert(xs: &[(f64, f64)], new: (f64, f64), at: f64, eps: f64) -> usize {
            let eps: NotNan<f64> = eps.try_into().unwrap();
            let y: NotNan<f64> = at.try_into().unwrap();
            let y0: NotNan<f64> = 0.0.try_into().unwrap();
            let y1: NotNan<f64> = 1.0.try_into().unwrap();
            let segs = mk_segs(xs);

            let x0: NotNan<f64> = new.0.try_into().unwrap();
            let x1: NotNan<f64> = new.1.try_into().unwrap();
            let new = Segment {
                start: Point::new(x0, y0),
                end: Point::new(x1, y1),
            };

            let line = WeakSweepLine {
                y,
                segs: (0..xs.len()).map(SegIdx).collect(),
                segs_that_changed_order: Vec::new(),
            };
            line.insertion_idx(&segs, &new, &eps)
        }

        let eps = 1.0 / 128.0;
        assert_eq!(
            insert(&[(-1.0, -1.0), (1.0, 1.0)], (-2.0, 0.0), 0.0, eps),
            0
        );
        assert_eq!(insert(&[(-1.0, -1.0), (1.0, 1.0)], (0.0, 0.0), 0.0, eps), 1);
        assert_eq!(insert(&[(-1.0, -1.0), (1.0, 1.0)], (2.0, 0.0), 0.0, eps), 2);

        assert_eq!(
            insert(
                &[(-1e6, 1e6), (-1.0, -1.0), (1.0, 1.0)],
                (0.0, 0.0),
                0.5,
                eps
            ),
            2
        );
        assert_eq!(
            insert(
                &[
                    (-1e6, 1e6),
                    (-1e6, 1e6),
                    (-1e6, 1e6),
                    (-1.0, -1.0),
                    (1.0, 1.0),
                    (-1e6, 1e6),
                    (-1e6, 1e6),
                    (-1e6, 1e6),
                ],
                (0.0, 0.0),
                0.5,
                eps
            ),
            4
        );
    }

    #[test]
    fn test_sweep() {
        let eps = NotNan::new(0.01).unwrap();

        let segs = mk_segs(&[(0.0, 0.0), (1.0, 1.0), (-2.0, 2.0)]);
        let lines = sweep(&segs, &eps);
        dbg!(&lines);
        assert_eq!(4, lines.len());
        dbg!(&weaks_to_sweeps_dense(&lines, &segs, &eps));
    }

    fn p<F: Float>(x: f32, y: f32) -> Point<F> {
        Point::new(F::from_f32(x), F::from_f32(y))
    }

    fn cyclic_pairs<T>(xs: &[T]) -> impl Iterator<Item = (&T, &T)> {
        xs.windows(2)
            .map(|pair| (&pair[0], &pair[1]))
            .chain(xs.last().zip(xs.first()))
    }

    fn run_perturbation<P: FloatPerturbation>(ps: Vec<Perturbation<P>>) {
        let base = vec![vec![
            p(0.0, 0.0),
            p(1.0, 1.0),
            p(1.0, -1.0),
            p(2.0, 0.0),
            p(1.0, 1.0),
            p(1.0, -1.0),
        ]];

        let perturbed_polylines = ps
            .iter()
            .map(|p| realize_perturbation(&base, p))
            .collect::<Vec<_>>();
        let segs = Segments {
            segs: perturbed_polylines
                .iter()
                .flat_map(|poly| {
                    cyclic_pairs(poly).map(|(p0, p1)| Segment {
                        start: p0.min(p1).clone(),
                        end: p0.max(p1).clone(),
                    })
                })
                .collect(),
            contour_prev: vec![],
            contour_next: vec![],
            orientation: vec![],
        };

        let eps = P::Float::from_f32(0.1);
        let weaks = sweep(&segs, &eps);
        let _sweeps = weaks_to_sweeps_dense(&weaks, &segs, &eps);
    }

    proptest! {
    #[test]
    fn perturbation_test_f64(perturbations in prop::collection::vec(perturbation(f64_perturbation(0.1)), 1..5)) {
        run_perturbation(perturbations);
    }

    #[test]
    fn perturbation_test_f32(perturbations in prop::collection::vec(perturbation(f32_perturbation(0.1)), 1..5)) {
        run_perturbation(perturbations);
    }

    #[test]
    fn perturbation_test_rational(perturbations in prop::collection::vec(perturbation(rational_perturbation(0.1.try_into().unwrap())), 1..5)) {
        run_perturbation(perturbations);
    }
    }
}
