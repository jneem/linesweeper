//! A sweep-line implementation using weak orderings.
//!
//! This algorithm is documented in `docs/sweep.typ`.

// TODO:
// - investigate better insertion heuristics: if there are a bunch of insertions at the same point, we
//   currently put them in some arbitrary order and then later have to process a bunch of intersections

use std::collections::BTreeSet;
use std::collections::HashMap;

use malachite::Rational;

use crate::{
    geom::Segment,
    num::Float,
    segments::{SegIdx, Segments},
    sweep::{SweepEvent, SweepEventKind},
};

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct SegmentOrderEntry {
    seg: SegIdx,
    exit: bool,
    enter: bool,
    // This is filled out during `compute_changed_segments`. Before that
    // happens, segments are marked for needing positions by putting them in the
    // `segs_needing_positions` list.
    needs_position: bool,
    old_idx: Option<usize>,
}

impl SegmentOrderEntry {
    fn reset_state(&mut self) {
        self.exit = false;
        self.enter = false;
        self.needs_position = false;
        self.old_idx = None;
    }

    fn set_old_idx_if_unset(&mut self, i: usize) {
        if self.old_idx.is_none() {
            self.old_idx = Some(i);
        }
    }
}

impl From<SegIdx> for SegmentOrderEntry {
    fn from(seg: SegIdx) -> Self {
        Self {
            seg,
            exit: false,
            enter: false,
            needs_position: false,
            old_idx: None,
        }
    }
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub(crate) struct SegmentOrder {
    pub(crate) segs: Vec<SegmentOrderEntry>,
}

impl SegmentOrder {
    fn seg(&self, i: usize) -> SegIdx {
        self.segs[i].seg
    }

    fn is_exit(&self, i: usize) -> bool {
        self.segs[i].exit
    }
}

impl FromIterator<SegIdx> for SegmentOrder {
    fn from_iter<T: IntoIterator<Item = SegIdx>>(iter: T) -> Self {
        SegmentOrder {
            segs: iter.into_iter().map(SegmentOrderEntry::from).collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EventQueue<F: Float> {
    inner: std::collections::BTreeSet<SweepEvent<F>>,
}

impl<F: Float> EventQueue<F> {
    /// Builds an event queue containing the starting and ending positions
    /// of all the  segments.
    ///
    /// The returned event queue will not contain any intersection events.
    pub fn from_segments(segments: &Segments<F>) -> Self {
        let mut inner = std::collections::BTreeSet::new();

        for seg in segments.indices() {
            if segments[seg].is_horizontal() {
                inner.insert(SweepEvent {
                    y: segments[seg].start.y.clone(),
                    kind: SweepEventKind::Horizontal(seg),
                });
            } else {
                let (a, b) = SweepEvent::from_segment(seg, segments);
                inner.insert(a);
                inner.insert(b);
            }
        }

        Self { inner }
    }

    pub fn push(&mut self, ev: SweepEvent<F>) {
        self.inner.insert(ev);
    }

    pub fn pop(&mut self) -> Option<SweepEvent<F>> {
        self.inner.pop_first()
    }

    pub fn next_y(&self) -> Option<&F> {
        self.inner.first().map(|ev| &ev.y)
    }

    fn next_kind(&self) -> Option<&SweepEventKind> {
        self.inner.first().map(|ev| &ev.kind)
    }

    pub fn next_is_stage_1(&self, y: &F) -> bool {
        self.next_y() == Some(y)
            && self.next_kind().is_some_and(|kind| match kind {
                SweepEventKind::Enter(_) | SweepEventKind::Horizontal(_) => true,
                SweepEventKind::Intersection { .. } | SweepEventKind::Exit(_) => false,
            })
    }

    pub fn next_is_stage_2(&self, y: &F) -> bool {
        self.next_y() == Some(y)
            && self.next_kind().is_some_and(|kind| match kind {
                SweepEventKind::Exit(_) => true,
                SweepEventKind::Enter(_)
                | SweepEventKind::Horizontal(_)
                | SweepEventKind::Intersection { .. } => false,
            })
    }

    pub fn next_is_at(&self, y: &F) -> bool {
        self.next_y() == Some(y)
    }
}

/// Encapsulates the state of the sweep-line algorithm and allows iterating over sweep lines.
#[derive(Clone, Debug)]
pub struct Sweeper<F: Float> {
    pub(crate) y: F,
    pub(crate) eps: F,
    pub(crate) line: SegmentOrder,
    pub(crate) events: EventQueue<F>,
    // TODO: maybe borrow Segments?
    pub(crate) segments: Segments<F>,

    // The collection of segments that we know need to be given explicit
    // positions in the current sweep line.
    //
    // These include:
    // - any segments that changed order with any other segments
    // - any segments that entered or exited
    // - any segments that are between the endpoints of contour-adjacent segments.
    //
    // These segments are identified by their index in the current order, so that
    // it's fast to find them. It means that we need to do some fixing-up if indices after
    // inserting all the new segments.
    pub(crate) segs_needing_positions: Vec<usize>,
    changed_intervals: Vec<(usize, usize)>,
}

impl<F: Float> Sweeper<F> {
    /// Creates a new sweeper for a collection of segments, and with a given tolerance.
    pub fn new(segments: &Segments<F>, eps: F) -> Self {
        let events = EventQueue::from_segments(segments);

        Sweeper {
            eps,
            line: SegmentOrder::default(),
            y: events.next_y().unwrap().clone(),
            events,
            segments: segments.clone(),
            segs_needing_positions: Vec::new(),
            changed_intervals: Vec::new(),
        }
    }

    /// Moves the sweep forward, returning the next sweep line.
    ///
    /// Returns `None` when sweeping is complete.
    pub fn next_line(&mut self) -> Option<SweepLine<'_, F>> {
        self.check_invariants();

        let y = self.events.next_y().cloned()?;
        self.advance(y.clone());
        self.check_invariants();

        // Process all the enter events at this y.
        while self.events.next_is_stage_1(&y) {
            self.step();
            self.check_invariants();
        }

        // Process all the exit events.
        while self.events.next_is_stage_2(&y) {
            self.step();
            self.check_invariants();
        }

        // Process all the intersection events at this y.
        while self.events.next_is_at(&y) {
            self.step();
            self.check_invariants();
        }

        self.compute_changed_intervals();
        Some(SweepLine { state: self })
    }

    fn advance(&mut self, y: F) {
        // All the exiting segments should be in segs_needing_positions, so find them all and remove them.
        self.segs_needing_positions
            .retain(|idx| self.line.segs[*idx].exit);

        // Reset the state flags for all segments. All segments with non-trivial state flags should
        // belong to the changed intervals. This needs to go before we remove the exiting segments,
        // because that messes up the indices.
        for (start, end) in self.changed_intervals.drain(..) {
            for seg in &mut self.line.segs[start..end] {
                seg.reset_state();
            }
        }

        // Remove the exiting segments in reverse order, so the indices stay good.
        self.segs_needing_positions.sort();
        self.segs_needing_positions.dedup();
        for &idx in self.segs_needing_positions.iter().rev() {
            self.line.segs.remove(idx);
        }
        self.y = y;
        self.segs_needing_positions.clear();
    }

    fn intersection_scan_right(&mut self, start_idx: usize) {
        let seg = &self.segments[self.line.seg(start_idx)];
        let y = &self.y;

        // We're allowed to take a potentially-smaller height bound by taking
        // into account the current queue. A larger height bound is still ok,
        // just a little slower.
        let mut height_bound = seg.end.y.clone();

        for j in (start_idx + 1)..self.line.segs.len() {
            if self.line.is_exit(j) {
                continue;
            }
            let other = &self.segments[self.line.seg(j)];
            height_bound = height_bound.min(other.end.y.clone());

            if let Some(int_y) = seg.crossing_y(other, &self.eps) {
                self.events.push(SweepEvent::intersection(
                    self.line.seg(start_idx),
                    self.line.seg(j),
                    int_y.clone().max(y.clone()),
                ));
                height_bound = int_y.min(height_bound);
            }

            // For the early stopping, we need to check whether `seg` is less than `other`'s lower
            // bound on the whole interesting `y` interval. Since they're lines, it's enough to check
            // at the two interval endpoints.
            let y1 = &height_bound;
            let threshold = self.eps.clone() / F::from_f32(4.0);
            if threshold <= other.lower(y, &self.eps) - seg.at_y(y)
                && threshold <= other.lower(y1, &self.eps) - seg.at_y(y1)
            {
                break;
            }
        }
    }

    fn intersection_scan_left(&mut self, start_idx: usize) {
        let seg = &self.segments[self.line.seg(start_idx)];
        let y = &self.y;

        let mut height_bound = seg.end.y.clone();

        for j in (0..start_idx).rev() {
            if self.line.is_exit(j) {
                continue;
            }
            let other = &self.segments[self.line.seg(j)];
            height_bound = height_bound.min(other.end.y.clone());
            if let Some(int_y) = other.crossing_y(seg, &self.eps) {
                self.events.push(SweepEvent::intersection(
                    self.line.seg(j),
                    self.line.seg(start_idx),
                    int_y.clone().max(self.y.clone()),
                ));
                height_bound = int_y.min(height_bound);
            }

            // For the early stopping, we need to check whether `seg` is greater than `other`'s upper
            // bound on the whole interesting `y` interval. Since they're lines, it's enough to check
            // at the two interval endpoints.
            let y1 = &height_bound;
            let threshold = self.eps.clone() / F::from_f32(4.0);
            if seg.at_y(y) - other.upper(y, &self.eps) > threshold
                && seg.at_y(y1) - other.upper(y1, &self.eps) > threshold
            {
                break;
            }
        }
    }

    fn scan_for_removal(&mut self, pos: usize) {
        if pos > 0 {
            self.intersection_scan_right(pos - 1);
            self.intersection_scan_left(pos - 1);
        }
    }

    fn insert(&mut self, pos: usize, seg: SegmentOrderEntry) {
        self.line.segs.insert(pos, seg);
        self.intersection_scan_right(pos);
        self.intersection_scan_left(pos);
    }

    /// Consume a single event from the event queue and process it.
    fn step(&mut self) {
        let Some(ev) = self.events.pop() else {
            return;
        };

        assert!(self.y == ev.y);
        match ev.kind {
            SweepEventKind::Enter(seg_idx) => {
                let new_seg = &self.segments[seg_idx];
                let pos = self
                    .line
                    .insertion_idx(&self.y, &self.segments, new_seg, &self.eps);
                let mut entry = SegmentOrderEntry::from(seg_idx);
                entry.enter = true;
                self.insert(pos, entry);

                // TODO: see if we can make this not quadratic
                for other_pos in &mut self.segs_needing_positions {
                    if *other_pos >= pos {
                        *other_pos += 1;
                    }
                }
                self.segs_needing_positions.push(pos);
            }
            SweepEventKind::Exit(seg_idx) => {
                let pos = self
                    .line
                    .position(seg_idx)
                    .expect("exit for a segment we don't have");
                // It's important that this goes before `scan_for_removal`, so that
                // the scan doesn't get confused by the segment that should be marked
                // for exit.
                self.line.segs[pos].exit = true;
                self.scan_for_removal(pos);
                self.segs_needing_positions.push(pos);
            }
            SweepEventKind::Intersection { left, right } => {
                let left_idx = self.line.position(left).unwrap();
                let right_idx = self.line.position(right).unwrap();
                if left_idx < right_idx {
                    self.segs_needing_positions.extend(left_idx..=right_idx);
                    for (i, entry) in self.line.segs[left_idx..=right_idx].iter_mut().enumerate() {
                        if entry.old_idx.is_none() {
                            entry.old_idx = Some(left_idx + i);
                        }
                    }

                    let left_seg = &self.segments[left];
                    let eps = &self.eps;
                    let y = &self.y;

                    // We're going to put `left_seg` after `right_seg` in the
                    // sweep line, and while doing so we need to "push" along
                    // all segments that are strictly bigger than `left_seg`
                    // (slight false positives are allowed; no false negatives).
                    let mut to_move = vec![(left_idx, self.line.segs[left_idx])];
                    let threshold = eps.clone() / F::from_f32(-4.0);
                    for j in (left_idx + 1)..right_idx {
                        let seg = &self.segments[self.line.seg(j)];
                        if seg.lower(y, eps) - left_seg.upper(y, eps) > threshold {
                            to_move.push((j, self.line.segs[j]));
                        }
                    }

                    // Remove them in reverse to make indexing easier.
                    for &(j, _) in to_move.iter().rev() {
                        self.line.segs.remove(j);
                        self.scan_for_removal(j);
                    }

                    // We want to insert them at what was previously `right_idx + 1`, but the
                    // index changed because of the removal.
                    let insertion_pos = right_idx + 1 - to_move.len();

                    for &(_, seg) in to_move.iter().rev() {
                        self.insert(insertion_pos, seg);
                    }
                }
            }
            SweepEventKind::Horizontal(seg_idx) => {
                let new_seg = &self.segments[seg_idx];
                let pos = self
                    .line
                    .insertion_idx(&self.y, &self.segments, new_seg, &self.eps);
                let mut entry = SegmentOrderEntry::from(seg_idx);
                entry.enter = true;
                entry.exit = true;
                self.insert(pos, entry);

                // TODO: see if we can make this not quadratic
                for other_pos in &mut self.segs_needing_positions {
                    if *other_pos >= pos {
                        *other_pos += 1;
                    }
                }

                self.segs_needing_positions.push(pos);
            }
        }
    }

    #[cfg(feature = "slow-asserts")]
    fn check_invariants(&self) {
        for ev in &self.events.inner {
            assert!(
                ev.y >= self.y,
                "at y={:?}, event {:?} occurred in the past",
                &self.y,
                &ev
            );
        }

        for &seg_entry in &self.line.segs {
            let seg_idx = seg_entry.seg;
            let seg = &self.segments[seg_idx];
            assert!(
                (&seg.start.y..=&seg.end.y).contains(&&self.y),
                "segment {seg:?} out of range at y={:?}",
                self.y
            );
        }

        // All segments marked as stering or exiting must be in `self.segs_needing_positions`
        for (idx, &seg_entry) in self.line.segs.iter().enumerate() {
            if seg_entry.exit || seg_entry.enter {
                assert!(self.segs_needing_positions.contains(&idx));
            }
        }

        assert!(self
            .line
            .find_invalid_order(&self.y, &self.segments, &self.eps)
            .is_none());

        let eps = self.eps.to_exact();
        for i in 0..self.line.segs.len() {
            if self.line.is_exit(i) {
                continue;
            }
            for j in (i + 1)..self.line.segs.len() {
                if self.line.is_exit(j) {
                    continue;
                }
                let segi = self.segments[self.line.seg(i)].to_exact();
                let segj = self.segments[self.line.seg(j)].to_exact();

                if let Some(y_int) = segi.exact_eps_crossing(&segj, &eps) {
                    if y_int >= self.y.to_exact() {
                        // Find an event between i and j.
                        let is_between = |idx: SegIdx| -> bool {
                            self.line
                                .position(idx)
                                .is_some_and(|pos| i <= pos && pos <= j)
                        };
                        let has_witness = self.events.inner.iter().any(|ev| {
                            let is_between = match &ev.kind {
                                SweepEventKind::Enter(_) | SweepEventKind::Horizontal(_) => false,
                                SweepEventKind::Intersection { left, right } => {
                                    is_between(*left) && is_between(*right)
                                }
                                SweepEventKind::Exit(seg_idx) => is_between(*seg_idx),
                            };
                            let before_y = ev.y.to_exact() <= y_int;
                            is_between && before_y
                        });
                        assert!(
                            has_witness,
                            "segments {:?} and {:?} cross at {:?}, but there is no witness",
                            self.line.segs[i], self.line.segs[j], y_int
                        );
                    }
                }
            }
        }
    }

    #[cfg(not(feature = "slow-asserts"))]
    fn check_invariants(&self) {}

    /// Updates our internal `changed_intervals` state based on the segments marked
    /// as needing positions.
    ///
    /// For each segment marked as needing a position, we expand it to a range of
    /// physically nearby segments, and then we deduplicate and merge those ranges.
    fn compute_changed_intervals(&mut self) {
        debug_assert!(self.changed_intervals.is_empty());
        let y = &self.y;

        // We compare horizontal positions to decide when to stop iterating. Those positions
        // are each accurate to eps / 8, so we compare them with slack eps / 4 to ensure no
        // false negatives.
        let eps = &self.eps;
        let segments = &self.segments;
        let slack = eps.clone() / F::from_f32(4.0);

        for &idx in &self.segs_needing_positions {
            if self.line.segs[idx].needs_position {
                continue;
            }
            // Ensure that every segment in the changed interval has `old_idx` set. This
            // isn't strictly necessary (because segments without `old_idx` set haven't
            // changed their indices), but it's more convenient to just say that all
            // segments in a changed interval must have `old_idx` set.
            self.line.segs[idx].set_old_idx_if_unset(idx);

            // unwrap: segs_needing_positions should all be in the line.
            let mut start_idx = idx;
            self.line.segs[idx].needs_position = true;
            let seg_idx = self.line.segs[idx].seg;
            let mut seg_min = segments[seg_idx].lower(y, eps);

            for i in (0..idx).rev() {
                let prev_seg_idx = self.line.seg(i);
                let prev_seg = &segments[prev_seg_idx];
                if seg_min - prev_seg.upper(y, eps) > slack {
                    break;
                } else {
                    seg_min = prev_seg.lower(y, eps);
                    self.line.segs[i].needs_position = true;
                    self.line.segs[i].set_old_idx_if_unset(i);
                    start_idx = i;
                }
            }

            let mut end_idx = idx + 1;
            let mut seg_max = segments[seg_idx].upper(y, eps);
            for i in (idx + 1)..self.line.segs.len() {
                let next_seg_idx = self.line.seg(i);
                let next_seg = &segments[next_seg_idx];
                if next_seg.lower(y, eps) - seg_max > slack {
                    break;
                } else {
                    seg_max = next_seg.upper(y, eps);
                    self.line.segs[i].needs_position = true;
                    self.line.segs[i].set_old_idx_if_unset(i);
                    end_idx = i + 1;
                }
            }
            self.changed_intervals.push((start_idx, end_idx))
        }
        self.changed_intervals.sort();
        self.changed_intervals.dedup();

        // By merging adjacent intervals, we ensure that there is no horizontal segment
        // that spans two ranges. That's because horizontal segments mark everything they
        // cross as needing a position. Any collection of subranges that are crossed by
        // a horizontal segment are therefore adjacent and will be merged here.
        merge_adjacent(&mut self.changed_intervals);
    }
}

impl Segment<Rational> {
    // The moment our lower bound crosses to the right of `other`'s upper bound.
    // (Actually, it could give too large a value right now, because it doesn't take the
    // "chamfers" into account.)
    #[cfg(feature = "slow-asserts")]
    pub(crate) fn exact_eps_crossing(&self, other: &Self, eps: &Rational) -> Option<Rational> {
        let y0 = self.start.y.clone().max(other.start.y.clone());
        let y1 = self.end.y.clone().min(other.end.y.clone());
        let scaled_eps = self.scaled_eps(eps);

        assert!(y1 >= y0);

        let dx0 = other.at_y(&y0) - self.at_y(&y0) + scaled_eps.clone() * Rational::from(2);
        let dx1 = other.at_y(&y1) - self.at_y(&y1) + scaled_eps * Rational::from(2);
        if dx0 == dx1 {
            // They're parallel.
            (dx0 == 0).then_some(y0)
        } else if dx1 < 0 {
            let t = &dx0 / (&dx0 - dx1);
            (0..=1).contains(&t).then(|| &y0 + t * (y1 - &y0))
        } else {
            None
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
        assert!(self.start.y <= self.end.y);
        if self.start.y == self.end.y {
            // See `scaled_eps_bound`
            return eps.clone();
        }

        let dx = (self.end.x.clone() - &self.start.x).abs();
        let dy = self.end.y.clone() - &self.start.y;

        if dx <= dy {
            eps.clone()
        } else {
            (dx * eps) / dy
        }
    }

    /// The lower envelope of this segment at the given height.
    ///
    /// In the write-up this was called `alpha^-_(y,epsilon)`.
    fn lower(&self, y: &F, eps: &F) -> F {
        let min_x = self.end.x.clone().min(self.start.x.clone());

        if self.is_horizontal() {
            // Special case for horizontal lines, because their
            // `at_y` function returns the larger x position, and
            // we want the smaller one here.
            self.start.x.clone() - eps
        } else {
            (self.at_y(y) - self.scaled_eps(eps)).max(min_x - eps)
        }
    }

    fn upper(&self, y: &F, eps: &F) -> F {
        let max_x = self.end.x.clone().max(self.start.x.clone());

        (self.at_y(y) + self.scaled_eps(eps)).min(max_x + eps)
    }
}

impl SegmentOrder {
    /// If the ordering invariants fail, returns a pair of indices witnessing that failure.
    /// Used in tests, and when enabling slow-asserts
    #[allow(dead_code)]
    fn find_invalid_order<F: Float>(
        &self,
        y: &F,
        segments: &Segments<F>,
        eps: &F,
    ) -> Option<(SegIdx, SegIdx)> {
        let eps = eps.to_exact();
        let y = y.to_exact();
        for i in 0..self.segs.len() {
            for j in (i + 1)..self.segs.len() {
                let segi = segments[self.seg(i)].to_exact();
                let segj = segments[self.seg(j)].to_exact();

                if segi.lower(&y, &eps) > segj.upper(&y, &eps) {
                    return Some((self.seg(i), self.seg(j)));
                }
            }
        }

        None
    }

    // Finds an index into this sweep line where it's ok to insert this new segment.
    fn insertion_idx<F: Float>(
        &self,
        y: &F,
        segments: &Segments<F>,
        seg: &Segment<F>,
        eps: &F,
    ) -> usize {
        // Checks if `other` is smaller than `seg` with no false negatives: if `other` is actually smaller than `seg`
        // it will definitely return true.
        let maybe_strictly_smaller = |other: &SegmentOrderEntry| -> bool {
            let other = &segments[other.seg];
            // `at_y` is guaranteed to have accuracy eps/8, and in order to have a strict segment inequality there
            // needs to be a difference of at least 2*eps (more, if there's a slope).
            other.at_y(y) < seg.at_y(y)
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
            let other = &segments[self.seg(i)];
            // `lower` is accurate to within eps/8, so if this test succeeds then other's lower bound is
            // definitely bigger.
            if other.lower(y, eps) - seg.lower(y, eps) >= eps.clone() / F::from_f32(4.0) {
                break;
            }
        }
        idx
    }

    // Find the position of the given segment in our array.
    //
    // TODO: if we're large, we could use a binary search.
    fn position(&self, seg: SegIdx) -> Option<usize> {
        self.segs.iter().position(|&x| x.seg == seg)
    }

    fn range_order(&self, range: (usize, usize)) -> HashMap<SegIdx, usize> {
        self.segs[range.0..range.1]
            .iter()
            .enumerate()
            .map(|(idx, seg)| (seg.seg, idx))
            .collect()
    }
}
/// Returns an iterator of the allowable x positions for a slice of segments.
fn horizontal_positions<'a, F: Float>(
    entries: &[SegmentOrderEntry],
    y: &F,
    segments: &'a Segments<F>,
    eps: &'a F,
) -> Vec<(SegmentOrderEntry, F, F)> {
    let mut max_so_far = entries
        .first()
        .map(|seg| segments[seg.seg].lower(y, eps))
        // If `self.segs` is empty our y doesn't matter; we're going to return
        // an empty vec.
        .unwrap_or(F::from_f32(0.0));

    let mut ret: Vec<_> = entries
        .iter()
        .map(move |seg| {
            let x = segments[seg.seg].lower(y, eps);
            max_so_far = max_so_far.clone().max(x);
            // Fill out the minimum allowed positions, with a placeholder for the maximum.
            (*seg, max_so_far.clone(), F::from_f32(0.0))
        })
        .collect();

    let mut min_so_far = entries
        .last()
        .map(|seg| segments[seg.seg].upper(y, eps))
        // If `self.segs` is empty our y doesn't matter; we're going to return
        // an empty vec.
        .unwrap_or(F::from_f32(0.0));
    for (entry, _, max_allowed) in ret.iter_mut().rev() {
        let x = segments[entry.seg].upper(y, eps);
        min_so_far = min_so_far.clone().min(x);
        *max_allowed = min_so_far.clone();
    }

    ret
}

/// A sweep-line, as output by a [`Sweeper`].
///
/// This contains all the information about how the input line segments interact
/// with a sweep-line, including which segments start here, which segments end here,
/// and which segments intersect here.
///
/// A sweep-line stores a `y` coordinate along with two (potentially) different
/// orderings of the segments: the ordering just above `y` and the ordering just
/// below `y`. If two line segments intersect at the sweep-line, their orderings
/// above `y` and below `y` will be different.
///
/// TODO: the public API is pretty gross right now and requires lots of allocation.
/// It will change.
#[derive(Clone, Debug)]
pub struct SweepLine<'a, F: Float> {
    state: &'a Sweeper<F>,
}

impl<F: Float> Copy for SweepLine<'_, F> {}

impl<F: Float> SweepLine<'_, F> {
    /// The vertical position of this sweep-line.
    pub fn y(&self) -> &F {
        &self.state.y
    }

    /// Get the line segment at position `idx` in the new order.
    pub fn line_segment(&self, idx: usize) -> SegIdx {
        self.state.line.segs[idx].seg
    }

    /// Returns the index ranges of segments in this sweep-line that need to be
    /// given explicit positions.
    ///
    /// Not every line segment that passes through a sweep-line needs to be
    /// subdivided at that sweep-line; in order to have a fast sweep-line
    /// implementation, we need to be able to ignore the segments that don't
    /// need subdivision.
    ///
    /// This method returns a list of ranges (in increasing order, non-overlapping,
    /// and non-adjacent). Each of those ranges indexes a range of segments
    /// that need to be subdivided at the current sweep-line.
    pub fn changed_intervals(&self) -> &[(usize, usize)] {
        &self.state.changed_intervals
    }

    /// Given a range of segments at this sweep line (see [`SweepLine::changed_intervals`]),
    /// returns maps for looking up segment orders.
    ///
    /// The first map corresponds to the old order at this height; the second map corresponds
    /// to the new order. If the range came from `SweepLine::changed_intervals`, both maps will
    /// have the same set of keys. The values in the maps are the positions of the segments in
    /// the sweep line.
    ///
    /// For example, if you ask for the range `4..7` and the old sweep line has segments `s42`,
    /// `s1`, `s77` in those positions then you'll get back the map `{ s42 -> 4, s1 -> 5, s77 -> 6 }`.
    pub fn range_orders(
        &self,
        range: (usize, usize),
    ) -> (HashMap<SegIdx, usize>, HashMap<SegIdx, usize>) {
        let mut old_order = HashMap::new();
        for (i, entry) in self.state.line.segs[range.0..range.1].iter().enumerate() {
            old_order.insert(entry.seg, entry.old_idx.unwrap_or(range.0 + i));
        }
        (old_order, self.state.line.range_order(range))
    }

    /// Returns an [`OutputEventBatcher`] for visiting and processing all positions within
    /// a range of segments.
    ///
    /// The range must have come from [`SweepLine::changed_intervals`].
    pub fn events_in_range(
        &self,
        range: (usize, usize),
        segments: &Segments<F>,
        eps: &F,
    ) -> OutputEventBatcher<F> {
        let mut old_line = vec![SegmentOrderEntry::from(SegIdx(42)); range.1 - range.0];
        for entry in &self.state.line.segs[range.0..range.1] {
            old_line[entry.old_idx.unwrap() - range.0] = *entry;
        }
        let old_xs = horizontal_positions(&old_line, &self.state.y, segments, eps);
        let new_xs = horizontal_positions(
            &self.state.line.segs[range.0..range.1],
            &self.state.y,
            segments,
            eps,
        );

        // The two positioning arrays should have the same segments, but possibly in a different
        // order. We build them up in the old-sweep-line order.
        // TODO: we should be able to build up the return value directly
        let mut seg_positions: Vec<(SegIdx, F, Option<F>, bool, bool)> =
            Vec::with_capacity(range.1 - range.0);
        // It would be natural to set max_so_far to -infinity, but a generic F: Float doesn't
        // have -infinity.
        let mut max_so_far = old_xs
            .first()
            .map_or(F::from_f32(0.0), |(_idx, min_x, _max_x)| {
                min_x.clone() - F::from_f32(1.0)
            });
        for (entry, min_x, max_x) in old_xs {
            let preferred_x = if entry.exit {
                // The best possible position is the true segment-ending position.
                // (This could change if we want to be more sophisticated at joining contours.)
                segments[entry.seg].end.x.clone()
            } else if entry.enter {
                // The best possible position is the true segment-starting position.
                // (This could change if we want to be more sophisticated at joining contours.)
                segments[entry.seg].start.x.clone()
            } else if max_so_far >= min_x {
                // The best possible position is snapping to the neighbor that we
                // just positioned.
                max_so_far.clone()
            } else {
                segments[entry.seg].at_y(&self.state.y)
            };
            let x = preferred_x
                .max(min_x.clone())
                .max(max_so_far.clone())
                .min(max_x.clone());
            max_so_far = x.clone();
            seg_positions.push((entry.seg, x, None, entry.enter, entry.exit));
        }

        let mut max_so_far = new_xs
            .first()
            .map_or(F::from_f32(0.0), |(_idx, min_x, _max_x)| {
                min_x.clone() - F::from_f32(1.0)
            });
        for (entry, min_x, max_x) in new_xs {
            let (seg_idx, old_x, new_x, _, _) =
                &mut seg_positions[entry.old_idx.unwrap() - range.0];
            debug_assert_eq!(*seg_idx, entry.seg);
            let preferred_x = if min_x <= *old_x && *old_x <= max_x {
                // Try snapping to the previous position if possible.
                old_x.clone()
            } else if max_so_far >= min_x {
                // Try snapping to the neighbor we just positioned.
                max_so_far.clone()
            } else {
                segments[entry.seg].at_y(&self.state.y)
            };
            let x = preferred_x
                .max(min_x.clone())
                .max(max_so_far.clone())
                .min(max_x.clone());
            max_so_far = x.clone();
            *new_x = Some(x);
        }

        let mut positions = Vec::with_capacity(range.1 - range.0);
        for (idx, x0, x1, entrance, exit) in seg_positions {
            let Some(x1) = x1 else {
                panic!("the two ranges should have the same segments");
            };

            let seg = &segments[idx];
            let ev = match (entrance, exit) {
                (true, true) => {
                    // A horizontal segment. We can ignore the inferred position, and
                    // go with the original bounds.
                    debug_assert!(seg.is_horizontal());
                    OutputEvent::new(idx, seg.start.x.clone(), false, seg.end.x.clone(), false)
                }
                (true, false) => {
                    // The segment enters at this sweep-line. The actual entrance position
                    // will be the segment's true start position, and then we will move
                    // to the adjusted position before leaving this sweep-line.
                    OutputEvent::new(idx, seg.start.x.clone(), false, x1, true)
                }
                (false, true) => {
                    // The segment terminates at this sweep-line. The actual final position
                    // will be the segment's true end position, but the segment will enter
                    // the sweep-line at the adjusted position.
                    OutputEvent::new(idx, x0, true, seg.end.x.clone(), false)
                }
                (false, false) => OutputEvent::new(idx, x0, true, x1, true),
            };

            positions.push(ev);
        }
        positions.sort();

        OutputEventBatcher {
            positions: positions.into_iter(),
            last_x: None,
            active_horizontals: BTreeSet::new(),
        }
    }
}

/// Given a sorted list of disjoint, endpoint-exclusive intervals, merge adjacent ones.
///
/// For example, [1..2, 4..5, 5..6] is turned into [1..2, 4..6].
fn merge_adjacent(intervals: &mut Vec<(usize, usize)>) {
    if intervals.is_empty() {
        return;
    }

    let mut write_idx = 0;
    for read_idx in 1..intervals.len() {
        let last_end = intervals[write_idx].1;
        let cur = intervals[read_idx];
        if last_end < cur.0 {
            write_idx += 1;
            intervals[write_idx] = cur;
        } else {
            intervals[write_idx].1 = cur.1;
        }
    }
    intervals.truncate(write_idx + 1);
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct HSeg<F: Float> {
    pub end: F,
    pub connected_at_start: bool,
    pub connected_at_end: bool,
    pub enter_first: bool,
    pub seg: SegIdx,
    pub start: F,
}

impl<F: Float> HSeg<F> {
    pub fn from_position(pos: OutputEvent<F>) -> Option<Self> {
        let OutputEvent {
            x0,
            connected_above,
            x1,
            connected_below,
            ..
        } = pos;

        if x0 == x1 {
            return None;
        }

        let enter_first = x0 < x1;
        let (start, end, connected_at_start, connected_at_end) = if enter_first {
            (x0, x1, connected_above, connected_below)
        } else {
            (x1, x0, connected_below, connected_above)
        };
        Some(HSeg {
            end,
            start,
            enter_first,
            seg: pos.seg_idx,
            connected_at_start,
            connected_at_end,
        })
    }
    pub fn connected_above_at(&self, x: &F) -> bool {
        (*x == self.start && self.enter_first && self.connected_at_start)
            || (*x == self.end && !self.enter_first && self.connected_at_end)
    }

    pub fn connected_below_at(&self, x: &F) -> bool {
        (*x == self.start && !self.enter_first && self.connected_at_start)
            || (*x == self.end && self.enter_first && self.connected_at_end)
    }
}

/// Describes the interaction between a line segment and a sweep-line.
///
/// In exact math, a non-horizontal line segment can interact with a sweep-line
/// in exactly one way: by intersecting it at a point. When dealing with inexact
/// math, intersections and re-orderings between line segments might force
/// our sweep-line algorithm to perturb the line segment. In that case, even
/// a non-horizontal line segment might enter and leave the sweep-line at two
/// different points.
///
/// `OutputEvent` is ordered by the smallest horizontal coordinate where
/// it intersects the sweep-line (i.e. [`OutputEvent::smaller_x`]).
///
/// The two points, `x0` and `x1`, are in sweep-line order. This doesn't
/// necessarily mean that `x0` is smaller than `x1`! Instead, it means that
/// when traversing the segment in sweep-line order (i.e. in increasing `y`,
/// and increasing `x` if the segment is horizontal) then it visits `x0`
/// before `x1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputEvent<F: Float> {
    /// The first horizontal coordinate on this sweep-line that we'd hit if
    /// we were traversing the segment in sweep-line orientation.
    pub x0: F,
    /// Does this line segment extend "above" (i.e. smaller `y`) this sweep-line?
    ///
    /// If so, it will extend up from `x0`, because that's what the "sweep-line order"
    /// means.
    pub connected_above: bool,
    /// The last horizontal coordinate on this sweep-line that we'd hit if
    /// we were traversing the segment in sweep-line orientation.
    pub x1: F,
    /// Does this line segment extend "below" (i.e. larger `y`) this sweep-line?
    ///
    /// If so, it will extend down from `x1`, because that's what the "sweep-line order"
    /// means.
    pub connected_below: bool,
    /// The segment that's interacting with the sweep line.
    pub seg_idx: SegIdx,
}

impl<F: Float> OutputEvent<F> {
    /// The smallest `x` coordinate at which the line segment touches the sweep-line.
    pub fn smaller_x(&self) -> &F {
        (&self.x0).min(&self.x1)
    }

    /// The largest `x` coordinate at which the line segment touches the sweep-line.
    pub fn larger_x(&self) -> &F {
        (&self.x0).max(&self.x1)
    }

    fn new(seg_idx: SegIdx, x0: F, connected_above: bool, x1: F, connected_below: bool) -> Self {
        Self {
            x0,
            connected_above,
            x1,
            connected_below,
            seg_idx,
        }
    }

    /// Does the line segment extend up from the horizontal coordinate `x`?
    pub fn connected_above_at(&self, x: &F) -> bool {
        x == &self.x0 && self.connected_above
    }

    /// Does the line segment extend down from the horizontal coordinate `x`?
    pub fn connected_below_at(&self, x: &F) -> bool {
        x == &self.x1 && self.connected_below
    }
}

impl<F: Float> Ord for OutputEvent<F> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.smaller_x()
            .cmp(other.smaller_x())
            .then_with(|| self.larger_x().cmp(other.larger_x()))
    }
}

impl<F: Float> PartialOrd for OutputEvent<F> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Emits output events for a single sub-range of a single sweep-line.
///
/// This is constructed using [`SweepLine::events_in_range`]. By repeatedly
/// calling `OutputEventBatcher::increase_x` you can iterate over all
/// interesting horizontal positions, left to right (i.e. smaller `x` to larger
/// `x`).
#[derive(Clone, Debug)]
pub struct OutputEventBatcher<F: Float> {
    last_x: Option<F>,
    positions: std::vec::IntoIter<OutputEvent<F>>,
    active_horizontals: BTreeSet<HSeg<F>>,
}

impl<F: Float> OutputEventBatcher<F> {
    /// The current horizontal position, or `None` if we're finished.
    pub fn x(&self) -> Option<&F> {
        match (
            self.active_horizontals.first(),
            self.positions.as_slice().first(),
        ) {
            (None, None) => None,
            (None, Some(pos)) => Some(pos.smaller_x()),
            (Some(h), None) => Some(&h.end),
            (Some(h), Some(pos)) => Some((&h.end).min(pos.smaller_x())),
        }
    }

    fn positions_at_x<'a, 'b: 'a>(
        &'b self,
        x: &'a F,
    ) -> impl Iterator<Item = &'b OutputEvent<F>> + 'a {
        self.positions
            .as_slice()
            .iter()
            .take_while(move |p| p.smaller_x() == x)
    }

    /// Iterates over all segments at the current position that are connected to
    /// something "above" (i.e. with smaller `y` than) the current sweep line.
    pub fn segments_connected_up(&self) -> impl Iterator<Item = SegIdx> + '_ {
        let maybe_iter = self.x().map(|x| {
            let horiz = self
                .active_horizontals
                .iter()
                .filter(|hseg| hseg.connected_above_at(x))
                .map(|hseg| hseg.seg);

            let posns = self
                .positions_at_x(x)
                .filter(move |pos| pos.connected_above_at(x))
                .map(|pos| pos.seg_idx);

            horiz.chain(posns)
        });

        maybe_iter.into_iter().flatten()
    }

    /// Iterates over all segments at the current position that are connected to
    /// something "below" (i.e. with larger `y` than) the current sweep line.
    pub fn segments_connected_down(&self) -> impl Iterator<Item = SegIdx> + '_ {
        let maybe_iter = self.x().map(|x| {
            let horiz = self
                .active_horizontals
                .iter()
                .filter(|hseg| hseg.connected_below_at(x))
                .map(|hseg| hseg.seg);

            let posns = self
                .positions_at_x(x)
                .filter(move |pos| pos.connected_below_at(x))
                .map(|pos| pos.seg_idx);

            horiz.chain(posns)
        });

        maybe_iter.into_iter().flatten()
    }

    /// Iterates over the horizontal segments that are active at the current position.
    ///
    /// This includes the segments that end here, but does not include the ones
    /// that start here.
    pub fn active_horizontals(&self) -> impl Iterator<Item = SegIdx> + '_ {
        self.active_horizontals.iter().map(|hseg| hseg.seg)
    }

    /// Returns the collection of all output events that end at the current
    /// position, or `None` if this batcher is finished.
    ///
    /// If this returns `None`, this batcher is finished.
    ///
    /// All the returned events start at the previous `x` position and end
    /// at the current `x` position. In particular, if you alternate between
    /// calling [`OutputEventBatcher::increase_x`] and this method, you'll
    /// receive non-overlapping batches of output events.
    pub fn events(&mut self) -> Option<Vec<OutputEvent<F>>> {
        let next_x = self.x()?.clone();

        let mut ret = Vec::new();
        for h in &self.active_horizontals {
            // unwrap: on the first event of this sweep line, active_horizontals is empty. So
            // we only get here after last_x is populated.
            let x0 = self.last_x.clone().unwrap();
            let x1 = next_x.clone().min(h.end.clone());
            let connected_end = x1 == h.end && h.connected_at_end;
            let connected_start = x0 == h.start && h.connected_at_start;
            if h.enter_first {
                ret.push(OutputEvent::new(
                    h.seg,
                    x0,
                    connected_start,
                    x1,
                    connected_end,
                ));
            } else {
                ret.push(OutputEvent::new(
                    h.seg,
                    x1,
                    connected_end,
                    x0,
                    connected_start,
                ));
            }
        }

        // Drain the active horizontals, throwing away horizontal segments that end before
        // the new x position.
        while let Some(h) = self.active_horizontals.first() {
            if h.end <= next_x {
                self.active_horizontals.pop_first();
            } else {
                break;
            }
        }

        while let Some(pos) = self.positions.as_slice().first() {
            if pos.smaller_x() > &next_x {
                break;
            }
            // unwrap: we just peeked at this element.
            let pos = self.positions.next().unwrap();

            if pos.x0 == pos.x1 {
                // We push output event for points immediately.
                ret.push(pos);
            } else if let Some(hseg) = HSeg::from_position(pos) {
                // For horizontal segments, we don't output anything straight
                // away. When we update the horizontal position and visit our
                // horizontal segments, we'll output something.
                self.active_horizontals.insert(hseg);
            }
        }
        self.last_x = Some(next_x);
        Some(ret)
    }

    /// Move along to the next horizontal position.
    pub fn increase_x(&mut self) {
        if let Some(x) = self.x().cloned() {
            self.drain_active_horizontals(&x);

            while let Some(p) = self.positions.as_slice().first() {
                if p.smaller_x() > &x {
                    break;
                }
                // unwrap: we just peeked at this element.
                let p = self.positions.next().unwrap();

                if let Some(hseg) = HSeg::from_position(p) {
                    self.active_horizontals.insert(hseg);
                }
            }
        }
    }

    fn drain_active_horizontals(&mut self, x: &F) {
        while let Some(h) = self.active_horizontals.first() {
            if h.end <= *x {
                self.active_horizontals.pop_first();
            } else {
                break;
            }
        }
    }
}

/// Runs the sweep-line algorithm, calling the provided callback on every output point.
pub fn sweep<F: Float, C: FnMut(F, OutputEvent<F>)>(
    segments: &Segments<F>,
    eps: &F,
    mut callback: C,
) {
    let mut state = Sweeper::new(segments, eps.clone());
    while let Some(line) = state.next_line() {
        for &(start, end) in line.changed_intervals() {
            let mut positions = line.events_in_range((start, end), segments, eps);
            while let Some(events) = positions.events() {
                for ev in events {
                    callback(line.state.y.clone(), ev);
                }
            }
        }
    }
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
            realize_perturbation, F64Perturbation, FloatPerturbation, Perturbation,
            PointPerturbation,
        },
        segments::Segments,
    };

    fn mk_segs(xs: &[(f64, f64)]) -> Segments<NotNan<f64>> {
        let y0: NotNan<f64> = 0.0.try_into().unwrap();
        let y1: NotNan<f64> = 1.0.try_into().unwrap();
        let mut segs = Segments::default();

        for &(x0, x1) in xs {
            segs.add_points([
                Point::new(x0.try_into().unwrap(), y0),
                Point::new(x1.try_into().unwrap(), y1),
            ]);
        }
        segs
    }

    #[test]
    fn merge_adjacent() {
        fn merge(v: impl IntoIterator<Item = (usize, usize)>) -> Vec<(usize, usize)> {
            let mut v = v.into_iter().collect();
            super::merge_adjacent(&mut v);
            v
        }

        assert_eq!(merge([(1, 2), (3, 4)]), vec![(1, 2), (3, 4)]);
        assert_eq!(merge([(1, 2), (2, 4)]), vec![(1, 4)]);
        assert_eq!(merge([(1, 2), (3, 4), (4, 5)]), vec![(1, 2), (3, 5)]);
        assert_eq!(merge([]), vec![]);
    }

    #[test]
    fn invalid_order() {
        fn check_order(xs: &[(f64, f64)], at: f64, eps: f64) -> Option<(usize, usize)> {
            let eps: NotNan<f64> = eps.try_into().unwrap();
            let y: NotNan<f64> = at.try_into().unwrap();
            let segs = mk_segs(xs);

            let line: SegmentOrder = (0..xs.len()).map(SegIdx).collect();

            line.find_invalid_order(&y, &segs, &eps)
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

            let line: SegmentOrder = (0..xs.len()).map(SegIdx).collect();
            line.insertion_idx(&y, &segs, &new, &eps)
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
        dbg!(&segs);
        sweep(&segs, &eps, |_, ev| {
            dbg!(ev);
        });
    }

    fn p<F: Float>(x: f32, y: f32) -> Point<F> {
        Point::new(F::from_f32(x), F::from_f32(y))
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
        let mut segs = Segments::default();
        for poly in perturbed_polylines {
            segs.add_cycle(poly);
        }
        let eps = P::Float::from_f32(0.1);
        sweep(&segs, &eps, |_, _| {});
    }

    #[derive(serde::Serialize)]
    struct Output {
        order: SegmentOrder,
        changed: Vec<(usize, usize)>,
    }
    fn snapshot_outputs(segs: Segments<NotNan<f64>>, eps: f64) -> Vec<Output> {
        let mut outputs = Vec::new();
        let mut sweeper = Sweeper::new(&segs, eps.try_into().unwrap());
        while let Some(line) = sweeper.next_line() {
            outputs.push(Output {
                order: line.state.line.clone(),
                changed: line.changed_intervals().to_owned(),
            });
        }
        outputs
    }

    #[test]
    fn square() {
        let segs =
            Segments::from_closed_cycle([p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0)]);
        insta::assert_ron_snapshot!(snapshot_outputs(segs, 0.01));
    }

    #[test]
    fn regression_position_state() {
        let ps = [Perturbation::Point {
            perturbation: PointPerturbation {
                x: F64Perturbation::Ulp(0),
                y: F64Perturbation::Ulp(-1),
            },
            idx: 14924312967467343829,
            next: Box::new(Perturbation::Base { idx: 0 }),
        }];
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
        let mut segs = Segments::default();
        for poly in perturbed_polylines {
            segs.add_cycle(poly);
        }
        insta::assert_ron_snapshot!(snapshot_outputs(segs, 0.1));
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
