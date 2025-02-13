//! A sweep-line implementation using weak orderings.
//!
//! This algorithm is documented in `docs/sweep.typ`.

use std::collections::BTreeSet;

use malachite::Rational;

use crate::{
    geom::Segment,
    num::Float,
    segments::{SegIdx, Segments},
    treevec::TreeVec,
};

use super::{OutputEvent, SweepLineRange, SweepLineRangeBuffers};

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct SegmentOrderEntry<F: Float> {
    seg: SegIdx,
    /// True if this segment is about to leave the sweep-line.
    ///
    /// We handle enter/exits like this:
    ///
    /// 1. insert the newly entered segments
    /// 2. mark the about-to-exit segments
    /// 3. process intersections and re-shuffle things
    /// 4. output the geometry
    /// 5. remove the exited segments that we marked in step 2.
    ///
    /// The reason we don't remove about-to-exit segments immediately in
    /// step 2 is to make it easier to compare the old and new orderings.
    /// We keep track of the re-shuffling in step 3 (see `old_idx` below),
    /// and the lack of deletions means that we don't need to worry about
    /// indices shifting.
    exit: bool,
    enter: bool,
    /// This is epsilon below this segment's smallest horizontal position. All
    /// horizontal positions smaller than this are guaranteed not to interact
    /// with this segment.
    lower_bound: F,
    /// This is epsilon above this segment's largest horizontal position. All
    /// horizontal positions larger than this are guaranteed not to interact
    /// with this segment.
    upper_bound: F,
    /// This is filled out during `compute_changed_intervals`, where we use it as
    /// a sort of "dirty" flag to avoid processing an entry twice.
    in_changed_interval: bool,
    /// We need to keep track of two sweep-line orders at once: the previous one
    /// and the current one. And if a large sweep-line has only one small change,
    /// we want the cost of this tracking to be small. We do this by keeping the
    /// sweep line in the "current" order, and then whenever some segments have
    /// their order changed, we remember their old positions.
    ///
    /// So, for example, say we're in a situation like this, where the dashed
    /// horizontal line is the position of the sweep:
    ///
    /// ```text
    /// s_1  s_3   s_5  s_7
    ///  │     ╲   ╱     ╲
    ///  │      ╲ ╱       ╲
    /// ╌│╌╌╌╌╌╌╌╳╌╌╌╌╌╌╌╌╌╲
    ///  │      ╱ ╲         ╲
    /// ```
    ///
    /// The old order is [s_1, s_3, s_5, s_7], and we start off with all the old_idx
    /// fields set to `None`. Then we swap `s_3` and `s_5`, so the current order
    /// is [s_1, s_5, s_3, s_7] and we set the old_idx fields to be
    /// [None, Some(2), Some(1), None]. This allows us to reconstruct the original
    /// order when we need to.
    old_idx: Option<usize>,
    /// When two segments are contour-adjacent, we allow them to "share" the same
    /// sweep-line slot. This helps performance (because we aren't constantly
    /// removing and inserting segments in the middle of the sweep-line), and makes
    /// it easier to generate good output (because if we have both contour-adjacent
    /// segments handy, it's easy to avoid unnecessary horizontal segments).
    ///
    /// So in a situation like this, for example:
    ///
    /// ```text
    /// s_1   s_3     s_7
    ///  │      ╲       ╲
    ///  │       ╲       ╲
    ///  │       ╱        ╲
    ///  │      ╱          ╲
    ///       s_5
    /// ```
    ///
    /// when the sweep-line hits the "kink" the new order will be [s_1, s_5, s_7]
    /// and the old_seg values will be [None, Some(s_3), None].
    ///
    /// Note that `old_seg` is not guaranteed to get used for all contour-adjacent
    /// segments, even if they're monotonic in y: if there are some other annoying
    /// segments nearby, the new segment and the old segment might get separated
    /// in the sweep-line. In this case, they will get their own entries.
    ///
    /// If this is `Some`, both `enter` and `exit` will be true.
    old_seg: Option<SegIdx>,
}

impl<F: Float> SegmentOrderEntry<F> {
    fn new(seg: SegIdx, segments: &Segments<F>, eps: &F) -> Self {
        let x0 = segments[seg].start.x.clone();
        let x1 = segments[seg].end.x.clone();
        Self {
            seg,
            exit: false,
            enter: false,
            lower_bound: x0.clone().min(x1.clone()) - eps,
            upper_bound: x0.max(x1) + eps,
            in_changed_interval: false,
            old_idx: None,
            old_seg: None,
        }
    }

    fn reset_state(&mut self) {
        self.exit = false;
        self.enter = false;
        self.in_changed_interval = false;
        self.old_idx = None;
        self.old_seg = None;
    }

    fn set_old_idx_if_unset(&mut self, i: usize) {
        if self.old_idx.is_none() {
            self.old_idx = Some(i);
        }
    }

    fn old_seg(&self) -> SegIdx {
        self.old_seg.unwrap_or(self.seg)
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct SegmentOrder<F: Float> {
    pub(crate) segs: TreeVec<SegmentOrderEntry<F>, 128>,
}

impl<F: Float> Default for SegmentOrder<F> {
    fn default() -> Self {
        Self {
            segs: TreeVec::new(),
        }
    }
}

impl<F: Float> SegmentOrder<F> {
    fn seg(&self, i: usize) -> SegIdx {
        self.segs[i].seg
    }

    fn is_exit(&self, i: usize) -> bool {
        let seg = &self.segs[i];
        seg.exit && seg.old_seg.is_none()
    }
}

#[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
struct IntersectionEvent<F: Float> {
    pub y: F,
    /// This segment used to be to the left, and after the intersection it will be to the right.
    ///
    /// In our sweep line intersection, this segment might have already been moved to the right by
    /// some other constraints. That's ok.
    pub left: SegIdx,
    /// This segment used to be to the right, and after the intersection it will be to the left.
    pub right: SegIdx,
}

#[derive(Clone, Debug)]
struct EventQueue<F: Float> {
    /// The enter events are stored in `Segments<F>`; this is the index of the first
    /// one that we haven't processed yet.
    next_enter_idx: usize,
    /// The index of the first exit event that we haven't processed yet.
    next_exit_idx: usize,
    intersection: std::collections::BTreeSet<IntersectionEvent<F>>,
}

impl<F: Float> Default for EventQueue<F> {
    fn default() -> Self {
        Self {
            next_enter_idx: 0,
            next_exit_idx: 0,
            intersection: BTreeSet::new(),
        }
    }
}

impl<F: Float> EventQueue<F> {
    pub fn push(&mut self, ev: IntersectionEvent<F>) {
        self.intersection.insert(ev);
    }

    pub fn next_y<'a>(&'a self, segments: &'a Segments<F>) -> Option<&'a F> {
        let enter_y = segments
            .entrances()
            .get(self.next_enter_idx)
            .map(|(y, _)| y);
        let exit_y = segments.exits().get(self.next_exit_idx).map(|(y, _)| y);
        let int_y = self.intersection.first().map(|i| &i.y);

        [enter_y, exit_y, int_y].into_iter().flatten().min()
    }

    pub fn entrances_at_y<'a>(&mut self, y: &F, segments: &'a Segments<F>) -> &'a [(F, SegIdx)] {
        let entrances = &segments.entrances()[self.next_enter_idx..];
        let count = entrances
            .iter()
            .position(|(enter_y, _)| enter_y > y)
            .unwrap_or(entrances.len());
        self.next_enter_idx += count;
        &entrances[..count]
    }

    pub fn exits_at_y<'a>(&mut self, y: &F, segments: &'a Segments<F>) -> &'a [(F, SegIdx)] {
        let exits = &segments.exits()[self.next_exit_idx..];
        let count = exits
            .iter()
            .position(|(exit_y, _)| exit_y > y)
            .unwrap_or(exits.len());
        self.next_exit_idx += count;
        &exits[..count]
    }

    pub fn next_intersection_at_y(&mut self, y: &F) -> Option<IntersectionEvent<F>> {
        if self.intersection.first().map(|i| &i.y) == Some(y) {
            self.intersection.pop_first()
        } else {
            None
        }
    }
}

/// Holds some buffers that are used when iterating over a sweep-line.
///
/// Save on re-allocation by allocating this once and reusing it in multiple calls to
/// [`Sweeper::next_line`].
#[derive(Clone, Debug)]
pub struct SweepLineBuffers<F: Float> {
    /// A subset of the old sweep-line.
    old_line: Vec<SegmentOrderEntry<F>>,
    /// A vector of (segment, min allowable horizontal position, max allowable horizontal position).
    positions: Vec<(SegmentOrderEntry<F>, F, F)>,
    output_events: Vec<OutputEvent<F>>,
}

impl<F: Float> Default for SweepLineBuffers<F> {
    fn default() -> Self {
        SweepLineBuffers {
            old_line: Vec::new(),
            positions: Vec::new(),
            output_events: Vec::new(),
        }
    }
}

/// Encapsulates the state of the sweep-line algorithm and allows iterating over sweep lines.
#[derive(Clone, Debug)]
pub struct Sweeper<'a, F: Float> {
    y: F,
    eps: F,
    line: SegmentOrder<F>,
    events: EventQueue<F>,
    segments: &'a Segments<F>,

    horizontals: Vec<SegIdx>,

    // The collection of segments that we know need to be given explicit
    // positions in the current sweep line.
    //
    // These include:
    // - any segments that changed order with any other segments
    // - any segments that entered or exited
    //
    // These segments are identified by their index in the current order, so that
    // it's fast to find them. It means that we need to do some fixing-up of indices after
    // inserting all the new segments.
    segs_needing_positions: Vec<usize>,
    changed_intervals: Vec<ChangedInterval>,
}

impl<'segs, F: Float> Sweeper<'segs, F> {
    /// Creates a new sweeper for a collection of segments, and with a given tolerance.
    pub fn new(segments: &'segs Segments<F>, eps: F) -> Self {
        let events = EventQueue::default();

        Sweeper {
            eps,
            line: SegmentOrder::default(),
            y: events.next_y(segments).unwrap().clone(),
            events,
            segments,
            segs_needing_positions: Vec::new(),
            changed_intervals: Vec::new(),
            horizontals: Vec::new(),
        }
    }

    /// Moves the sweep forward, returning the next sweep line.
    ///
    /// Returns `None` when sweeping is complete.
    pub fn next_line<'slf, 'buf>(
        &'slf mut self,
        bufs: &'buf mut SweepLineBuffers<F>,
    ) -> Option<SweepLine<'buf, 'slf, 'segs, F>> {
        self.check_invariants();

        let y = self.events.next_y(self.segments).cloned()?;
        self.advance(y.clone());
        self.check_invariants();

        // Process all the enter events at this y.
        {
            let enters = self.events.entrances_at_y(&y, self.segments);
            for (enter_y, idx) in enters {
                debug_assert_eq!(enter_y, &y);
                self.handle_enter(*idx);
                self.check_invariants();
            }
        }

        // Process all the exit events.
        {
            let exits = self.events.exits_at_y(&y, self.segments);
            for (exit_y, idx) in exits {
                debug_assert_eq!(exit_y, &y);
                self.handle_exit(*idx);
                self.check_invariants();
            }
        }

        // Process all the intersection events at this y.
        while let Some(intersection) = self.events.next_intersection_at_y(&y) {
            self.handle_intersection(intersection.left, intersection.right);
            self.check_invariants();
        }

        self.compute_changed_intervals();
        Some(SweepLine {
            state: self,
            next_changed_interval: 0,
            bufs,
        })
    }

    fn advance(&mut self, y: F) {
        // All the exiting segments should be in segs_needing_positions, so find them all and remove them.
        self.segs_needing_positions
            .retain(|idx| self.line.segs[*idx].exit && self.line.segs[*idx].old_seg.is_none());

        // Reset the state flags for all segments. All segments with non-trivial state flags should
        // belong to the changed intervals. This needs to go before we remove the exiting segments,
        // because that messes up the indices.
        for r in self.changed_intervals.drain(..) {
            for seg in self.line.segs.range_mut(r.segs) {
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
        self.horizontals.clear();
    }

    fn intersection_scan_right(&mut self, start_idx: usize) {
        let seg = &self.segments[self.line.seg(start_idx)];
        let y = &self.y;
        let two_eps = self.eps.clone() * F::from_f32(2.0);

        // We're allowed to take a potentially-smaller height bound by taking
        // into account the current queue. A larger height bound is still ok,
        // just a little slower.
        let mut height_bound = seg.end.y.clone();

        for j in (start_idx + 1)..self.line.segs.len() {
            if self.line.is_exit(j) {
                continue;
            }
            let other = &self.segments[self.line.seg(j)];
            if seg.quick_left_of(other, &two_eps) {
                break;
            }
            height_bound = height_bound.min(other.end.y.clone());

            if let Some(int_y) = seg.crossing_y(other, &self.eps) {
                let int_y = int_y.max(y.clone());
                self.events.push(IntersectionEvent {
                    y: int_y.clone().max(y.clone()),
                    left: self.line.seg(start_idx),
                    right: self.line.seg(j),
                });
                height_bound = int_y.min(height_bound);
            }

            // For the early stopping, we need to check whether `seg` is less than `other`'s lower
            // bound on the whole interesting `y` interval. Since they're lines, it's enough to check
            // at the two interval endpoints.
            let y1 = &height_bound;
            let threshold = self.eps.clone() / F::from_f32(4.0);
            let scaled_eps = other.scaled_eps(&self.eps);
            if threshold <= other.lower_with_scaled_eps(y, &self.eps, &scaled_eps) - seg.at_y(y)
                && threshold
                    <= other.lower_with_scaled_eps(y1, &self.eps, &scaled_eps) - seg.at_y(y1)
            {
                break;
            }
        }
    }

    fn intersection_scan_left(&mut self, start_idx: usize) {
        let seg = &self.segments[self.line.seg(start_idx)];
        let y = &self.y;
        let two_eps = self.eps.clone() * F::from_f32(2.0);

        let mut height_bound = seg.end.y.clone();

        for j in (0..start_idx).rev() {
            if self.line.is_exit(j) {
                continue;
            }
            let other = &self.segments[self.line.seg(j)];
            if other.quick_left_of(seg, &two_eps) {
                break;
            }
            height_bound = height_bound.min(other.end.y.clone());
            if let Some(int_y) = other.crossing_y(seg, &self.eps) {
                let int_y = int_y.max(y.clone());
                self.events.push(IntersectionEvent {
                    left: self.line.seg(j),
                    right: self.line.seg(start_idx),
                    y: int_y.clone().max(self.y.clone()),
                });
                height_bound = int_y.min(height_bound);
            }

            // For the early stopping, we need to check whether `seg` is greater than `other`'s upper
            // bound on the whole interesting `y` interval. Since they're lines, it's enough to check
            // at the two interval endpoints.
            let y1 = &height_bound;
            let scaled_eps = other.scaled_eps(&self.eps);
            let threshold = self.eps.clone() / F::from_f32(4.0);
            if seg.at_y(y) - other.upper_with_scaled_eps(y, &self.eps, &scaled_eps) > threshold
                && seg.at_y(y1) - other.upper_with_scaled_eps(y1, &self.eps, &scaled_eps)
                    > threshold
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

    fn insert(&mut self, pos: usize, seg: SegmentOrderEntry<F>) {
        self.line.segs.insert(pos, seg);
        self.intersection_scan_right(pos);
        self.intersection_scan_left(pos);
    }

    fn handle_enter(&mut self, seg_idx: SegIdx) {
        let new_seg = &self.segments[seg_idx];

        if new_seg.is_horizontal() {
            self.horizontals.push(seg_idx);
            return;
        }

        let pos = self
            .line
            .insertion_idx(&self.y, self.segments, new_seg, &self.eps);
        let contour_prev = if self.segments.positively_oriented(seg_idx) {
            self.segments.contour_prev(seg_idx)
        } else {
            self.segments.contour_next(seg_idx)
        };
        if let Some(contour_prev) = contour_prev {
            if self.segments[contour_prev].start.y < self.y {
                if pos < self.line.segs.len() && self.line.segs[pos].seg == contour_prev {
                    self.handle_contour_continuation(seg_idx, new_seg, pos);
                    return;
                }
                if pos > 0 && self.line.segs[pos - 1].seg == contour_prev {
                    self.handle_contour_continuation(seg_idx, new_seg, pos - 1);
                    return;
                }
            }
        }

        let mut entry = SegmentOrderEntry::new(seg_idx, self.segments, &self.eps);
        entry.enter = true;
        entry.exit = false;
        self.insert(pos, entry);
        self.add_seg_needing_position(pos, true);
    }

    fn add_seg_needing_position(&mut self, pos: usize, insert: bool) {
        // Fix up the index of any other segments that we got inserted before
        // (at this point, segs_needing_positions only contains newly-inserted
        // segments, and it's sorted increasing).
        //
        // We sorted all the to-be-inserted segments by horizontal position
        // before inserting them, so we expect these two loops to be short most
        // of the time.
        if insert {
            for other_pos in self.segs_needing_positions.iter_mut().rev() {
                if *other_pos >= pos {
                    *other_pos += 1;
                } else {
                    break;
                }
            }
        }
        let insert_pos = self
            .segs_needing_positions
            .iter()
            .rposition(|p| *p < pos)
            .map(|p| p + 1)
            .unwrap_or(0);
        self.segs_needing_positions.insert(insert_pos, pos);
        debug_assert!(self.segs_needing_positions.is_sorted());
    }

    // A special case of handle-enter, in which the entering segment is
    // continuing the contour of an exiting segment.
    fn handle_contour_continuation(&mut self, seg_idx: SegIdx, seg: &Segment<F>, pos: usize) {
        let x0 = seg.start.x.clone();
        let x1 = seg.end.x.clone();
        self.line.segs[pos].old_seg = Some(self.line.segs[pos].seg);
        self.line.segs[pos].seg = seg_idx;
        self.line.segs[pos].enter = true;
        self.line.segs[pos].exit = true;
        self.line.segs[pos].lower_bound = x0.clone().min(x1.clone()) - &self.eps;
        self.line.segs[pos].upper_bound = x0.max(x1) + &self.eps;
        self.intersection_scan_right(pos);
        self.intersection_scan_left(pos);
        self.add_seg_needing_position(pos, false);
    }

    /// Marks a segment as needing to exit, but doesn't actually remove it
    /// from the sweep-line. See `SegmentOrderEntry::exit` for an explanation.
    fn handle_exit(&mut self, seg_idx: SegIdx) {
        let Some(pos) = self
            .line
            .position(seg_idx, self.segments, &self.y, &self.eps)
        else {
            // It isn't an error if we don't find the segment that's exiting: it
            // might have been marked as a contour continuation, in which case
            // it's now the `old_seg` of some sweep-line entry and not the `seg`.
            return;
        };
        // It's important that this goes before `scan_for_removal`, so that
        // the scan doesn't get confused by the segment that should be marked
        // for exit.
        self.line.segs[pos].exit = true;
        self.scan_for_removal(pos);
        self.segs_needing_positions.push(pos);
    }

    fn handle_intersection(&mut self, left: SegIdx, right: SegIdx) {
        let left_idx = self
            .line
            .position(left, self.segments, &self.y, &self.eps)
            .unwrap();
        let right_idx = self
            .line
            .position(right, self.segments, &self.y, &self.eps)
            .unwrap();
        if left_idx < right_idx {
            self.segs_needing_positions.extend(left_idx..=right_idx);
            for (i, entry) in self.line.segs.range_mut(left_idx..=right_idx).enumerate() {
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
            let mut to_move = vec![(left_idx, self.line.segs[left_idx].clone())];
            let threshold = eps.clone() / F::from_f32(-4.0);
            for j in (left_idx + 1)..right_idx {
                let seg = &self.segments[self.line.seg(j)];
                if seg.lower(y, eps) - left_seg.upper(y, eps) > threshold {
                    to_move.push((j, self.line.segs[j].clone()));
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

            for (_, seg) in to_move.into_iter().rev() {
                self.insert(insertion_pos, seg);
            }
        }
    }

    #[cfg(feature = "slow-asserts")]
    fn check_invariants(&self) {
        for seg_entry in self.line.segs.iter() {
            let seg_idx = seg_entry.seg;
            let seg = &self.segments[seg_idx];
            assert!(
                (&seg.start.y..=&seg.end.y).contains(&&self.y),
                "segment {seg:?} out of range at y={:?}",
                self.y
            );
        }

        // All segments marked as stering or exiting must be in `self.segs_needing_positions`
        for (idx, seg_entry) in self.line.segs.iter().enumerate() {
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
                                .position(idx, self.segments, &self.y, &self.eps)
                                .is_some_and(|pos| i <= pos && pos <= j)
                        };
                        let has_exit_witness = self.line.segs.range(i..=j).any(|seg_entry| {
                            self.segments[seg_entry.seg].end.y.to_exact() <= y_int
                        });

                        let has_intersection_witness = self.events.intersection.iter().any(|ev| {
                            let is_between = is_between(ev.left) && is_between(ev.right);
                            let before_y = ev.y.to_exact() <= y_int;
                            is_between && before_y
                        });
                        let has_witness = has_exit_witness || has_intersection_witness;
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

    fn compute_horizontal_changed_intervals(&mut self) {
        self.horizontals
            .sort_by_key(|seg_idx| self.segments[*seg_idx].start.x.clone());

        for (idx, &seg_idx) in self.horizontals.iter().enumerate() {
            let seg = &self.segments[seg_idx];

            // Find the index of some segment that might overlap with this
            // horizontal segment, but the index before definitely doesn't.
            let start_idx = self.line.segs.partition_point(|other_entry| {
                let other_seg = &self.segments[other_entry.seg];
                seg.start.x > other_seg.upper(&self.y, &self.eps)
            });

            let mut end_idx = start_idx;
            for j in start_idx..self.line.segs.len() {
                let other_entry = &mut self.line.segs[j];
                let other_seg = &self.segments[other_entry.seg];
                if other_seg.lower(&self.y, &self.eps) <= seg.end.x {
                    // Ensure that every segment in the changed interval has `old_idx` set;
                    // see also `compute_changed_intervals`.
                    self.line.segs[j].set_old_idx_if_unset(j);
                    end_idx = j + 1;
                } else {
                    break;
                }
            }
            let changed = ChangedInterval {
                segs: start_idx..end_idx,
                horizontals: Some(idx..(idx + 1)),
            };
            self.changed_intervals.push(changed);
        }
    }

    /// Updates our internal `changed_intervals` state based on the segments marked
    /// as needing positions.
    ///
    /// For each segment marked as needing a position, we expand it to a range of
    /// physically nearby segments, and then we deduplicate and merge those ranges.
    fn compute_changed_intervals(&mut self) {
        debug_assert!(self.changed_intervals.is_empty());
        self.compute_horizontal_changed_intervals();

        let y = &self.y;

        // We compare horizontal positions to decide when to stop iterating. Those positions
        // are each accurate to eps / 8, so we compare them with slack eps / 4 to ensure no
        // false negatives.
        let eps = &self.eps;
        let segments = &self.segments;
        let slack = eps.clone() / F::from_f32(4.0);

        for &idx in &self.segs_needing_positions {
            if self.line.segs[idx].in_changed_interval {
                continue;
            }
            // Ensure that every segment in the changed interval has `old_idx` set. This
            // isn't strictly necessary (because segments without `old_idx` set haven't
            // changed their indices), but it's more convenient to just say that all
            // segments in a changed interval must have `old_idx` set.
            self.line.segs[idx].set_old_idx_if_unset(idx);

            self.line.segs[idx].in_changed_interval = true;
            let seg_idx = self.line.segs[idx].seg;
            let mut start_idx = idx;
            let mut end_idx = idx + 1;
            let mut seg_min = segments[seg_idx].lower(y, eps);
            let mut seg_max = segments[seg_idx].upper(y, eps);

            for i in (idx + 1)..self.line.segs.len() {
                let next_seg_idx = self.line.seg(i);
                let next_seg = &segments[next_seg_idx];
                if next_seg.lower(y, eps) - &seg_max > slack {
                    break;
                } else {
                    seg_max = next_seg.upper(y, eps);
                    self.line.segs[i].in_changed_interval = true;
                    self.line.segs[i].set_old_idx_if_unset(i);

                    end_idx = i + 1;
                }
            }

            for i in (0..start_idx).rev() {
                let prev_seg_idx = self.line.seg(i);
                let prev_seg = &segments[prev_seg_idx];
                if seg_min.clone() - prev_seg.upper(y, eps) > slack {
                    break;
                } else {
                    seg_min = prev_seg.lower(y, eps);
                    self.line.segs[i].in_changed_interval = true;
                    self.line.segs[i].set_old_idx_if_unset(i);

                    start_idx = i;
                }
            }
            self.changed_intervals
                .push(ChangedInterval::from_seg_interval(start_idx..end_idx));
        }
        self.changed_intervals.sort_by_key(|r| r.segs.start);

        // By merging adjacent intervals, we ensure that there is no horizontal segment
        // that spans two ranges. That's because horizontal segments mark everything they
        // cross as needing a position. Any collection of subranges that are crossed by
        // a horizontal segment are therefore adjacent and will be merged here.
        merge_adjacent(&mut self.changed_intervals);
    }
}

/// Represents a sub-interval of the sweep-line where some subdivisions need to happen.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChangedInterval {
    // Indices into the sweep-line's segment array.
    pub(crate) segs: std::ops::Range<usize>,
    // Indices into the array of horizontal segments.
    pub(crate) horizontals: Option<std::ops::Range<usize>>,
}

impl ChangedInterval {
    fn merge(&mut self, other: &ChangedInterval) {
        self.segs.end = self.segs.end.max(other.segs.end);
        self.segs.start = self.segs.start.min(other.segs.start);

        self.horizontals = match (self.horizontals.clone(), other.horizontals.clone()) {
            (None, None) => None,
            (None, Some(r)) | (Some(r), None) => Some(r),
            (Some(a), Some(b)) => Some(a.start.min(b.start)..a.end.max(b.end)),
        };
    }

    fn from_seg_interval(segs: std::ops::Range<usize>) -> Self {
        Self {
            segs,
            horizontals: None,
        }
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

impl<F: Float> SegmentOrder<F> {
    /// If the ordering invariants fail, returns a pair of indices witnessing that failure.
    /// Used in tests, and when enabling slow-asserts
    #[allow(dead_code)]
    fn find_invalid_order(
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
    fn insertion_idx(&self, y: &F, segments: &Segments<F>, seg: &Segment<F>, eps: &F) -> usize {
        let seg_x = seg.start.x.clone();

        // Fast path: we first do a binary search just with the horizontal bounding intervals.
        // `pos` is an index where we're to the left of that segment's right-most point, and
        // we're to the right of the segment at `pos - 1`'s right-most point.
        // (It isn't necessarily the first such index because things aren't quite sorted; see
        // below about the binary search.)
        let pos = self
            .segs
            .partition_point(|entry| seg_x >= entry.upper_bound);

        if pos >= self.segs.len() {
            return pos;
        } else if pos + 1 >= self.segs.len() || self.segs[pos + 1].lower_bound >= seg_x {
            // At this point, the segment before pos is definitely to our left, and the segment
            // after pos is definitely to our right (or there is no such segment). That means
            // we can insert either before or after `pos`. We'll choose based on the position.
            // (We could also try checking whether the new segment is a contour neighbor of
            // the segment at pos. That should be a pretty common case.)
            let other = &segments[self.segs[pos].seg];
            if seg_x < other.at_y(y) {
                return pos;
            } else {
                return pos + 1;
            }
        }

        // Horizontal evaluation is accurate to within eps / 8, so if we want to compare
        // two different horizontal coordinates, we need a slack of eps / 4.
        let slack = eps.clone() / F::from_f32(4.0);

        // A predicate that tests `other.upper(y) <= seg(y)` with no false positives.
        // This is called `p` in the write-up.
        let lower_pred = |other: &SegmentOrderEntry<F>| -> bool {
            let other = &segments[other.seg];
            // This is `other.upper(y, eps) < seg_y - slack`, but rearranged for better accuracy.
            seg_x.clone() - other.upper(y, eps) > slack
        };

        // The rust stdlib docs say that we're not allowed to do this, because
        // our array isn't sorted with respect to `lower_pred`.
        // But for now at least, their implementation does a normal
        // binary search and so it's guaranteed to return an index where
        // `lower_pred` fails but the index before it succeeds.
        //
        // `search_start` is `i_- + 1` in the write-up; it's the first index
        // where the predicate returns false.
        let search_start = self.segs.partition_point(lower_pred);
        let mut idx = search_start;
        // A predicate that tests `other.lower(y) <= seg(y)`, with no false negatives (and
        // also some guarantees for the false positives). This is called `q` in the write-up.
        let upper_pred = |other: &SegmentOrderEntry<F>| -> bool {
            let other = &segments[other.seg];
            // This is `other.lower(y, eps) < seg(y) + slack`, but rearranged for accuracy
            other.lower(y, eps) - &seg_x < slack
        };
        for i in search_start..self.segs.len() {
            if upper_pred(&self.segs[i]) {
                idx = i + 1;
            } else {
                break;
            }
        }
        idx
    }

    // Find the position of the given segment in our array.
    fn position(&self, seg_idx: SegIdx, segments: &Segments<F>, y: &F, eps: &F) -> Option<usize> {
        if self.segs.len() <= 32 {
            return self.segs.iter().position(|x| x.seg == seg_idx);
        }

        let seg = &segments[seg_idx];
        let seg_lower = seg.lower(y, eps);
        let seg_upper = seg.upper(y, eps);

        // start_idx points to a segment whose upper bound is bigger than `seg`'s start
        // position (so it could potentially be `seg`). But the segment at `start_idx - 1`
        // has an upper bound less than `seg`'s start position, so `seg` cannot be at
        // `start_idx - 1` or anywhere before it.
        let mut start_idx = self
            .segs
            .partition_point(|entry| entry.upper_bound <= seg_lower);

        // end_idx points to something that's definitely after `seg`.
        let mut end_idx = self
            .segs
            .partition_point(|entry| entry.lower_bound <= seg_upper);

        if end_idx <= start_idx {
            return None;
        }

        // If the bounds are reasonable, we'll just do a linear search between them.
        // If they're too far apart, try to refine them first.
        if end_idx - start_idx > 32 {
            start_idx = self.segs.partition_point(|entry| {
                let other_seg = &segments[entry.seg];
                other_seg.upper(y, eps) <= seg_lower
            });

            end_idx = self.segs.partition_point(|entry| {
                let other_seg = &segments[entry.seg];
                other_seg.lower(y, eps) <= seg_upper
            });
        }

        if end_idx <= start_idx {
            return None;
        }
        self.segs
            .range(start_idx..)
            .position(|x| x.seg == seg_idx)
            .map(|i| i + start_idx)
    }
}
/// Computes the allowable x positions for a slice of segments.
fn horizontal_positions<'a, F: Float, G: Fn(&SegmentOrderEntry<F>) -> SegIdx>(
    entries: &[SegmentOrderEntry<F>],
    entry_seg: G,
    y: &F,
    segments: &'a Segments<F>,
    eps: &'a F,
    out: &mut Vec<(SegmentOrderEntry<F>, F, F)>,
) {
    out.clear();
    let mut max_so_far = entries
        .first()
        .map(|entry| segments[entry_seg(entry)].lower(y, eps))
        // If `self.segs` is empty our y doesn't matter; we're going to return
        // an empty vec.
        .unwrap_or(F::from_f32(0.0));

    for entry in entries {
        let x = segments[entry_seg(entry)].lower(y, eps);
        max_so_far = max_so_far.clone().max(x);
        // Fill out the minimum allowed positions, with a placeholder for the maximum.
        out.push((entry.clone(), max_so_far.clone(), F::from_f32(0.0)))
    }

    let mut min_so_far = entries
        .last()
        .map(|entry| segments[entry_seg(entry)].upper(y, eps))
        // If `self.segs` is empty our y doesn't matter; we're going to return
        // an empty vec.
        .unwrap_or(F::from_f32(0.0));

    for (entry, _, max_allowed) in out.iter_mut().rev() {
        let x = segments[entry_seg(entry)].upper(y, eps);
        min_so_far = min_so_far.clone().min(x);
        *max_allowed = min_so_far.clone();
    }
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
#[derive(Debug)]
pub struct SweepLine<'buf, 'state, 'segs, F: Float> {
    state: &'state Sweeper<'segs, F>,
    bufs: &'buf mut SweepLineBuffers<F>,
    // Index into state.changed_intervals
    next_changed_interval: usize,
}

impl<'segs, F: Float> SweepLine<'_, '_, 'segs, F> {
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
    pub fn cur_interval(&self) -> Option<&ChangedInterval> {
        self.next_changed_interval
            .checked_sub(1)
            .and_then(|idx| self.state.changed_intervals.get(idx))
    }

    fn next_range_single_seg<'a, 'bufs>(
        &'a mut self,
        bufs: &'bufs mut SweepLineRangeBuffers<F>,
        segments: &Segments<F>,
        idx: usize,
    ) -> Option<SweepLineRange<'bufs, 'a, 'segs, F>> {
        bufs.clear();
        self.bufs.output_events.clear();

        let entry = &self.state.line.segs[idx];
        let x = segments[entry.seg].at_y(self.y());
        if let Some(old_seg) = entry.old_seg {
            // This entry is on a contour, where one segment ends and the next begins.
            // We ouput two events (one per segment) at the same position.
            self.bufs.output_events.push(OutputEvent {
                x0: x.clone(),
                connected_above: true,
                x1: x.clone(),
                connected_below: false,
                seg_idx: old_seg,
                sweep_idx: None,
                old_sweep_idx: entry.old_idx,
            });

            self.bufs.output_events.push(OutputEvent {
                x0: x.clone(),
                connected_above: false,
                x1: x.clone(),
                connected_below: true,
                seg_idx: entry.seg,
                sweep_idx: Some(idx),
                old_sweep_idx: None,
            });
        } else {
            // It's a single segment either entering or exiting at this height.
            // We can handle them both in a single case.
            self.bufs.output_events.push(OutputEvent {
                x0: x.clone(),
                connected_above: !entry.enter,
                x1: x.clone(),
                connected_below: !entry.exit,
                seg_idx: entry.seg,
                sweep_idx: Some(idx),
                old_sweep_idx: entry.old_idx,
            });
        }

        let changed_interval = ChangedInterval {
            segs: idx..(idx + 1),
            horizontals: None,
        };

        Some(SweepLineRange::new(
            self,
            &self.bufs.output_events,
            bufs,
            changed_interval,
        ))
    }

    /// Returns a [`SweepLineRange`] for visiting and processing all positions within
    /// a range of segments.
    pub fn next_range<'a, 'bufs>(
        &'a mut self,
        bufs: &'bufs mut SweepLineRangeBuffers<F>,
        segments: &Segments<F>,
        eps: &F,
    ) -> Option<SweepLineRange<'bufs, 'a, 'segs, F>> {
        let range = self
            .state
            .changed_intervals
            .get(self.next_changed_interval)?
            .clone();
        self.next_changed_interval += 1;

        debug_assert!(!range.segs.is_empty());
        if range.segs.len() == 1 && range.horizontals.is_none() {
            return self.next_range_single_seg(bufs, segments, range.segs.start);
        }

        let dummy_entry = SegmentOrderEntry {
            seg: SegIdx(424242),
            exit: false,
            enter: false,
            lower_bound: F::from_f32(0.0),
            upper_bound: F::from_f32(0.0),
            in_changed_interval: false,
            old_idx: None,
            old_seg: None,
        };

        let buffers = &mut self.bufs;
        buffers
            .old_line
            .resize(range.segs.end - range.segs.start, dummy_entry.clone());
        for entry in self.state.line.segs.range(range.segs.clone()) {
            buffers.old_line[entry.old_idx.unwrap() - range.segs.start] = entry.clone();
        }

        horizontal_positions(
            &buffers.old_line,
            |entry| entry.old_seg(),
            &self.state.y,
            segments,
            eps,
            &mut buffers.positions,
        );

        // The two positioning arrays should have the same segments, but possibly in a different
        // order. We build them up in the old-sweep-line order.
        buffers.output_events.clear();
        let events = &mut buffers.output_events;
        // It would be natural to set max_so_far to -infinity, but a generic F: Float doesn't
        // have -infinity.
        let mut max_so_far = buffers
            .positions
            .first()
            .map_or(F::from_f32(0.0), |(_idx, min_x, _max_x)| {
                min_x.clone() - F::from_f32(1.0)
            });
        for (entry, min_x, max_x) in &buffers.positions {
            let preferred_x = if entry.exit {
                // The best possible position is the true segment-ending position.
                // (This could change if we want to be more sophisticated at joining contours.)
                segments[entry.old_seg()].end.x.clone()
            } else if entry.enter {
                // The best possible position is the true segment-starting position.
                // (This could change if we want to be more sophisticated at joining contours.)
                segments[entry.seg].start.x.clone()
            } else {
                segments[entry.seg].at_y(&self.state.y)
            };
            let x = preferred_x
                .max(min_x.clone())
                .max(max_so_far.clone())
                .min(max_x.clone());
            max_so_far = x.clone();
            events.push(OutputEvent {
                x0: x,
                connected_above: entry.old_seg.is_some() || !entry.enter,
                // This will be filled out when we traverse new_xs.
                x1: F::from_f32(42.42),
                connected_below: !entry.exit,
                seg_idx: entry.old_seg(),
                sweep_idx: None,
                old_sweep_idx: entry.old_idx,
            });
        }

        buffers.old_line.clear();
        buffers
            .old_line
            .extend(self.state.line.segs.range(range.segs.clone()).cloned());
        horizontal_positions(
            &buffers.old_line,
            |entry| entry.seg,
            &self.state.y,
            segments,
            eps,
            &mut buffers.positions,
        );
        let mut max_so_far = buffers
            .positions
            .first()
            .map_or(F::from_f32(0.0), |(_idx, min_x, _max_x)| {
                min_x.clone() - F::from_f32(1.0)
            });
        for (idx, (entry, min_x, max_x)) in buffers.positions.iter().enumerate() {
            let ev = &mut events[entry.old_idx.unwrap() - range.segs.start];
            ev.sweep_idx = Some(range.segs.start + idx);
            debug_assert_eq!(ev.seg_idx, entry.old_seg());
            let preferred_x = if *min_x <= ev.x0 && ev.x0 <= *max_x {
                // Try snapping to the previous position if possible.
                ev.x0.clone()
            } else {
                segments[entry.seg].at_y(&self.state.y)
            };
            ev.x1 = preferred_x
                .max(min_x.clone())
                .max(max_so_far.clone())
                .min(max_x.clone());
            max_so_far = ev.x1.clone();

            let x1 = ev.x1.clone();
            if entry.old_seg.is_some() {
                events.push(OutputEvent {
                    x0: x1.clone(),
                    x1,
                    connected_above: false,
                    // This will be filled out when we traverse new_xs.
                    connected_below: true,
                    seg_idx: entry.seg,
                    sweep_idx: Some(range.segs.start + idx),
                    old_sweep_idx: None,
                });
            }
        }

        // Modify the positions so that entering and exiting segments get their exact position.
        //
        // This is the easiest way to maintain the continuity of contours, but
        // eventually we should change this to minimize horizontal jank. But
        // first, we should add a test for continuity of contours (TODO).
        for ev in &mut *events {
            let seg = &segments[ev.seg_idx];
            if !ev.connected_above {
                ev.x0 = seg.start.x.clone();
            }
            if !ev.connected_below {
                ev.x1 = seg.end.x.clone();
            }
        }

        if let Some(range) = &range.horizontals {
            for &seg_idx in &self.state.horizontals[range.clone()] {
                let seg = &self.state.segments[seg_idx];
                events.push(OutputEvent {
                    x0: seg.start.x.clone(),
                    connected_above: false,
                    x1: seg.end.x.clone(),
                    connected_below: false,
                    seg_idx,
                    sweep_idx: None,
                    old_sweep_idx: None,
                });
            }
        }
        events.sort();
        bufs.clear();

        Some(SweepLineRange::new(
            self,
            &self.bufs.output_events,
            bufs,
            range,
        ))
    }
}

/// Given a list of intervals, sorted by starting point, merge the overlapping ones.
///
/// For example, [1..2, 4..5, 5..6] is turned into [1..2, 4..6].
fn merge_adjacent(intervals: &mut Vec<ChangedInterval>) {
    if intervals.is_empty() {
        return;
    }

    let mut write_idx = 0;
    for read_idx in 1..intervals.len() {
        let last_end = intervals[write_idx].segs.end;
        let cur = intervals[read_idx].clone();
        if last_end < cur.segs.start {
            write_idx += 1;
            intervals[write_idx] = cur;
        } else if last_end < cur.segs.end {
            intervals[write_idx].merge(&cur);
        }
    }
    intervals.truncate(write_idx + 1);
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
        fn merge(
            v: impl IntoIterator<Item = std::ops::Range<usize>>,
        ) -> Vec<std::ops::Range<usize>> {
            let mut v = v
                .into_iter()
                .map(ChangedInterval::from_seg_interval)
                .collect();
            super::merge_adjacent(&mut v);
            v.into_iter().map(|ci| ci.segs).collect()
        }

        assert_eq!(merge([1..2, 3..4]), vec![1..2, 3..4]);
        assert_eq!(merge([1..2, 2..4]), vec![1..4]);
        assert_eq!(merge([1..2, 3..4, 4..5]), vec![1..2, 3..5]);
        assert_eq!(merge([1..2, 1..3, 4..5]), vec![1..3, 4..5]);
        assert_eq!(merge([1..2, 1..4, 4..5]), vec![1..5]);
        assert_eq!(merge([1..4, 1..2, 4..5]), vec![1..5]);
        assert_eq!(merge([]), Vec::<std::ops::Range<_>>::new());
    }

    #[test]
    fn invalid_order() {
        fn check_order(xs: &[(f64, f64)], at: f64, eps: f64) -> Option<(usize, usize)> {
            let eps: NotNan<f64> = eps.try_into().unwrap();
            let y: NotNan<f64> = at.try_into().unwrap();
            let segs = mk_segs(xs);

            let line: SegmentOrder<NotNan<f64>> = SegmentOrder {
                segs: (0..xs.len())
                    .map(|i| SegmentOrderEntry::new(SegIdx(i), &segs, &eps))
                    .collect(),
            };

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
            let mut xs: Vec<_> = xs.to_owned();
            xs.push(new);
            let segs = mk_segs(&xs);

            let x0: NotNan<f64> = new.0.try_into().unwrap();
            let x1: NotNan<f64> = new.1.try_into().unwrap();
            let new = Segment::new(Point::new(x0, y0), Point::new(x1, y1));

            let mut line: SegmentOrder<NotNan<f64>> = SegmentOrder {
                segs: (0..(xs.len() - 1))
                    .map(|i| SegmentOrderEntry::new(SegIdx(i), &segs, &eps))
                    .collect(),
            };
            let idx = line.insertion_idx(&y, &segs, &new, &eps);

            assert!(line.find_invalid_order(&y, &segs, &eps).is_none());
            line.segs.insert(
                idx,
                SegmentOrderEntry::new(SegIdx(xs.len() - 1), &segs, &eps),
            );
            assert!(line.find_invalid_order(&y, &segs, &eps).is_none());
            idx
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

        insert(&[(2.0, 2.0), (-100.0, 100.0)], (1.0, 1.0), 0.5, 0.25);
    }

    #[test]
    fn test_sweep() {
        let eps = NotNan::new(0.01).unwrap();

        let segs = mk_segs(&[(0.0, 0.0), (1.0, 1.0), (-2.0, 2.0)]);
        dbg!(&segs);
        crate::sweep::sweep(&segs, &eps, |_, ev| {
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
        crate::sweep::sweep(&segs, &eps, |_, _| {});
    }

    #[derive(serde::Serialize, Debug)]
    struct Output {
        order: SegmentOrder<NotNan<f64>>,
        changed: Vec<ChangedInterval>,
    }
    fn snapshot_outputs(segs: Segments<NotNan<f64>>, eps: f64) -> Vec<Output> {
        let mut outputs = Vec::new();
        let mut sweeper = Sweeper::new(&segs, eps.try_into().unwrap());
        let mut line_bufs = SweepLineBuffers::default();
        while let Some(line) = sweeper.next_line(&mut line_bufs) {
            outputs.push(Output {
                order: line.state.line.clone(),
                changed: line.state.changed_intervals.clone(),
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

    #[test]
    fn two_squares() {
        let a = vec![p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0)];
        let b = vec![p(-0.5, -0.5), p(0.5, -0.5), p(0.5, 0.5), p(-0.5, 0.5)];
        let mut segs = Segments::<NotNan<f64>>::default();
        segs.add_cycle(a);
        segs.add_cycle(b);
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
