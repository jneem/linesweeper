//! A sweep-line implementation using weak orderings.
//!
//! This algorithm is documented in `docs/sweep.typ`.

use std::collections::BTreeSet;

use malachite::Rational;

use crate::{
    geom::Segment,
    num::Float,
    segments::{SegIdx, Segments},
};

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct SegmentOrderEntry {
    seg: SegIdx,
    exit: bool,
    enter: bool,
    // This is filled out during `compute_changed_intervals`, where we use it to detect
    // if this segment was already marked as needing a position because it was near
    // some other segment that needs a position.
    in_changed_interval: bool,
    old_idx: Option<usize>,
}

impl SegmentOrderEntry {
    fn reset_state(&mut self) {
        self.exit = false;
        self.enter = false;
        self.in_changed_interval = false;
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
            in_changed_interval: false,
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

#[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub struct IntersectionEvent<F: Float> {
    pub y: F,
    /// This segment used to be to the left, and after the intersection it will be to the right.
    ///
    /// In our sweep line intersection, this segment might have already been moved to the right by
    /// some other constraints. That's ok.
    pub left: SegIdx,
    /// This segment used to be to the right, and after the intersection it will be to the left.
    pub right: SegIdx,
}

// This is probably too clever, and we should just use ranges.
#[derive(Debug)]
pub struct SliceGuard<'a, T> {
    vec: &'a mut Vec<T>,
    start: usize,
}

impl<T> std::ops::Deref for SliceGuard<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.vec[self.start..]
    }
}

impl<T> Drop for SliceGuard<'_, T> {
    fn drop(&mut self) {
        self.vec.truncate(self.start)
    }
}

#[derive(Clone, Debug)]
pub struct EventQueue<F: Float> {
    /// The enter events are stored in `Segments<F>`; this is the index of the first
    /// one that we haven't processed yet.
    next_enter_idx: usize,
    /// The index of the first exit event that we haven't processed yet.
    next_exit_idx: usize,
    intersection: std::collections::BTreeSet<IntersectionEvent<F>>,
}

impl<F: Float> EventQueue<F> {
    /// Builds an event queue containing the starting and ending positions
    /// of all the  segments.
    ///
    /// The returned event queue will not contain any intersection events.
    pub fn from_segments(segments: &Segments<F>) -> Self {
        let mut enter = Vec::with_capacity(segments.len());
        let mut exit = Vec::with_capacity(segments.len());
        let intersection = BTreeSet::new();

        for seg in segments.indices() {
            enter.push((segments[seg].start.y.clone(), seg));
            if !segments[seg].is_horizontal() {
                exit.push((segments[seg].end.y.clone(), seg));
            }
        }

        // We sort the enter segments by reversed y position (so that we can consume them
        // in chunks from the end of the vector), and then by horizontal start position so
        // that they're fairly likely to get inserted in the sweep-line in order (which makes
        // the indexing fix-ups faster).
        enter.sort_by(|(y1, seg1), (y2, seg2)| {
            y1.cmp(y2)
                .reverse()
                .then_with(|| segments[*seg1].at_y(y1).cmp(&segments[*seg2].at_y(y1)))
        });
        exit.sort_by(|x, y| x.cmp(y).reverse());

        Self {
            next_enter_idx: 0,
            next_exit_idx: 0,
            intersection,
        }
    }

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

// Holds some buffers that are used when iterating over a sweep-line.
#[derive(Clone, Debug)]
struct SweepLineBuffers<F: Float> {
    /// A subset of the old sweep-line.
    old_line: Vec<SegmentOrderEntry>,
    /// A vector of (segment, min allowable horizontal position, max allowable horizontal position).
    positions: Vec<(SegmentOrderEntry, F, F)>,
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

/// Holds some buffers that are used when iterating over a sweep-line.
///
/// Save on re-allocation by allocating this once and reusing it in multiple calls to
/// [`SweepLine::next_range`].
#[derive(Clone, Debug)]
pub struct SweepLineRangeBuffers<F: Float> {
    active_horizontals: Vec<HSeg<F>>,
}

impl<F: Float> Default for SweepLineRangeBuffers<F> {
    fn default() -> Self {
        SweepLineRangeBuffers {
            active_horizontals: Vec::new(),
        }
    }
}

/// Encapsulates the state of the sweep-line algorithm and allows iterating over sweep lines.
#[derive(Clone, Debug)]
pub struct Sweeper<'a, F: Float> {
    y: F,
    eps: F,
    line: SegmentOrder,
    events: EventQueue<F>,
    segments: &'a Segments<F>,

    // These buffers are only used in `events_in_range`. Keeping them here allows us to avoid
    // allocating and re-allocating them.
    buffers: SweepLineBuffers<F>,

    horizontals: Vec<SegIdx>,

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
    segs_needing_positions: Vec<usize>,
    changed_intervals: Vec<ChangedInterval>,
}

impl<'segs, F: Float> Sweeper<'segs, F> {
    /// Creates a new sweeper for a collection of segments, and with a given tolerance.
    pub fn new(segments: &'segs Segments<F>, eps: F) -> Self {
        let events = EventQueue::from_segments(segments);

        Sweeper {
            eps,
            line: SegmentOrder::default(),
            y: events.next_y(segments).unwrap().clone(),
            events,
            segments,
            segs_needing_positions: Vec::new(),
            changed_intervals: Vec::new(),
            horizontals: Vec::new(),
            buffers: SweepLineBuffers::default(),
        }
    }

    /// Moves the sweep forward, returning the next sweep line.
    ///
    /// Returns `None` when sweeping is complete.
    pub fn next_line<'slf>(&'slf mut self) -> Option<SweepLine<'slf, 'segs, F>> {
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
        })
    }

    fn advance(&mut self, y: F) {
        // All the exiting segments should be in segs_needing_positions, so find them all and remove them.
        self.segs_needing_positions
            .retain(|idx| self.line.segs[*idx].exit);

        // Reset the state flags for all segments. All segments with non-trivial state flags should
        // belong to the changed intervals. This needs to go before we remove the exiting segments,
        // because that messes up the indices.
        for r in self.changed_intervals.drain(..) {
            for seg in &mut self.line.segs[r.segs] {
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

    fn insert(&mut self, pos: usize, seg: SegmentOrderEntry) {
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
        let mut entry = SegmentOrderEntry::from(seg_idx);
        entry.enter = true;
        entry.exit = false;
        self.insert(pos, entry);

        // Fix up the index of any other segments that we got inserted before
        // (at this point, segs_needing_positions only contains newly-inserted
        // segments, and it's sorted increasing).
        //
        // We sorted all the to-be-inserted segments by horizontal position
        // before inserting them, so we expect these two loops to be short most
        // of the time.
        for other_pos in self.segs_needing_positions.iter_mut().rev() {
            if *other_pos >= pos {
                *other_pos += 1;
            } else {
                break;
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

    // TODO: explain somewhere why this doesn't actually remove the segment
    // yet
    fn handle_exit(&mut self, seg_idx: SegIdx) {
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

    fn handle_intersection(&mut self, left: SegIdx, right: SegIdx) {
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

    #[cfg(feature = "slow-asserts")]
    fn check_invariants(&self) {
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
                        let has_exit_witness = self.line.segs[i..=j].iter().any(|seg_entry| {
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
        let seg_y = seg.at_y(y);
        // Horizontal evaluation is accurate to within eps / 8, so if we want to compare
        // two different horizontal coordinates, we need a slack of eps / 4.
        let slack = eps.clone() / F::from_f32(4.0);

        // A predicate that tests `other.upper(y) <= seg(y)` with no false positives.
        // This is called `p` in the write-up.
        let lower_pred = |other: &SegmentOrderEntry| -> bool {
            let other = &segments[other.seg];
            // This is `other.upper(y, eps) < seg_y - slack`, but rearranged for better accuracy.
            seg_y.clone() - other.upper(y, eps) > slack
        };

        // The rust stdlib docs say that we're not allowed to do this, because
        // our array isn't sorted with respect to `maybe_strictly_smaller`.
        // But for now at least, their implementation does a normal
        // binary search and so it's guaranteed to return an index where
        // `maybe_strictly_smaller` fails but the index before it succeeds.
        //
        // `search_start` is `i_- + 1` in the write-up; it's the first index
        // where the predicate returns false.
        let search_start = self.segs.partition_point(lower_pred);
        let mut idx = search_start;
        // A predicate that tests `other.lower(y) <= seg(y)`, with no false negatives (and
        // also some guarantees for the false positives). This is called `q` in the write-up.
        let upper_pred = |other: &SegmentOrderEntry| -> bool {
            let other = &segments[other.seg];
            // This is `other.lower(y, eps) < seg(y) + slack`, but rearranged for accuracy
            other.lower(y, eps) - &seg_y < slack
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
    //
    // TODO: if we're large, we could use a binary search.
    fn position(&self, seg: SegIdx) -> Option<usize> {
        self.segs.iter().position(|&x| x.seg == seg)
    }
}
/// Computes the allowable x positions for a slice of segments.
fn horizontal_positions<'a, F: Float>(
    entries: &[SegmentOrderEntry],
    y: &F,
    segments: &'a Segments<F>,
    eps: &'a F,
    out: &mut Vec<(SegmentOrderEntry, F, F)>,
) {
    out.clear();
    let mut max_so_far = entries
        .first()
        .map(|seg| segments[seg.seg].lower(y, eps))
        // If `self.segs` is empty our y doesn't matter; we're going to return
        // an empty vec.
        .unwrap_or(F::from_f32(0.0));

    for seg in entries {
        let x = segments[seg.seg].lower(y, eps);
        max_so_far = max_so_far.clone().max(x);
        // Fill out the minimum allowed positions, with a placeholder for the maximum.
        out.push((*seg, max_so_far.clone(), F::from_f32(0.0)))
    }

    let mut min_so_far = entries
        .last()
        .map(|seg| segments[seg.seg].upper(y, eps))
        // If `self.segs` is empty our y doesn't matter; we're going to return
        // an empty vec.
        .unwrap_or(F::from_f32(0.0));

    for (entry, _, max_allowed) in out.iter_mut().rev() {
        let x = segments[entry.seg].upper(y, eps);
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
///
/// TODO: the public API is pretty gross right now and requires lots of allocation.
/// It will change.
#[derive(Debug)]
pub struct SweepLine<'state, 'segs, F: Float> {
    state: &'state mut Sweeper<'segs, F>,
    // Index into state.changed_intervals
    next_changed_interval: usize,
}

impl<'segs, F: Float> SweepLine<'_, 'segs, F> {
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

        let buffers = &mut self.state.buffers;
        buffers
            .old_line
            .resize(range.segs.end - range.segs.start, SegIdx(424242).into());
        for entry in &self.state.line.segs[range.segs.clone()] {
            buffers.old_line[entry.old_idx.unwrap() - range.segs.start] = *entry;
        }

        horizontal_positions(
            &buffers.old_line,
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
                segments[entry.seg].end.x.clone()
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
                connected_above: !entry.enter,
                // This will be filled out when we traverse new_xs.
                x1: F::from_f32(42.42),
                connected_below: !entry.exit,
                seg_idx: entry.seg,
                sweep_idx: None,
                old_sweep_idx: entry.old_idx,
            });
        }

        horizontal_positions(
            &self.state.line.segs[range.segs.clone()],
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
            debug_assert_eq!(ev.seg_idx, entry.seg);
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
        bufs.active_horizontals.clear();

        Some(SweepLineRange {
            output_event_idx: 0,
            last_x: None,
            changed_interval: range,
            line: self,
            bufs,
        })
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

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct HSeg<F: Float> {
    pub end: F,
    pub connected_at_start: bool,
    pub connected_at_end: bool,
    pub enter_first: bool,
    pub seg: SegIdx,
    pub start: F,
    pub sweep_idx: Option<usize>,
    pub old_sweep_idx: Option<usize>,
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
            sweep_idx: pos.sweep_idx,
            old_sweep_idx: pos.old_sweep_idx,
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
    /// The segment's index in the new sweep line (`None` for horizontal segments).
    pub sweep_idx: Option<usize>,
    /// The segment's index in the old sweep line (`None` for horizontal segments).
    pub old_sweep_idx: Option<usize>,
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

    fn new(
        seg_idx: SegIdx,
        x0: F,
        connected_above: bool,
        x1: F,
        connected_below: bool,
        sweep_idx: Option<usize>,
        old_sweep_idx: Option<usize>,
    ) -> Self {
        Self {
            x0,
            connected_above,
            x1,
            connected_below,
            seg_idx,
            sweep_idx,
            old_sweep_idx,
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
/// This is constructed using [`SweepLine::next_range`]. By repeatedly
/// calling `SweepLineRange::increase_x` you can iterate over all
/// interesting horizontal positions, left to right (i.e. smaller `x` to larger
/// `x`).
#[derive(Debug)]
pub struct SweepLineRange<'bufs, 'state, 'segs, F: Float> {
    last_x: Option<F>,
    line: &'state SweepLine<'state, 'segs, F>,
    bufs: &'bufs mut SweepLineRangeBuffers<F>,
    changed_interval: ChangedInterval,
    output_event_idx: usize,
}

impl<'segs, F: Float> SweepLineRange<'_, '_, 'segs, F> {
    fn output_events(&self) -> &[OutputEvent<F>] {
        &self.line.state.buffers.output_events[self.output_event_idx..]
    }

    /// The current horizontal position, or `None` if we're finished.
    pub fn x(&self) -> Option<&F> {
        match (
            self.bufs.active_horizontals.first(),
            self.output_events().first(),
        ) {
            (None, None) => None,
            (None, Some(pos)) => Some(pos.smaller_x()),
            (Some(h), None) => Some(&h.end),
            (Some(h), Some(pos)) => Some((&h.end).min(pos.smaller_x())),
        }
    }

    fn positions_at_x<'c, 'b: 'c>(
        &'b self,
        x: &'c F,
    ) -> impl Iterator<Item = &'b OutputEvent<F>> + 'c {
        self.output_events()
            .iter()
            .take_while(move |p| p.smaller_x() == x)
    }

    /// Updates a [`SegmentsConnectedAtX`] to reflect the current horizontal position.
    pub fn update_segments_at_x(&self, segs: &mut SegmentsConnectedAtX) {
        segs.connected_up.clear();
        segs.connected_down.clear();

        let Some(x) = self.x() else {
            return;
        };

        for hseg in &self.bufs.active_horizontals {
            if hseg.connected_above_at(x) {
                segs.connected_up
                    .push((hseg.seg, hseg.old_sweep_idx.unwrap()));
            }

            if hseg.connected_below_at(x) {
                segs.connected_down
                    .push((hseg.seg, hseg.sweep_idx.unwrap()));
            }
        }

        for pos in self.positions_at_x(x) {
            if pos.connected_above_at(x) {
                segs.connected_up
                    .push((pos.seg_idx, pos.old_sweep_idx.unwrap()));
            }

            if pos.connected_below_at(x) {
                segs.connected_down
                    .push((pos.seg_idx, pos.sweep_idx.unwrap()));
            }
        }

        segs.connected_up
            .sort_by_key(|(_seg_idx, sweep_idx)| *sweep_idx);
        segs.connected_down
            .sort_by_key(|(_seg_idx, sweep_idx)| *sweep_idx);
    }

    /// Iterates over the horizontal segments that are active at the current position.
    ///
    /// This includes the segments that end here, but does not include the ones
    /// that start here.
    pub fn active_horizontals(&self) -> impl Iterator<Item = SegIdx> + '_ {
        self.bufs.active_horizontals.iter().map(|hseg| hseg.seg)
    }

    /// Returns the collection of all output events that end at the current
    /// position, or `None` if this batcher is finished.
    ///
    /// If this returns `None`, this batcher is finished.
    ///
    /// All the returned events start at the previous `x` position and end
    /// at the current `x` position. In particular, if you alternate between
    /// calling [`SweepLineRange::increase_x`] and this method, you'll
    /// receive non-overlapping batches of output events.
    pub fn events(&mut self) -> Option<Vec<OutputEvent<F>>> {
        let next_x = self.x()?.clone();

        let mut ret = Vec::new();
        for h in &self.bufs.active_horizontals {
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
                    h.sweep_idx,
                    h.old_sweep_idx,
                ));
            } else {
                ret.push(OutputEvent::new(
                    h.seg,
                    x1,
                    connected_end,
                    x0,
                    connected_start,
                    h.sweep_idx,
                    h.old_sweep_idx,
                ));
            }
        }

        // Drain the active horizontals, throwing away horizontal segments that end before
        // the new x position.
        self.drain_active_horizontals(&next_x);

        // Move along to the next horizontal position, processing the x events at the current
        // position and either emitting them immediately or saving them as horizontals.
        while let Some(ev) = self
            .line
            .state
            .buffers
            .output_events
            .get(self.output_event_idx)
        {
            if ev.smaller_x() > &next_x {
                break;
            }
            self.output_event_idx += 1;

            if ev.x0 == ev.x1 {
                // We push output event for points immediately.
                ret.push(ev.clone());
            } else if let Some(hseg) = HSeg::from_position(ev.clone()) {
                // For horizontal segments, we don't output anything straight
                // away. When we update the horizontal position and visit our
                // horizontal segments, we'll output something.
                self.bufs.active_horizontals.push(hseg);
            }
        }
        self.bufs.active_horizontals.sort();
        self.last_x = Some(next_x);
        Some(ret)
    }

    /// Move along to the next horizontal position.
    pub fn increase_x(&mut self) {
        if let Some(x) = self.x().cloned() {
            self.drain_active_horizontals(&x);

            while let Some(ev) = self
                .line
                .state
                .buffers
                .output_events
                .get(self.output_event_idx)
            {
                if ev.smaller_x() > &x {
                    break;
                }
                self.output_event_idx += 1;

                if let Some(hseg) = HSeg::from_position(ev.clone()) {
                    self.bufs.active_horizontals.push(hseg);
                }
            }
        }
        self.bufs.active_horizontals.sort();
    }

    fn drain_active_horizontals(&mut self, x: &F) {
        let new_start = self
            .bufs
            .active_horizontals
            .iter()
            .position(|h| h.end > *x)
            .unwrap_or(self.bufs.active_horizontals.len());
        self.bufs.active_horizontals.drain(..new_start);
    }

    /// The indices within the sweep line represented by this range.
    pub fn seg_range(&self) -> ChangedInterval {
        self.changed_interval.clone()
    }

    /// The sweep line that this is a range of.
    pub fn line(&self) -> &SweepLine<'_, 'segs, F> {
        self.line
    }
}

/// Runs the sweep-line algorithm, calling the provided callback on every output point.
pub fn sweep<F: Float, C: FnMut(F, OutputEvent<F>)>(
    segments: &Segments<F>,
    eps: &F,
    mut callback: C,
) {
    let mut state = Sweeper::new(segments, eps.clone());
    let mut range_bufs = SweepLineRangeBuffers::default();
    while let Some(mut line) = state.next_line() {
        let y = line.state.y.clone();
        while let Some(mut range) = line.next_range(&mut range_bufs, segments, eps) {
            while let Some(events) = range.events() {
                for ev in events {
                    callback(y.clone(), ev);
                }
            }
        }
    }
}

/// A re-usable struct for collecting segments at a single position on a sweep-line.
///
/// See [`SweepLineRange::update_segments_at_x`] for where this is used. At
/// any given time, this struct is implicitly associated to a single horizontal
/// position: the position of the `SweepLineRange` last time we were updated.
#[derive(Debug, Default)]
pub struct SegmentsConnectedAtX {
    connected_up: Vec<(SegIdx, usize)>,
    connected_down: Vec<(SegIdx, usize)>,
}

impl SegmentsConnectedAtX {
    /// The segments that are connected up to a previous sweep-line at the
    /// current horizontal position.
    ///
    /// The returned iterator is sorted by the old sweep-line order. In other
    /// words, it will return segments clockwise when viewed from the current
    /// position.
    pub fn connected_up(&self) -> impl Iterator<Item = SegIdx> + '_ {
        self.connected_up.iter().map(|x| x.0)
    }

    /// The segments that are connected down to a subsequent sweep-line at the
    /// current horizontal position.
    ///
    /// The returned iterator is sorted by the new sweep-line order. In other
    /// words, it will return segments counter-clockwise when viewed from the current
    /// position.
    pub fn connected_down(&self) -> impl Iterator<Item = SegIdx> + '_ {
        self.connected_down.iter().map(|x| x.0)
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
            let mut xs: Vec<_> = xs.to_owned();
            xs.push(new);
            let segs = mk_segs(&xs);

            let x0: NotNan<f64> = new.0.try_into().unwrap();
            let x1: NotNan<f64> = new.1.try_into().unwrap();
            let new = Segment::new(Point::new(x0, y0), Point::new(x1, y1));

            let mut line: SegmentOrder = (0..(xs.len() - 1)).map(SegIdx).collect();
            let idx = line.insertion_idx(&y, &segs, &new, &eps);

            assert!(line.find_invalid_order(&y, &segs, &eps).is_none());
            line.segs.insert(idx, SegIdx(xs.len() - 1).into());
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

    #[derive(serde::Serialize, Debug)]
    struct Output {
        order: SegmentOrder,
        changed: Vec<ChangedInterval>,
    }
    fn snapshot_outputs(segs: Segments<NotNan<f64>>, eps: f64) -> Vec<Output> {
        let mut outputs = Vec::new();
        let mut sweeper = Sweeper::new(&segs, eps.try_into().unwrap());
        while let Some(line) = sweeper.next_line() {
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
