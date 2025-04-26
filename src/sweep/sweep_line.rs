//! A sweep-line implementation using weak orderings.
//!
//! This algorithm is documented in `docs/sweep.typ`.

use std::{cell::RefCell, collections::HashMap, ops::DerefMut};

use kurbo::{ParamCurve as _, ParamCurveNearest};

use crate::{
    curve::{self, CurveOrder, Order},
    geom::Segment,
    num::CheapOrderedFloat,
    segments::{SegIdx, Segments},
    treevec::TreeVec,
};

use super::{OutputEvent, SweepLineRange, SweepLineRangeBuffers};

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct SegmentOrderEntry {
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
    lower_bound: f64,
    /// This is epsilon above this segment's largest horizontal position. All
    /// horizontal positions larger than this are guaranteed not to interact
    /// with this segment.
    upper_bound: f64,
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

impl SegmentOrderEntry {
    fn new(seg: SegIdx, segments: &Segments, eps: f64) -> Self {
        let x0 = segments[seg].min_x();
        let x1 = segments[seg].max_x();
        Self {
            seg,
            exit: false,
            enter: false,
            lower_bound: x0.min(x1) - eps,
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
pub(crate) struct SegmentOrder {
    pub(crate) segs: TreeVec<SegmentOrderEntry, 128>,
}

impl Default for SegmentOrder {
    fn default() -> Self {
        Self {
            segs: TreeVec::new(),
        }
    }
}

impl SegmentOrder {
    fn seg(&self, i: usize) -> SegIdx {
        self.segs[i].seg
    }

    fn is_exit(&self, i: usize) -> bool {
        let seg = &self.segs[i];
        seg.exit && seg.old_seg.is_none()
    }
}

#[derive(Clone, Debug, PartialEq)]
struct IntersectionEvent {
    pub y: f64,
    /// This segment used to be to the left, and after the intersection it will be to the right.
    ///
    /// In our sweep line intersection, this segment might have already been moved to the right by
    /// some other constraints. That's ok.
    pub left: SegIdx,
    /// This segment used to be to the right, and after the intersection it will be to the left.
    pub right: SegIdx,
}

impl Eq for IntersectionEvent {}

impl Ord for IntersectionEvent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (CheapOrderedFloat::from(self.y), self.left, self.right).cmp(&(
            CheapOrderedFloat::from(other.y),
            other.left,
            other.right,
        ))
    }
}

impl PartialOrd for IntersectionEvent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug, Default)]
struct EventQueue {
    /// The enter events are stored in `Segments`; this is the index of the first
    /// one that we haven't processed yet.
    next_enter_idx: usize,
    /// The index of the first exit event that we haven't processed yet.
    next_exit_idx: usize,
    intersection: std::collections::BTreeSet<IntersectionEvent>,
}

impl EventQueue {
    pub fn push(&mut self, ev: IntersectionEvent) {
        self.intersection.insert(ev);
    }

    pub fn next_y<'a>(&'a self, segments: &'a Segments) -> Option<f64> {
        let enter_y = segments
            .entrances()
            .get(self.next_enter_idx)
            .map(|(y, _)| *y);
        let exit_y = segments.exits().get(self.next_exit_idx).map(|(y, _)| *y);
        let int_y = self.intersection.first().map(|i| i.y);

        [enter_y, exit_y, int_y]
            .into_iter()
            .flatten()
            .min_by_key(|y| CheapOrderedFloat::from(*y))
    }

    pub fn entrances_at_y<'a>(&mut self, y: &f64, segments: &'a Segments) -> &'a [(f64, SegIdx)] {
        let entrances = &segments.entrances()[self.next_enter_idx..];
        let count = entrances
            .iter()
            .position(|(enter_y, _)| enter_y > y)
            .unwrap_or(entrances.len());
        self.next_enter_idx += count;
        &entrances[..count]
    }

    pub fn exits_at_y<'a>(&mut self, y: &f64, segments: &'a Segments) -> &'a [(f64, SegIdx)] {
        let exits = &segments.exits()[self.next_exit_idx..];
        let count = exits
            .iter()
            .position(|(exit_y, _)| exit_y > y)
            .unwrap_or(exits.len());
        self.next_exit_idx += count;
        &exits[..count]
    }

    pub fn next_intersection_at_y(&mut self, y: f64) -> Option<IntersectionEvent> {
        if self.intersection.first().map(|i| i.y) == Some(y) {
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
#[derive(Clone, Debug, Default)]
pub struct SweepLineBuffers {
    /// A subset of the old sweep-line.
    old_line: Vec<SegmentOrderEntry>,
    /// A vector of (segment, min allowable horizontal position, max allowable horizontal position).
    positions: Vec<(SegmentOrderEntry, f64, f64)>,
    output_events: Vec<OutputEvent>,
}

#[derive(Clone, Debug)]
pub struct ComparisonCache {
    inner: HashMap<(SegIdx, SegIdx), CurveOrder>,
    accuracy: f64,
    tolerance: f64,
}

impl ComparisonCache {
    pub fn new(tolerance: f64, accuracy: f64) -> Self {
        ComparisonCache {
            inner: HashMap::new(),
            accuracy,
            tolerance,
        }
    }

    // FIXME: less cloning
    pub fn compare_segments(&mut self, segments: &Segments, i: SegIdx, j: SegIdx) -> CurveOrder {
        if let Some(order) = self.inner.get(&(i, j)) {
            return order.clone();
        }

        let segi = &segments[i];
        let segj = &segments[j];

        let forward = curve::intersect_cubics(
            segi.to_kurbo(),
            segj.to_kurbo(),
            self.tolerance,
            self.accuracy,
        )
        .with_y_slop(self.tolerance);
        let reverse = forward.flip();
        self.inner.insert((j, i), reverse);
        self.inner.entry((i, j)).insert_entry(forward).get().clone()
    }
}

/// Encapsulates the state of the sweep-line algorithm and allows iterating over sweep lines.
#[derive(Clone, Debug)]
pub struct Sweeper<'a> {
    y: f64,
    eps: f64,
    line: SegmentOrder,
    events: EventQueue,
    segments: &'a Segments,

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

    comparisons: RefCell<ComparisonCache>,
    conservative_comparisons: RefCell<ComparisonCache>,
}

impl<'segs> Sweeper<'segs> {
    /// Creates a new sweeper for a collection of segments, and with a given tolerance.
    pub fn new(segments: &'segs Segments, eps: f64) -> Self {
        let events = EventQueue::default();

        Sweeper {
            eps,
            line: SegmentOrder::default(),
            y: events.next_y(segments).unwrap(),
            events,
            segments,
            segs_needing_positions: Vec::new(),
            changed_intervals: Vec::new(),
            horizontals: Vec::new(),
            comparisons: RefCell::new(ComparisonCache::new(eps, eps / 2.0)),
            conservative_comparisons: RefCell::new(ComparisonCache::new(4.0 * eps, eps / 2.0)),
        }
    }

    fn compare_segments(&self, i: SegIdx, j: SegIdx) -> CurveOrder {
        self.comparisons
            .borrow_mut()
            .compare_segments(self.segments, i, j)
    }

    fn compare_segments_conservatively(&self, i: SegIdx, j: SegIdx) -> CurveOrder {
        self.conservative_comparisons
            .borrow_mut()
            .compare_segments(self.segments, i, j)
    }

    /// Moves the sweep forward, returning the next sweep line.
    ///
    /// Returns `None` when sweeping is complete.
    pub fn next_line<'slf, 'buf>(
        &'slf mut self,
        bufs: &'buf mut SweepLineBuffers,
    ) -> Option<SweepLine<'buf, 'slf, 'segs>> {
        self.check_invariants();

        let y = self.events.next_y(self.segments)?;
        self.advance(y);
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
        while let Some(intersection) = self.events.next_intersection_at_y(y) {
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

    fn advance(&mut self, y: f64) {
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
        let seg_idx = self.line.seg(start_idx);
        let y = self.y;

        // We're allowed to take a potentially-smaller height bound by taking
        // into account the current queue. A larger height bound is still ok,
        // just a little slower.
        // let mut height_bound = seg.end.y;

        // TODO: reinstate early-exit
        for j in (start_idx + 1)..self.line.segs.len() {
            if self.line.is_exit(j) {
                continue;
            }
            let other_idx = self.line.seg(j);
            //let other = &self.segments[other_idx];
            // if seg.quick_left_of(other, two_eps) {
            //     break;
            // }
            // height_bound = height_bound.min(other.end.y);

            // TODO: there's a choice to be made here: do we distinguish
            // in the event queue between actual intersections and near-intersections
            // that need to be recorded? For now, no. We will distinguish them
            // in handle_intersection

            let cmp = self.compare_segments(seg_idx, other_idx);

            if let Some(touch) = cmp.next_touch_after(y) {
                let int = match touch {
                    curve::NextTouch::Cross(cross_y) => Some((y.max(cross_y), start_idx, j)),
                    curve::NextTouch::Touch(touch_y) => {
                        (touch_y > y).then_some((touch_y, j, start_idx))
                    }
                };
                // TODO: the ordering of left/right is a little confusing here, because IntersectionEvent
                // was originally designed under the assumption that every intersection is a crossing.
                // Maybe the naming would be better if `IntersectionEvent::left` was the segment on the
                // left after intersecting?
                if let Some((int_y, left_idx, right_idx)) = int {
                    self.events.push(IntersectionEvent {
                        y: int_y,
                        left: self.line.seg(left_idx),
                        right: self.line.seg(right_idx),
                    });
                }
                //height_bound = int_y.min(height_bound);
            }

            // For the early stopping, we need to check whether `seg` is less than `other`'s lower
            // bound on the whole interesting `y` interval. Since they're lines, it's enough to check
            // at the two interval endpoints.
            // let y1 = height_bound;
            // let threshold = self.eps / 4.0;
            // let scaled_eps = other.scaled_eps(self.eps);
            // if threshold <= other.lower_with_scaled_eps(y, self.eps, scaled_eps) - seg.at_y(y)
            //     && threshold <= other.lower_with_scaled_eps(y1, self.eps, scaled_eps) - seg.at_y(y1)
            // {
            //     break;
            // }
        }
    }

    fn intersection_scan_left(&mut self, start_idx: usize) {
        let seg_idx = self.line.seg(start_idx);
        let y = self.y;

        for j in (0..start_idx).rev() {
            if self.line.is_exit(j) {
                continue;
            }
            let other_idx = self.line.seg(j);

            let cmp = self.compare_segments(other_idx, seg_idx);
            if let Some(touch) = cmp.next_touch_after(y) {
                let int = match touch {
                    curve::NextTouch::Cross(cross_y) => Some((y.max(cross_y), j, start_idx)),
                    curve::NextTouch::Touch(touch_y) => {
                        (touch_y > y).then_some((touch_y, start_idx, j))
                    }
                };
                if let Some((int_y, left_idx, right_idx)) = int {
                    self.events.push(IntersectionEvent {
                        y: int_y,
                        left: self.line.seg(left_idx),
                        right: self.line.seg(right_idx),
                    });
                }
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

        let pos = self.line.insertion_idx(
            self.y,
            self.segments,
            seg_idx,
            &mut self.comparisons.borrow_mut(),
        );
        let contour_prev = if self.segments.positively_oriented(seg_idx) {
            self.segments.contour_prev(seg_idx)
        } else {
            self.segments.contour_next(seg_idx)
        };
        if let Some(contour_prev) = contour_prev {
            if self.segments[contour_prev].p0.y < self.y {
                debug_assert_eq!(self.segments[contour_prev].p3, new_seg.p0);
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

        let mut entry = SegmentOrderEntry::new(seg_idx, self.segments, self.eps);
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
    fn handle_contour_continuation(&mut self, seg_idx: SegIdx, seg: &Segment, pos: usize) {
        let x0 = seg.p0.x;
        let x1 = seg.p3.x;
        self.line.segs[pos].old_seg = Some(self.line.segs[pos].seg);
        self.line.segs[pos].seg = seg_idx;
        self.line.segs[pos].enter = true;
        self.line.segs[pos].exit = true;
        self.line.segs[pos].lower_bound = x0.min(x1) - self.eps;
        self.line.segs[pos].upper_bound = x0.max(x1) + self.eps;
        self.intersection_scan_right(pos);
        self.intersection_scan_left(pos);
        self.add_seg_needing_position(pos, false);
    }

    /// Marks a segment as needing to exit, but doesn't actually remove it
    /// from the sweep-line. See `SegmentOrderEntry::exit` for an explanation.
    fn handle_exit(&mut self, seg_idx: SegIdx) {
        let pos = self
            .line
            .position(
                seg_idx,
                self.segments,
                self.comparisons.borrow_mut().deref_mut(),
                self.y,
            )
            .unwrap();

        if self.line.segs[pos].old_seg == Some(seg_idx) {
            return;
        }

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
            .position(
                left,
                self.segments,
                self.comparisons.borrow_mut().deref_mut(),
                self.y,
            )
            .unwrap();
        let right_idx = self
            .line
            .position(
                right,
                self.segments,
                self.comparisons.borrow_mut().deref_mut(),
                self.y,
            )
            .unwrap();
        if left_idx < right_idx {
            self.segs_needing_positions.extend(left_idx..=right_idx);
            for (i, entry) in self.line.segs.range_mut(left_idx..=right_idx).enumerate() {
                if entry.old_idx.is_none() {
                    entry.old_idx = Some(left_idx + i);
                }
            }

            // We're going to put `left_seg` after `right_seg` in the
            // sweep line, and while doing so we need to "push" along
            // all segments that are strictly bigger than `left_seg`
            // (slight false positives are allowed; no false negatives).
            let mut to_move = vec![(left_idx, self.line.segs[left_idx])];
            for j in (left_idx + 1)..right_idx {
                let seg_j_idx = self.line.seg(j);
                let cmp = self.compare_segments(left, seg_j_idx);
                if cmp.order_at(self.y) == Order::Right {
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

            for (_, seg) in to_move.into_iter().rev() {
                self.insert(insertion_pos, seg);
            }
        }
    }

    // If we have a segment in a changed interval, then every other segment
    // that it compares "Ish" to should also be in that changed interval.
    #[cfg(feature = "slow-asserts")]
    fn check_changed_interval_closeness(&self) {
        let y = self.y;
        for iv in &self.changed_intervals {
            for j in self.line.segs.range(iv.segs.clone()) {
                for i in self.line.segs.range(..iv.segs.start) {
                    let cmp = self.compare_segments(i.seg, j.seg);
                    assert_ne!(cmp.order_at(y), Order::Right);
                }

                for k in self.line.segs.range(iv.segs.end..) {
                    let cmp = self.compare_segments(j.seg, k.seg);
                    assert_ne!(cmp.order_at(y), Order::Right);
                }
            }
        }
    }

    #[cfg(feature = "slow-asserts")]
    fn check_invariants(&self) {
        for seg_entry in self.line.segs.iter() {
            let seg_idx = seg_entry.seg;
            let seg = &self.segments[seg_idx];
            assert!(
                (&seg.p0.y..=&seg.p3.y).contains(&&self.y),
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
            .find_invalid_order(
                self.y,
                &self.segments,
                self.comparisons.borrow_mut().deref_mut(),
            )
            .is_none());

        for i in 0..self.line.segs.len() {
            if self.line.is_exit(i) {
                continue;
            }
            for j in (i + 1)..self.line.segs.len() {
                if self.line.is_exit(j) {
                    continue;
                }
                let segi = self.line.seg(i);
                let segj = self.line.seg(j);

                if let Some(next_touch) = self.compare_segments(segi, segj).next_touch_after(self.y)
                {
                    let (y_int, really) = match next_touch {
                        curve::NextTouch::Cross(y) => (y, true),
                        curve::NextTouch::Touch(y) => (y, y > self.y),
                    };
                    if y_int >= self.y && really {
                        // Find an event between i and j.
                        let is_between = |idx: SegIdx| -> bool {
                            self.line
                                .position(
                                    idx,
                                    self.segments,
                                    self.comparisons.borrow_mut().deref_mut(),
                                    self.y,
                                )
                                .is_some_and(|pos| i <= pos && pos <= j)
                        };
                        let has_exit_witness = self
                            .line
                            .segs
                            .range(i..=j)
                            .any(|seg_entry| self.segments[seg_entry.seg].p3.y <= y_int);

                        let has_intersection_witness = self.events.intersection.iter().any(|ev| {
                            let is_between = is_between(ev.left) && is_between(ev.right);
                            let before_y = ev.y <= y_int;
                            is_between && before_y
                        });
                        let has_witness = has_exit_witness || has_intersection_witness;
                        assert!(
                            has_witness,
                            "segments {:?} and {:?} cross at {:?}, but there is no witness. y={}, intersections={:?}",
                            self.line.segs[i].seg, self.line.segs[j].seg, y_int, self.y, self.events.intersection
                        );
                    }
                }
            }
        }

        self.check_changed_interval_closeness();
    }

    #[cfg(not(feature = "slow-asserts"))]
    fn check_invariants(&self) {}

    fn compute_horizontal_changed_intervals(&mut self) {
        self.horizontals
            .sort_by_key(|seg_idx| CheapOrderedFloat::from(self.segments[*seg_idx].p0.x));

        for (idx, &seg_idx) in self.horizontals.iter().enumerate() {
            let seg = &self.segments[seg_idx];

            // Find the index of some segment that might overlap with this
            // horizontal segment, but the index before definitely doesn't.
            let start_idx = self.line.segs.partition_point(|other_entry| {
                let other_seg = &self.segments[other_entry.seg];

                // We test using Euclidean distance instead of just comparing
                // x coordinates because if other_seg is almost horizontal then
                // a tiny error in solving it for y will make a big error in
                // its x coordinate.
                let p = kurbo::Point::new(seg.p0.x, seg.p0.y);
                let other_seg = other_seg.to_kurbo();
                let nearest = other_seg.nearest(p, self.eps / 2.0);

                // "other's nearest point to p is to the left of p" is a reasonable
                // and robust proxy for "other crosses self.y to the left of p", because
                // "other" is monotonic in y
                nearest.distance_sq > 4.0 * self.eps * self.eps
                    && other_seg.eval(nearest.t).x < seg.p0.x
            });

            let mut end_idx = start_idx;
            for j in start_idx..self.line.segs.len() {
                let other_entry = &mut self.line.segs[j];
                let other_seg = &self.segments[other_entry.seg];

                let p = kurbo::Point::new(seg.p3.x, seg.p3.y);
                let other_seg = other_seg.to_kurbo();
                let nearest = other_seg.nearest(p, self.eps / 2.0);

                if nearest.distance_sq <= 4.0 * self.eps * self.eps
                    || other_seg.eval(nearest.t).x < seg.p3.x
                {
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
            let mut start_idx = idx;
            let mut end_idx = idx + 1;

            for i in (idx + 1)..self.line.segs.len() {
                // Note that end_idx is mutated in the loop, so this points to the segment
                // that we most recently added to the changed interval. Modulo some subtleties
                // involving conservative/non-conservative comparisons, this will usually be
                // `i - 1`.
                let seg_idx = self.line.seg(end_idx - 1);
                let next_seg_idx = self.line.seg(i);
                let strong_cmp = self.compare_segments_conservatively(seg_idx, next_seg_idx);
                if strong_cmp.order_at(self.y) == Order::Left {
                    break;
                } else {
                    debug_assert_eq!(strong_cmp.order_at(self.y), Order::Ish);
                    let cmp = self.compare_segments(seg_idx, next_seg_idx);
                    debug_assert_ne!(cmp.order_at(self.y), Order::Right);
                    if cmp.order_at(self.y) == Order::Ish {
                        for j in end_idx..=i {
                            self.line.segs[j].in_changed_interval = true;
                            self.line.segs[j].set_old_idx_if_unset(j);
                        }

                        end_idx = i + 1;
                    }
                }
            }

            for i in (0..start_idx).rev() {
                let prev_seg_idx = self.line.seg(i);
                // Note that start_idx is mutated in the loop, so this points to the segment
                // that we most recently added to the changed interval.
                let seg_idx = self.line.seg(start_idx);
                let strong_cmp = self.compare_segments_conservatively(prev_seg_idx, seg_idx);
                if strong_cmp.order_at(self.y) == Order::Left {
                    break;
                } else {
                    debug_assert_eq!(strong_cmp.order_at(self.y), Order::Ish);
                    let cmp = self.compare_segments(prev_seg_idx, seg_idx);
                    debug_assert_ne!(cmp.order_at(self.y), Order::Right);
                    if cmp.order_at(self.y) == Order::Ish {
                        for j in i..start_idx {
                            self.line.segs[j].in_changed_interval = true;
                            self.line.segs[j].set_old_idx_if_unset(j);
                        }
                        start_idx = i;
                    }
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

impl SegmentOrder {
    // If the ordering invariants fail, returns a pair of indices witnessing that failure.
    // Used in tests, and when enabling slow-asserts
    #[allow(dead_code)]
    fn find_invalid_order(
        &self,
        y: f64,
        segments: &Segments,
        cmp: &mut ComparisonCache,
    ) -> Option<(SegIdx, SegIdx)> {
        for i in 0..self.segs.len() {
            for j in (i + 1)..self.segs.len() {
                let segi = self.seg(i);
                let segj = self.seg(j);

                let order = cmp.compare_segments(segments, segi, segj);
                if order.order_at(y) == Order::Right {
                    dbg!(&order);
                    return Some((segi, segj));
                }
            }
        }

        None
    }

    // Finds an index into this sweep line where it's ok to insert this new segment.
    fn insertion_idx(
        &self,
        y: f64,
        segments: &Segments,
        seg_idx: SegIdx,
        comparer: &mut ComparisonCache,
    ) -> usize {
        //dbg!(self, seg_idx);
        let seg = &segments[seg_idx];
        let seg_x = seg.p0.x;

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
            // we can insert either before or after `pos`. We'll choose based on what the order
            // will be in the future, so as to minimize swapping.
            // (We could also try checking whether the new segment is a contour neighbor of
            // the segment at pos. That should be a pretty common case.)
            let cmp = comparer.compare_segments(segments, seg_idx, self.segs[pos].seg);
            return match cmp.order_after(y) {
                Order::Right => pos + 1,
                Order::Ish => pos,
                Order::Left => pos,
            };
        }

        // A predicate that tests whether `other` is definitely to the left of `seg` at `y`.
        let lower_pred = |other: &SegmentOrderEntry| -> bool {
            let cmp = comparer.compare_segments(segments, other.seg, seg_idx);
            cmp.order_at(y) == Order::Left
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
        for i in search_start..self.segs.len() {
            let cmp = comparer.compare_segments(segments, self.segs[i].seg, seg_idx);
            match cmp.order_at(y) {
                Order::Right => {
                    // The segment at i is definitely to the right of the new one.
                    break;
                }
                Order::Ish => {
                    // We could go either way, and we'll update our preference based on the
                    // future ordering.
                    if cmp.order_after(y) == Order::Left {
                        idx = i + 1;
                    }
                }
                Order::Left => {
                    // The segment at i is definitely to the left of the new one.
                    idx = i + 1;
                }
            }
        }
        idx
    }

    // Find the position of the given segment in our array.
    //
    // Returns an index pointing to a `SegmentOrderEntry` for which
    // `seg_idx` is either the `seg` or the `old_seg`.
    fn position(
        &self,
        seg_idx: SegIdx,
        segments: &Segments,
        comparer: &mut ComparisonCache,
        y: f64,
    ) -> Option<usize> {
        if self.segs.len() <= 32 {
            return self
                .segs
                .iter()
                .position(|x| x.seg == seg_idx || x.old_seg == Some(seg_idx));
        }

        // start_idx points to a segment that might be close to our segment (or
        // could even be our segment). But the segment at `start_idx - 1` is
        // definitely to our right.
        let start_idx = self.segs.partition_point(|entry| {
            comparer
                .compare_segments(segments, entry.seg, seg_idx)
                .order_at(y)
                == Order::Left
        });

        // end_idx points to something that's definitely after us.
        let end_idx = self.segs.partition_point(|entry| {
            comparer
                .compare_segments(segments, entry.seg, seg_idx)
                .order_at(y)
                != Order::Right
        });

        if end_idx <= start_idx {
            return None;
        }

        self.segs
            .range(start_idx..)
            .position(|x| x.seg == seg_idx || x.old_seg == Some(seg_idx))
            .map(|i| i + start_idx)
    }
}

/// Finds a feasible `x` position for each segment.
///
/// The point is that we've decided on a horizontal ordering of the segments, but
/// their numerical positions might not completely agree with that ordering. For
/// each segment, this function computes a range of `x` coordinates with the
/// guarantee that if you go from left to right (or right to left) and assign
/// each segment an `x` coordinate within its range then you won't paint
/// yourself into a corner: subsequent points can be positioned with the right
/// ordering *and* within the designated range.
fn feasible_horizontal_positions<G: Fn(&SegmentOrderEntry) -> SegIdx>(
    entries: &[SegmentOrderEntry],
    entry_seg: G,
    y: f64,
    segments: &Segments,
    eps: f64,
    out: &mut Vec<(SegmentOrderEntry, f64, f64)>,
) {
    out.clear();
    let mut max_so_far = f64::NEG_INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut min = f64::INFINITY;

    for entry in entries {
        let seg = &segments[entry_seg(entry)];
        max_so_far = max_so_far.max(seg.lower(y, eps));
        // Fill out the minimum allowed positions, with a placeholder for the maximum.
        out.push((*entry, max_so_far, 0.0));

        let x = seg.at_y(y);
        max = max.max(x);
        min = min.min(x);
    }

    let mut min_so_far = f64::INFINITY;

    for (entry, min_allowed, max_allowed) in out.iter_mut().rev() {
        let x = segments[entry_seg(entry)].upper(y, eps);
        min_so_far = min_so_far.min(x);
        *max_allowed = min_so_far.min(max);
        *min_allowed = min_allowed.max(min);
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
pub struct SweepLine<'buf, 'state, 'segs> {
    state: &'state Sweeper<'segs>,
    bufs: &'buf mut SweepLineBuffers,
    // Index into state.changed_intervals
    next_changed_interval: usize,
}

impl<'segs> SweepLine<'_, '_, 'segs> {
    /// The vertical position of this sweep-line.
    pub fn y(&self) -> f64 {
        self.state.y
    }

    /// Get the line segment at position `idx` in the new order.
    pub fn line_segment(&self, idx: usize) -> SegIdx {
        self.state.line.segs[idx].seg
    }

    pub fn compare_segments(&self, i: SegIdx, j: SegIdx) -> CurveOrder {
        self.state.compare_segments(i, j)
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
        bufs: &'bufs mut SweepLineRangeBuffers,
        segments: &Segments,
        idx: usize,
    ) -> Option<SweepLineRange<'bufs, 'a, 'segs>> {
        bufs.clear();
        self.bufs.output_events.clear();

        let entry = &self.state.line.segs[idx];
        let x = segments[entry.seg].at_y(self.y());
        if let Some(old_seg) = entry.old_seg {
            // This entry is on a contour, where one segment ends and the next begins.
            // We ouput two events (one per segment) at the same position.
            self.bufs.output_events.push(OutputEvent {
                x0: x,
                connected_above: true,
                x1: x,
                connected_below: false,
                seg_idx: old_seg,
                sweep_idx: None,
                old_sweep_idx: entry.old_idx,
            });

            self.bufs.output_events.push(OutputEvent {
                x0: x,
                connected_above: false,
                x1: x,
                connected_below: true,
                seg_idx: entry.seg,
                sweep_idx: Some(idx),
                old_sweep_idx: None,
            });
        } else {
            // It's a single segment either entering or exiting at this height.
            // We can handle them both in a single case.
            self.bufs.output_events.push(OutputEvent {
                x0: x,
                connected_above: !entry.enter,
                x1: x,
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
        bufs: &'bufs mut SweepLineRangeBuffers,
        segments: &Segments,
        eps: f64,
    ) -> Option<SweepLineRange<'bufs, 'a, 'segs>> {
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
            lower_bound: 0.0,
            upper_bound: 0.0,
            in_changed_interval: false,
            old_idx: None,
            old_seg: None,
        };

        let buffers = &mut self.bufs;
        buffers
            .old_line
            .resize(range.segs.end - range.segs.start, dummy_entry);
        for entry in self.state.line.segs.range(range.segs.clone()) {
            buffers.old_line[entry.old_idx.unwrap() - range.segs.start] = *entry;
        }

        // Assign horizontal positions to all the points in the old sweep line.
        // First, compute the feasible positions.
        feasible_horizontal_positions(
            &buffers.old_line,
            |entry| entry.old_seg(),
            self.state.y,
            segments,
            eps,
            &mut buffers.positions,
        );

        // Now go through and assign the actual positions.
        //
        // The two positioning arrays should have the same segments, but possibly in a different
        // order. We build them up in the old-sweep-line order.
        buffers.output_events.clear();
        let events = &mut buffers.output_events;
        let mut max_so_far = f64::NEG_INFINITY;
        for (entry, min_x, max_x) in &buffers.positions {
            let preferred_x = if entry.exit {
                // The best possible position is the true segment-ending position.
                // (This could change if we want to be more sophisticated at joining contours.)
                segments[entry.old_seg()].p3.x
            } else if entry.enter {
                // The best possible position is the true segment-starting position.
                // (This could change if we want to be more sophisticated at joining contours.)
                segments[entry.seg].p0.x
            } else {
                segments[entry.seg].at_y(self.state.y)
            };
            let x = preferred_x.max(*min_x).max(max_so_far).min(*max_x);
            max_so_far = x;
            events.push(OutputEvent {
                x0: x,
                connected_above: entry.old_seg.is_some() || !entry.enter,
                // This will be filled out when we traverse new_xs.
                x1: 42.42,
                connected_below: !entry.exit,
                seg_idx: entry.old_seg(),
                sweep_idx: None,
                old_sweep_idx: entry.old_idx,
            });
        }

        // And now we repeat for the new sweep line: compute the feasible positions
        // and then choose the actual positions.
        buffers.old_line.clear();
        buffers
            .old_line
            .extend(self.state.line.segs.range(range.segs.clone()).cloned());
        feasible_horizontal_positions(
            &buffers.old_line,
            |entry| entry.seg,
            self.state.y,
            segments,
            eps,
            &mut buffers.positions,
        );
        let mut max_so_far = f64::NEG_INFINITY;
        for (idx, (entry, min_x, max_x)) in buffers.positions.iter().enumerate() {
            let ev = &mut events[entry.old_idx.unwrap() - range.segs.start];
            ev.sweep_idx = Some(range.segs.start + idx);
            debug_assert_eq!(ev.seg_idx, entry.old_seg());
            let preferred_x = if *min_x <= ev.x0 && ev.x0 <= *max_x {
                // Try snapping to the previous position if possible.
                ev.x0
            } else {
                segments[entry.seg].at_y(self.state.y)
            };
            ev.x1 = preferred_x.max(*min_x).max(max_so_far).min(*max_x);
            max_so_far = ev.x1;

            let x1 = ev.x1;
            if entry.old_seg.is_some() {
                events.push(OutputEvent {
                    x0: x1,
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
                ev.x0 = seg.p0.x;
            }
            if !ev.connected_below {
                ev.x1 = seg.p3.x;
            }
        }

        if let Some(range) = &range.horizontals {
            for &seg_idx in &self.state.horizontals[range.clone()] {
                let seg = &self.state.segments[seg_idx];
                events.push(OutputEvent {
                    x0: seg.p0.x,
                    connected_above: false,
                    x1: seg.p3.x,
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

    use proptest::prelude::*;

    use crate::{
        geom::Point,
        perturbation::{
            f64_perturbation, perturbation, realize_perturbation, F64Perturbation, Perturbation,
            PointPerturbation,
        },
        segments::Segments,
    };

    fn mk_segs(xs: &[(f64, f64)]) -> Segments {
        let mut segs = Segments::default();

        for &(x0, x1) in xs {
            segs.add_points([Point::new(x0, 0.0), Point::new(x1, 1.0)]);
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
            let y = at;
            let segs = mk_segs(xs);

            let line: SegmentOrder = SegmentOrder {
                segs: (0..xs.len())
                    .map(|i| SegmentOrderEntry::new(SegIdx(i), &segs, eps))
                    .collect(),
            };

            let mut cmp = ComparisonCache::new(eps, eps / 2.0);
            line.find_invalid_order(y, &segs, &mut cmp)
                .map(|(a, b)| (a.0, b.0))
        }

        let crossing = &[(-1.0, 1.0), (1.0, -1.0)];
        let eps = 1.0 / 128.0;
        assert!(check_order(crossing, 0.0, eps).is_none());
        assert!(check_order(crossing, 0.5, eps).is_none());
        assert_eq!(check_order(crossing, 1.0, eps), Some((0, 1)));

        let not_quite_crossing = &[(-0.5 * eps, 0.5 * eps), (0.5 * eps, -0.5 * eps)];
        assert!(check_order(not_quite_crossing, 0.0, eps).is_none());
        assert!(check_order(not_quite_crossing, 0.5, eps).is_none());
        assert!(check_order(not_quite_crossing, 1.0, eps).is_none());

        let barely_crossing = &[(-1.5 * eps, 1.5 * eps), (1.5 * eps, -1.5 * eps)];
        assert!(check_order(barely_crossing, 0.0, eps).is_none());
        assert!(check_order(barely_crossing, 0.5, eps).is_none());
        assert_eq!(check_order(barely_crossing, 1.0, eps), Some((0, 1)));

        let non_adj_crossing = &[(-eps, eps), (0.0, 0.0), (eps, -eps)];
        assert!(check_order(non_adj_crossing, 0.0, eps).is_none());
        assert!(check_order(non_adj_crossing, 0.5, eps).is_none());
        assert_eq!(check_order(non_adj_crossing, 1.0, eps), Some((0, 2)));

        let flat_crossing = &[(-1e6, 1e6), (-10.0 * eps, -10.0 * eps)];
        assert_eq!(check_order(flat_crossing, 0.5 - eps, eps), None);

        let end_crossing_bevel = &[(2.5 * eps, 2.5 * eps), (-1e6, 0.0)];
        assert_eq!(check_order(end_crossing_bevel, 1.0, eps), Some((0, 1)));

        let start_crossing_bevel = &[(2.5 * eps, 2.5 * eps), (0.0, -1e6)];
        assert_eq!(check_order(start_crossing_bevel, 1.0, eps), Some((0, 1)));
    }

    #[test]
    fn insertion_idx() {
        fn insert(xs: &[(f64, f64)], new: (f64, f64), at: f64, eps: f64) -> usize {
            let y = at;
            let mut xs: Vec<_> = xs.to_owned();
            xs.push(new);
            let segs = mk_segs(&xs);

            let new_idx = SegIdx(xs.len() - 1);

            let mut line: SegmentOrder = SegmentOrder {
                segs: (0..(xs.len() - 1))
                    .map(|i| SegmentOrderEntry::new(SegIdx(i), &segs, eps))
                    .collect(),
            };
            let mut comparer = ComparisonCache::new(eps, eps / 2.0);
            let idx = line.insertion_idx(y, &segs, new_idx, &mut comparer);

            dbg!(&line);
            assert!(dbg!(line.find_invalid_order(y, &segs, &mut comparer)).is_none());
            line.segs.insert(
                idx,
                SegmentOrderEntry::new(SegIdx(xs.len() - 1), &segs, eps),
            );
            assert!(line.find_invalid_order(y, &segs, &mut comparer,).is_none());
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
                0.5 - 1e-6,
                eps
            ),
            2
        );
        // This test doesn't really work any more, now that we've tightened
        // up our invalid order test.
        // assert_eq!(
        //     insert(
        //         &[
        //             (-1e6, 1e6),
        //             (-1e6, 1e6),
        //             (-1e6, 1e6),
        //             (-1.0, -1.0),
        //             (1.0, 1.0),
        //             (-1e6, 1e6),
        //             (-1e6, 1e6),
        //             (-1e6, 1e6),
        //         ],
        //         (0.0, 0.0),
        //         dbg!(0.5 - 1e-6 / 2.0),
        //         eps
        //     ),
        //     4
        // );

        insert(&[(2.0, 2.0), (-100.0, 100.0)], (1.0, 1.0), 0.51, 0.25);
    }

    #[test]
    fn test_sweep() {
        let eps = 0.01;

        let segs = mk_segs(&[(0.0, 0.0), (1.0, 1.0), (-2.0, 2.0)]);
        dbg!(&segs);
        crate::sweep::sweep(&segs, eps, |_, ev| {
            dbg!(ev);
        });
    }

    fn p(x: f64, y: f64) -> Point {
        Point::new(x, y)
    }

    fn run_perturbation(ps: Vec<Perturbation<F64Perturbation>>) {
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
        let eps = 0.1;
        crate::sweep::sweep(&segs, eps, |_, _| {});
    }

    #[derive(serde::Serialize, Debug)]
    struct Output {
        order: SegmentOrder,
        changed: Vec<ChangedInterval>,
    }
    fn snapshot_outputs(segs: Segments, eps: f64) -> Vec<Output> {
        let mut outputs = Vec::new();
        let mut sweeper = Sweeper::new(&segs, eps);
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
        let mut segs = Segments::default();
        segs.add_cycle(a);
        segs.add_cycle(b);
        insta::assert_ron_snapshot!(snapshot_outputs(segs, 0.1));
    }

    proptest! {
    #[test]
    fn perturbation_test_f64(perturbations in prop::collection::vec(perturbation(f64_perturbation(0.1)), 1..5)) {
        run_perturbation(perturbations);
    }

    }

    #[test]
    fn bug() {
        use Perturbation::*;
        let perturbations: Vec<Perturbation<F64Perturbation>> = vec![
            Base { idx: 0 },
            Superimposition {
                left: Box::new(Base { idx: 0 }),
                right: Box::new(Base { idx: 0 }),
            },
            Base { idx: 0 },
            Superimposition {
                left: Box::new(Base { idx: 0 }),
                right: Box::new(Base { idx: 0 }),
            },
        ];
        run_perturbation(perturbations);
    }
}
