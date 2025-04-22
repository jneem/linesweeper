//! Utilities for computing topological properties of closed polylines.
//!
//! This consumes the output of the sweep-line algorithm and does things
//! like winding number computations and boolean operations.

use std::collections::VecDeque;

use kurbo::{BezPath, Rect};

use crate::{
    geom::Point,
    positioning_graph,
    segments::{SegIdx, Segments},
    sweep::{
        ComparisonCache, SegmentsConnectedAtX, SweepLineBuffers, SweepLineRange,
        SweepLineRangeBuffers, Sweeper,
    },
};

/// We support boolean operations, so a "winding number" for us is two winding
/// numbers, one for each shape.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Default, serde::Serialize)]
pub struct WindingNumber {
    /// The winding number of the first shape.
    pub shape_a: i32,
    /// The winding number of the second shape.
    pub shape_b: i32,
}

impl std::fmt::Debug for WindingNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}a + {}b", self.shape_a, self.shape_b)
    }
}

/// For a segment, we store two winding numbers (one on each side of the segment).
///
/// For simple segments, the winding numbers on two sides only differ by one. Once
/// we merge segments, they can differ by more.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Default, serde::Serialize)]
pub struct HalfSegmentWindingNumbers {
    /// This half-segment is incident to a point. Imagine you're standing at
    /// that point, looking out along the segment. This is the winding number of
    /// the area just counter-clockwise (to the left, from your point of view)
    /// of the segment.
    pub counter_clockwise: WindingNumber,
    /// This half-segment is incident to a point. Imagine you're standing at
    /// that point, looking out along the segment. This is the winding number of
    /// the area just clockwise (to the right, from your point of view) of the segment.
    pub clockwise: WindingNumber,
}

impl HalfSegmentWindingNumbers {
    /// A half-segment's winding numbers are trivial if they're the same on both sides.
    /// In this case, the segment is invisible to the topology of the sets.
    fn is_trivial(&self) -> bool {
        self.counter_clockwise == self.clockwise
    }

    /// Returns the winding numbers of our opposite half-segment.
    fn flipped(self) -> Self {
        Self {
            counter_clockwise: self.clockwise,
            clockwise: self.counter_clockwise,
        }
    }
}

impl std::fmt::Debug for HalfSegmentWindingNumbers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} | {:?}", self.clockwise, self.counter_clockwise)
    }
}

/// An index into the set of output segments.
///
/// There's no compile-time magic preventing misuse of this index, but you
/// should only use this to index into the [`Topology`] that you got it from.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, serde::Serialize)]
pub struct OutputSegIdx(usize);

impl OutputSegIdx {
    /// Returns an index to the first half of this output segment.
    pub fn first_half(self) -> HalfOutputSegIdx {
        HalfOutputSegIdx {
            idx: self,
            first_half: true,
        }
    }

    /// Returns an index to the second half of this output segment.
    pub fn second_half(self) -> HalfOutputSegIdx {
        HalfOutputSegIdx {
            idx: self,
            first_half: false,
        }
    }
}

/// An index that refers to one end of an output segment.
///
/// The two ends of an output segment are sweep-line ordered: the "first" half
/// has a smaller `y` coordinate (or smaller `x` coordinate if the `y`s are
/// tied) than the "second" half.
#[derive(Clone, Copy, Hash, PartialEq, Eq, serde::Serialize)]
pub struct HalfOutputSegIdx {
    idx: OutputSegIdx,
    first_half: bool,
}

impl HalfOutputSegIdx {
    pub fn other_half(self) -> Self {
        Self {
            idx: self.idx,
            first_half: !self.first_half,
        }
    }

    pub fn is_first_half(self) -> bool {
        self.first_half
    }
}

impl std::fmt::Debug for HalfOutputSegIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.first_half {
            write!(f, "s{}->", self.idx.0)
        } else {
            write!(f, "s{}<-", self.idx.0)
        }
    }
}

/// A vector indexed by half-output segments.
#[derive(Clone, Hash, PartialEq, Eq, serde::Serialize)]
pub struct HalfOutputSegVec<T> {
    start: Vec<T>,
    end: Vec<T>,
}

impl<T> HalfOutputSegVec<T> {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            start: Vec::with_capacity(cap),
            end: Vec::with_capacity(cap),
        }
    }
}

impl<T: Default> HalfOutputSegVec<T> {
    pub fn with_size(cap: usize) -> Self {
        Self {
            start: std::iter::from_fn(|| Some(T::default()))
                .take(cap)
                .collect(),
            end: std::iter::from_fn(|| Some(T::default()))
                .take(cap)
                .collect(),
        }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for HalfOutputSegVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct Entry<'a, T> {
            idx: usize,
            start: &'a T,
            end: &'a T,
        }

        impl<T: std::fmt::Debug> std::fmt::Debug for Entry<'_, T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(
                    f,
                    "{idx:4}: {start:?} -> {end:?}",
                    idx = self.idx,
                    start = self.start,
                    end = self.end
                )
            }
        }

        let mut list = f.debug_list();
        for (idx, (start, end)) in self.start.iter().zip(&self.end).enumerate() {
            list.entry(&Entry { idx, start, end });
        }
        list.finish()
    }
}

impl<T> Default for HalfOutputSegVec<T> {
    fn default() -> Self {
        Self {
            start: Vec::new(),
            end: Vec::new(),
        }
    }
}

impl<T> std::ops::Index<HalfOutputSegIdx> for HalfOutputSegVec<T> {
    type Output = T;

    fn index(&self, index: HalfOutputSegIdx) -> &Self::Output {
        if index.first_half {
            &self.start[index.idx.0]
        } else {
            &self.end[index.idx.0]
        }
    }
}

impl<T> std::ops::IndexMut<HalfOutputSegIdx> for HalfOutputSegVec<T> {
    fn index_mut(&mut self, index: HalfOutputSegIdx) -> &mut T {
        if index.first_half {
            &mut self.start[index.idx.0]
        } else {
            &mut self.end[index.idx.0]
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize)]
pub struct OutputSegVec<T> {
    inner: Vec<T>,
}

impl<T> OutputSegVec<T> {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            inner: Vec::with_capacity(cap),
        }
    }

    pub fn indices(&self) -> impl Iterator<Item = OutputSegIdx> {
        (0..self.inner.len()).map(OutputSegIdx)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<T: Default> OutputSegVec<T> {
    pub fn with_size(cap: usize) -> Self {
        Self {
            inner: std::iter::from_fn(|| Some(T::default()))
                .take(cap)
                .collect(),
        }
    }
}

impl<T> Default for OutputSegVec<T> {
    fn default() -> Self {
        Self { inner: Vec::new() }
    }
}

impl<T> std::ops::Index<OutputSegIdx> for OutputSegVec<T> {
    type Output = T;

    fn index(&self, index: OutputSegIdx) -> &Self::Output {
        &self.inner[index.0]
    }
}

impl<T> std::ops::IndexMut<OutputSegIdx> for OutputSegVec<T> {
    fn index_mut(&mut self, index: OutputSegIdx) -> &mut T {
        &mut self.inner[index.0]
    }
}

impl<T> std::ops::Index<HalfOutputSegIdx> for OutputSegVec<T> {
    type Output = T;

    fn index(&self, index: HalfOutputSegIdx) -> &Self::Output {
        &self.inner[index.idx.0]
    }
}

/// An index into the set of points.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, serde::Serialize)]
pub struct PointIdx(usize);

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize)]
struct PointVec<T> {
    inner: Vec<T>,
}

impl<T> PointVec<T> {
    fn with_capacity(cap: usize) -> Self {
        Self {
            inner: Vec::with_capacity(cap),
        }
    }
}

impl<T> Default for PointVec<T> {
    fn default() -> Self {
        Self { inner: Vec::new() }
    }
}

impl<T> std::ops::Index<PointIdx> for PointVec<T> {
    type Output = T;

    fn index(&self, index: PointIdx) -> &Self::Output {
        &self.inner[index.0]
    }
}

impl<T> std::ops::IndexMut<PointIdx> for PointVec<T> {
    fn index_mut(&mut self, index: PointIdx) -> &mut Self::Output {
        &mut self.inner[index.0]
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, serde::Serialize)]
struct PointNeighbors {
    clockwise: HalfOutputSegIdx,
    counter_clockwise: HalfOutputSegIdx,
}

impl std::fmt::Debug for PointNeighbors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} o {:?}", self.counter_clockwise, self.clockwise)
    }
}

/// Consumes sweep-line output and computes topology.
///
/// Computes winding numbers and boolean operations. In principle this could be extended
/// to support more-than-boolean operations, but it only does boolean for now. Also,
/// it currently requires all input paths to be closed; it could be extended to support
/// things like clipping a potentially non-closed path to a closed path.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Topology {
    /// Indexed by `SegIdx`.
    shape_a: Vec<bool>,
    /// Indexed by `SegIdx`.
    ///
    /// For each input segment, this is the list of output segments that we've started
    /// recording but haven't finished with. There can be up to three of them, because
    /// consider a segment that passes through a sweep-line like this:
    ///
    /// ```text
    ///           /
    ///          /
    /// (*) /---/
    ///    /
    ///   /
    /// ```
    ///
    /// When we come to process the sweep-line at height (*), we'll already have the
    /// unfinished output segment coming from above. But before dealing with it, we'll
    /// first encounter the output segment pointing down and add an unfinished segment
    /// for that. Then we'll add an output segment for the horizontal line and so
    /// at that point there will be three unfinished output segments.
    open_segs: Vec<VecDeque<OutputSegIdx>>,
    /// Winding numbers of each segment.
    ///
    /// This is sort of logically indexed by `HalfOutputSegIdx`, because we can look at the
    /// `HalfSegmentWindingNumbers` for each `HalfOutputSegIdx`. But since the two halves of
    /// the winding numbers are determined by one another, we only store the winding numbers
    /// for the start half of the output segment.
    winding: OutputSegVec<HalfSegmentWindingNumbers>,
    /// The output points.
    points: PointVec<Point>,
    /// The segment endpoints, as indices into `points`.
    point_idx: HalfOutputSegVec<PointIdx>,
    /// For each output half-segment, its neighboring segments are the ones that share a point with it.
    point_neighbors: HalfOutputSegVec<PointNeighbors>,
    /// Marks the output segments that have been deleted due to merges of coindident segments.
    deleted: OutputSegVec<bool>,
    /// The map from a segment to its scan-left neighbor is always strictly decreasing (in the
    /// index). This ensures that a scan will always terminate, and it also means that we can
    /// build the contours in increasing `OutputSegIdx` order.
    scan_west: OutputSegVec<Option<OutputSegIdx>>,
    pub close_segments: Vec<positioning_graph::Node>,
    pub orig_seg: OutputSegVec<SegIdx>,
    // TODO: probably don't have this owned
    #[serde(skip)]
    pub segments: Segments,
}

impl Topology {
    /// We're working on building up a list of half-segments that all meet at a point.
    /// Maybe we've done a few already, but there's a region where we may add more.
    /// Something like this:
    ///
    /// ```text
    ///         \
    ///          \  ?
    ///           \  ?
    ///       -----o  ?
    ///           /|  ?
    ///          / |?
    ///         /  |
    /// ```
    ///
    /// `first_seg` is the most-counter-clockwise segment we've added so far
    /// (the one pointing down in the picture above, and `last_seg` is the
    /// most-clockwise one (pointing up-left in the picture above). These can be
    /// `None` if we haven't actually added any segments left.
    ///
    /// This method adds more segments to the picture, starting at `last_seg`
    /// and working clockwise. It works only with segment indices, so it's the
    /// caller's responsibility to ensure that the geometry is correct, and that
    /// the provided segments actually go in clockwise order (relative to each
    /// other, and also relative to the segments we've already placed).
    fn add_segs_clockwise(
        &mut self,
        first_seg: &mut Option<HalfOutputSegIdx>,
        last_seg: &mut Option<HalfOutputSegIdx>,
        segs: impl Iterator<Item = HalfOutputSegIdx>,
        p: PointIdx,
    ) {
        for seg in segs {
            self.point_idx[seg] = p;
            if first_seg.is_none() {
                *first_seg = Some(seg);
            }
            if let Some(last) = last_seg {
                self.point_neighbors[*last].clockwise = seg;
                self.point_neighbors[seg].counter_clockwise = *last;
            }
            *last_seg = Some(seg);
        }
        if let Some((first, last)) = first_seg.zip(*last_seg) {
            self.point_neighbors[last].clockwise = first;
            self.point_neighbors[first].counter_clockwise = last;
        }
    }

    /// Like `add_segs_clockwise`, but adds them on the other side.
    fn add_segs_counter_clockwise(
        &mut self,
        first_seg: &mut Option<HalfOutputSegIdx>,
        last_seg: &mut Option<HalfOutputSegIdx>,
        segs: impl Iterator<Item = HalfOutputSegIdx>,
        p: PointIdx,
    ) {
        for seg in segs {
            self.point_idx[seg] = p;
            if last_seg.is_none() {
                *last_seg = Some(seg);
            }
            if let Some(first) = first_seg {
                self.point_neighbors[*first].counter_clockwise = seg;
                self.point_neighbors[seg].clockwise = *first;
            }
            *first_seg = Some(seg);
        }
        if let Some((first, last)) = first_seg.zip(*last_seg) {
            self.point_neighbors[last].clockwise = first;
            self.point_neighbors[first].counter_clockwise = last;
        }
    }

    /// Takes some segments where we've already placed the first half, and
    /// gets ready to place the second half.
    ///
    /// The state-tracking is subtle and should be re-considered. The basic
    /// issue is that (as discussed in the documentation for `open_segs`) a
    /// single segment index can have three open half-segments at any one time,
    /// so how do we know which one is ready for its second half? The short
    /// answer is that we use a double-ended queue, and see `new_half_seg`
    /// for how we use it.
    fn second_half_segs<'a, 'slf: 'a>(
        &'slf mut self,
        segs: impl Iterator<Item = SegIdx> + 'a,
    ) -> impl Iterator<Item = HalfOutputSegIdx> + 'a {
        segs.map(|s| {
            self.open_segs[s.0]
                .pop_front()
                .expect("should be open")
                .second_half()
        })
    }

    /// Creates a new half-segment.
    ///
    /// This needs to update the open segment state to be compatible with `second_half_segs`.
    ///
    /// The key is that we know the order that the segments are processed: any
    /// horizontal segments will be closed first, followed by segments coming
    /// from an earlier sweep-line, followed by segments extending down from
    /// this sweep-line (which won't be closed until we move on to the next
    /// sweep-line). Therefore, we push horizontal half-segments to the front
    /// of the queue so that they can be taken next. We push non-horizontal
    /// half-segments to the back of the queue, so that the older ones (coming
    /// from the previous sweep-line) will get taken before the new ones.
    fn new_half_seg(
        &mut self,
        idx: SegIdx,
        p: PointIdx,
        winding: HalfSegmentWindingNumbers,
        horizontal: bool,
    ) -> OutputSegIdx {
        let out_idx = OutputSegIdx(self.winding.inner.len());
        if horizontal {
            self.open_segs[idx.0].push_front(out_idx);
        } else {
            self.open_segs[idx.0].push_back(out_idx);
        }
        self.point_idx.start.push(p);
        self.point_idx
            .end
            // TODO: maybe an option instead of this weird sentinel
            .push(PointIdx(usize::MAX));

        let no_nbrs = PointNeighbors {
            clockwise: out_idx.first_half(),
            counter_clockwise: out_idx.first_half(),
        };
        self.point_neighbors.start.push(no_nbrs);
        self.point_neighbors.end.push(no_nbrs);
        self.winding.inner.push(winding);
        self.deleted.inner.push(false);
        self.scan_west.inner.push(None);
        self.orig_seg.inner.push(idx);
        out_idx
    }

    pub fn from_paths(
        set_a: impl IntoIterator<Item = BezPath>,
        set_b: impl IntoIterator<Item = BezPath>,
        eps: f64,
    ) -> Self {
        let mut segments = Segments::default();
        let mut shape_a = Vec::new();
        segments.add_bez_paths(set_a);
        shape_a.resize(segments.len(), true);
        segments.add_bez_paths(set_b);
        shape_a.resize(segments.len(), false);
        segments.check_invariants();
        Self::from_segments(segments, shape_a, eps)
    }

    /// Creates a new `Topology` for a collection of segments and a given tolerance.
    ///
    /// The segments must contain only closed polylines. For the purpose of boolean ops,
    /// the first closed polyline determines the first set and all the other polylines determine
    /// the other set. (Obviously this isn't flexible, and it will be changed. TODO)
    pub fn from_polylines(
        set_a: impl IntoIterator<Item = impl IntoIterator<Item = Point>>,
        set_b: impl IntoIterator<Item = impl IntoIterator<Item = Point>>,
        eps: f64,
    ) -> Self {
        let mut segments = Segments::default();
        let mut shape_a = Vec::new();
        segments.add_cycles(set_a);
        shape_a.resize(segments.len(), true);
        segments.add_cycles(set_b);
        shape_a.resize(segments.len(), false);
        Self::from_segments(segments, shape_a, eps)
    }

    fn from_segments(segments: Segments, shape_a: Vec<bool>, eps: f64) -> Self {
        let mut ret = Self {
            shape_a,
            open_segs: vec![VecDeque::new(); segments.len()],

            // We have at least as many output segments as input segments, so preallocate enough for them.
            winding: OutputSegVec::with_capacity(segments.len()),
            points: PointVec::with_capacity(segments.len()),
            point_idx: HalfOutputSegVec::with_capacity(segments.len()),
            point_neighbors: HalfOutputSegVec::with_capacity(segments.len()),
            deleted: OutputSegVec::with_capacity(segments.len()),
            scan_west: OutputSegVec::with_capacity(segments.len()),
            close_segments: Vec::new(),
            orig_seg: OutputSegVec::with_capacity(segments.len()),
            segments: Segments::default(),
        };
        let mut sweep_state = Sweeper::new(&segments, eps);
        let mut range_bufs = SweepLineRangeBuffers::default();
        let mut line_bufs = SweepLineBuffers::default();
        //dbg!(&segments);
        while let Some(mut line) = sweep_state.next_line(&mut line_bufs) {
            while let Some(positions) = line.next_range(&mut range_bufs, &segments, eps) {
                let range = positions.seg_range();
                let scan_west_seg = if range.segs.start == 0 {
                    None
                } else {
                    let prev_seg = positions.line().line_segment(range.segs.start - 1);
                    debug_assert!(!ret.open_segs[prev_seg.0].is_empty());
                    ret.open_segs[prev_seg.0].front().copied()
                };
                ret.process_sweep_line_range(positions, &segments, scan_west_seg);
            }
        }
        ret.merge_coincident();
        ret.update_close_intervals();
        ret.segments = segments;
        ret
    }

    fn process_sweep_line_range(
        &mut self,
        mut pos: SweepLineRange,
        segments: &Segments,
        mut scan_west: Option<OutputSegIdx>,
    ) {
        let y = pos.line().y();
        let mut winding = scan_west
            .map(|idx| self.winding[idx].counter_clockwise)
            .unwrap_or_default();

        // A re-usable buffer for holding temporary lists of segment indices. We
        // need to collect the output of second_half_segments because it holds
        // a &mut self reference.
        let mut seg_buf = Vec::new();
        let mut connected_segs = SegmentsConnectedAtX::default();
        // A pair (SegIdx, OutputSegIdx) where both indices refer to the last segment
        // we saw that points up (or down) from this sweep line.
        let mut last_connected_up_seg: Option<(SegIdx, OutputSegIdx)> = None;
        let mut last_connected_down_seg: Option<(SegIdx, OutputSegIdx)> = None;

        while let Some(next_x) = pos.x() {
            let p = PointIdx(self.points.inner.len());
            self.points.inner.push(Point::new(next_x, y));
            // The first segment at our current point, in clockwise order.
            let mut first_seg = None;
            // The last segment at our current point, in clockwise order.
            let mut last_seg = None;

            // Close off the horizontal segments from the previous point in this sweep-line.
            let hsegs = pos.active_horizontals();
            seg_buf.clear();
            seg_buf.extend(self.second_half_segs(hsegs));
            self.add_segs_clockwise(&mut first_seg, &mut last_seg, seg_buf.iter().copied(), p);

            // Find all the segments that are connected to something above this sweep-line at next_x.
            pos.update_segments_at_x(&mut connected_segs);
            seg_buf.clear();
            seg_buf.extend(self.second_half_segs(connected_segs.connected_up()));
            self.add_segs_clockwise(&mut first_seg, &mut last_seg, seg_buf.iter().copied(), p);

            for (&out_idx, idx) in seg_buf.iter().zip(connected_segs.connected_up()) {
                if let Some((prev_idx, prev_out_idx)) = last_connected_up_seg {
                    let cmp = pos.line().compare_segments(prev_idx, idx);

                    if let Some((y_start, _)) = cmp.close_interval_at(y) {
                        self.close_segments.push(positioning_graph::Node {
                            left_seg: prev_out_idx,
                            right_seg: out_idx.idx,
                            y0: y_start,
                            y1: y,
                        });
                    }
                }
                last_connected_up_seg = Some((idx, out_idx.idx));
            }

            // Then: gather the output segments from half-segments starting here and moving
            // to later sweep-lines. Allocate new output segments for them.
            // Also, calculate their winding numbers and update `winding`.
            seg_buf.clear();
            for new_seg in connected_segs.connected_down() {
                let winding_dir = if segments.positively_oriented(new_seg) {
                    1
                } else {
                    -1
                };
                let prev_winding = winding;
                if self.shape_a[new_seg.0] {
                    winding.shape_a += winding_dir;
                } else {
                    winding.shape_b += winding_dir;
                }
                let windings = HalfSegmentWindingNumbers {
                    clockwise: prev_winding,
                    counter_clockwise: winding,
                };
                let half_seg = self.new_half_seg(new_seg, p, windings, false);
                self.scan_west[half_seg] = scan_west;
                scan_west = Some(half_seg);
                seg_buf.push(half_seg.first_half());

                if let Some((prev_idx, prev_out_idx)) = last_connected_down_seg {
                    let cmp = pos.line().compare_segments(prev_idx, new_seg);

                    if let Some((_, y_end)) = cmp.close_interval_at(y) {
                        dbg!(&cmp, y, y_end);
                        self.close_segments.push(positioning_graph::Node {
                            left_seg: prev_out_idx,
                            right_seg: half_seg,
                            y0: y,
                            y1: y_end,
                        });
                    }
                }
                last_connected_down_seg = Some((new_seg, half_seg));
            }
            self.add_segs_counter_clockwise(
                &mut first_seg,
                &mut last_seg,
                seg_buf.iter().copied(),
                p,
            );

            // Bump the current x position, which will get rid of horizontals ending here
            // and add any horizontals starting here.
            pos.increase_x();

            // Finally, gather the output segments from horizontal segments starting here.
            // Allocate new output segments for them and calculate their winding numbers.
            let hsegs = pos.active_horizontals();

            // We don't want to update our "global" winding number state because that's supposed
            // to keep track of the winding number below the current sweep line.
            let mut w = winding;
            seg_buf.clear();
            for new_seg in hsegs {
                let winding_dir = if segments.positively_oriented(new_seg) {
                    1
                } else {
                    -1
                };
                let prev_w = w;
                if self.shape_a[new_seg.0] {
                    w.shape_a += winding_dir;
                } else {
                    w.shape_b += winding_dir;
                }
                let windings = HalfSegmentWindingNumbers {
                    counter_clockwise: w,
                    clockwise: prev_w,
                };
                let half_seg = self.new_half_seg(new_seg, p, windings, true);
                self.scan_west[half_seg] = scan_west;
                seg_buf.push(half_seg.first_half());
            }
            self.add_segs_counter_clockwise(
                &mut first_seg,
                &mut last_seg,
                seg_buf.iter().copied(),
                p,
            );
        }
    }

    fn delete_half(&mut self, half_seg: HalfOutputSegIdx) {
        let nbr = self.point_neighbors[half_seg];
        self.point_neighbors[nbr.clockwise].counter_clockwise = nbr.counter_clockwise;
        self.point_neighbors[nbr.counter_clockwise].clockwise = nbr.clockwise;
    }

    fn delete(&mut self, seg: OutputSegIdx) {
        self.deleted[seg] = true;
        self.delete_half(seg.first_half());
        self.delete_half(seg.second_half());
    }

    /// After generating the topology, there's a good chance we end up with
    /// coincident output segments. This method removes coincident segments. If
    /// a collection of coincident segments has a net winding number of zero,
    /// this method just removes them all. Otherwise, they are replaced by a
    /// single segment.
    ///
    /// In principle, we could do this as we build the topology. The thing that
    /// makes it a little bit tricky is that (except for horizontal segments)
    /// we don't know whether two segments are coincident until we've processed
    /// their second endpoint.
    fn merge_coincident(&mut self) {
        for idx in 0..self.winding.inner.len() {
            let idx = OutputSegIdx(idx);
            if self.deleted[idx] {
                continue;
            }
            let cc_nbr = self.point_neighbors[idx.first_half()].clockwise;
            if self.point_idx[idx.second_half()] == self.point_idx[cc_nbr.other_half()] {
                // All output segments are in sweep line order, so if they're
                // coincident then they'd better both be first halves.
                debug_assert!(cc_nbr.first_half);
                self.delete(idx);
                self.winding[cc_nbr.idx].counter_clockwise = self.winding[idx].counter_clockwise;

                if self.winding[cc_nbr.idx].is_trivial() {
                    self.delete(cc_nbr.idx);
                }
            }
        }
    }

    /// Iterates over indices of all output segments.
    pub fn segment_indices(&self) -> impl Iterator<Item = OutputSegIdx> + '_ {
        (0..self.winding.inner.len())
            .map(OutputSegIdx)
            .filter(|i| !self.deleted[*i])
    }

    /// Returns the winding numbers of an output half-segment.
    pub fn winding(&self, idx: HalfOutputSegIdx) -> HalfSegmentWindingNumbers {
        if idx.first_half {
            self.winding[idx.idx]
        } else {
            self.winding[idx.idx].flipped()
        }
    }

    /// Returns the endpoint of an output half-segment.
    pub fn point(&self, idx: HalfOutputSegIdx) -> &Point {
        &self.points[self.point_idx[idx]]
    }

    /// Returns the contours of some set defined by this topology.
    ///
    /// The callback function `inside` takes a winding number and returns `true`
    /// if a point with that winding number should be in the resulting set. For example,
    /// to compute a boolean "and" using the non-zero winding rule, `inside` should be
    /// `|w| w.shape_a != 0 && w.shape_b != 0`.
    pub fn contours(&self, inside: impl Fn(WindingNumber) -> bool) -> Contours {
        // We walk contours in sweep-line order of their smallest point. This mostly ensures
        // that we visit outer contours before we visit their children. However, when the inner
        // and outer contours share a point, we run into a problem. For example:
        //
        // /------------------\
        // |        /\        |
        // |       /  \       |
        // \       \  /      /
        //  \       \/      /
        //   \             /
        //    -------------
        // (where the top-middle point is supposed to have 4 segments coming out of it; it's
        // a hard to draw it in ASCII). In this case, we'll "notice" the inner contour when
        // we realize that we've visited a point twice. At that point, we extract the inner part
        // into a separate contour and mark it as a child of the outer one. This requires some
        // slightly sketch indexing, because we need to refer to the outer contour even though
        // we haven't finished generating it. We solve this by reserving a slot for the unfinished
        // outer contour as soon as we start walking it.
        let mut ret = Contours::default();
        let mut seg_contour: Vec<Option<ContourIdx>> = vec![None; self.winding.inner.len()];

        let bdy = |idx: OutputSegIdx| -> bool {
            inside(self.winding[idx].clockwise) != inside(self.winding[idx].counter_clockwise)
        };

        let mut visited = vec![false; self.winding.inner.len()];
        // Keep track of the points that were visited on this walk, so that if we re-visit a
        // point we can split out an additional contour.
        let mut last_visit = PointVec::with_capacity(self.points.inner.len());
        last_visit.inner.resize(self.points.inner.len(), None);
        for idx in self.segment_indices() {
            if visited[idx.0] {
                continue;
            }

            if !bdy(idx) {
                continue;
            }

            // We found a boundary segment. Let's start by scanning left to figure out where we
            // are relative to existing contours.
            let contour_idx = ContourIdx(ret.contours.len());
            let mut contour = Contour::default();
            let mut west_seg = self.scan_west[idx];
            while let Some(left) = west_seg {
                if self.deleted[left] || !bdy(left) {
                    west_seg = self.scan_west[left];
                } else {
                    break;
                }
            }
            if let Some(west) = west_seg {
                if let Some(west_contour) = seg_contour[west.0] {
                    // Is the thing just to our left inside or outside the output set?
                    let outside = !inside(self.winding(west.first_half()).counter_clockwise);
                    if outside == ret.contours[west_contour.0].outer {
                        // They're an outer contour, and there's exterior between us and them,
                        // or they're an inner contour and there's interior between us.
                        // That means they're our sibling.
                        contour.parent = ret.contours[west_contour.0].parent;
                        contour.outer = outside;
                        debug_assert!(outside || contour.parent.is_some());
                    } else {
                        contour.parent = Some(west_contour);
                        contour.outer = !ret.contours[west_contour.0].outer;
                    }
                } else {
                    panic!("I'm {idx:?}, west is {west:?}. Y u no have contour?");
                }
            };
            // Reserve a space for the unfinished outer contour, as described above.
            ret.contours.push(contour);

            // First, arrange the orientation so that the interior is on our
            // left as we walk.
            let (start, mut next) = if inside(self.winding[idx].counter_clockwise) {
                (idx.first_half(), idx.second_half())
            } else {
                (idx.second_half(), idx.first_half())
            };
            // `segs` collects the endpoint of each segment as we walk along it.
            // "endpoint" here means "as we walk the contour;" it could be either a first_half
            // or a second_half as far as `HalfOutputSegIdx` is concerned.
            let mut segs = Vec::new();
            last_visit[self.point_idx[start]] = Some(0);

            debug_assert!(inside(self.winding(start).counter_clockwise));
            loop {
                visited[next.idx.0] = true;

                debug_assert!(inside(self.winding(next).clockwise));
                debug_assert!(!inside(self.winding(next).counter_clockwise));

                // Walk clockwise around the point until we find the next segment
                // that's on the boundary.
                let mut nbr = self.point_neighbors[next].clockwise;
                debug_assert!(inside(self.winding(nbr).counter_clockwise));
                while inside(self.winding(nbr).clockwise) {
                    nbr = self.point_neighbors[nbr].clockwise;
                }

                segs.push(next);
                if nbr == start {
                    break;
                }

                let p = self.point_idx[nbr];
                let last_visit_idx = last_visit[p]
                    // We don't clean up `last_visit` when we finish walking a
                    // contour because it could be expensive if there are many
                    // small contours. This means that `last_visit[p]` could be
                    // a false positive from an earlier contours. We handle this
                    // by double-checking that it was actually a time when this
                    // contours visited `p`.
                    .filter(|&idx| idx < segs.len() && self.point_idx[segs[idx].other_half()] == p);
                if let Some(seg_idx) = last_visit_idx {
                    // We repeated a point, meaning that we've found an inner contour. Extract
                    // it and remove it from the current contour.

                    // seg_idx should point to the end of a segment whose start is at p.
                    debug_assert_eq!(self.point_idx[segs[seg_idx].other_half()], p);

                    let loop_contour_idx = ContourIdx(ret.contours.len());
                    for &seg in &segs[seg_idx..] {
                        seg_contour[seg.idx.0] = Some(loop_contour_idx);
                    }
                    let points = segs[seg_idx..]
                        .iter()
                        .map(|s| *self.point(s.other_half()))
                        .collect();
                    let contour_segs = segs[seg_idx..].to_vec();
                    ret.contours.push(Contour {
                        points,
                        segs: contour_segs,
                        parent: Some(contour_idx),
                        outer: !ret.contours[contour_idx.0].outer,
                    });
                    segs.truncate(seg_idx);
                    // In principle, we should also be unsetting `last_visit`
                    // for all points in the contour we just removed. I *think*
                    // we don't need to, because it's impossible for the outer
                    // contour to visit any of them anyway. Should check this
                    // more carefully.
                } else {
                    last_visit[p] = Some(segs.len());
                }

                next = nbr.other_half();
            }
            for &seg in &segs {
                seg_contour[seg.idx.0] = Some(contour_idx);
            }
            ret.contours[contour_idx.0].points =
                segs.iter().map(|s| *self.point(s.other_half())).collect();
            ret.contours[contour_idx.0].segs = segs.to_vec();
        }

        ret
    }

    pub fn bounding_box(&self) -> kurbo::Rect {
        let mut rect = Rect::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        );
        for seg in self.segments.segments() {
            rect = rect.union_pt(seg.p0.to_kurbo());
            rect = rect.union_pt(seg.p1.to_kurbo());
            rect = rect.union_pt(seg.p2.to_kurbo());
            rect = rect.union_pt(seg.p3.to_kurbo());
        }
        rect
    }

    // While initially creating the topology, the safe and close intervals
    // were just based on the curve comparisons. But the output segments might
    // be shorter than the intervals, so fix them up.
    //
    // TODO: also check sanity
    fn update_close_intervals(&mut self) {
        self.close_segments.retain_mut(|node| {
            let left_p0 = self.point_idx[node.left_seg.first_half()];
            let left_p1 = self.point_idx[node.left_seg.second_half()];
            let right_p0 = self.point_idx[node.right_seg.first_half()];
            let right_p1 = self.point_idx[node.right_seg.second_half()];
            let y0 = self.points[left_p0].y.max(self.points[right_p0].y);
            let y1 = self.points[left_p1].y.min(self.points[right_p1].y);

            debug_assert!(y0 < y1);
            node.y0 = node.y0.max(y0);
            node.y1 = node.y1.min(y1);
            node.y0 < node.y1
        })
    }

    pub fn compute_positions(&self, eps: f64) -> OutputSegVec<BezPath> {
        // TODO: reuse the cache from the sweep-line
        let mut cmp = ComparisonCache::default();
        let mut endpoints = HalfOutputSegVec::with_size(self.orig_seg.len());
        for idx in self.orig_seg.indices() {
            endpoints[idx.first_half()] = self.points[self.point_idx[idx.first_half()]].to_kurbo();
            endpoints[idx.second_half()] =
                self.points[self.point_idx[idx.second_half()]].to_kurbo();
        }

        crate::position::compute_positions(
            &self.segments,
            &self.orig_seg,
            &self.close_segments,
            &mut cmp,
            &endpoints,
            eps,
        )
    }
}

/// An index for a [`Contour`] within [`Contours`].
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, serde::Serialize)]
pub struct ContourIdx(pub usize);

/// A simple, closed polyline.
///
/// A contour has no repeated points, and its segments do not intersect.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Contour {
    /// The points making up this contour.
    ///
    /// If you're drawing a contour with line segments, don't forget to close it: the last point
    /// should be connected to the first point.
    pub points: Vec<Point>,

    /// `segs[i]` is the segment from `points[i]` to `points[(i + 1) % points.len()]`.
    pub segs: Vec<HalfOutputSegIdx>,

    // TODO: also gather the output segment indices. (So that we can do positioning)
    /// A contour can have a parent, so that sets with holes can be represented as nested contours.
    /// For example, the shaded set below:
    ///
    /// ```text
    ///   ----------------------
    ///   |xxxxxxxxxxxxxxxxxxxx|
    ///   |xxxxxxxxxxxxxxxxxxxx|
    ///   |xxxxxxxxx/\xxxxxxxxx|
    ///   |xxxxxxxx/  \xxxxxxxx|
    ///   |xxxxxxx/    \xxxxxxx|
    ///   |xxxxxxx\    /xxxxxxx|
    ///   |xxxxxxxx\  /xxxxxxxx|
    ///   |xxxxxxxxx\/xxxxxxxxx|
    ///   |xxxxxxxxxxxxxxxxxxxx|
    ///   |xxxxxxxxxxxxxxxxxxxx|
    ///   ----------------------
    /// ```
    ///
    /// is represented as a square contour with no parent, and a diamond contour with the square
    /// as its parent.
    ///
    /// A contour can share at most one point with its parent. For example, if you translate the
    /// diamond above upwards until it touches the top of the square, it will share that top point
    /// with its parent. You can't make them share two points, though: if you try to translate the
    /// diamond to a corner...
    ///
    /// ```text
    ///   ----------------------
    ///   |xx/\xxxxxxxxxxxxxxxx|
    ///   |x/  \xxxxxxxxxxxxxxx|
    ///   |/    \xxxxxxxxxxxxxx|
    ///   |\    /xxxxxxxxxxxxxx|
    ///   |x\  /xxxxxxxxxxxxxxx|
    ///   |xx\/xxxxxxxxxxxxxxxx|
    ///   |xxxxxxxxxxxxxxxxxxxx|
    ///   |xxxxxxxxxxxxxxxxxxxx|
    ///   |xxxxxxxxxxxxxxxxxxxx|
    ///   |xxxxxxxxxxxxxxxxxxxx|
    ///   ----------------------
    /// ```
    ///
    /// ...then it will be interpreted as two contours without a parent/child relationship:
    ///
    /// ```text
    ///   ----
    ///   |xx/
    ///   |x/
    ///   |/
    /// ```
    ///
    /// and
    ///
    /// ```text
    ///       ------------------
    ///       \xxxxxxxxxxxxxxxx|
    ///        \xxxxxxxxxxxxxxx|
    ///         \xxxxxxxxxxxxxx|
    ///   |\    /xxxxxxxxxxxxxx|
    ///   |x\  /xxxxxxxxxxxxxxx|
    ///   |xx\/xxxxxxxxxxxxxxxx|
    ///   |xxxxxxxxxxxxxxxxxxxx|
    ///   |xxxxxxxxxxxxxxxxxxxx|
    ///   |xxxxxxxxxxxxxxxxxxxx|
    ///   |xxxxxxxxxxxxxxxxxxxx|
    ///   ----------------------
    /// ```
    ///
    pub parent: Option<ContourIdx>,

    /// Whether this contour is "outer" or not. A contour with no parent is "outer", and
    /// then they alternate: a contour is "outer" if and only if its parent isn't.
    ///
    /// As you walk along a contour, the "occupied" part of the set it represents is
    /// on your left. This means that outer contours wind counter-clockwise and inner
    /// contours wind clockwise.
    pub outer: bool,
}

impl Default for Contour {
    fn default() -> Self {
        Self {
            points: Vec::default(),
            segs: Vec::default(),
            outer: true,
            parent: None,
        }
    }
}

/// A collection of [`Contour`]s.
///
/// Can be indexed with a [`ContourIdx`].
#[derive(Clone, Debug, serde::Serialize, Default)]
pub struct Contours {
    contours: Vec<Contour>,
}

impl Contours {
    /// Returns all of the contour indices, grouped by containment.
    ///
    /// For each of the inner vecs, the first element is an outer contour with
    /// no parent. All of the other contours in that inner vec lie inside that
    /// outer contour.
    pub fn grouped(&self) -> Vec<Vec<ContourIdx>> {
        let mut children = vec![Vec::new(); self.contours.len()];
        let mut top_level = Vec::new();
        for i in 0..self.contours.len() {
            if let Some(parent) = self.contours[i].parent {
                children[parent.0].push(ContourIdx(i));
            } else {
                top_level.push(ContourIdx(i));
            }
        }

        let mut ret = Vec::with_capacity(top_level.len());
        for top in top_level {
            let mut tree = Vec::new();
            fn visit(idx: ContourIdx, children: &[Vec<ContourIdx>], acc: &mut Vec<ContourIdx>) {
                acc.push(idx);
                for &child in &children[idx.0] {
                    visit(child, children, acc);
                }
            }
            visit(top, &children, &mut tree);
            ret.push(tree);
        }

        ret
    }

    /// Iterates over all of the contours.
    pub fn contours(&self) -> impl Iterator<Item = &Contour> + '_ {
        self.contours.iter()
    }
}

impl std::ops::Index<ContourIdx> for Contours {
    type Output = Contour;

    fn index(&self, index: ContourIdx) -> &Self::Output {
        &self.contours[index.0]
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use crate::{
        geom::Point,
        perturbation::{
            f64_perturbation, perturbation, realize_perturbation, F64Perturbation, Perturbation,
        },
    };

    use super::Topology;

    fn p(x: f64, y: f64) -> Point {
        Point::new(x, y)
    }

    const EMPTY: [[Point; 0]; 0] = [];

    #[test]
    fn square() {
        let segs = [[p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0)]];
        let eps = 0.01;
        let top = Topology::from_polylines(segs, EMPTY, eps);
        //check_intersections(&top);

        insta::assert_ron_snapshot!(top);
    }

    #[test]
    fn diamond() {
        let segs = [[p(0.0, 0.0), p(1.0, 1.0), p(0.0, 2.0), p(-1.0, 1.0)]];
        let eps = 0.01;
        let top = Topology::from_polylines(segs, EMPTY, eps);
        //check_intersections(&top);

        insta::assert_ron_snapshot!(top);
    }

    #[test]
    fn square_and_diamond() {
        let square = [[p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0)]];
        let diamond = [[p(0.0, 0.0), p(1.0, 1.0), p(0.0, 2.0), p(-1.0, 1.0)]];
        let eps = 0.01;
        let top = Topology::from_polylines(square, diamond, eps);
        //check_intersections(&top);

        insta::assert_ron_snapshot!(top);
    }

    #[test]
    fn square_with_double_back() {
        let segs = [[
            p(0.0, 0.0),
            p(0.5, 0.0),
            p(0.5, 0.5),
            p(0.5, 0.0),
            p(1.0, 0.0),
            p(1.0, 1.0),
            p(0.0, 1.0),
        ]];
        let eps = 0.01;
        let top = Topology::from_polylines(segs, EMPTY, eps);
        //check_intersections(&top);

        insta::assert_ron_snapshot!(top);
    }

    #[test]
    fn nested_squares() {
        let outer = [[p(-2.0, -2.0), p(2.0, -2.0), p(2.0, 2.0), p(-2.0, 2.0)]];
        let inner = [[p(-1.0, -1.0), p(1.0, -1.0), p(1.0, 1.0), p(-1.0, 1.0)]];
        let eps = 0.01;
        let top = Topology::from_polylines(outer, inner, eps);
        let contours = top.contours(|w| (w.shape_a + w.shape_b) % 2 != 0);

        insta::assert_ron_snapshot!((top, contours));
    }

    #[test]
    fn inner_loop() {
        let outer = [[p(-2.0, -2.0), p(2.0, -2.0), p(2.0, 2.0), p(-2.0, 2.0)]];
        let inners = [
            [p(-1.5, -1.0), p(0.0, 2.0), p(1.5, -1.0)],
            [p(-0.1, 0.0), p(0.0, 2.0), p(0.1, 0.0)],
        ];
        let eps = 0.01;
        let top = Topology::from_polylines(outer, inners, eps);
        let contours = top.contours(|w| (w.shape_a + w.shape_b) % 2 != 0);

        insta::assert_ron_snapshot!((top, contours));
    }

    // Checks that all output segments intersect one another only at endpoints.
    // fn check_intersections(top: &Topology) {
    //     for i in 0..top.winding.inner.len() {
    //         for j in (i + 1)..top.winding.inner.len() {
    //             let i = OutputSegIdx(i);
    //             let j = OutputSegIdx(j);

    //             let p0 = top.point(i.first_half()).clone();
    //             let p1 = top.point(i.second_half()).clone();
    //             let q0 = top.point(j.first_half()).clone();
    //             let q1 = top.point(j.second_half()).clone();

    //             let s = Segment::new(p0.clone().min(p1.clone()), p1.max(p0)).to_exact();
    //             let t = Segment::new(q0.clone().min(q1.clone()), q1.max(q0)).to_exact();

    //             if s.end.y >= t.start.y && t.end.y >= s.start.y {
    //                 // FIXME
    //                 // if let Some(y) = s.exact_intersection_y(&t) {
    //                 //     dbg!(y);
    //                 //     assert!(
    //                 //         s.start == t.start
    //                 //             || s.start == t.end
    //                 //             || s.end == t.start
    //                 //             || s.end == t.end
    //                 //     );
    //                 // }
    //             }
    //         }
    //     }
    // }

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
        let eps = 0.1;
        let _top = Topology::from_polylines(perturbed_polylines, EMPTY, eps);
        //check_intersections(&top);
    }

    proptest! {
    #[test]
    fn perturbation_test_f64(perturbations in prop::collection::vec(perturbation(f64_perturbation(0.1)), 1..5)) {
        run_perturbation(perturbations);
    }
    }
}
