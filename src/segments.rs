use kurbo::BezPath;

use crate::{
    geom::{monotonic_pieces, Point, Segment},
    num::CheapOrderedFloat,
};

/// An index into our segment arena.
///
/// Throughout this library, we assign identities to segments, so that we may
/// consider segments as different even if they have the same start- and end-points.
///
/// This index is used to identify a segment, whose data can be retrieved by looking
/// it up in [`Segments`]. (Of course, this index-as-identifier breaks down if there are
/// multiple `Segments` in flight. Just be careful not to mix them up.)
#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash, serde::Serialize)]
pub struct SegIdx(pub usize);

impl std::fmt::Debug for SegIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "s_{}", self.0)
    }
}

/// An arena of line segments.
///
/// Segments are indexed by [`SegIdx`] and can be retrieved by indexing (i.e. with square brackets).
#[derive(Debug, Clone, Default)]
pub struct Segments {
    segs: Vec<Segment>,
    contour_prev: Vec<Option<SegIdx>>,
    contour_next: Vec<Option<SegIdx>>,
    /// For each segment, stores true if the sweep-line order (small y to big y)
    /// is the same as the orientation in its original contour.
    orientation: Vec<bool>,

    /// All the entrance heights, of segments, ordered by height.
    /// This includes horizontal segments.
    enter: Vec<(f64, SegIdx)>,
    /// All the exit heights of segments, ordered by height.
    /// This does not include horizontal segments.
    exit: Vec<(f64, SegIdx)>,
}

fn cyclic_pairs<T>(xs: &[T]) -> impl Iterator<Item = (&T, &T)> {
    pairs(xs).chain(xs.last().zip(xs.first()))
}

fn pairs<T>(xs: &[T]) -> impl Iterator<Item = (&T, &T)> {
    xs.windows(2).map(|pair| (&pair[0], &pair[1]))
}

impl Segments {
    /// The number of line segments in this arena.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.segs.len()
    }

    /// Iterate over all indices that can be used to index into this arena.
    pub fn indices(&self) -> impl Iterator<Item = SegIdx> {
        (0..self.segs.len()).map(SegIdx)
    }

    /// Iterate over all segments in this arena.
    pub fn segments(&self) -> impl Iterator<Item = &Segment> {
        self.segs.iter()
    }

    /// Returns the starting point of the segment at `idx`, relative to the segment's original orientation.
    ///
    /// The start and end points of the segment itself are stored in sweep-line
    /// order (i.e. `start` has the smaller `y` coordinate), regardless of the
    /// original orientation of the segment. Use this method to retrieve the
    /// segment's original start point.
    pub fn oriented_start(&self, idx: SegIdx) -> &Point {
        if self.orientation[idx.0] {
            &self[idx].p0
        } else {
            &self[idx].p3
        }
    }

    /// Returns the ending point of the segment at `idx`, relative to the segment's original orientation.
    ///
    /// The start and end points of the segment itself are stored in sweep-line
    /// order (i.e. `start` has the smaller `y` coordinate), regardless of the
    /// original orientation of the segment. Use this method to retrieve the
    /// segment's original end point.
    pub fn oriented_end(&self, idx: SegIdx) -> &Point {
        if self.orientation[idx.0] {
            &self[idx].p3
        } else {
            &self[idx].p0
        }
    }

    /// Returns the index of the segment following `idx`.
    ///
    /// If `idx` is part of a non-closed polyline and it is the last segment,
    /// this returns `None`. If `idx` is part of a closed polyline, this will
    /// always return `Some`, and you might need to be careful to avoid looping
    /// infinitely.
    pub fn contour_next(&self, idx: SegIdx) -> Option<SegIdx> {
        self.contour_next[idx.0]
    }

    /// Returns the index of the segment preceding `idx`.
    ///
    /// If `idx` is part of a non-closed polyline and it is the first segment,
    /// this returns `None`. If `idx` is part of a closed polyline, this will
    /// always return `Some`, and you might need to be careful to avoid looping
    /// infinitely.
    pub fn contour_prev(&self, idx: SegIdx) -> Option<SegIdx> {
        self.contour_prev[idx.0]
    }

    /// Does the sweep-line orientation of `idx` agree with its original orientation?
    pub fn positively_oriented(&self, idx: SegIdx) -> bool {
        self.orientation[idx.0]
    }

    /// Add a (non-closed) polyline to this arena.
    pub fn add_points<P: Into<Point>>(&mut self, ps: impl IntoIterator<Item = P>) {
        let old_len = self.segs.len();

        let ps: Vec<_> = ps.into_iter().map(|p| p.into()).collect();
        if ps.len() <= 1 {
            return;
        }

        for (p, q) in pairs(&ps) {
            let (a, b, orient) = if p < q { (p, q, true) } else { (q, p, false) };
            self.segs.push(Segment::straight(*a, *b));
            self.orientation.push(orient);
            self.contour_prev
                .push(Some(SegIdx(self.segs.len().saturating_sub(2))));
            self.contour_next.push(Some(SegIdx(self.segs.len())));
        }

        if let Some(first) = self.contour_prev.get_mut(old_len) {
            *first = None;
        }
        if let Some(last) = self.contour_next.last_mut() {
            *last = None;
        }

        self.update_enter_exit(old_len);
    }

    /// Add a collection of closed polylines to this arena.
    ///
    /// This can be much faster than calling `add_cycles` repeatedly.
    pub fn add_cycles<P: Into<Point>>(
        &mut self,
        ps: impl IntoIterator<Item = impl IntoIterator<Item = P>>,
    ) {
        let old_len = self.segs.len();
        for p in ps {
            self.add_cycle_without_updating_enter_exit(p);
        }
        self.update_enter_exit(old_len);
    }

    /// Add a closed polyline to this arena.
    pub fn add_cycle<P: Into<Point>>(&mut self, ps: impl IntoIterator<Item = P>) {
        let old_len = self.segs.len();
        self.add_cycle_without_updating_enter_exit(ps);
        self.update_enter_exit(old_len);
    }

    pub fn add_bez_paths(&mut self, ps: impl IntoIterator<Item = BezPath>) {
        let old_len = self.segs.len();
        for p in ps {
            self.add_path_without_updating_enter_exit(&p);
        }
        self.update_enter_exit(old_len);
    }

    fn add_path_without_updating_enter_exit(&mut self, p: &BezPath) {
        let old_len = self.segs.len();
        for seg in p.segments() {
            // TODO: could have a fast path for line segments
            let cubic = seg.to_cubic();
            let cubics = monotonic_pieces(cubic);
            for c in cubics {
                let (p0, p1, p2, p3, orient) = if c.p0.y <= c.p3.y {
                    (c.p0, c.p1, c.p2, c.p3, true)
                } else {
                    (c.p3, c.p2, c.p1, c.p0, false)
                };
                self.segs
                    .push(Segment::new(p0.into(), p1.into(), p2.into(), p3.into()));
                self.orientation.push(orient);
                self.contour_prev
                    .push(Some(SegIdx(self.segs.len().saturating_sub(2))));
                self.contour_next.push(Some(SegIdx(self.segs.len())));
            }
        }

        if let Some(first) = self.contour_prev.get_mut(old_len) {
            *first = Some(SegIdx(self.segs.len() - 1));
        }
        if let Some(last) = self.contour_next.last_mut() {
            *last = Some(SegIdx(old_len));
        }
    }

    fn add_cycle_without_updating_enter_exit<P: Into<Point>>(
        &mut self,
        ps: impl IntoIterator<Item = P>,
    ) {
        let old_len = self.segs.len();

        let ps: Vec<_> = ps.into_iter().map(|p| p.into()).collect();
        if ps.len() <= 1 {
            return;
        }

        for (p, q) in cyclic_pairs(&ps) {
            let (a, b, orient) = if p < q { (p, q, true) } else { (q, p, false) };
            self.segs.push(Segment::straight(*a, *b));
            self.orientation.push(orient);
            self.contour_prev
                .push(Some(SegIdx(self.segs.len().saturating_sub(2))));
            self.contour_next.push(Some(SegIdx(self.segs.len())));
        }

        if let Some(first) = self.contour_prev.get_mut(old_len) {
            *first = Some(SegIdx(self.segs.len() - 1));
        }
        if let Some(last) = self.contour_next.last_mut() {
            *last = Some(SegIdx(old_len));
        }
    }

    /// Construct a segment arena from a single closed polyline.
    pub fn from_closed_cycle<P: Into<Point>>(ps: impl IntoIterator<Item = P>) -> Self {
        let mut ret = Self::default();
        ret.add_cycle(ps);
        ret
    }

    fn update_enter_exit(&mut self, old_len: usize) {
        for idx in old_len..self.len() {
            let seg_idx = SegIdx(idx);
            let seg = &self.segs[seg_idx.0];

            self.enter.push((seg.p0.y, seg_idx));
            if !seg.is_horizontal() {
                self.exit.push((seg.p3.y, seg_idx));
            }
        }

        // We sort the enter segments by y position, and then by horizontal
        // start position so that they're fairly likely to get inserted in the
        // sweep-line in order (which makes the indexing fix-ups faster).
        self.enter.sort_by(|(y1, seg1), (y2, seg2)| {
            CheapOrderedFloat::from(*y1)
                .cmp(&CheapOrderedFloat::from(*y2))
                .then_with(|| {
                    CheapOrderedFloat::from(self.segs[seg1.0].at_y(*y1))
                        .cmp(&CheapOrderedFloat::from(self.segs[seg2.0].at_y(*y1)))
                })
        });
        self.exit.sort_by(|(y1, _), (y2, _)| {
            CheapOrderedFloat::from(*y1).cmp(&CheapOrderedFloat::from(*y2))
        });
    }

    /// All the entrance heights of segments, ordered by height.
    ///
    /// Includes horizontal segments.
    pub fn entrances(&self) -> &[(f64, SegIdx)] {
        &self.enter
    }

    /// All the exit heights of segments, ordered by height.
    ///
    /// Does not include horizontal segments.
    pub fn exits(&self) -> &[(f64, SegIdx)] {
        &self.exit
    }
}

impl std::ops::Index<SegIdx> for Segments {
    type Output = Segment;

    fn index(&self, index: SegIdx) -> &Self::Output {
        &self.segs[index.0]
    }
}
