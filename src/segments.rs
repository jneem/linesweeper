use crate::{
    geom::{Point, Segment},
    num::Float,
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
#[derive(Debug, Clone)]
pub struct Segments<F: Float> {
    segs: Vec<Segment<F>>,
    contour_prev: Vec<Option<SegIdx>>,
    contour_next: Vec<Option<SegIdx>>,
    /// For each segment, stores true if the sweep-line order (small y to big y)
    /// is the same as the orientation in its original contour.
    orientation: Vec<bool>,
}

impl<F: Float> Default for Segments<F> {
    fn default() -> Self {
        Self {
            segs: Default::default(),
            contour_prev: Default::default(),
            contour_next: Default::default(),
            orientation: Default::default(),
        }
    }
}

fn cyclic_pairs<T>(xs: &[T]) -> impl Iterator<Item = (&T, &T)> {
    pairs(xs).chain(xs.last().zip(xs.first()))
}

fn pairs<T>(xs: &[T]) -> impl Iterator<Item = (&T, &T)> {
    xs.windows(2).map(|pair| (&pair[0], &pair[1]))
}

impl<F: Float> Segments<F> {
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
    pub fn segments(&self) -> impl Iterator<Item = &Segment<F>> {
        self.segs.iter()
    }

    /// Returns the starting point of the segment at `idx`, relative to the segment's original orientation.
    ///
    /// The start and end points of the segment itself are stored in sweep-line
    /// order (i.e. `start` has the smaller `y` coordinate), regardless of the
    /// original orientation of the segment. Use this method to retrieve the
    /// segment's original start point.
    pub fn oriented_start(&self, idx: SegIdx) -> &Point<F> {
        if self.orientation[idx.0] {
            &self[idx].start
        } else {
            &self[idx].end
        }
    }

    /// Returns the ending point of the segment at `idx`, relative to the segment's original orientation.
    ///
    /// The start and end points of the segment itself are stored in sweep-line
    /// order (i.e. `start` has the smaller `y` coordinate), regardless of the
    /// original orientation of the segment. Use this method to retrieve the
    /// segment's original end point.
    pub fn oriented_end(&self, idx: SegIdx) -> &Point<F> {
        if self.orientation[idx.0] {
            &self[idx].end
        } else {
            &self[idx].start
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
    pub fn add_points<P: Into<Point<F>>>(&mut self, ps: impl IntoIterator<Item = P>) {
        let old_len = self.segs.len();

        let ps: Vec<_> = ps.into_iter().map(|p| p.into()).collect();
        if ps.len() <= 1 {
            return;
        }

        for (p, q) in pairs(&ps) {
            let (a, b, orient) = if p < q { (p, q, true) } else { (q, p, false) };
            self.segs.push(Segment::new(a.clone(), b.clone()));
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
    }

    /// Add a closed polyline to this arena.
    pub fn add_cycle<P: Into<Point<F>>>(&mut self, ps: impl IntoIterator<Item = P>) {
        let old_len = self.segs.len();

        let ps: Vec<_> = ps.into_iter().map(|p| p.into()).collect();
        if ps.len() <= 1 {
            return;
        }

        for (p, q) in cyclic_pairs(&ps) {
            let (a, b, orient) = if p < q { (p, q, true) } else { (q, p, false) };
            self.segs.push(Segment::new(a.clone(), b.clone()));
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
    pub fn from_closed_cycle<P: Into<Point<F>>>(ps: impl IntoIterator<Item = P>) -> Self {
        let mut ret = Self::default();
        ret.add_cycle(ps);
        ret
    }
}

impl<F: Float> std::ops::Index<SegIdx> for Segments<F> {
    type Output = Segment<F>;

    fn index(&self, index: SegIdx) -> &Self::Output {
        &self.segs[index.0]
    }
}
