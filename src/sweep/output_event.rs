use crate::{num::Float, SegIdx};

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

    pub(crate) fn new(
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
