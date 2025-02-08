use crate::{num::Float, SegIdx};

use super::{ChangedInterval, OutputEvent, SweepLine};

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

/// Emits output events for a single sub-range of a single sweep-line.
///
/// This is constructed using [`SweepLine::next_range`]. By repeatedly
/// calling `SweepLineRange::increase_x` you can iterate over all
/// interesting horizontal positions, left to right (i.e. smaller `x` to larger
/// `x`).
#[derive(Debug)]
pub struct SweepLineRange<'bufs, 'state, 'segs, F: Float> {
    last_x: Option<F>,
    line: &'state SweepLine<'state, 'state, 'segs, F>,
    bufs: &'bufs mut SweepLineRangeBuffers<F>,
    changed_interval: ChangedInterval,
    output_events: &'state [OutputEvent<F>],
}

impl<'bufs, 'state, 'segs, F: Float> SweepLineRange<'bufs, 'state, 'segs, F> {
    pub(crate) fn new(
        line: &'state SweepLine<'state, 'state, 'segs, F>,
        output_events: &'state [OutputEvent<F>],
        bufs: &'bufs mut SweepLineRangeBuffers<F>,
        changed_interval: ChangedInterval,
    ) -> Self {
        Self {
            last_x: None,
            output_events,
            changed_interval,
            line,
            bufs,
        }
    }

    fn output_events(&self) -> &[OutputEvent<F>] {
        self.output_events
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
        while let Some(ev) = self.output_events.first() {
            if ev.smaller_x() > &next_x {
                break;
            }
            self.output_events = &self.output_events[1..];

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

            while let Some(ev) = self.output_events.first() {
                if ev.smaller_x() > &x {
                    break;
                }
                self.output_events = &self.output_events[1..];

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
    pub fn line(&self) -> &SweepLine<'_, '_, 'segs, F> {
        self.line
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

impl<F: Float> SweepLineRangeBuffers<F> {
    pub(crate) fn clear(&mut self) {
        self.active_horizontals.clear();
    }
}
