//! The sweep-line implementation.
//!
//! The details of this implementation are described in the `docs` directory.
//The main entry ! point is [`Sweeper`], which computes all the intersection
//points between a collection of ! line segments, and makes them available
//sweep-line by sweep-line.

mod weak_ordering;

pub use weak_ordering::{
    sweep, ChangedInterval, OutputEvent, SegmentsConnectedAtX, SweepLine, SweepLineRange, Sweeper,
};
