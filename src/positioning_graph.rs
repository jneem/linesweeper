use crate::{topology::OutputSegIdx, SegIdx};

struct Node {
    left_seg: OutputSegIdx,
    right_seg: OutputSegIdx,
    y0: f64,
    y1: f64,
}

pub struct PositioningGraph {}
