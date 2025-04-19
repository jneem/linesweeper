use crate::{
    topology::{OutputSegIdx, OutputSegVec},
    SegIdx,
};

fn overlaps((x0, x1): (f64, f64), (y0, y1): (f64, f64)) -> bool {
    x0 <= y1 && y0 <= x1
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Node {
    pub left_seg: OutputSegIdx,
    pub right_seg: OutputSegIdx,
    pub y0: f64,
    pub y1: f64,
}

impl Node {
    pub fn overlaps_y(&self, other: &Node) -> bool {
        overlaps((self.y0, self.y1), (other.y0, other.y1))
    }
}

pub struct PositioningGraph {
    pub nodes: Vec<Node>,
    /// An edge list representation. This vector has the same length
    /// as `nodes`, and indices point into `nodes`.
    pub edges: Vec<Vec<usize>>,
}

impl PositioningGraph {
    pub fn new(output_segs: usize, nodes: Vec<Node>) -> Self {
        let mut edges = vec![Vec::new(); nodes.len()];

        // We could have better asymptotic complexity if we made a segment tree per node.
        let mut by_node: OutputSegVec<Vec<usize>> = OutputSegVec::with_size(output_segs);
        for (i, node) in nodes.iter().enumerate() {
            for other_idx in &by_node[node.left_seg] {
                let other = &nodes[*other_idx];
                if node.overlaps_y(other) {
                    edges[i].push(*other_idx);
                    edges[*other_idx].push(i);
                }
            }

            by_node[node.left_seg].push(i);
            by_node[node.right_seg].push(i);
        }

        Self { nodes, edges }
    }
}
