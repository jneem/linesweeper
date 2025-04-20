use crate::{
    segment_tree::SegmentTree,
    topology::{OutputSegIdx, OutputSegVec},
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

    pub fn connected_components(&self) -> Vec<SegmentTree<OutputSegIdx>> {
        let mut ret = Vec::new();
        let mut seen = vec![false; self.nodes.len()];
        let mut stack: Vec<usize> = Vec::new();
        let mut component: Vec<usize> = Vec::new();

        for i in 0..self.nodes.len() {
            if seen[i] {
                continue;
            }

            component.clear();
            stack.clear();
            stack.push(i);

            while let Some(i) = stack.pop() {
                if seen[i] {
                    continue;
                }

                component.push(i);
                seen[i] = true;
                for &j in &self.edges[i] {
                    stack.push(j);
                }
            }

            debug_assert!(!component.is_empty());
            let intervals: Vec<_> = component
                .iter()
                .flat_map(|idx| {
                    let node = &self.nodes[*idx];
                    [
                        (node.y0, node.y1, node.left_seg),
                        (node.y0, node.y1, node.right_seg),
                    ]
                })
                .collect();
            ret.push(SegmentTree::new(&intervals));
        }

        ret
    }
}

#[cfg(test)]
mod tests {
    use crate::topology::OutputSegVec;

    use super::{Node, PositioningGraph};

    #[test]
    fn basic() {
        let indices = OutputSegVec::<bool>::with_size(3)
            .indices()
            .collect::<Vec<_>>();
        let s0 = indices[0];
        let s1 = indices[1];
        let s2 = indices[2];
        let nodes = vec![
            Node {
                left_seg: s0,
                right_seg: s1,
                y0: 0.0,
                y1: 1.0,
            },
            Node {
                left_seg: s1,
                right_seg: s2,
                y0: 0.5,
                y1: 1.5,
            },
        ];

        let graph = PositioningGraph::new(3, nodes);
        let components = graph.connected_components();
        assert_eq!(components.len(), 1);
        let segs = &components[0];
        let mut iter = segs.iter();
        assert_eq!(iter.next_payloads(), Some((0.0, 0.5, &[s1, s0][..])));
        assert_eq!(
            iter.next_payloads(),
            Some((0.5, 1.0, &[s1, s0, s2, s1][..]))
        );
        assert_eq!(iter.next_payloads(), Some((1.0, 1.5, &[s2, s1][..])));

        // An example with two connected components.
        let nodes = vec![
            Node {
                left_seg: s0,
                right_seg: s1,
                y0: 0.0,
                y1: 1.0,
            },
            Node {
                left_seg: s1,
                right_seg: s2,
                y0: 1.5,
                y1: 2.5,
            },
        ];
        let graph = PositioningGraph::new(3, nodes);
        let components = graph.connected_components();
        assert_eq!(components.len(), 2);
    }
}
