// Could probably halve the size of this by
// compressing the payload indices to u32.
#[derive(Clone, Debug, Default)]
struct SegmentTreeNode {
    // TODO: could probably have some niche optimization here
    x: Option<f64>,
    payload_idx: usize,
    payload_len: usize,
}

#[derive(Clone, Debug)]
pub struct SegmentTree<T> {
    tree: Vec<SegmentTreeNode>,
    payloads: Vec<T>,
    min: f64,
    max: f64,
}

fn fill_heap(heap: &mut [SegmentTreeNode], heap_idx: usize, values: &[f64]) {
    if values.is_empty() {
        return;
    }

    let idx = values.len() / 2;
    heap[heap_idx].x = Some(values[idx]);

    fill_heap(heap, heap_idx * 2 + 1, &values[..idx]);
    fill_heap(heap, heap_idx * 2 + 2, &values[(idx + 1)..]);
}

fn find_nodes(
    tree: &[SegmentTreeNode],
    interval: (f64, f64),
    tree_bounds: (f64, f64),
    out: &mut Vec<usize>,
) {
    fn find_nodes_inner(
        tree: &[SegmentTreeNode],
        idx: usize,
        interval: (f64, f64),
        bounds: (f64, f64),
        out: &mut Vec<usize>,
    ) {
        if interval.0 <= bounds.0 && bounds.1 <= interval.1 {
            out.push(idx);
            return;
        }

        let x = tree[idx].x.unwrap();
        if x > interval.0 {
            find_nodes_inner(tree, 2 * idx + 1, interval, (bounds.0, x), out);
        }
        if x < interval.1 {
            find_nodes_inner(tree, 2 * idx + 2, interval, (x, bounds.1), out);
        }
    }

    find_nodes_inner(tree, 0, interval, tree_bounds, out)
}

impl<T: Clone> SegmentTree<T> {
    pub fn new(intervals: &[(f64, f64, T)]) -> Self {
        let mut points: Vec<f64> = Vec::with_capacity(intervals.len() * 2);
        points.extend(intervals.iter().map(|(start, _, _)| start));
        points.extend(intervals.iter().map(|(_, end, _)| end));
        points.sort_by(|x, y| x.partial_cmp(y).unwrap());
        points.dedup();

        debug_assert!(points.len() >= 2);
        let min = points[0];
        let max = *points.last().unwrap();

        let mut tree = vec![SegmentTreeNode::default(); (2 * points.len() - 3).next_power_of_two()];
        fill_heap(&mut tree, 0, &points[1..points.len() - 1]);

        let mut buf = Vec::new();
        for (x0, x1, _) in intervals {
            buf.clear();
            find_nodes(&tree, (*x0, *x1), (min, max), &mut buf);
            for &idx in &buf {
                tree[idx].payload_len += 1;
            }
        }

        let mut sum = 0;
        for node in &mut tree {
            sum += node.payload_len;
            node.payload_idx = sum;
        }

        // TODO: could do some sort of MaybeUninitialized dance?
        let mut payloads = vec![intervals[0].2.clone(); sum];
        for (x0, x1, payload) in intervals {
            buf.clear();
            find_nodes(&tree, (*x0, *x1), (min, max), &mut buf);
            for &idx in &buf {
                let node = &mut tree[idx];
                node.payload_idx = node.payload_idx.checked_sub(1).unwrap();
                payloads[node.payload_idx] = payload.clone();
            }
        }

        Self {
            tree,
            payloads,
            min,
            max,
        }
    }

    pub fn iter(&self) -> SegmentTreeIter<T> {
        SegmentTreeIter {
            tree: self,
            payload: Vec::new(),
            stack: Vec::new(),
        }
    }
}

pub struct SegmentTreeIter<'a, T> {
    tree: &'a SegmentTree<T>,
    payload: Vec<T>,
    // The stack points to the heap node that we've just visited (whose
    // elements are at the end of the `payload` array).
    stack: Vec<(usize, f64, f64)>,
}

impl<T: Clone> SegmentTreeIter<'_, T> {
    pub fn next_payloads(&mut self) -> Option<(f64, f64, &[T])> {
        // We don't distinguish between "we're done" and  "we haven't started yet".
        // In other words, this isn't a "fused" iterator.
        if self.stack.is_empty() {
            self.fill_stack_to_leaf(0, (self.tree.min, self.tree.max));
        } else if let Some(next_idx) = self.empty_stack_to_unfinished_parent() {
            let &(parent_idx, _, parent_end) = self.stack.last().unwrap();
            self.fill_stack_to_leaf(
                next_idx,
                (self.tree.tree[parent_idx].x.unwrap(), parent_end),
            );
        } else {
            return None;
        }

        let &(_, start, end) = self.stack.last().unwrap();
        Some((start, end, self.payload.as_slice()))
    }

    fn fill_stack_to_leaf(&mut self, idx: usize, interval: (f64, f64)) {
        if idx >= self.tree.tree.len() {
            return;
        }
        let node = &self.tree.tree[idx];
        self.stack.push((idx, interval.0, interval.1));
        self.payload.extend_from_slice(
            &self.tree.payloads[node.payload_idx..(node.payload_idx + node.payload_len)],
        );

        if let Some(x) = node.x {
            self.fill_stack_to_leaf(2 * idx + 1, (interval.0, x));
        }
    }

    fn empty_stack_to_unfinished_parent(&mut self) -> Option<usize> {
        let idx = self.stack.pop()?.0;
        let payload_len = self.tree.tree[idx].payload_len;
        self.payload
            .truncate(self.payload.len().checked_sub(payload_len).unwrap());

        // A left child is always an odd index. If the right sibling is
        // non-empty, we've found where to stop unwinding the stack.
        if (idx % 2) != 0 && idx + 1 < self.tree.tree.len() {
            return Some(idx + 1);
        }

        self.empty_stack_to_unfinished_parent()
    }
}

#[cfg(test)]
mod tests {
    use super::SegmentTree;
    use proptest::prelude::*;

    #[test]
    fn basic() {
        let intervals = [(0.0, 10.0, 0), (1.0, 9.0, 1), (0.0, 2.0, 2)];
        let tree = SegmentTree::new(&intervals);
        let mut iter = tree.iter();
        assert_eq!(iter.next_payloads(), Some((0.0, 1.0, [0, 2].as_slice())));
        assert_eq!(iter.next_payloads(), Some((1.0, 2.0, [0, 2, 1].as_slice())));
        assert_eq!(iter.next_payloads(), Some((2.0, 9.0, [0, 1].as_slice())));
        assert_eq!(iter.next_payloads(), Some((9.0, 10.0, [0].as_slice())));
    }

    proptest! {
        #[test]
        fn segment_tree(mut intervals: Vec<(f64, f64)>) {
            intervals.retain_mut(|(x0, x1)| {
                if !x0.is_finite() || !x1.is_finite() || x0 == x1 {
                    return false;
                }
                let min = x0.min(*x1);
                let max = x0.max(*x1);

                *x0 = min;
                *x1 = max;
                true
            });

            if intervals.is_empty() {
                return Ok(());
            }

            let indexed = intervals.into_iter().enumerate().map(|(i, (x0, x1))| (x0, x1, i)).collect::<Vec<_>>();
            let tree = SegmentTree::new(&indexed);
            let mut iter = tree.iter();
            while let Some((x0, x1, payloads)) = iter.next_payloads() {
                let mut payloads_found = payloads.to_vec();
                payloads_found.sort();

                let mut payloads_expected = indexed.iter().filter_map(|&(y0, y1, i)| {
                    (y0 <= x0 && x1 <= y1).then_some(i)
                }).collect::<Vec<_>>();
                payloads_expected.sort();
                assert_eq!(payloads_found, payloads_expected);
            }
        }
    }
}
