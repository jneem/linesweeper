use arrayvec::ArrayVec;
use serde::ser::SerializeSeq;

#[derive(Clone, Debug)]
pub struct TreeVec<T, const B: usize> {
    root: Box<Node<T, B>>,
}

impl<T: serde::Serialize, const B: usize> serde::Serialize for TreeVec<T, B> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for x in self.iter() {
            seq.serialize_element(x)?;
        }
        seq.end()
    }
}

#[derive(Clone, Debug, serde::Serialize)]
enum Node<T, const B: usize> {
    Leaf {
        data: ArrayVec<T, B>,
    },
    Internal {
        size: ArrayVec<usize, B>,
        children: ArrayVec<Box<Node<T, B>>, B>,
    },
}

enum InsertResult<T, const B: usize> {
    Done,
    Split(Box<Node<T, B>>),
}

enum RemoveResult {
    Done,
    Undersize,
}

enum MergeResult {
    Absorbed,
    Rebalanced,
}

fn child_idx(sizes: &[usize], mut offset: usize) -> Option<(usize, usize)> {
    for (idx, &size) in sizes.iter().enumerate() {
        if size > offset {
            return Some((idx, offset));
        }
        offset -= size;
    }
    None
}

impl<T, const B: usize> Node<T, B> {
    fn subtree_size(&self) -> usize {
        match self {
            Node::Leaf { data } => data.len(),
            Node::Internal { size, .. } => size.iter().copied().sum(),
        }
    }

    // offset is the offset relative to this node
    fn get(&self, offset: usize) -> Option<&T> {
        match self {
            Node::Leaf { data } => data.get(offset),
            Node::Internal { size, children } => {
                let (idx, offset) = child_idx(size, offset)?;
                children[idx].get(offset)
            }
        }
    }

    // offset is the offset relative to this node
    fn get_mut(&mut self, offset: usize) -> Option<&mut T> {
        match self {
            Node::Leaf { data } => data.get_mut(offset),
            Node::Internal { size, children } => {
                let (idx, offset) = child_idx(size, offset)?;
                children[idx].get_mut(offset)
            }
        }
    }

    fn insert(&mut self, offset: usize, element: T) -> InsertResult<T, B> {
        match self {
            Node::Leaf { data } => {
                if data.is_full() {
                    let mut second_half: ArrayVec<T, B> = data.drain(B / 2..).collect();
                    if offset <= B / 2 {
                        data.insert(offset, element)
                    } else {
                        second_half.insert(offset - B / 2, element)
                    }
                    InsertResult::Split(Box::new(Node::Leaf { data: second_half }))
                } else {
                    data.insert(offset, element);
                    InsertResult::Done
                }
            }
            Node::Internal { size, children } => {
                let (idx, offset) = if offset > 0 {
                    // unwrap: if this fails, it's out-of-bounds
                    let (idx, offset) = child_idx(size, offset - 1).unwrap();
                    (idx, offset + 1)
                } else {
                    (0, 0)
                };
                match children[idx].insert(offset, element) {
                    InsertResult::Done => {
                        size[idx] += 1;
                        InsertResult::Done
                    }
                    InsertResult::Split(node) => {
                        size[idx] = children[idx].subtree_size();

                        if children.is_full() {
                            let mut second_half_children: ArrayVec<_, B> =
                                children.drain(B / 2..).collect();
                            let mut second_half_size: ArrayVec<_, B> =
                                size.drain(B / 2..).collect();
                            if idx < B / 2 {
                                size.insert(idx + 1, node.subtree_size());
                                children.insert(idx + 1, node);
                            } else {
                                second_half_size.insert(idx + 1 - B / 2, node.subtree_size());
                                second_half_children.insert(idx + 1 - B / 2, node);
                            }
                            InsertResult::Split(Box::new(Node::Internal {
                                size: second_half_size,
                                children: second_half_children,
                            }))
                        } else {
                            size.insert(idx + 1, node.subtree_size());
                            children.insert(idx + 1, node);
                            InsertResult::Done
                        }
                    }
                }
            }
        }
    }

    fn merge_from_right(&mut self, right_sibling: &mut Node<T, B>) -> MergeResult {
        match (self, right_sibling) {
            (Node::Leaf { data: left_data }, Node::Leaf { data: right_data }) => {
                debug_assert!(right_data.len() >= left_data.len());
                if left_data.len() + right_data.len() <= B {
                    left_data.extend(right_data.drain(..));
                    MergeResult::Absorbed
                } else {
                    let count = (right_data.len() - left_data.len()) / 2;
                    debug_assert!(count > 0);
                    left_data.extend(right_data.drain(..count));
                    MergeResult::Rebalanced
                }
            }
            (
                Node::Internal {
                    size: left_size,
                    children: left_children,
                },
                Node::Internal {
                    size: right_size,
                    children: right_children,
                },
            ) => {
                if left_children.len() + right_children.len() <= B {
                    left_size.extend(right_size.drain(..));
                    left_children.extend(right_children.drain(..));
                    MergeResult::Absorbed
                } else {
                    let count = (right_children.len() - left_children.len()) / 2;
                    debug_assert!(count > 0);
                    left_children.extend(right_children.drain(..count));
                    left_size.extend(right_size.drain(..count));
                    MergeResult::Rebalanced
                }
            }
            _ => unreachable!(),
        }
    }

    fn merge_from_left(&mut self, left_sibling: &mut Node<T, B>) -> MergeResult {
        match (left_sibling, self) {
            (Node::Leaf { data: left_data }, Node::Leaf { data: right_data }) => {
                debug_assert!(right_data.len() <= left_data.len());
                if left_data.len() + right_data.len() <= B {
                    left_data.extend(right_data.drain(..));
                    std::mem::swap(left_data, right_data);
                    MergeResult::Absorbed
                } else {
                    // Unlike merge_from_right, here we only move a single element
                    // from the left to the right. This is just because safe rust
                    // makes it tricky to efficiently move more; ideally we'd also
                    // be rebalancing here.
                    right_data.insert(0, left_data.pop().unwrap());
                    MergeResult::Rebalanced
                }
            }
            (
                Node::Internal {
                    size: left_size,
                    children: left_children,
                },
                Node::Internal {
                    size: right_size,
                    children: right_children,
                },
            ) => {
                if left_children.len() + right_children.len() <= B {
                    left_size.extend(right_size.drain(..));
                    left_children.extend(right_children.drain(..));
                    std::mem::swap(left_children, right_children);
                    std::mem::swap(left_size, right_size);
                    MergeResult::Absorbed
                } else {
                    right_children.insert(0, left_children.pop().unwrap());
                    right_size.insert(0, left_size.pop().unwrap());
                    MergeResult::Rebalanced
                }
            }
            _ => unreachable!(),
        }
    }

    fn remove(&mut self, offset: usize) -> RemoveResult {
        match self {
            Node::Leaf { data } => {
                data.remove(offset);
                if data.len() < B / 2 {
                    RemoveResult::Undersize
                } else {
                    RemoveResult::Done
                }
            }
            Node::Internal { size, children } => {
                let (idx, offset) = child_idx(size, offset).unwrap();
                size[idx] -= 1;
                match children[idx].remove(offset) {
                    RemoveResult::Done => RemoveResult::Done,
                    RemoveResult::Undersize => {
                        if idx + 1 < children.len() {
                            let (a, b) = children.split_at_mut(idx + 1);
                            let cur = a.last_mut().unwrap();
                            let next = b.first_mut().unwrap();

                            match cur.merge_from_right(next) {
                                MergeResult::Absorbed => {
                                    size[idx] = cur.subtree_size();

                                    children.remove(idx + 1);
                                    size.remove(idx + 1);

                                    if children.len() < B / 2 {
                                        RemoveResult::Undersize
                                    } else {
                                        RemoveResult::Done
                                    }
                                }
                                MergeResult::Rebalanced => {
                                    size[idx] = cur.subtree_size();
                                    size[idx + 1] = next.subtree_size();
                                    RemoveResult::Done
                                }
                            }
                        } else {
                            debug_assert!(idx > 0);

                            let (a, b) = children.split_at_mut(idx);
                            let prev = a.last_mut().unwrap();
                            let cur = b.first_mut().unwrap();

                            match cur.merge_from_left(prev) {
                                MergeResult::Absorbed => {
                                    size[idx] = cur.subtree_size();

                                    children.remove(idx - 1);
                                    size.remove(idx - 1);
                                    if children.len() < B / 2 {
                                        RemoveResult::Undersize
                                    } else {
                                        RemoveResult::Done
                                    }
                                }
                                MergeResult::Rebalanced => {
                                    size[idx - 1] = prev.subtree_size();
                                    size[idx] = cur.subtree_size();
                                    RemoveResult::Done
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn check_invariants(&self, is_root: bool) {
        match self {
            Node::Leaf { data } => {
                if !is_root {
                    assert!(data.len() >= B / 2);
                }
            }
            Node::Internal { size, children } => {
                assert_eq!(size.len(), children.len());
                if !is_root {
                    assert!(size.len() >= B / 2);
                }

                for (child, size) in children.iter().zip(size) {
                    assert_eq!(child.subtree_size(), *size);

                    child.check_invariants(false);
                }
            }
        }
    }
}

impl<T, const B: usize> Default for TreeVec<T, B> {
    fn default() -> Self {
        Self {
            root: Box::new(Node::Leaf {
                data: ArrayVec::new(),
            }),
        }
    }
}

impl<T, const B: usize> TreeVec<T, B> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        match &*self.root {
            Node::Leaf { data } => data.is_empty(),
            Node::Internal { .. } => false,
        }
    }

    pub fn len(&self) -> usize {
        self.root.subtree_size()
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.root.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.root.get_mut(index)
    }

    pub fn insert(&mut self, index: usize, element: T) {
        match self.root.insert(index, element) {
            InsertResult::Done => {}
            InsertResult::Split(node) => {
                let mut root = Box::new(Node::Internal {
                    size: ArrayVec::new(),
                    children: ArrayVec::new(),
                });
                std::mem::swap(&mut root, &mut self.root);

                let Node::Internal { size, children } = &mut *self.root else {
                    unreachable!();
                };
                size.push(root.subtree_size());
                size.push(node.subtree_size());
                children.push(root);
                children.push(node);
            }
        }
    }

    pub fn remove(&mut self, index: usize) {
        self.root.remove(index);

        if let Node::Internal { children, .. } = &mut *self.root {
            if children.len() == 1 {
                // unwrap: an internal node always has children
                self.root = children.pop().unwrap()
            }
        }
    }

    pub fn check_invariants(&self) {
        self.root.check_invariants(true);
    }

    pub fn iter(&self) -> Iter<'_, T, B> {
        let mut ret = Iter {
            stack: Vec::new(),
            leaf: [].iter(),
            remaining: self.len(),
        };
        ret.descend(&*self.root);
        ret
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, T, B> {
        let mut ret = IterMut {
            stack: Vec::new(),
            leaf: [].iter_mut(),
            remaining: self.len(),
        };
        ret.descend(&mut *self.root);
        ret
    }

    pub fn partition_point<P>(&self, mut pred: P) -> usize
    where
        P: FnMut(&T) -> bool,
    {
        if let Node::Leaf { data } = &*self.root {
            return data.partition_point(pred);
        }

        // Our own slow, manual binary search. Could be sped up by using the tree structure.
        let mut end = self.len();
        if end == 0 {
            return 0;
        }
        let mut start = 0usize;

        while end > start + 1 {
            let mid = (start + end) / 2;
            let val = pred(&self[mid]);

            if val {
                start = mid;
            } else {
                end = mid;
            }
        }
        if pred(&self[start]) {
            start + 1
        } else {
            start
        }
    }

    pub fn range(&self, range: impl std::ops::RangeBounds<usize>) -> Iter<'_, T, B> {
        let start = match range.start_bound() {
            std::ops::Bound::Included(x) => *x,
            std::ops::Bound::Excluded(x) => *x + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Included(x) => *x + 1,
            std::ops::Bound::Excluded(x) => *x,
            std::ops::Bound::Unbounded => self.len(),
        };

        if end > self.len() {
            panic!("out of bounds");
        }
        let mut ret = Iter {
            stack: Vec::new(),
            leaf: [].iter(),
            remaining: end - start,
        };
        ret.descend_to(&*self.root, start);
        ret
    }

    pub fn range_mut(&mut self, range: impl std::ops::RangeBounds<usize>) -> IterMut<'_, T, B> {
        let start = match range.start_bound() {
            std::ops::Bound::Included(x) => *x,
            std::ops::Bound::Excluded(x) => *x + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Included(x) => *x + 1,
            std::ops::Bound::Excluded(x) => *x,
            std::ops::Bound::Unbounded => self.len(),
        };
        if end > self.len() {
            panic!("out of bounds");
        }
        let mut ret = IterMut {
            stack: Vec::new(),
            leaf: [].iter_mut(),
            remaining: end - start,
        };
        ret.descend_to(&mut *self.root, start);
        ret
    }
}

impl<T, const B: usize> FromIterator<T> for TreeVec<T, B> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        // TODO: a faster implementation
        let mut ret = TreeVec::new();
        for (idx, x) in iter.into_iter().enumerate() {
            ret.insert(idx, x);
        }
        ret
    }
}

impl<T, const B: usize> std::ops::Index<usize> for TreeVec<T, B> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        self.get(index).unwrap()
    }
}

impl<T, const B: usize> std::ops::IndexMut<usize> for TreeVec<T, B> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.get_mut(index).unwrap()
    }
}

pub struct Iter<'a, T, const B: usize> {
    stack: Vec<std::slice::Iter<'a, Box<Node<T, B>>>>,
    leaf: std::slice::Iter<'a, T>,
    remaining: usize,
}

impl<'a, T, const B: usize> Iter<'a, T, B> {
    fn descend(&mut self, mut node: &'a Node<T, B>) {
        loop {
            match node {
                Node::Leaf { data } => {
                    self.leaf = data.iter();
                    return;
                }
                Node::Internal { children, .. } => {
                    let mut children = children.iter();
                    // unwrap: internal nodes are always non-empty
                    node = children.next().unwrap();
                    self.stack.push(children);
                }
            }
        }
    }

    fn descend_to(&mut self, mut node: &'a Node<T, B>, mut offset: usize) {
        loop {
            match node {
                Node::Leaf { data } => {
                    self.leaf = data[offset..].iter();
                    return;
                }
                Node::Internal { children, size } => {
                    let Some((idx, child_offset)) = child_idx(size, offset) else {
                        return;
                    };
                    offset = child_offset;
                    let mut children = children[idx..].iter();
                    // unwrap: child_idx always returns a valid index into children
                    node = children.next().unwrap();
                    self.stack.push(children);
                }
            }
        }
    }
}

impl<'a, T, const B: usize> Iterator for Iter<'a, T, B> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            None
        } else {
            self.remaining -= 1;
            if let Some(ret) = self.leaf.next() {
                Some(ret)
            } else {
                loop {
                    let stack_top = self.stack.last_mut()?;

                    let Some(next_node) = stack_top.next() else {
                        self.stack.pop();
                        continue;
                    };

                    self.descend(next_node);
                    return self.leaf.next();
                }
            }
        }
    }
}

pub struct IterMut<'a, T, const B: usize> {
    stack: Vec<std::slice::IterMut<'a, Box<Node<T, B>>>>,
    leaf: std::slice::IterMut<'a, T>,
    remaining: usize,
}

impl<'a, T, const B: usize> IterMut<'a, T, B> {
    fn descend(&mut self, mut node: &'a mut Node<T, B>) {
        loop {
            match node {
                Node::Leaf { data } => {
                    self.leaf = data.iter_mut();
                    return;
                }
                Node::Internal { children, .. } => {
                    let mut children = children.iter_mut();
                    // unwrap: internal nodes are always non-empty
                    node = children.next().unwrap();
                    self.stack.push(children);
                }
            }
        }
    }

    fn descend_to(&mut self, mut node: &'a mut Node<T, B>, mut offset: usize) {
        loop {
            match node {
                Node::Leaf { data } => {
                    self.leaf = data[offset..].iter_mut();
                    return;
                }
                Node::Internal { children, size } => {
                    let Some((idx, child_offset)) = child_idx(size, offset) else {
                        return;
                    };
                    offset = child_offset;
                    let mut children = children[idx..].iter_mut();
                    // unwrap: child_idx always returns a valid index into children
                    node = children.next().unwrap();
                    self.stack.push(children);
                }
            }
        }
    }
}

impl<'a, T, const B: usize> Iterator for IterMut<'a, T, B> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            None
        } else {
            self.remaining -= 1;
            if let Some(ret) = self.leaf.next() {
                Some(ret)
            } else {
                loop {
                    let stack_top = self.stack.last_mut()?;

                    let Some(next_node) = stack_top.next() else {
                        self.stack.pop();
                        continue;
                    };

                    self.descend(next_node);
                    return self.leaf.next();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get() {
        let mut vec = TreeVec::<i32, 4>::default();
        vec.insert(0, 1);
        vec.insert(0, 2);
        vec.insert(0, 3);
        vec.insert(0, 4);
        assert_eq!(*vec.get(0).unwrap(), 4);
        assert_eq!(*vec.get(1).unwrap(), 3);
        assert_eq!(*vec.get(2).unwrap(), 2);
        assert_eq!(*vec.get(3).unwrap(), 1);
        vec.check_invariants();

        vec.insert(0, 1);
        vec.insert(0, 2);
        vec.insert(0, 3);
        vec.insert(0, 4);
        vec.check_invariants();
        assert_eq!(*vec.get(0).unwrap(), 4);
        assert_eq!(*vec.get(1).unwrap(), 3);
        assert_eq!(*vec.get(2).unwrap(), 2);
        assert_eq!(*vec.get(3).unwrap(), 1);
        assert_eq!(*vec.get(4).unwrap(), 4);
        assert_eq!(*vec.get(5).unwrap(), 3);
        assert_eq!(*vec.get(6).unwrap(), 2);
        assert_eq!(*vec.get(7).unwrap(), 1);
    }

    #[test]
    fn insert_remove() {
        let mut vec = TreeVec::<i32, 4>::default();
        vec.insert(0, 1);
        vec.insert(0, 2);
        vec.insert(0, 3);
        vec.insert(0, 4);
        vec.remove(1);
        vec.check_invariants();
        assert_eq!(*vec.get(0).unwrap(), 4);
        assert_eq!(*vec.get(1).unwrap(), 2);
        assert_eq!(*vec.get(2).unwrap(), 1);

        vec.insert(0, 1);
        vec.insert(0, 2);
        vec.insert(0, 3);
        vec.insert(0, 4);
        vec.remove(5);
        vec.check_invariants();
        assert_eq!(*vec.get(0).unwrap(), 4);
        assert_eq!(*vec.get(1).unwrap(), 3);
        assert_eq!(*vec.get(2).unwrap(), 2);
        assert_eq!(*vec.get(3).unwrap(), 1);
        assert_eq!(*vec.get(4).unwrap(), 4);
        assert_eq!(*vec.get(5).unwrap(), 1);
    }

    #[test]
    fn iter() {
        let mut vec = TreeVec::<i32, 4>::default();
        assert_eq!(vec.iter().cloned().collect::<Vec<_>>(), Vec::<i32>::new());
        vec.insert(0, 1);
        assert_eq!(vec.iter().cloned().collect::<Vec<_>>(), vec![1]);
        vec.insert(1, 2);
        assert_eq!(vec.iter().cloned().collect::<Vec<_>>(), vec![1, 2]);
        vec.insert(0, 1);
        vec.insert(0, 1);
        vec.insert(0, 1);
        vec.insert(0, 1);
        vec.insert(0, 1);
        assert_eq!(
            vec.iter().cloned().collect::<Vec<_>>(),
            vec![1, 1, 1, 1, 1, 1, 2]
        );
    }
}
