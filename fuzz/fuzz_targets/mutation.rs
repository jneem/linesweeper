#![no_main]

use arbitrary::{Arbitrary, Unstructured};

use libfuzzer_sys::fuzz_target;
use linesweeper::treevec::TreeVec;

#[derive(Arbitrary, Debug)]
enum Op {
    Insert { idx: usize, val: i32 },
    Remove { idx: usize },
}

impl Op {
    fn apply_to_vec(&self, vec: &mut Vec<i32>) {
        match self {
            Op::Insert { idx, val } => {
                vec.insert(*idx % (vec.len() + 1), *val);
            }
            Op::Remove { idx } => {
                if !vec.is_empty() {
                    vec.remove(*idx % vec.len());
                }
            }
        }
    }

    fn apply_to_treevec<const B: usize>(&self, vec: &mut TreeVec<i32, B>) {
        match self {
            Op::Insert { idx, val } => {
                vec.insert(*idx % (vec.len() + 1), *val);
            }
            Op::Remove { idx } => {
                if !vec.is_empty() {
                    vec.remove(*idx % vec.len());
                }
            }
        }
    }
}

fn arbitrary_ops(mut u: Unstructured) -> Result<(), arbitrary::Error> {
    let len = u.arbitrary_len::<Op>()?;
    let mut vec = Vec::new();
    let mut tree_vec = TreeVec::<_, 4>::new();
    for _ in 0..len {
        let op: Op = u.arbitrary()?;
        op.apply_to_vec(&mut vec);
        op.apply_to_treevec(&mut tree_vec);
        tree_vec.check_invariants();

        assert_eq!(tree_vec.len(), vec.len());
        assert_eq!(tree_vec.iter().cloned().collect::<Vec<_>>(), vec);
    }
    Ok(())
}

fuzz_target!(|data: &[u8]| {
    let u = Unstructured::new(data);
    let _ = arbitrary_ops(u);
});
