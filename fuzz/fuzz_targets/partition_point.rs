#![no_main]

use arbitrary::Unstructured;

use libfuzzer_sys::fuzz_target;
use linesweeper::treevec::TreeVec;

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let len = u.arbitrary_len::<i32>().unwrap();
    let mut vec: Vec<i32> = std::iter::repeat_with(|| u.arbitrary().unwrap())
        .take(len)
        .collect();
    vec.sort();

    let tree_vec = vec.iter().copied().collect::<TreeVec<i32, 4>>();

    let search: i32 = u.arbitrary().unwrap();

    assert_eq!(
        vec.partition_point(|x| x <= &search),
        tree_vec.partition_point(|x| x <= &search)
    );
});
