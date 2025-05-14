#![no_main]

use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    linesweeper::curve::arbtests::solve_for_t(&mut Unstructured::new(data)).unwrap();
});
