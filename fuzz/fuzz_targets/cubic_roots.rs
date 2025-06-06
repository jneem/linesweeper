#![no_main]

use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    linesweeper::curve::arbtests::cubic_roots(&mut Unstructured::new(data)).unwrap();
});
