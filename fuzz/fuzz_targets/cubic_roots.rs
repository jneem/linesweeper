#![no_main]

use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let c = linesweeper::arbitrary::cubic(1e8, &mut u).unwrap();

    // How much relative accuracy do we expect?
    let accuracy = 1e-11;
    let range = linesweeper::arbitrary::float_in_range(1.0, 1e4, &mut u).unwrap();

    let size =
        c.c3.abs() * range.powi(3) + c.c2.abs() * range.powi(2) + c.c1.abs() * range + c.c0.abs();
    let threshold = accuracy * size;
    let roots = c.roots_between(-range, range, threshold);
    if c.eval(-range).signum() != c.eval(range).signum() {
        assert!(!roots.is_empty());
    }
    for r in roots {
        assert!(c.eval(r).abs() <= threshold);
    }
});
