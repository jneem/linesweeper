#![no_main]

use arbitrary::Unstructured;
use kurbo::CubicBez;
use libfuzzer_sys::fuzz_target;
use linesweeper::curve::intersect_cubics;

fn monotonic_beziers(u: &mut Unstructured<'_>) -> Result<(CubicBez, CubicBez), arbitrary::Error> {
    let c0 = linesweeper::arbitrary::monotonic_bezier(u)?;
    let c1 = linesweeper::arbitrary::another_monotonic_bezier(u, &c0)?;
    Ok((c0, c1))
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    if let Ok((c0, c1)) = monotonic_beziers(&mut u) {
        // Filter out purely-horizontal curves.
        if c0.p0.y != c0.p3.y && c1.p0.y != c1.p3.y {
            let _ = intersect_cubics(c0, c1, 1e-6, 1e-6);
        }
    }
});
