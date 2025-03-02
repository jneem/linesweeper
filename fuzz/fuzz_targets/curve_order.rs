#![no_main]

use arbitrary::Unstructured;
use kurbo::{CubicBez, Point};
use libfuzzer_sys::fuzz_target;
use linesweeper::curve::intersect_cubics;

fn float_in_range(start: f64, end: f64, u: &mut Unstructured<'_>) -> f64 {
    let num: u32 = u.arbitrary().unwrap();
    let t = num as f64 / u32::MAX as f64;
    (1.0 - t) * start + t * end
}

fn monotonic_beziers(u: &mut Unstructured<'_>) -> Result<(CubicBez, CubicBez), arbitrary::Error> {
    let same_start: bool = u.arbitrary()?;
    let same_start_tangent: bool = u.arbitrary()?;
    let same_end: bool = u.arbitrary()?;
    let same_end_tangent: bool = u.arbitrary()?;
    let horizontal_start: bool = u.arbitrary()?;
    let horizontal_end: bool = u.arbitrary()?;

    let p0 = Point::new(0.5, 0.0);
    let q0 = if same_start {
        p0
    } else {
        Point::new(float_in_range(0.0, 1.0, u), float_in_range(0.0, 0.1, u))
    };

    let p1 = if horizontal_start {
        Point::new(float_in_range(0.0, 1.0, u), 0.0)
    } else {
        Point::new(float_in_range(0.0, 1.0, u), float_in_range(0.0, 1.0, u))
    };

    let q1 = if same_start && same_start_tangent {
        p1
    } else {
        Point::new(float_in_range(0.0, 1.0, u), float_in_range(q0.y, 1.0, u))
    };

    let p2 = if horizontal_end {
        Point::new(float_in_range(0.0, 1.0, u), 1.0)
    } else {
        Point::new(float_in_range(0.0, 1.0, u), float_in_range(p1.y, 1.0, u))
    };

    let q2 = if same_end && same_end_tangent && p2.y >= q1.y {
        p2
    } else {
        Point::new(float_in_range(0.0, 1.0, u), float_in_range(q1.y, 1.0, u))
    };

    let p3 = Point::new(float_in_range(0.0, 1.0, u), 1.0);
    let q3 = if same_end {
        p3
    } else {
        Point::new(float_in_range(0.0, 1.0, u), float_in_range(q2.y, 1.0, u))
    };

    Ok((CubicBez::new(p0, p1, p2, p3), CubicBez::new(q0, q1, q2, q3)))
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
