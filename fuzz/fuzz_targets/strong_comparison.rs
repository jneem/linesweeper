#![no_main]

use arbitrary::Unstructured;
use kurbo::Shape;
use libfuzzer_sys::fuzz_target;
use linesweeper::{
    arbitrary::{another_monotonic_bezier, float_in_range, monotonic_bezier},
    curve::{Order, intersect_cubics},
};

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let c0 = monotonic_bezier(&mut u).unwrap();
    let c1 = another_monotonic_bezier(&mut u, &c0).unwrap();
    let c2 = another_monotonic_bezier(&mut u, &c1).unwrap();

    let y0 = c0.p0.y.max(c1.p0.y).max(c2.p0.y);
    let y1 = c0.p3.y.min(c1.p3.y).min(c2.p3.y);

    if y0 >= y1 {
        return;
    }

    let bbox = c0
        .bounding_box()
        .union(c1.bounding_box())
        .union(c2.bounding_box());
    let max_coord = bbox
        .max_x()
        .max(bbox.max_y())
        .max(bbox.min_x().abs())
        .max(bbox.min_y().abs());

    let eps = 1e-6 * max_coord.max(1.0);
    let cmp01 = intersect_cubics(c0, c1, eps, eps / 2.0).with_y_slop(eps / 2.0);
    let cmp02 = intersect_cubics(c0, c2, eps, eps / 2.0).with_y_slop(eps / 2.0);
    let cmp12 = intersect_cubics(c1, c2, eps, eps / 2.0).with_y_slop(eps / 2.0);

    let y = float_in_range(y0, y1, &mut u).unwrap();
    // dbg!(&c0, &c1, &c2, y, y0, y1);
    // dbg!(&cmp01, &cmp12, &cmp02);
    let c01 = cmp01.order_at(y);
    let c12 = cmp12.order_at(y);
    let c20 = cmp02.order_at(y).flip();

    if c01 == c12 && c12 == c20 && c01 != Order::Ish {
        panic!("cycle!");
    }

    // TODO: make a stronger comparison also, and test that
});
