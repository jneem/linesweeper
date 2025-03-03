#![no_main]

use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;
use linesweeper::{
    arbitrary::{another_monotonic_bezier, float_in_range, monotonic_bezier},
    curve::Order,
    order::Comparison,
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

    let eps = 1e-6;
    let cmp01 = Comparison::new(c0, c1, eps);
    let cmp02 = Comparison::new(c0, c2, eps);
    let cmp12 = Comparison::new(c1, c2, eps);

    let y = float_in_range(y0, y1, &mut u).unwrap();
    let c01 = cmp01.order.order_at(y);
    let c12 = cmp12.order.order_at(y);
    let c20 = cmp02.order.order_at(y).flip();

    if c01 == c12 && c12 == c20 && c01 != Order::Ish {
        panic!("cycle");
    }

    let c01_bound = cmp01.bound.order_at(y);

    // If c2 crosses to the left of c0, it does so *after* crossing to the left
    // of c1.
    if c01_bound == Order::Left && c12 != Order::Right {
        if let Some((cross_y, _, _)) = cmp02
            .order
            .iter()
            .skip_while(|(start, _end, _order)| *start < y)
            .find(|(_start, _end, order)| *order == Order::Right)
        {
            if cross_y <= y1 {
                assert_eq!(cmp12.order.order_at(cross_y), Order::Right);
            }
        }
    }
});
