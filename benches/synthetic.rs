use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ordered_float::OrderedFloat;

use linesweeper::{
    boolean_op, topology::Topology, BooleanOp, FillRule, Float as _, Point, Segments,
};

type Float = OrderedFloat<f64>;
type Contours = Vec<Vec<Point<Float>>>;

fn cubes((x0, y0): (f64, f64), size: f64, offset: f64, count: usize) -> Contours {
    let mut ret = Vec::new();
    let x0 = OrderedFloat::from(x0);
    let y0 = OrderedFloat::from(y0);
    let size = OrderedFloat::from(size);
    for i in 0..count {
        let x = x0 + Float::from_f32(i as f32) * offset;
        for j in 0..count {
            let y = y0 + Float::from_f32(j as f32) * offset;
            ret.push(vec![
                Point::new(x, y),
                Point::new(x, y + size),
                Point::new(x + size, y + size),
                Point::new(x + size, y),
            ]);
        }
    }

    ret
}

fn checkerboard(n: usize) -> (Contours, Contours) {
    (
        cubes((0.0, 0.0), 30.0, 40.0, n),
        cubes((20.0, 20.0), 30.0, 40.0, n - 1),
    )
}

fn just_the_sweep(c: &mut Criterion) {
    let (contours_even, contours_odd) = checkerboard(10);

    let eps = OrderedFloat::from(0.01f64);
    let mut segs = Segments::default();
    for c in contours_even.into_iter().chain(contours_odd) {
        segs.add_cycle(c);
    }

    c.bench_function("just the sweep", |b| {
        b.iter(|| linesweeper::sweep::sweep(&segs, &eps, |_, _| {}))
    });
    // c.bench_function("just the sweep", |b| {
    //     b.iter(|| {
    //         for _ in 0..10_000 {
    //             linesweeper::sweep::sweep(&segs, &eps, |_, _| {})
    //         }
    //     })
    // });
}

fn build_topology(c: &mut Criterion) {
    let (contours_even, contours_odd) = checkerboard(10);

    let eps = OrderedFloat::from(0.01f64);
    c.bench_function("build topology", |b| {
        b.iter(|| {
            black_box(Topology::new(
                contours_even.clone(),
                contours_odd.clone(),
                &eps,
            ))
        });
    });
}

fn xor(c: &mut Criterion) {
    let to_floats = |contours: Contours| -> Vec<_> {
        contours
            .into_iter()
            .map(|ps| {
                ps.into_iter()
                    .map(|p| (p.x.into_inner(), p.y.into_inner()))
                    .collect()
            })
            .collect()
    };

    let (contours_even, contours_odd) = checkerboard(10);
    let contours_even = to_floats(contours_even);
    let contours_odd = to_floats(contours_odd);

    c.bench_function("xor", |b| {
        b.iter(|| {
            black_box(boolean_op(
                &contours_even,
                &contours_odd,
                FillRule::EvenOdd,
                BooleanOp::Xor,
            ))
        });
    });

    let to_float_arrays = |contours: Vec<Vec<(f64, f64)>>| -> Vec<Vec<_>> {
        contours
            .into_iter()
            .map(|ps| ps.into_iter().map(|(x, y)| [x, y]).collect())
            .collect()
    };
    let contours_even = to_float_arrays(contours_even);
    let contours_odd = to_float_arrays(contours_odd);

    c.bench_function("xor i_overlay", |b| {
        b.iter(|| {
            use i_overlay::float::single::SingleFloatOverlay;
            contours_even.overlay(
                &contours_odd,
                i_overlay::core::overlay_rule::OverlayRule::Xor,
                i_overlay::core::fill_rule::FillRule::EvenOdd,
            );
        });
    });
}

criterion_group!(benches, build_topology, just_the_sweep, xor);
criterion_main!(benches);
