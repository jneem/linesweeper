use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ordered_float::NotNan;

use linesweeper::{
    boolean_op, topology::Topology, BooleanOp, FillRule, Float as _, Point, Segments,
};

type Float = NotNan<f64>;

fn cubes((x0, y0): (f64, f64), size: f64, offset: f64, count: usize) -> Vec<Vec<Point<Float>>> {
    let mut ret = Vec::new();
    let x0 = NotNan::try_from(x0).unwrap();
    let y0 = NotNan::try_from(y0).unwrap();
    let size = NotNan::try_from(size).unwrap();
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

fn checkerboard(n: usize) -> Vec<Vec<Point<Float>>> {
    let mut ret = cubes((0.0, 0.0), 30.0, 40.0, n);
    ret.extend(cubes((20.0, 20.0), 30.0, 40.0, n - 1));

    ret
}

fn just_the_sweep(c: &mut Criterion) {
    let contours = checkerboard(10);

    let eps = NotNan::try_from(0.01f64).unwrap();
    let mut segs = Segments::default();
    for c in contours {
        segs.add_cycle(c);
    }

    c.bench_function("just the sweep", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                linesweeper::sweep::sweep(&segs, &eps, |_, _| {})
            }
        })
    });
}

fn build_topology(c: &mut Criterion) {
    let contours = checkerboard(10);

    let eps = NotNan::try_from(0.01f64).unwrap();

    const EMPTY: [[Point<NotNan<f64>>; 0]; 0] = [];

    c.bench_function("build topology", |b| {
        b.iter(|| black_box(Topology::new(contours.clone(), EMPTY, &eps)));
    });
}

fn xor(c: &mut Criterion) {
    let contours: Vec<_> = checkerboard(10)
        .into_iter()
        .map(|ps| {
            ps.into_iter()
                .map(|p| (p.x.into_inner(), p.y.into_inner()))
                .collect()
        })
        .collect();

    c.bench_function("xor", |b| {
        b.iter(|| {
            black_box(boolean_op(
                &contours,
                &Vec::new(),
                FillRule::EvenOdd,
                BooleanOp::Xor,
            ))
        });
    });
}

criterion_group!(benches, build_topology, just_the_sweep, xor);
criterion_main!(benches);
