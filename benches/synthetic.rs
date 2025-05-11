use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use linesweeper::{
    boolean_op,
    generators::{checkerboard, slanted_checkerboard, slanties},
    topology::Topology,
    BooleanOp, FillRule, Point, Segments,
};

type Contours = Vec<Vec<Point>>;

fn just_the_sweep(c: &mut Criterion) {
    let (contours_even, contours_odd) = checkerboard(10);

    let eps = 0.01f64;
    let mut segs = Segments::default();
    segs.add_cycles(contours_even.into_iter().chain(contours_odd));

    c.bench_function("checkerboard: just the sweep", |b| {
        b.iter(|| linesweeper::sweep::sweep(&segs, eps, |_, _| {}))
    });

    let (contours_even, contours_odd) = slanted_checkerboard(10);

    let eps = 0.01f64;
    let mut segs = Segments::default();
    segs.add_cycles(contours_even.into_iter().chain(contours_odd));

    c.bench_function("slanted_checkerboard: just the sweep", |b| {
        b.iter(|| linesweeper::sweep::sweep(&segs, eps, |_, _| {}))
    });
}

fn build_topology(c: &mut Criterion) {
    let (contours_even, contours_odd) = checkerboard(10);

    let eps = 0.01f64;
    c.bench_function("checkerboard: build topology", |b| {
        b.iter(|| {
            black_box(Topology::from_polylines(
                contours_even.clone(),
                contours_odd.clone(),
                eps,
            ))
        });
    });

    let (contours_even, contours_odd) = slanted_checkerboard(10);

    let eps = 0.01f64;
    c.bench_function("slanted_checkerboard: build topology", |b| {
        b.iter(|| {
            black_box(Topology::from_polylines(
                contours_even.clone(),
                contours_odd.clone(),
                eps,
            ))
        });
    });
}

fn xor(c: &mut Criterion) {
    let to_floats = |contours: Contours| -> Vec<_> {
        contours
            .into_iter()
            .map(|ps| ps.into_iter().map(|p| (p.x, p.y)).collect())
            .collect()
    };

    let mut group = c.benchmark_group("checkerboard: xor");
    for size in [10, 100, 500] {
        let (contours_even, contours_odd) = checkerboard(size);
        let contours_even = to_floats(contours_even);
        let contours_odd = to_floats(contours_odd);

        if size > 100 {
            group.sample_size(10);
        }
        group.bench_with_input(BenchmarkId::new("linesweeper", size), &size, |b, _size| {
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

        group.bench_with_input(BenchmarkId::new("i_overlay", size), &size, |b, _size| {
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
    group.finish();

    // TODO: copy-paste
    let mut group = c.benchmark_group("slanties: xor");
    for size in [10, 100, 500] {
        let (contours_even, contours_odd) = slanties(size);
        let contours_even = to_floats(contours_even);
        let contours_odd = to_floats(contours_odd);

        if size > 100 {
            group.sample_size(10);
        }
        group.bench_with_input(BenchmarkId::new("linesweeper", size), &size, |b, _size| {
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

        group.bench_with_input(BenchmarkId::new("i_overlay", size), &size, |b, _size| {
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
    group.finish();

    let (contours_even, contours_odd) = slanted_checkerboard(10);
    let contours_even = to_floats(contours_even);
    let contours_odd = to_floats(contours_odd);

    c.bench_function("slanted_checkerboard: xor", |b| {
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

    c.bench_function("slanted_checkerboard: xor i_overlay", |b| {
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
