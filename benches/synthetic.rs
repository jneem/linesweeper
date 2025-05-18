use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kurbo::BezPath;
use linesweeper::{
    boolean_op,
    generators::{checkerboard, slanted_checkerboard, slanties},
    topology::Topology,
    BooleanOp, FillRule, Point, Segments,
};

fn to_bez(points: &[Vec<Point>]) -> BezPath {
    fn one(points: &[Point]) -> BezPath {
        let mut points = points.iter();
        let p = points.next().unwrap();
        let mut ret = BezPath::default();
        ret.move_to(p.to_kurbo());
        for q in points {
            ret.line_to(q.to_kurbo());
        }
        ret.line_to(p.to_kurbo());
        ret
    }

    let mut ret = BezPath::default();
    for ps in points {
        ret.extend(one(ps));
    }
    ret
}

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
    let path_even = to_bez(&contours_even);
    let path_odd = to_bez(&contours_odd);

    let eps = 0.01f64;
    c.bench_function("checkerboard: build topology", |b| {
        b.iter(|| black_box(Topology::from_paths_binary(&path_even, &path_odd, eps)));
    });

    let (contours_even, contours_odd) = slanted_checkerboard(10);

    let eps = 0.01f64;
    c.bench_function("slanted_checkerboard: build topology", |b| {
        b.iter(|| {
            black_box(Topology::from_polylines_binary(
                contours_even.clone(),
                contours_odd.clone(),
                eps,
            ))
        });
    });
}

fn xor(c: &mut Criterion) {
    let to_float_arrays = |contours: Vec<Vec<Point>>| -> Vec<Vec<_>> {
        contours
            .into_iter()
            .map(|ps| ps.into_iter().map(|p| [p.x, p.y]).collect())
            .collect()
    };
    let mut group = c.benchmark_group("checkerboard: xor");
    for size in [10, 100, 500] {
        let (contours_even, contours_odd) = checkerboard(size);
        let path_even = to_bez(&contours_even);
        let path_odd = to_bez(&contours_odd);

        if size > 100 {
            group.sample_size(10);
        }
        group.bench_with_input(BenchmarkId::new("linesweeper", size), &size, |b, _size| {
            b.iter(|| {
                black_box(boolean_op(
                    &path_even,
                    &path_odd,
                    FillRule::EvenOdd,
                    BooleanOp::Xor,
                ))
            });
        });

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
        let path_even = to_bez(&contours_even);
        let path_odd = to_bez(&contours_odd);

        if size > 100 {
            group.sample_size(10);
        }
        group.bench_with_input(BenchmarkId::new("linesweeper", size), &size, |b, _size| {
            b.iter(|| {
                black_box(boolean_op(
                    &path_even,
                    &path_odd,
                    FillRule::EvenOdd,
                    BooleanOp::Xor,
                ))
            });
        });

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
    let path_even = to_bez(&contours_even);
    let path_odd = to_bez(&contours_odd);

    c.bench_function("slanted_checkerboard: xor", |b| {
        b.iter(|| {
            black_box(boolean_op(
                &path_even,
                &path_odd,
                FillRule::EvenOdd,
                BooleanOp::Xor,
            ))
        });
    });

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
