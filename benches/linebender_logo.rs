use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ordered_float::NotNan;

use linesweeper::{topology::Topology, Point, Segments};

type Float = NotNan<f64>;

fn svg_to_contours(tree: &usvg::Tree) -> Vec<Vec<Point<Float>>> {
    let mut ret = Vec::new();

    fn pt(p: usvg::tiny_skia_path::Point) -> kurbo::Point {
        kurbo::Point::new(p.x as f64, p.y as f64)
    }

    fn add_group(group: &usvg::Group, ret: &mut Vec<Vec<Point<Float>>>) {
        for child in group.children() {
            match child {
                usvg::Node::Group(group) => add_group(group, ret),
                usvg::Node::Path(path) => {
                    let data = path.data().clone().transform(path.abs_transform()).unwrap();
                    let kurbo_els = data.segments().map(|seg| match seg {
                        usvg::tiny_skia_path::PathSegment::MoveTo(p) => {
                            kurbo::PathEl::MoveTo(pt(p))
                        }
                        usvg::tiny_skia_path::PathSegment::LineTo(p) => {
                            kurbo::PathEl::LineTo(pt(p))
                        }
                        usvg::tiny_skia_path::PathSegment::QuadTo(p0, p1) => {
                            kurbo::PathEl::QuadTo(pt(p0), pt(p1))
                        }
                        usvg::tiny_skia_path::PathSegment::CubicTo(p0, p1, p2) => {
                            kurbo::PathEl::CurveTo(pt(p0), pt(p1), pt(p2))
                        }
                        usvg::tiny_skia_path::PathSegment::Close => kurbo::PathEl::ClosePath,
                    });

                    let mut points = Vec::<Point<Float>>::new();
                    kurbo::flatten(kurbo_els, 1e-6, |el| match el {
                        kurbo::PathEl::MoveTo(p) => {
                            // Even if it wasn't closed in the svg, we close it.
                            if !points.is_empty() {
                                ret.push(points.split_off(0));
                            }
                            points
                                .push(Point::new(p.x.try_into().unwrap(), p.y.try_into().unwrap()));
                        }
                        kurbo::PathEl::LineTo(p) => {
                            points
                                .push(Point::new(p.x.try_into().unwrap(), p.y.try_into().unwrap()));
                        }
                        kurbo::PathEl::ClosePath => {
                            let p = points.first().cloned();
                            if !points.is_empty() {
                                ret.push(points.split_off(0));
                            }
                            if let Some(p) = p {
                                points.push(p);
                            }
                        }
                        kurbo::PathEl::QuadTo(..) | kurbo::PathEl::CurveTo(..) => unreachable!(),
                    });

                    if !points.is_empty() {
                        ret.push(points);
                    }
                }
                _ => {}
            }
        }
    }

    add_group(tree.root(), &mut ret);
    ret
}

fn just_the_sweep(c: &mut Criterion) {
    let input = include_str!("../examples/linebender.svg");
    let tree = usvg::Tree::from_str(input, &usvg::Options::default()).unwrap();
    let contours = svg_to_contours(&tree);

    let eps = NotNan::try_from(0.01f64).unwrap();
    let mut segs = Segments::default();
    for c in contours {
        segs.add_cycle(c);
    }

    c.bench_function("just the sweep", |b| {
        b.iter(|| linesweeper::sweep::sweep(&segs, &eps, |_, _| {}))
    });
}

fn build_topology(c: &mut Criterion) {
    let input = include_str!("../examples/linebender.svg");
    let tree = usvg::Tree::from_str(input, &usvg::Options::default()).unwrap();
    let contours = svg_to_contours(&tree);

    let eps = NotNan::try_from(0.01f64).unwrap();

    const EMPTY: [[Point<NotNan<f64>>; 0]; 0] = [];

    c.bench_function("build topology", |b| {
        b.iter(|| black_box(Topology::new(contours.clone(), EMPTY, &eps)));
    });
}

criterion_group!(benches, build_topology, just_the_sweep);
criterion_main!(benches);
