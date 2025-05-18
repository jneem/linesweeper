use criterion::{black_box, criterion_group, criterion_main, Criterion};

use kurbo::BezPath;
use linesweeper::{boolean_op, topology::Topology, BooleanOp, FillRule, Point, Segments};
use linesweeper_util::svg_to_bezpaths;

fn svg_to_contours(tree: &usvg::Tree) -> Vec<Vec<Point>> {
    let mut ret = Vec::new();

    fn pt(p: usvg::tiny_skia_path::Point) -> kurbo::Point {
        kurbo::Point::new(p.x as f64, p.y as f64)
    }

    fn add_group(group: &usvg::Group, ret: &mut Vec<Vec<Point>>) {
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

                    let mut points = Vec::<Point>::new();
                    kurbo::flatten(kurbo_els, 1e-6, |el| match el {
                        kurbo::PathEl::MoveTo(p) => {
                            // Even if it wasn't closed in the svg, we close it.
                            if !points.is_empty() {
                                ret.push(points.split_off(0));
                            }
                            points.push(Point::new(p.x, p.y));
                        }
                        kurbo::PathEl::LineTo(p) => {
                            points.push(Point::new(p.x, p.y));
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
    let path = svg_to_bezpaths(&tree)
        .into_iter()
        .flatten()
        .collect::<BezPath>();

    let mut segs = Segments::default();
    segs.add_bez_path(&path);

    c.bench_function("logo: just the sweep", |b| {
        b.iter(|| linesweeper::sweep::sweep(&segs, 0.01, |_, _| {}))
    });
}

fn build_topology(c: &mut Criterion) {
    let input = include_str!("../examples/linebender.svg");
    let tree = usvg::Tree::from_str(input, &usvg::Options::default()).unwrap();
    let path = svg_to_bezpaths(&tree)
        .into_iter()
        .flatten()
        .collect::<BezPath>();

    c.bench_function("logo: build topology", |b| {
        b.iter(|| black_box(Topology::from_paths_binary(&path, &BezPath::new(), 0.01)));
    });
}

fn xor(c: &mut Criterion) {
    let input = include_str!("../examples/linebender.svg");
    let tree = usvg::Tree::from_str(input, &usvg::Options::default()).unwrap();
    let contours = svg_to_contours(&tree);
    let bezpaths = svg_to_bezpaths(&tree);

    let first_path = bezpaths.first().unwrap().clone();
    let second_path = bezpaths[1..]
        .iter()
        .flat_map(|b| b.elements().iter().cloned())
        .collect();

    c.bench_function("logo: xor", |b| {
        b.iter(|| {
            black_box(boolean_op(
                &first_path,
                &second_path,
                FillRule::EvenOdd,
                BooleanOp::Xor,
            ))
        });
    });

    let to_float_arrays = |contours: Vec<Vec<Point>>| -> Vec<Vec<_>> {
        contours
            .into_iter()
            .map(|ps| ps.into_iter().map(|p| [p.x, p.y]).collect())
            .collect()
    };
    let first_contour = to_float_arrays(vec![contours.first().unwrap().clone()]);
    let other_contours = to_float_arrays(contours[1..].to_vec());

    c.bench_function("logo: xor i_overlay", |b| {
        b.iter(|| {
            use i_overlay::float::single::SingleFloatOverlay;
            first_contour.overlay(
                &other_contours,
                i_overlay::core::overlay_rule::OverlayRule::Xor,
                i_overlay::core::fill_rule::FillRule::EvenOdd,
            );
        });
    });
}

criterion_group!(benches, build_topology, just_the_sweep, xor);
criterion_main!(benches);
