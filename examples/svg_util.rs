//! Utilities for the examples that work with svg.

use linesweeper::Point;

pub fn svg_to_contours(tree: &usvg::Tree) -> Vec<Vec<Point>> {
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
                    kurbo::flatten(kurbo_els, 1e-3, |el| match el {
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
