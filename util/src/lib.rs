use kurbo::{BezPath, ParamCurve as _, PathEl, Rect, Shape};
use skrifa::{
    instance::Location,
    outline::{pen::PathElement, DrawSettings},
    OutlineGlyph,
};
use std::path::{Path, PathBuf};

// TODO: this function also decomposes all the bezier paths so that
// there are no internal `MoveTo`s. We needed that at some point, but I think
// not anymore?
pub fn svg_to_bezpaths(tree: &usvg::Tree) -> Vec<BezPath> {
    let mut ret = Vec::new();

    fn pt(p: usvg::tiny_skia_path::Point) -> kurbo::Point {
        kurbo::Point::new(p.x as f64, p.y as f64)
    }

    fn add_group(group: &usvg::Group, ret: &mut Vec<BezPath>) {
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

                    let mut path = BezPath::new();
                    for el in kurbo_els {
                        match el {
                            kurbo::PathEl::MoveTo(p) => {
                                // Even if it wasn't closed in the svg, we close it.
                                if !path.is_empty() {
                                    ret.push(std::mem::take(&mut path));
                                }
                                path.move_to(p);
                            }
                            kurbo::PathEl::ClosePath => {
                                path.close_path();
                                let p = path.segments().next().map(|s| s.start());
                                if let Some(p) = p {
                                    ret.push(std::mem::take(&mut path));
                                    path.move_to(p);
                                }
                            }
                            el => {
                                path.push(el);
                            }
                        }
                    }

                    if !path.is_empty() {
                        ret.push(path);
                    }
                }
                _ => {}
            }
        }
    }

    add_group(tree.root(), &mut ret);
    ret
}

pub fn bezier_bounding_box<'a>(paths: impl Iterator<Item = &'a BezPath>) -> Rect {
    let mut rect = Rect::new(
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );

    for p in paths {
        rect = rect.union(p.bounding_box());
    }
    rect
}

fn outline_elts(outline: OutlineGlyph) -> Vec<PathElement> {
    let mut ret: Vec<PathElement> = Vec::new();
    outline
        .draw(
            DrawSettings::unhinted(skrifa::prelude::Size::unscaled(), &Location::default()),
            &mut ret,
        )
        .unwrap();
    ret
}

fn skrifa_to_kurbo(elt: PathElement) -> PathEl {
    let p = |x: f32, y: f32| -> kurbo::Point { kurbo::Point::new(x.into(), y.into()) };
    match elt {
        PathElement::MoveTo { x, y } => PathEl::MoveTo(p(x, y)),
        PathElement::LineTo { x, y } => PathEl::LineTo(p(x, y)),
        PathElement::QuadTo { cx0, cy0, x, y } => PathEl::QuadTo(p(cx0, cy0), p(x, y)),
        PathElement::CurveTo {
            cx0,
            cy0,
            cx1,
            cy1,
            x,
            y,
        } => PathEl::CurveTo(p(cx0, cy0), p(cx1, cy1), p(x, y)),
        PathElement::Close => PathEl::ClosePath,
    }
}

pub fn outline_to_bezpath(outline: OutlineGlyph) -> BezPath {
    kurbo::Affine::scale_non_uniform(1.0, -1.0)
        * outline_elts(outline)
            .into_iter()
            .map(skrifa_to_kurbo)
            .collect::<BezPath>()
}

pub fn saved_snapshot_path_for(input_path: &Path) -> PathBuf {
    let mut ws: PathBuf = std::env::var_os("CARGO_MANIFEST_DIR").unwrap().into();
    ws.push("tests/snapshots/snapshots");
    ws.push(input_path);
    ws.set_extension("png");
    ws
}
