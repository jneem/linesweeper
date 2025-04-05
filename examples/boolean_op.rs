use std::{path::PathBuf, str::FromStr, sync::Arc};

use clap::{Args, Parser};
use kurbo::{BezPath, CubicBez, ParamCurve as _};
use ordered_float::NotNan;
use svg::Document;

use linesweeper::{
    curve::{solve_t_for_y, y_subsegment},
    generators,
    num::CheapOrderedFloat,
    topology::Topology,
    Point,
};

mod svg_util;

#[derive(Copy, Clone, Debug)]
enum Op {
    Union,
    Intersection,
    Xor,
    Difference,
    ReverseDifference,
}

impl FromStr for Op {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "union" => Ok(Op::Union),
            "intersection" => Ok(Op::Intersection),
            "xor" => Ok(Op::Xor),
            "difference" => Ok(Op::Difference),
            "reverse_difference" => Ok(Op::ReverseDifference),
            _ => Err(format!("unknown op {s}")),
        }
    }
}

#[derive(Copy, Clone, Debug, clap::ValueEnum)]
enum Example {
    Checkerboard,
    SlantedCheckerboard,
    Slanties,
}

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    output: PathBuf,

    #[command(flatten)]
    input: Input,

    #[arg(long)]
    non_zero: bool,

    #[arg(long)]
    epsilon: Option<f64>,
}

#[derive(Args, Debug)]
#[group(required = true, multiple = false)]
struct Input {
    input: Option<PathBuf>,

    #[arg(long)]
    example: Option<Example>,
}

fn points_to_bez(ps: Vec<Point>) -> BezPath {
    let mut ps = ps.into_iter();
    let mut ret = BezPath::new();
    if let Some(p) = ps.next() {
        ret.move_to((p.x, p.y));
    }
    for p in ps {
        ret.line_to((p.x, p.y));
    }
    ret.close_path();
    ret
}

fn contours_to_bezs(
    (cs0, cs1): (Vec<Vec<Point>>, Vec<Vec<Point>>),
) -> (Vec<BezPath>, Vec<BezPath>) {
    (
        cs0.into_iter().map(points_to_bez).collect(),
        cs1.into_iter().map(points_to_bez).collect(),
    )
}

fn get_contours(input: &Input) -> anyhow::Result<(Vec<BezPath>, Vec<BezPath>)> {
    match (&input.input, &input.example) {
        (Some(path), None) => {
            let input = std::fs::read_to_string(path)?;
            let tree = usvg::Tree::from_str(&input, &usvg::Options::default())?;
            let mut contours = svg_util::svg_to_bezpaths(&tree);
            let rest = contours.split_off(1);
            Ok((contours, rest))
        }
        (None, Some(example)) => match example {
            Example::Checkerboard => Ok(contours_to_bezs(generators::checkerboard(10))),
            Example::SlantedCheckerboard => {
                Ok(contours_to_bezs(generators::slanted_checkerboard(10)))
            }
            Example::Slanties => Ok(contours_to_bezs(generators::slanties(10))),
        },
        _ => unreachable!(),
    }
}

pub fn main() -> anyhow::Result<()> {
    let args = Cli::parse();
    let (shape_a, shape_b) = get_contours(&args.input)?;

    let eps = args.epsilon.unwrap_or(0.1);
    let top = Topology::from_paths(shape_a.clone(), shape_b.clone(), eps);
    let bbox = top.bounding_box();
    let min_x = bbox.min_x();
    let min_y = bbox.min_y();
    let max_x = bbox.max_x();
    let max_y = bbox.max_y();
    let pad = 1.0 + eps;
    let one_width = max_x - min_x + 2.0 * pad;
    let one_height = max_y - min_y + 2.0 * pad;
    let stroke_width = (max_y - min_y).max(max_x - max_y) / 512.0;
    let mut document = svg::Document::new().set(
        "viewBox",
        (min_x - pad, min_y - pad, one_width * 3.0, one_height * 2.0),
    );

    // Draw the original document.
    for c in shape_a.into_iter().chain(shape_b) {
        let p = c.segments().next().unwrap().start();
        let mut data = svg::node::element::path::Data::new();
        data = data.move_to((p.x, p.y));

        for s in c.segments() {
            let c = s.to_cubic();
            data = data.cubic_curve_to((c.p1.x, c.p1.y, c.p2.x, c.p2.y, c.p3.x, c.p3.y));
        }

        let path = svg::node::element::Path::new()
            .set("stroke", "black")
            .set("stroke-width", stroke_width)
            .set("stroke-linecap", "round")
            .set("stroke-linejoin", "round")
            .set("opacity", 0.2)
            .set("fill", "none")
            .set("d", data);
        document = document.add(path);
    }

    document = add_op(
        document,
        Op::Union,
        args.non_zero,
        &top,
        one_width,
        0.0,
        stroke_width,
    );
    document = add_op(
        document,
        Op::Intersection,
        args.non_zero,
        &top,
        one_width * 2.0,
        0.0,
        stroke_width,
    );
    document = add_op(
        document,
        Op::Xor,
        args.non_zero,
        &top,
        0.0,
        one_height,
        stroke_width,
    );
    document = add_op(
        document,
        Op::Difference,
        args.non_zero,
        &top,
        one_width,
        one_height,
        stroke_width,
    );
    document = add_op(
        document,
        Op::ReverseDifference,
        args.non_zero,
        &top,
        one_width * 2.0,
        one_height,
        stroke_width,
    );

    svg::save(&args.output, &document)?;

    Ok(())
}

fn add_op(
    mut doc: Document,
    op: Op,
    non_zero: bool,
    top: &Topology,
    x_off: f64,
    y_off: f64,
    stroke_width: f64,
) -> Document {
    let contours = top.contours(|w| {
        let inside = |winding| {
            if non_zero {
                winding != 0
            } else {
                winding % 2 != 0
            }
        };

        match op {
            Op::Union => inside(w.shape_a) || inside(w.shape_b),
            Op::Intersection => inside(w.shape_a) && inside(w.shape_b),
            Op::Xor => inside(w.shape_a) != inside(w.shape_b),
            Op::Difference => inside(w.shape_a) && !inside(w.shape_b),
            Op::ReverseDifference => inside(w.shape_b) && !inside(w.shape_a),
        }
    });

    let colors = [
        "#005F73", "#0A9396", "#94D2BD", "#E9D8A6", "#EE9B00", "#CA6702", "#BB3E03", "#AE2012",
        "#9B2226",
    ];

    let mut color_idx = 0;
    for group in contours.grouped() {
        let mut data = svg::node::element::path::Data::new();

        for contour_idx in group {
            let segs = contours[contour_idx].segs.iter().cloned();
            let mut first = true;

            for seg in segs {
                let p0 = top.point(seg.other_half());
                let p1 = top.point(seg);

                if first {
                    first = !first;
                    data = data.move_to((p0.x + x_off, p0.y + y_off));
                }

                let (y0, y1) = top.safe_intervals[seg];
                let y0 = y0.max(p0.y.min(p1.y));
                let y1 = y1.min(p0.y.max(p1.y));
                if y0 < y1 {
                    let seg_idx = top.orig_seg[seg];
                    let curve = top.segments[seg_idx].to_kurbo();
                    let mut curve = y_subsegment(curve, y0, y1);

                    if p0.y > p1.y {
                        curve = CubicBez::new(curve.p3, curve.p2, curve.p1, curve.p0);
                    }

                    if curve.p0 != p0.to_kurbo() {
                        data = data.line_to((curve.p0.x + x_off, curve.p0.y + y_off));
                    }
                    data = data.cubic_curve_to((
                        curve.p1.x + x_off,
                        curve.p1.y + y_off,
                        curve.p2.x + x_off,
                        curve.p2.y + y_off,
                        curve.p3.x + x_off,
                        curve.p3.y + y_off,
                    ));
                    if curve.p3 != p1.to_kurbo() {
                        data = data.line_to((p1.x + x_off, p1.y + y_off));
                    }
                } else {
                    data = data.line_to((p1.x + x_off, p1.y + y_off));
                }
            }
            data = data.close();
        }
        let path = svg::node::element::Path::new()
            .set("d", data)
            .set("stroke", "black")
            .set("stroke-width", stroke_width)
            .set("stroke-linecap", "round")
            .set("stroke-linejoin", "round")
            .set("fill", colors[color_idx]);
        doc = doc.add(path);
        color_idx = (color_idx + 1) % colors.len();
    }
    doc
}
