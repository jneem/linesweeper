use std::{path::PathBuf, str::FromStr};

use clap::{Args, Parser};
use ordered_float::NotNan;
use svg::Document;

use linesweeper::{generators, num::CheapOrderedFloat, topology::Topology, Point};

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

fn get_contours(input: &Input) -> anyhow::Result<(Vec<Vec<Point>>, Vec<Vec<Point>>)> {
    match (&input.input, &input.example) {
        (Some(path), None) => {
            let input = std::fs::read_to_string(path)?;
            let tree = usvg::Tree::from_str(&input, &usvg::Options::default())?;
            let mut contours = svg_util::svg_to_contours(&tree);
            let rest = contours.split_off(1);
            Ok((contours, rest))
        }
        (None, Some(example)) => match example {
            Example::Checkerboard => Ok(generators::checkerboard(10)),
            Example::SlantedCheckerboard => Ok(generators::slanted_checkerboard(10)),
            Example::Slanties => Ok(generators::slanties(10)),
        },
        _ => unreachable!(),
    }
}

pub fn main() -> anyhow::Result<()> {
    let args = Cli::parse();
    let (shape_a, shape_b) = get_contours(&args.input)?;

    let eps = args.epsilon.unwrap_or(0.1);
    let top = Topology::new(shape_a.clone(), shape_b.clone(), eps);
    let mut contours = shape_a;
    contours.extend(shape_b);

    let ys: Vec<_> = contours.iter().flatten().map(|p| p.y).collect();
    let xs: Vec<_> = contours.iter().flatten().map(|p| p.x).collect();
    let min_x = xs
        .iter()
        .min_by_key(|x| CheapOrderedFloat::from(**x))
        .unwrap();
    let max_x = xs
        .iter()
        .max_by_key(|x| CheapOrderedFloat::from(**x))
        .unwrap();
    let min_y = ys
        .iter()
        .min_by_key(|x| CheapOrderedFloat::from(**x))
        .unwrap();
    let max_y = ys
        .iter()
        .max_by_key(|x| CheapOrderedFloat::from(**x))
        .unwrap();
    let pad = 1.0 + eps;
    let one_width = max_x - min_x + 2.0 * pad;
    let one_height = max_y - min_y + 2.0 * pad;
    let stroke_width = (max_y - min_y).max(max_x - max_y) / 512.0;
    let mut document = svg::Document::new().set(
        "viewBox",
        (min_x - pad, min_y - pad, one_width * 3.0, one_height * 2.0),
    );

    // Draw the original document.
    for c in contours {
        let p = c.first().unwrap();
        let mut data = svg::node::element::path::Data::new();
        data = data.move_to((p.x, p.y));

        for p in &c[1..] {
            data = data.line_to((p.x, p.y));
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
            let mut contour = contours[contour_idx].points.iter().cloned();
            let Some(p) = contour.next() else {
                continue;
            };

            let (x, y) = (p.x, p.y);
            data = data.move_to((x + x_off, y + y_off));
            for p in contour {
                let (x, y) = (p.x, p.y);
                data = data.line_to((x + x_off, y + y_off));
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
