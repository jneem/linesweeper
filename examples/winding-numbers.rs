use std::path::PathBuf;

use clap::Parser;

use linesweeper::topology::Topology;

mod svg_util;

#[derive(Parser)]
struct Args {
    input: PathBuf,
    output: PathBuf,

    #[arg(long)]
    epsilon: Option<f64>,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let input = std::fs::read_to_string(&args.input)?;
    let tree = usvg::Tree::from_str(&input, &usvg::Options::default())?;
    let contours = svg_util::svg_to_contours(&tree);

    let eps = args.epsilon.unwrap_or(0.1).try_into().unwrap();
    let top = Topology::new([contours[0].clone()], contours[1..].iter().cloned(), &eps);

    let ys: Vec<_> = contours.iter().flatten().map(|p| p.y).collect();
    let xs: Vec<_> = contours.iter().flatten().map(|p| p.x).collect();
    let min_x = xs.iter().min().unwrap().into_inner();
    let max_x = xs.iter().max().unwrap().into_inner();
    let min_y = ys.iter().min().unwrap().into_inner();
    let max_y = ys.iter().max().unwrap().into_inner();
    let pad = 1.0 + eps.into_inner();
    let stroke_width = (max_y - min_y).max(max_x - max_y) / 512.0;
    let dot_radius = stroke_width * 1.5;
    let mut document = svg::Document::new().set(
        "viewBox",
        (
            min_x - pad,
            min_y - pad,
            max_x - min_x + 2.0 * pad,
            max_y - min_y + 2.0 * pad,
        ),
    );

    let text_size = "1px";

    for seg in top.segment_indices() {
        let p0 = &top.point(seg.first_half());
        let p1 = &top.point(seg.second_half());
        let (x0, y0) = (p0.x.into_inner(), p0.y.into_inner());
        let (x1, y1) = (p1.x.into_inner(), p1.y.into_inner());
        let c = svg::node::element::Circle::new()
            .set("r", dot_radius)
            .set("cy", y0)
            .set("cx", x0)
            .set("opacity", 0.5)
            .set("fill", "blue");
        document = document.add(c);

        let c = svg::node::element::Circle::new()
            .set("r", dot_radius)
            .set("cy", y1)
            .set("cx", x1)
            .set("opacity", 0.5)
            .set("fill", "blue");
        document = document.add(c);

        let data = svg::node::element::path::Data::new()
            .move_to((x0, y0))
            .line_to((x1, y1));
        let path = svg::node::element::Path::new()
            .set("stroke", "black")
            .set("stroke-width", stroke_width / 2.0)
            .set("stroke-opacity", "0.5")
            .set("d", data);
        document = document.add(path);

        let nx = y1 - y0;
        let ny = x0 - x1;
        let norm = ((nx * nx) + (ny * ny)).sqrt();

        let nx = nx / norm * 3.0 * dot_radius;
        let ny = ny / norm * 3.0 * dot_radius;

        let text = svg::node::element::Text::new(format!(
            "{:?}",
            top.winding(seg.first_half()).counter_clockwise
        ))
        .set("font-size", text_size)
        .set("text-anchor", "start")
        .set("x", (x0 + x1) / 2.0 + nx)
        .set("y", (y0 + y1) / 2.0 + ny);
        document = document.add(text);

        let text =
            svg::node::element::Text::new(format!("{:?}", top.winding(seg.first_half()).clockwise))
                .set("font-size", text_size)
                .set("text-anchor", "end")
                .set("x", (x0 + x1) / 2.0 - nx)
                .set("y", (y0 + y1) / 2.0 - ny);
        document = document.add(text);
    }

    svg::save(&args.output, &document)?;

    Ok(())
}
