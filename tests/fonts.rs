use std::path::PathBuf;

use kurbo::{BezPath, PathEl};
use libtest_mimic::{Arguments, Failed, Trial};
use linesweeper::topology::Topology;
use skrifa::{
    instance::Location,
    outline::{pen::PathElement, DrawSettings},
    prelude::Size,
    FontRef, MetadataProvider, OutlineGlyph,
};

fn main() {
    let args = Arguments::from_args();
    let tests = glyph_tests();

    libtest_mimic::run(&args, tests).exit();
}

fn glyph_tests() -> Vec<Trial> {
    let ws = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let paths = glob::glob(&format!("{ws}/tests/fonts/**/*.ttf")).unwrap();
    paths
        .into_iter()
        .flat_map(|p| {
            let p = p.unwrap();
            let name = p.strip_prefix(&ws).unwrap().display().to_string();
            font_glyph_tests(name, p)
        })
        .collect()
}

fn font_glyph_tests(name: String, path: PathBuf) -> Vec<Trial> {
    let data = std::fs::read(&path).unwrap();
    let font_ref = FontRef::new(&data).unwrap();
    let names = font_ref.glyph_names();
    font_ref
        .outline_glyphs()
        .iter()
        .zip(font_ref.outline_glyphs().iter().skip(1))
        .map(|((id1, outline1), (_id2, outline2))| {
            let glyph_name = names.get(id1).unwrap();
            let elts1 = outline_elts(outline1);
            let elts2 = outline_elts(outline2);
            Trial::test(format!("{name}-{glyph_name}"), || {
                test_outline(elts1, elts2)
            })
        })
        .collect()
}

fn outline_elts(outline: OutlineGlyph) -> Vec<PathElement> {
    let mut ret: Vec<PathElement> = Vec::new();
    outline
        .draw(
            DrawSettings::unhinted(Size::unscaled(), &Location::default()),
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

fn test_outline(elts1: Vec<PathElement>, elts2: Vec<PathElement>) -> Result<(), Failed> {
    if elts1.is_empty() || elts2.is_empty() {
        return Ok(());
    }

    let path1: BezPath = elts1.into_iter().map(skrifa_to_kurbo).collect();
    let path2: BezPath = elts2.into_iter().map(skrifa_to_kurbo).collect();
    let top = Topology::from_paths_binary(&path1, &path2, 1e-3);
    let _out_paths = top.compute_positions();
    Ok(())
}
