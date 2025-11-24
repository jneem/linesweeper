use kurbo::BezPath;
use libtest_mimic::{Arguments, Failed, Trial};
use linesweeper::binary_op;
use linesweeper::topology::Contours;
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};
use tiny_skia::{Pixmap, Transform};

#[derive(Serialize, Deserialize, Debug)]
enum FillRule {
   EvenOdd,
   NonZero,
}

#[derive(Serialize, Deserialize, Debug)]
enum BinaryOp {
    Union,
    Intersection,
    Difference,
    Xor,
}

#[derive(Serialize, Deserialize, Debug)]
enum Assertion {
   NoPanic,
   Snapshot {width: u16, height: u16}
}

#[derive(Serialize, Deserialize, Debug)]
struct RegressionCaseDeclaration {
    svg_path_1: String,
    svg_path_2: String,
    fill_rule: FillRule,
    op: BinaryOp,
    assert: Option<Assertion>
}

impl RegressionCaseDeclaration  {
    fn linesweeper_fill_rule(&self) -> linesweeper::FillRule {
        match self.fill_rule {
            FillRule::EvenOdd => linesweeper::FillRule::EvenOdd,
            FillRule::NonZero => linesweeper::FillRule::NonZero,
        }
    }

    fn linesweeper_binary_op(&self) -> linesweeper::BinaryOp {
        match self.op {
            BinaryOp::Union => linesweeper::BinaryOp::Union,
            BinaryOp::Intersection => linesweeper::BinaryOp::Intersection,
            BinaryOp::Difference => linesweeper::BinaryOp::Difference,
            BinaryOp::Xor => linesweeper::BinaryOp::Xor,
        }
    }
}

fn main() {
    let args = Arguments::from_args();
    let tests = regression_tests();

    libtest_mimic::run(&args, tests).exit();
}

fn regression_tests() -> Vec<Trial> {
    let ws = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let file_paths = glob::glob(&format!("{ws}/tests/regression/**/*.yml")).unwrap();

    file_paths
        .into_iter()
        .map(|p| {
            let p = p.unwrap();
            let name = input_path_base(&p).display().to_string();
            Trial::test(name, || generate_regression_test(p))
        })
        .collect()
}

fn input_path_base(input_path: &Path) -> &Path {
    let ws = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let base = format!("{ws}/tests/regression");
    input_path.strip_prefix(base).unwrap()
}

fn generate_regression_test(path: PathBuf) -> Result<(), Failed> {
    let input = std::fs::read_to_string(&path).unwrap();
    let case: RegressionCaseDeclaration = serde_yaml::from_str(&input).unwrap();
    let p0 = BezPath::from_svg(case.svg_path_1.as_str()).unwrap();
    let p1 = BezPath::from_svg(case.svg_path_2.as_str()).unwrap();
    let contours = binary_op(
        &p0,
        &p1,
        case.linesweeper_fill_rule(),
        case.linesweeper_binary_op()
    )
    .unwrap();

    if let Assertion::Snapshot {width: width, height: height} = case.assert.unwrap_or(Assertion::NoPanic) {
        assert_regression_snapshot(&contours, width, height)?;
    }

    Ok(())
}

fn assert_regression_snapshot(contours: &Contours, width: u16, height: u16) -> Result<(), Failed> {

    let mut ws: PathBuf = std::env::var_os("CARGO_MANIFEST_DIR").unwrap().into();

    println!("\n\nws: {}\n\n", ws.display());

    /*
    let snapshot_path = path.with_extension("svg");
    let mut snapshot_svg = String::new();
    for contour in contours {
        snapshot_svg.push_str(&contour.to_svg_path_data());
    }
    if snapshot_path.exists() {
        let expected_svg = std::fs::read_to_string(&snapshot_path).unwrap();
        if expected_svg != snapshot_svg {
            return Err(Failed::from(format!(
                "Snapshot mismatch for {}. To update the snapshot, copy the contents of {} to {}",
                input_path_base(&path).display(),
                path.display(),
                snapshot_path.display()
            )));
        }
    } else {
        std::fs::write(&snapshot_path, snapshot_svg).unwrap();
        return Err(Failed::from(format!(
            "Created new snapshot for {} at {}",
            input_path_base(&path).display(),
            snapshot_path.display()
        )));
    }
    */
    Ok(())
}

/*
fn to_pixels_from_svg_path(svg_path: &str, width: u16, height: u16) -> Result<Vec<u8>, Failed> {
    let opt = usvg::Options::default();
    let svg_str = r#"
    <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
      <path d="{svg_path}"/>
    </svg>
    "#;
    let usvg_tree = usvg::Tree::from_str(&svg_str, &opt).unwrap();
    let width = usvg_tree.size().width().floor() as u32;
    let height = usvg_tree.size().height().floor() as u32;
    let mut pixmap = Pixmap::new(width, height).unwrap();

    resvg::render(&usvg_tree, Transform::default(), &mut pixmap.as_mut());

    Ok(pixmap.pixels().iter().map(to_argb_u32).map(normalize_argb_u32_pixel_as_a8).collect())
}

fn to_argb_u32(p: &tiny_skia::PremultipliedColorU8) -> u32 {
    let d = p.demultiply();
    u32::from_be_bytes([d.alpha(), d.red(), d.green(), d.blue()])
}

pub fn normalize_argb_u32_pixel_as_a8(pixel: u32) -> u8 {
    let a = (pixel >> 24) & 0xffu32;
    let r = (pixel >> 16) & 0xffu32;
    let g = (pixel >> 8) & 0xffu32;
    let b = (pixel >> 0) & 0xffu32;

    if is_pixel_foreground(r as u8, g as u8, b as u8, Some(a as u8)) {
        0
    } else {
        255
    }
}

fn is_pixel_foreground(r: u8, g: u8, b: u8, a: Option<u8>) -> bool {
    // If r=g=b=255, then it's white -> background.
    let is_white: bool = r == 255 && g == 255 && b == 255;
    let is_transparent: bool = if let Some(alpha) = a {
        alpha == 0
    } else {
        false
    };

    let is_background =  is_white || is_transparent;

    return !is_background;
}
*/
