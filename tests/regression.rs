use kompari::image;
use kurbo::BezPath;
use libtest_mimic::{Arguments, Failed, Trial};
use linesweeper::binary_op;
use linesweeper::topology::Contours;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
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
    Snapshot { width: u16, height: u16 },
}

#[derive(Serialize, Deserialize, Debug)]
struct RegressionCaseDeclaration {
    svg_path_1: String,
    svg_path_2: String,
    fill_rule: FillRule,
    op: BinaryOp,
    assert: Option<Assertion>,
}

impl RegressionCaseDeclaration {
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
        case.linesweeper_binary_op(),
    )
    .unwrap();

    if let Assertion::Snapshot { width, height } = case.assert.unwrap_or(Assertion::NoPanic) {
        assert_regression_snapshot(&path, &contours, width, height)?;
    }

    Ok(())
}

fn assert_regression_snapshot(
    path: &PathBuf,
    contours: &Contours,
    width: u16,
    height: u16,
) -> Result<(), Failed> {
    let mut bezpath = BezPath::new();

    for contour in contours.contours() {
        bezpath.extend(contour.path.elements().iter().cloned());
    }

    let svg_path_data = bezpath.to_svg();

    let actual_pixmap = to_pixmap_from_svg_path(&svg_path_data, width, height)?;

    let case_name = path.file_prefix().unwrap().to_str().unwrap();
    let snapshot_rel_name = format!("regression/{case_name}");
    let snapshot_rel_path = Path::new(&snapshot_rel_name);

    let snapshot_path = linesweeper_util::saved_snapshot_path_for(snapshot_rel_path);

    if snapshot_path.exists() {
        let png_data = actual_pixmap.encode_png().unwrap();
        let actual_image = image::load_from_memory(&png_data).unwrap().into_rgba8();
        let actual_snapshot = kompari::Image::from_raw(
            actual_image.width(),
            actual_image.height(),
            actual_image.into_raw(),
        )
        .unwrap();
        let expected_snapshot = kompari::load_image(&snapshot_path)?;

        return match kompari::compare_images(&expected_snapshot, &actual_snapshot) {
            kompari::ImageDifference::None => Ok(()),
            _ => Err("image comparison failed".into()),
        };
    } else {
        std::fs::create_dir_all(snapshot_path.parent().unwrap()).unwrap();
        actual_pixmap.save_png(&snapshot_path).unwrap();
        Ok(())
    }
}

fn to_pixmap_from_svg_path(svg_path: &str, width: u16, height: u16) -> Result<Pixmap, Failed> {
    let opt = usvg::Options::default();
    let svg_str = format!(
        r#"
    <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
      <path d="{svg_path}"/>
    </svg>
    "#
    );
    let usvg_tree = usvg::Tree::from_str(&svg_str, &opt).unwrap();
    let width = usvg_tree.size().width().floor() as u32;
    let height = usvg_tree.size().height().floor() as u32;
    let mut pixmap = Pixmap::new(width, height).unwrap();

    resvg::render(&usvg_tree, Transform::default(), &mut pixmap.as_mut());

    Ok(pixmap)
}
