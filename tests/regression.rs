use kurbo::BezPath;
use libtest_mimic::{Arguments, Failed, Trial};
use linesweeper::binary_op;
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};

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
   Snapshot
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
    binary_op(
        &p0,
        &p1,
        case.linesweeper_fill_rule(),
        case.linesweeper_binary_op()
    )
    .unwrap();
    Ok(())
}
