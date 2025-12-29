use criterion::{criterion_group, criterion_main, Criterion};
use kurbo::BezPath;

use linesweeper::{binary_op, BinaryOp, FillRule};

fn painted_dreams(c: &mut Criterion) {
    let path_a = BezPath::from_svg("M0,340C161.737914,383.575765 107.564182,490.730587 273,476 C419,463 481.741198,514.692273 481.333333,768 C481.333333,768 -0,768 -0,768 C-0,768 0,340 0,340 Z").unwrap();
    let path_b = BezPath::from_svg(
		"M458.370270,572.165771C428.525848,486.720093 368.618805,467.485992 273,476 C107.564178,490.730591 161.737915,383.575775 0,340 C0,340 0,689 0,689 C56,700 106.513901,779.342590 188,694.666687 C306.607422,571.416260 372.033966,552.205139 458.370270,572.165771 Z",
	).unwrap();

    c.bench_function("painted dreams", |b| {
        b.iter(|| binary_op(&path_a, &path_b, FillRule::EvenOdd, BinaryOp::Union))
    });
}

criterion_group!(benches, painted_dreams);
criterion_main!(benches);
