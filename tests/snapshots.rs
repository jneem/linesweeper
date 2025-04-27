use kompari::DirDiffConfig;
use kurbo::{Affine, BezPath, Point, Rect, Vec2};
use linesweeper::{
    sweep::{SweepLineBuffers, SweepLineRange, SweepLineRangeBuffers, Sweeper},
    Segment, Segments,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
use tiny_skia::Pixmap;

fn path_color(idx: usize) -> tiny_skia::Color {
    let palette = [
        tiny_skia::Color::from_rgba8(0x00, 0x5F, 0x73, 0xFF),
        tiny_skia::Color::from_rgba8(0x94, 0xD2, 0xBD, 0xFF),
        tiny_skia::Color::from_rgba8(0xE9, 0xD8, 0xA6, 0xFF),
        tiny_skia::Color::from_rgba8(0xEE, 0x9B, 0x00, 0xFF),
        tiny_skia::Color::from_rgba8(0xCA, 0x67, 0x02, 0xFF),
        tiny_skia::Color::from_rgba8(0xBB, 0x3E, 0x03, 0xFF),
        tiny_skia::Color::from_rgba8(0xAE, 0x20, 0x12, 0xFF),
    ];
    palette[idx % palette.len()]
}

fn sweep_line_color() -> tiny_skia::Color {
    tiny_skia::Color::from_rgba8(0x9B, 0x22, 0x26, 0xFF)
}

#[test]
fn sweep_snapshot_diffs() {
    let ws = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let paths = glob::glob(&format!("{ws}/tests/snapshots/inputs/sweep/**/*.svg")).unwrap();

    for p in paths {
        generate_sweep_snapshot(p.unwrap());
    }

    let stored_snapshots = PathBuf::from(format!("{ws}/tests/snapshots/snapshots"));
    let new_snapshots = PathBuf::from(format!("{ws}/target/snapshots/snapshots"));
    let diff_config = DirDiffConfig::new(stored_snapshots, new_snapshots);
    let diff = diff_config.create_diff().unwrap();

    assert!(diff.results().is_empty());
}

fn input_path_base(input_path: &Path) -> &Path {
    let ws = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let base = format!("{ws}/tests/snapshots/inputs");
    input_path.strip_prefix(base).unwrap()
}

fn output_path_for(input_path: &Path) -> PathBuf {
    let mut ws: PathBuf = std::env::var_os("CARGO_MANIFEST_DIR").unwrap().into();
    ws.push("target/snapshots/snapshots");
    ws.push(input_path);
    ws.set_extension("png");
    ws
}

fn skia_path(elts: impl IntoIterator<Item = kurbo::PathEl>) -> tiny_skia::Path {
    let mut pb = tiny_skia::PathBuilder::new();
    for elt in elts {
        match elt {
            kurbo::PathEl::MoveTo(p) => pb.move_to(p.x as f32, p.y as f32),
            kurbo::PathEl::LineTo(p) => pb.line_to(p.x as f32, p.y as f32),
            kurbo::PathEl::QuadTo(p0, p1) => {
                pb.quad_to(p0.x as f32, p0.y as f32, p1.x as f32, p1.y as f32)
            }
            kurbo::PathEl::CurveTo(p0, p1, p2) => pb.cubic_to(
                p0.x as f32,
                p0.y as f32,
                p1.x as f32,
                p1.y as f32,
                p2.x as f32,
                p2.y as f32,
            ),
            kurbo::PathEl::ClosePath => pb.close(),
        }
    }
    pb.finish().unwrap()
}

fn skia_segment(s: &Segment) -> tiny_skia::Path {
    let mut pb = tiny_skia::PathBuilder::new();
    pb.move_to(s.p0.x as f32, s.p0.y as f32);
    pb.cubic_to(
        s.p1.x as f32,
        s.p1.y as f32,
        s.p2.x as f32,
        s.p2.y as f32,
        s.p3.x as f32,
        s.p3.y as f32,
    );
    pb.finish().unwrap()
}

fn line(p: impl Into<Point>, q: impl Into<Point>) -> tiny_skia::Path {
    let mut pb = tiny_skia::PathBuilder::new();
    let p = p.into();
    pb.move_to(p.x as f32, p.y as f32);
    let q = q.into();
    pb.line_to(q.x as f32, q.y as f32);
    pb.finish().unwrap()
}

fn two_lines(p: impl Into<Point>, q: impl Into<Point>, r: impl Into<Point>) -> tiny_skia::Path {
    let mut pb = tiny_skia::PathBuilder::new();
    let p = p.into();
    pb.move_to(p.x as f32, p.y as f32);
    let q = q.into();
    pb.line_to(q.x as f32, q.y as f32);
    let r = r.into();
    pb.line_to(r.x as f32, r.y as f32);
    pb.finish().unwrap()
}

fn draw_orig_path(pixmap: &mut Pixmap, path: &BezPath, offset: kurbo::Point) {
    let p = skia_path(path);
    let mut paint = tiny_skia::Paint::default();
    paint.set_color_rgba8(0, 0, 0, 255);
    let stroke = tiny_skia::Stroke {
        width: 1.0,
        ..Default::default()
    };
    let transform = tiny_skia::Transform::from_translate(offset.x as f32, offset.y as f32);

    pixmap.stroke_path(&p, &paint, &stroke, transform, None);
}

fn adjust_x_positions(orig: &[f64], padding: f64, min_x: f64, max_x: f64) -> Vec<f64> {
    if orig.is_empty() {
        return Vec::new();
    }

    let mut padded = Vec::with_capacity(orig.len());
    let mut max_so_far = f64::NEG_INFINITY;
    for &x in orig {
        let x = x.max(max_so_far + padding);
        padded.push(x);
        max_so_far = x;
    }

    let orig_first = *orig.first().unwrap();
    let orig_last = *orig.last().unwrap();
    let padded_min = *padded.first().unwrap();
    let padded_max = *padded.last().unwrap();

    let mut mid_shift = (orig_first + orig_last - padded_min - padded_max) / 2.0;
    mid_shift = mid_shift.clamp(min_x - padded_min, max_x - padded_max);
    if padded_max - padded_min > max_x - min_x {
        mid_shift = (max_x + min_x - padded_max - padded_min) / 2.0;
    }
    for x in &mut padded {
        *x += mid_shift;
    }
    padded
}

fn color(c: tiny_skia::Color) -> tiny_skia::Paint<'static> {
    let mut p = tiny_skia::Paint::default();
    p.set_color(c);
    p
}

fn draw_sweep_line_range(
    pixmap: &mut Pixmap,
    segments: &Segments,
    range: SweepLineRange<'_, '_, '_>,
    bbox: kurbo::Rect,
    padding: f64,
) {
    let y = range.line().y();
    let mut paint = tiny_skia::Paint::default();
    paint.set_color(sweep_line_color());
    let stroke = tiny_skia::Stroke {
        width: 3.0,
        ..Default::default()
    };
    let thick_stroke = tiny_skia::Stroke {
        width: 6.0,
        ..Default::default()
    };
    let s_line = line(
        (bbox.min_x() - padding, bbox.min_y() + y),
        (bbox.max_x() + padding, bbox.min_y() + y),
    );

    pixmap.stroke_path(
        &s_line,
        &paint,
        &stroke,
        tiny_skia::Transform::identity(),
        None,
    );

    let old_segs: Vec<_> = range.old_segment_range().collect();
    let new_segs: Vec<_> = range.segment_range().collect();
    let really_new_segs = new_segs.iter().filter(|s| !old_segs.contains(s));
    let seg_color: HashMap<_, _> = old_segs
        .iter()
        .chain(really_new_segs.clone())
        .enumerate()
        .map(|(color_idx, seg_idx)| (seg_idx, color_idx))
        .collect();
    let origin = bbox.origin();

    for seg_idx in old_segs.iter().chain(really_new_segs) {
        let color_idx = seg_color[seg_idx];
        let seg = &segments[*seg_idx];
        let p = skia_segment(seg);
        let mut paint = tiny_skia::Paint::default();
        paint.set_color(path_color(color_idx));
        let transform = tiny_skia::Transform::from_translate(origin.x as f32, origin.y as f32);
        pixmap.stroke_path(&p, &paint, &stroke, transform, None);
    }

    let old_seg_positions: Vec<_> = old_segs.iter().map(|s| segments[*s].at_y(y)).collect();
    let padded_old_seg_positions: Vec<_> = adjust_x_positions(
        &old_seg_positions,
        padding / 2.0,
        bbox.min_x() - padding,
        bbox.max_x() + padding,
    );

    for ((&px, &x), seg_idx) in padded_old_seg_positions
        .iter()
        .zip(&old_seg_positions)
        .zip(&old_segs)
    {
        let p = two_lines(
            (bbox.min_x() + px, bbox.min_y()),
            (bbox.min_x() + px, bbox.min_y() + y - padding / 2.0),
            (bbox.min_x() + x, bbox.min_y() + y),
        );
        let color_idx = seg_color[seg_idx];
        let c = path_color(color_idx);

        pixmap.stroke_path(
            &p,
            &color(c),
            &thick_stroke,
            tiny_skia::Transform::identity(),
            None,
        );
    }
    let new_seg_positions: Vec<_> = new_segs.iter().map(|s| segments[*s].at_y(y)).collect();
    let padded_new_seg_positions: Vec<_> = adjust_x_positions(
        &new_seg_positions,
        padding / 2.0,
        bbox.min_x() - padding,
        bbox.max_x() + padding,
    );
    for ((&px, &x), seg_idx) in padded_new_seg_positions
        .iter()
        .zip(&new_seg_positions)
        .zip(&new_segs)
    {
        let p = two_lines(
            (bbox.min_x() + x, bbox.min_y() + y),
            (bbox.min_x() + px, bbox.min_y() + y + padding / 2.0),
            (bbox.min_x() + px, bbox.max_y()),
        );
        let color_idx = seg_color[seg_idx];
        let c = path_color(color_idx);

        pixmap.stroke_path(
            &p,
            &color(c),
            &thick_stroke,
            tiny_skia::Transform::identity(),
            None,
        );
    }

    // TODO: draw the rest
}

fn generate_sweep_snapshot(path: PathBuf) {
    let input = std::fs::read_to_string(&path).unwrap();
    let tree = usvg::Tree::from_str(&input, &usvg::Options::default()).unwrap();
    let bezs = linesweeper_util::svg_to_bezpaths(&tree);
    let bbox = linesweeper_util::bezier_bounding_box(bezs.iter());
    let bezs: Vec<_> = bezs
        .into_iter()
        .map(|p| Affine::translate(-bbox.origin().to_vec2()) * p)
        .collect();

    let mut segments = Segments::default();
    segments.add_bez_paths(bezs.clone());
    segments.check_invariants();

    let eps = 16.0;
    let mut range_bufs = SweepLineRangeBuffers::default();
    let mut line_bufs = SweepLineBuffers::default();

    // Run through once just to count the ranges.
    let mut sweep_state = Sweeper::new(&segments, eps);
    let mut num_ranges = 0;
    while let Some(mut line) = sweep_state.next_line(&mut line_bufs) {
        while line.next_range(&mut range_bufs, &segments).is_some() {
            num_ranges += 1;
        }
    }

    let pad = 32.0;
    let mut sweep_state = Sweeper::new(&segments, eps);
    let mut pixmap = Pixmap::new(
        (bbox.width() + 2.0 * pad).ceil() as u32,
        ((bbox.height() + pad) * num_ranges as f64 + pad).ceil() as u32,
    )
    .unwrap();

    let mut b = Rect::new(pad, pad, pad + bbox.width(), pad + bbox.height());
    while let Some(mut line) = sweep_state.next_line(&mut line_bufs) {
        while let Some(range) = line.next_range(&mut range_bufs, &segments) {
            for p in &bezs {
                draw_orig_path(&mut pixmap, p, b.origin());
            }
            draw_sweep_line_range(&mut pixmap, &segments, range, b, pad);

            b = b + Vec2::new(0.0, bbox.height() + pad);
        }
    }

    let out_path = output_path_for(input_path_base(&path));
    std::fs::create_dir_all(out_path.parent().unwrap()).unwrap();
    pixmap.save_png(out_path).unwrap();
}
