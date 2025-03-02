use std::any::Any;

use arbitrary::Unstructured;
use kurbo::{CubicBez, Point};

use crate::geom;

fn float_in_range(start: f64, end: f64, u: &mut Unstructured<'_>) -> Result<f64, arbitrary::Error> {
    let num: u32 = u.arbitrary()?;
    let t = num as f64 / u32::MAX as f64;
    Ok((1.0 - t) * start + t * end)
}

fn float(u: &mut Unstructured<'_>) -> Result<f64, arbitrary::Error> {
    float_in_range(-1e6, 1e6, u)
}

fn float_at_least(min: f64, u: &mut Unstructured<'_>) -> Result<f64, arbitrary::Error> {
    float_in_range(min, 1e6, u)
}

fn float_at_most(max: f64, u: &mut Unstructured<'_>) -> Result<f64, arbitrary::Error> {
    float_in_range(-1e6, max, u)
}

fn point(u: &mut Unstructured<'_>) -> Result<Point, arbitrary::Error> {
    Ok(Point::new(float(u)?, float(u)?))
}

fn point_at_least(min: f64, u: &mut Unstructured<'_>) -> Result<Point, arbitrary::Error> {
    Ok(Point::new(float(u)?, float_at_least(min, u)?))
}

fn point_at_most(max: f64, u: &mut Unstructured<'_>) -> Result<Point, arbitrary::Error> {
    Ok(Point::new(float(u)?, float_at_most(max, u)?))
}

pub fn monotonic_bezier(u: &mut Unstructured<'_>) -> Result<CubicBez, arbitrary::Error> {
    let p0 = point(u)?;
    let p3 = point(u)?;
    let (p0, p3) = if p0.y < p3.y { (p0, p3) } else { (p3, p0) };

    let p1 = point_at_least(p0.y, u)?;
    let p2 = point_at_most(p3.y, u)?;

    let ret = CubicBez::new(p0, p1, p2, p3);
    Ok(geom::monotonic_pieces(ret).into_iter().next().unwrap())
}

pub fn another_monotonic_bezier(
    u: &mut Unstructured<'_>,
    first: &CubicBez,
) -> Result<CubicBez, arbitrary::Error> {
    let same_start: bool = u.arbitrary()?;
    let same_start_tangent: bool = same_start && u.arbitrary()?;

    let p0 = if same_start { first.p0 } else { point(u)? };

    let p1 = if same_start_tangent {
        first.p1
    } else {
        point_at_least(p0.y, u)?
    };

    let p3 = point_at_least(p0.y.max(first.p0.y), u)?;
    let p2 = point_at_most(p3.y, u)?;

    let ret = CubicBez::new(p0, p1, p2, p3);
    Ok(geom::monotonic_pieces(ret).into_iter().next().unwrap())
}
