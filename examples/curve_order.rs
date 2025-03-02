use arbitrary::Unstructured;
use kurbo::{Affine, CubicBez, ParamCurve, Point};
use linesweeper::curve::{intersect_cubics, solve_t_for_y, Order};

const DATA: &str = r#"
Trăm năm trong cõi người ta
Chữ tài, chữ mệnh khéo là ghét nhau
Trải qua một cuộc bể dâu
Những điều trông thấy mà đau đớn lòng
Lạ gì bỉ sắc tư phong
Trời xanh quen thói má hồng đánh ghen
Cảo thơm lần giở trước đèn
Phong tình cổ lục còn truyền sử xanh
Rằng: năm Gia Tĩnh triều Minh
Bốn phương phẳng lặng, hai kinh vững vàng.

Có nhà viên ngoại họ Vương
Gia tư nghĩ cũng thường thường bậc trung
Một trai con thứ rốt lòng
Vương Quan là chữ, nối dòng nho gia
Đầu lòng hai ả tố nga
Thúy Kiều là chị, em là Thúy Vân
Mai cốt cách, tuyết tình thần
Mỗi người một vẻ, mười phân vẹn mười.

Vân xem trang trọng khác vời
Khuôn trăng đầy đặn, nét ngài nở nang
Hoa cười, ngọc thốt, đoan trang
Mây thua nước tóc, tuyết nhường màu da.

Kiều càng sắc sảo mặn mà
So bề tài sắc vẫn là phần hơn
Làn thu thủy, nét xuân sơn
Hoa ghen thua thắm, liễu hờn kém xanh
Một hai nghiêng nước nghiêng thành
Sắc đành đòi một, tài đành hoạ hai.

Thông minh vốn sẵn tính trời
Pha nghề thi hoạ đủ mùi ca ngâm
Cung, thương làu bậc ngũ âm
Nghề riêng ăn đứt Hồ cầm một trương
Khúc nhà tay lựa nên chương
Một thiên “Bạc mệnh” lại càng não nhân
Phong lưu rất mực hồng quần (35)
Xuân xanh xấp xỉ, tới tuần cập kê
Êm đềm trướng rủ màn che
Tường đông ong bướm đi về mặc ai.
"#;

fn float_in_range(start: f64, end: f64, u: &mut Unstructured<'_>) -> f64 {
    let num: u32 = u.arbitrary().unwrap();
    let t = num as f64 / u32::MAX as f64;
    (1.0 - t) * start + t * end
}

fn monotonic_beziers(u: &mut Unstructured<'_>) -> (CubicBez, CubicBez) {
    let same_start: bool = u.arbitrary().unwrap();
    let same_start_tangent: bool = u.arbitrary().unwrap();
    let same_end: bool = u.arbitrary().unwrap();
    let same_end_tangent: bool = u.arbitrary().unwrap();
    let horizontal_start: bool = u.arbitrary().unwrap();
    let horizontal_end: bool = u.arbitrary().unwrap();

    let p0 = Point::new(0.5, 0.0);
    let q0 = if same_start {
        p0
    } else {
        Point::new(float_in_range(0.0, 1.0, u), float_in_range(0.0, 0.1, u))
    };

    let p1 = if horizontal_start {
        Point::new(float_in_range(0.0, 1.0, u), 0.0)
    } else {
        Point::new(float_in_range(0.0, 1.0, u), float_in_range(0.0, 1.0, u))
    };

    let q1 = if same_start && same_start_tangent {
        p1
    } else {
        Point::new(float_in_range(0.0, 1.0, u), float_in_range(q0.y, 1.0, u))
    };

    let p2 = if horizontal_end {
        Point::new(float_in_range(0.0, 1.0, u), 1.0)
    } else {
        Point::new(float_in_range(0.0, 1.0, u), float_in_range(p1.y, 1.0, u))
    };

    let q2 = if same_end && same_end_tangent && p2.y >= q1.y {
        p2
    } else {
        Point::new(float_in_range(0.0, 1.0, u), float_in_range(q1.y, 1.0, u))
    };

    let p3 = Point::new(float_in_range(0.0, 1.0, u), 1.0);
    let q3 = if same_end {
        p3
    } else {
        Point::new(float_in_range(0.0, 1.0, u), float_in_range(q2.y, 1.0, u))
    };

    (CubicBez::new(p0, p1, p2, p3), CubicBez::new(q0, q1, q2, q3))
}

fn cubic_svg(c: CubicBez, y0: f64, y1: f64) -> svg::node::element::path::Data {
    let t0 = solve_t_for_y(c, y0);
    let t1 = solve_t_for_y(c, y1);
    let c = c.subsegment(t0..t1);
    svg::node::element::path::Data::new()
        .move_to((c.p0.x, c.p0.y))
        .cubic_curve_to((c.p1.x, c.p1.y, c.p2.x, c.p2.y, c.p3.x, c.p3.y))
}

fn add_curves(
    x: f64,
    y: f64,
    stroke_width: f64,
    c0: CubicBez,
    c1: CubicBez,
    mut doc: svg::Document,
) -> svg::Document {
    let orders = intersect_cubics(c0, c1, 0.05, 0.01);
    let trans = Affine::translate((x, y));

    for (y0, y1, order) in orders.iter() {
        let data0 = cubic_svg(trans * c0, y0 + y, y1 + y);
        let color0 = match order {
            Order::Right => "mediumblue",
            Order::Ish => "red",
            Order::Left => "lightskyblue",
        };
        let path0 = svg::node::element::Path::new()
            .set("stroke", color0)
            .set("stroke-width", stroke_width)
            .set("style", "fill:none")
            .set("d", data0);

        let data1 = cubic_svg(trans * c1, y0 + y, y1 + y);
        let color1 = match order {
            Order::Right => "chartreuse",
            Order::Ish => "darkorange",
            Order::Left => "olivedrab",
        };
        let path1 = svg::node::element::Path::new()
            .set("stroke", color1)
            .set("stroke-width", stroke_width)
            .set("style", "fill:none")
            .set("d", data1);

        doc = doc.add(path0);
        doc = doc.add(path1);
    }

    doc
}

pub fn main() {
    // let c0 = CubicBez::new((0., 0.), (1.4, 0.2), (1.2, 0.8), (0., 1.));
    // let c1 = CubicBez::new((1., 0.), (0.7, 0.2), (0.8, 0.7), (0., 1.));

    let stroke_width = 0.005;
    let pad = 0.2;

    let mut document = svg::Document::new().set(
        "viewBox",
        (0.0 - pad, 0.0 - pad, 10.0 + 2.0 * pad, 10.0 + 2.0 * pad),
    );

    let mut data = Vec::new();
    for _ in 0..50 {
        data.extend_from_slice(DATA.as_bytes());
    }

    let mut u = Unstructured::new(&data);
    for i in 0..10 {
        for j in 0..10 {
            let (c0, c1) = monotonic_beziers(&mut u);
            document = add_curves(i as f64, j as f64, stroke_width, c0, c1, document);
        }
    }

    svg::save("out.svg", &document).unwrap();
}
