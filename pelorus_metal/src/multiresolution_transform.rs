//! CPU-side map insertion transform, independently testable without Metal.

pub(crate) fn transform_sample(q: [f64; 4], translation: [f64; 3], point: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = point.map(f64::from);
    let [qx, qy, qz, qw] = q;
    let tx = 2.0 * (qy * z - qz * y);
    let ty = 2.0 * (qz * x - qx * z);
    let tz = 2.0 * (qx * y - qy * x);
    [
        (x + qw * tx + (qy * tz - qz * ty) + translation[0]) as f32,
        (y + qw * ty + (qz * tx - qx * tz) + translation[1]) as f32,
        (z + qw * tz + (qx * ty - qy * tx) + translation[2]) as f32,
    ]
}
