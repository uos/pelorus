// Same neighborhood, pooled covariance, eigenvalue fit and residual gate as
// IVoxHashMap::plane_fit_from_distributions and the CPU registration path.
//
// The multiresolution map amplifies single rounding differences into
// centimetres of trajectory, so the shader reproduces the CPU's f32 arithmetic
// operation by operation: the same evaluation order, no fused multiply-add,
// correctly rounded division and square root, and the CPU's f64 eigensolver in
// double-float arithmetic.
struct Cell {
    mean_count: vec4<f32>,
    cov0: vec4<f32>,
    cov1: vec4<f32>,
    cov2: vec4<f32>,
}
struct Level {
    resolution: f32,
    inverse_resolution: f32,
    table_offset: u32,
    table_mask: u32,
}
struct Params {
    pose_q: vec4<f32>,
    pose_t: vec4<f32>,
    offset_q: vec4<f32>,
    offset_t: vec4<f32>,
    count: u32,
    level_count: u32,
    max_thickness_squared: f32,
    // Always zero. See rounded().
    zero: u32,
    levels: array<Level, 4>,
}
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> cells: array<Cell>;
// xyz stores full pyramid grid keys (or the single-level hash in x).
// w stores index+1. Zero denotes an empty slot. Tables are separate per level.
@group(0) @binding(2) var<storage, read> table: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> planes: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: Params;

const neighbors = array<vec3<i32>, 27>(
    vec3(0,0,0), vec3(-1,0,0), vec3(1,0,0), vec3(0,1,0), vec3(0,-1,0), vec3(0,0,-1), vec3(0,0,1),
    vec3(1,1,0), vec3(-1,1,0), vec3(1,-1,0), vec3(-1,-1,0),
    vec3(1,0,1), vec3(-1,0,1), vec3(1,0,-1), vec3(-1,0,-1),
    vec3(0,1,1), vec3(0,-1,1), vec3(0,1,-1), vec3(0,-1,-1),
    vec3(1,1,1), vec3(-1,1,1), vec3(1,-1,1), vec3(1,1,-1),
    vec3(-1,-1,1), vec3(-1,1,-1), vec3(1,-1,-1), vec3(-1,-1,-1)
);

// WGSL lets the compiler fuse a product into a following addition and
// reassociate sums and products, which changes the rounding. Passing an
// intermediate result through an opaque identity pins the order of operations
// and the separate roundings the CPU performs.
fn rounded(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) ^ params.zero);
}
fn rounded3(v: vec3<f32>) -> vec3<f32> {
    return bitcast<vec3<f32>>(bitcast<vec3<u32>>(v) ^ vec3(params.zero));
}
// Division and square root on GPUs are approximations. The exact residuals of
// fma correct them to the correctly rounded result the CPU computes.
fn divide(a: f32, b: f32) -> f32 {
    var q = rounded(a / b);
    for (var i = 0u; i < 2u; i++) {
        q = rounded(q + rounded(fma(-q, b, a)) / b);
    }
    return q;
}
fn divide3(a: vec3<f32>, b: f32) -> vec3<f32> {
    return vec3(divide(a.x, b), divide(a.y, b), divide(a.z, b));
}
fn square_root(x: f32) -> f32 {
    if (!(x > 0.0) || x > 3.402823e38) { return sqrt(x); }
    var s = rounded(sqrt(x));
    for (var i = 0u; i < 2u; i++) {
        let below = bitcast<f32>(bitcast<u32>(s) - 1u);
        let above = bitcast<f32>(bitcast<u32>(s) + 1u);
        let error = abs(fma(-s, s, x));
        if (abs(fma(-below, below, x)) < error) { s = below; }
        else if (abs(fma(-above, above, x)) < error) { s = above; }
    }
    return s;
}
// nalgebra's Vector3::dot: (x + y) + z.
fn dot3(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return rounded(rounded(a.x*b.x) + rounded(a.y*b.y)) + rounded(a.z*b.z);
}
fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3(
        rounded(a.y*b.z) - rounded(a.z*b.y),
        rounded(a.z*b.x) - rounded(a.x*b.z),
        rounded(a.x*b.y) - rounded(a.y*b.x));
}
// nalgebra's UnitQuaternion * Vector3: (t*w + cross) + v.
fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = rounded3(cross3(q.xyz, v) * 2.0);
    return rounded3(rounded3(t*q.w) + cross3(q.xyz, t)) + v;
}

fn hash_grid(g: vec3<i32>) -> u32 {
    let u = bitcast<vec3<u32>>(g);
    return u.x * 73856093u + u.y * 471943u + u.z * 83492791u;
}
fn hash_key(key: u32) -> u32 {
    var h = key;
    h = (h ^ (h >> 16u)) * 0x85ebca6bu;
    h = (h ^ (h >> 13u)) * 0xc2b2ae35u;
    return h ^ (h >> 16u);
}
fn lookup(grid: vec3<i32>, level: Level) -> u32 {
    var key = bitcast<vec3<u32>>(grid);
    let hash = hash_grid(grid);
    if (params.level_count == 1u) { key = vec3(hash, 0u, 0u); }
    var slot = hash_key(hash) & level.table_mask;
    for (var i = 0u; i <= level.table_mask; i++) {
        let entry = table[level.table_offset + slot];
        if (entry.w == 0u || all(entry.xyz == key)) { return entry.w; }
        slot = (slot + 1u) & level.table_mask;
    }
    return 0u;
}
fn saturating_neighbor(value: i32, offset: i32) -> i32 {
    if ((value == 2147483647 && offset > 0) || (value == (-2147483647 - 1) && offset < 0)) {
        return value;
    }
    return value + offset;
}
// VoxelDistribution::pool, element by element.
fn pool(a: Cell, b: Cell) -> Cell {
    let n1 = a.mean_count.w;
    let n2 = b.mean_count.w;
    if (n2 == 0.0) { return a; }
    if (n1 == 0.0) { return b; }
    let total = n1 + n2;
    let weight = divide(rounded(n1*n2), total);
    let delta = a.mean_count.xyz - b.mean_count.xyz;
    let ca = mat3x3(a.cov0.xyz, a.cov1.xyz, a.cov2.xyz);
    let cb = mat3x3(b.cov0.xyz, b.cov1.xyz, b.cov2.xyz);
    var cov: mat3x3<f32>;
    for (var c = 0u; c < 3u; c++) {
        let outer = rounded3(delta*delta[c]);
        cov[c] = divide3(rounded3(rounded3(n1*ca[c]) + rounded3(n2*cb[c])) + rounded3(outer*weight), total);
    }
    let mean = divide3(rounded3(rounded3(n1*a.mean_count.xyz) + rounded3(n2*b.mean_count.xyz)), total);
    return Cell(vec4(mean, total), vec4(cov[0], 0.0), vec4(cov[1], 0.0), vec4(cov[2], 0.0));
}
fn neighborhood(world: vec3<f32>, level: Level) -> Cell {
    var grid = vec3<i32>(floor(divide3(world, level.resolution)));
    if (params.level_count == 1u) {
        grid = vec3<i32>(floor(world * level.inverse_resolution));
    }
    var pooled = Cell(vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0));
    for (var n = 0u; n < 27u; n++) {
        var offset = neighbors[n];
        if (params.level_count > 1u) {
            // Match the CPU pyramid's x/y/z pooling order.
            offset = vec3<i32>(i32(n / 9u) - 1, i32((n / 3u) % 3u) - 1, i32(n % 3u) - 1);
        }
        let neighbor = vec3(
            saturating_neighbor(grid.x, offset.x),
            saturating_neighbor(grid.y, offset.y),
            saturating_neighbor(grid.z, offset.z));
        let found = lookup(neighbor, level);
        if (found == 0u) { continue; }
        pooled = pool(pooled, cells[found-1u]);
    }
    return pooled;
}
fn eigenvalues(a: mat3x3<f32>) -> vec3<f32> {
    let a00 = a[0][0]; let a01 = a[0][1]; let a02 = a[0][2];
    let a11 = a[1][1]; let a12 = a[1][2]; let a22 = a[2][2];
    let trace_a = a00 + a11 + a22;
    let c2 = (a00*a11 + a00*a22 + a11*a22) - (a01*a01 + a02*a02 + a12*a12);
    let det_a = a00*(a11*a22-a12*a12) - a01*(a01*a22-a12*a02) + a02*(a01*a12-a11*a02);
    let p = -trace_a; let q = c2; let r = -det_a;
    let ad = (3.0*q-p*p)/3.0;
    let bd = (2.0*p*p*p-9.0*p*q+27.0*r)/27.0;
    var angle = 0.0;
    if (ad < -1e-7) { angle = acos(clamp((-bd/2.0)/sqrt(-ad*ad*ad/27.0), -1.0, 1.0)); }
    let two_s = 2.0*sqrt(max(0.0, -ad/3.0));
    return two_s * cos((vec3(angle) + vec3(0.0, 6.28318530718, 12.56637061436))/3.0) - vec3(p/3.0);
}

// Double-float numbers: x is the value rounded to f32 and y the f32 remainder,
// about 44 significant bits, for the f64 arithmetic of the CPU's plane_eigen.
fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = rounded(a + b);
    let v = rounded(s - a);
    return vec2(s, rounded(a - rounded(s - v)) + rounded(b - v));
}
fn quick_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = rounded(a + b);
    return vec2(s, b - rounded(s - a));
}
fn df_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = two_sum(a.x, b.x);
    let t = two_sum(a.y, b.y);
    let u = quick_two_sum(s.x, rounded(s.y + t.x));
    return quick_two_sum(u.x, rounded(u.y + t.y));
}
fn df_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return df_add(a, -b);
}
fn df_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let p = rounded(a.x * b.x);
    let e = rounded(fma(a.x, b.x, -p)) + rounded(rounded(a.x * b.y) + rounded(a.y * b.x));
    return quick_two_sum(p, rounded(e));
}
fn df_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let q1 = rounded(a.x / b.x);
    let r = df_sub(a, df_mul(b, vec2(q1, 0.0)));
    let q2 = rounded(r.x / b.x);
    let r2 = df_sub(r, df_mul(b, vec2(q2, 0.0)));
    let q3 = r2.x / b.x;
    return df_add(quick_two_sum(q1, q2), vec2(q3, 0.0));
}
fn df_sqrt(a: vec2<f32>) -> vec2<f32> {
    if (!(a.x > 0.0)) { return vec2(0.0); }
    let x = rounded(sqrt(a.x));
    let r = df_sub(a, df_mul(vec2(x, 0.0), vec2(x, 0.0)));
    return quick_two_sum(x, rounded(r.x / (2.0 * x)));
}
fn df_abs(a: vec2<f32>) -> vec2<f32> {
    return select(a, -a, a.x < 0.0);
}
fn df_less(a: vec2<f32>, b: vec2<f32>) -> bool {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
}
fn df_value(a: vec2<f32>) -> f32 {
    return rounded(a.x + a.y);
}
// The pyramid needs the two tangent eigenvectors as well as the normal.
// Same pivot, rotation and sort rules as plane_eigen on the CPU, in double-float
// arithmetic. Element (r, c) is at index 3*r + c.
struct Eigen {
    values: vec3<f32>,
    vectors: mat3x3<f32>,
}
fn plane_eigen(covariance: mat3x3<f32>) -> Eigen {
    var a: array<vec2<f32>, 9>;
    var v: array<vec2<f32>, 9>;
    for (var r = 0u; r < 3u; r++) {
        for (var c = 0u; c < 3u; c++) {
            a[3u*r + c] = vec2(covariance[c][r], 0.0);
            v[3u*r + c] = vec2(select(0.0, 1.0, r == c), 0.0);
        }
    }
    let one = vec2(1.0, 0.0);
    for (var iteration = 0u; iteration < 24u; iteration++) {
        var p = 0u; var q = 1u;
        if (df_less(df_abs(a[3u*p + q]), df_abs(a[2]))) { p = 0u; q = 2u; }
        if (df_less(df_abs(a[3u*p + q]), df_abs(a[5]))) { p = 1u; q = 2u; }
        let apq = a[3u*p + q];
        let app = a[4u*p];
        let aqq = a[4u*q];
        let diagonal = df_add(df_add(df_abs(a[0]), df_abs(a[4])), df_abs(a[8]));
        if (df_less(df_abs(apq), df_mul(vec2(1e-14, 0.0), df_add(diagonal, vec2(1e-20, 0.0))))) { break; }
        let tau = df_div(df_sub(aqq, app), df_add(apq, apq));
        let root = df_sqrt(df_add(one, df_mul(tau, tau)));
        let t = df_div(select(one, -one, tau.x < 0.0), df_add(df_abs(tau), root));
        let c = df_div(one, df_sqrt(df_add(one, df_mul(t, t))));
        let s = df_mul(t, c);
        a[4u*p] = df_sub(app, df_mul(t, apq));
        a[4u*q] = df_add(aqq, df_mul(t, apq));
        a[3u*p + q] = vec2(0.0);
        a[3u*q + p] = vec2(0.0);
        for (var k = 0u; k < 3u; k++) {
            if (k != p && k != q) {
                let kp = a[3u*k + p]; let kq = a[3u*k + q];
                a[3u*k + p] = df_sub(df_mul(c, kp), df_mul(s, kq));
                a[3u*p + k] = a[3u*k + p];
                a[3u*k + q] = df_add(df_mul(s, kp), df_mul(c, kq));
                a[3u*q + k] = a[3u*k + q];
            }
            let vp = v[3u*k + p]; let vq = v[3u*k + q];
            v[3u*k + p] = df_sub(df_mul(c, vp), df_mul(s, vq));
            v[3u*k + q] = df_add(df_mul(s, vp), df_mul(c, vq));
        }
    }
    var order = array(0u, 1u, 2u);
    for (var i = 0u; i < 3u; i++) {
        for (var j = i + 1u; j < 3u; j++) {
            if (df_less(a[4u*order[j]], a[4u*order[i]])) {
                let temp = order[i]; order[i] = order[j]; order[j] = temp;
            }
        }
    }
    var vectors: mat3x3<f32>;
    for (var c = 0u; c < 3u; c++) {
        for (var r = 0u; r < 3u; r++) {
            vectors[c][r] = df_value(v[3u*r + order[c]]);
        }
    }
    return Eigen(vec3(df_value(a[4u*order[0]]), df_value(a[4u*order[1]]), df_value(a[4u*order[2]])),
        vectors);
}
fn finite(v: vec3<f32>) -> bool {
    return all(abs(v) <= vec3(3.402823e38));
}
fn finite1(x: f32) -> bool {
    return abs(x) <= 3.402823e38;
}
// MultiResolutionMap::fit and query.
fn multiresolution_plane(world: vec3<f32>) -> vec4<f32> {
    for (var level = 0u; level < params.level_count; level++) {
        let pooled = neighborhood(world, params.levels[level]);
        if (pooled.mean_count.w < 6.0) { continue; }
        let cov = mat3x3(pooled.cov0.xyz, pooled.cov1.xyz, pooled.cov2.xyz);
        if (!(finite(cov[0]) && finite(cov[1]) && finite(cov[2]))) { continue; }
        let eigen = plane_eigen(cov);
        let thickness = max(eigen.values.x, 0.0);
        let spread = eigen.values.y;
        if (!finite1(spread) || !finite1(eigen.values.z) || spread < 1e-4) { continue; }
        if (thickness > params.max_thickness_squared || thickness > 0.1*spread) { continue; }
        let mean = pooled.mean_count.xyz;
        let delta = world - mean;
        let resolution = params.levels[0].resolution;
        let extents = vec2(
            rounded(rounded(3.0*square_root(max(eigen.values.y, 0.0))) + resolution),
            rounded(rounded(3.0*square_root(max(eigen.values.z, 0.0))) + resolution));
        if (abs(dot3(eigen.vectors[1], delta)) > extents.x || abs(dot3(eigen.vectors[2], delta)) > extents.y) { continue; }
        let normal = eigen.vectors[0];
        return vec4(normal, -dot3(normal, mean));
    }
    return vec4(0.0);
}
fn single_resolution_plane(world: vec3<f32>) -> vec4<f32> {
    let pooled = neighborhood(world, params.levels[0]);
    let count = pooled.mean_count.w;
    let mean = pooled.mean_count.xyz;
    var cov = mat3x3(pooled.cov0.xyz, pooled.cov1.xyz, pooled.cov2.xyz);
    if (count < 3.0) { return vec4(0.0); }
    let ev = eigenvalues(cov);
    let lo = min(ev.x, min(ev.y, ev.z));
    let hi = max(ev.x, max(ev.y, ev.z));
    if (!(hi >= 1e-9) || !(lo/hi <= 0.5)) { return vec4(0.0); }
    cov[0][0] -= lo; cov[1][1] -= lo; cov[2][2] -= lo;
    let v1 = cross(cov[0], cov[1]);
    let v2 = cross(cov[1], cov[2]);
    let v3 = cross(cov[2], cov[0]);
    let m1 = dot(v1,v1); let m2 = dot(v2,v2); let m3 = dot(v3,v3);
    var normal = v3;
    if (m1 > m2 && m1 > m3) { normal = v1; }
    else if (m2 > m3) { normal = v2; }
    let mag = length(normal);
    if (!(mag > 1e-12)) { return vec4(0.0); }
    normal /= mag;
    let d = -dot(normal, mean);
    return vec4(normal, d);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= params.count) { return; }
    planes[index] = vec4(0.0);
    let local = points[index].xyz;
    let imu = rounded3(rotate(params.offset_q, local) + params.offset_t.xyz);
    let world = rounded3(rotate(params.pose_q, imu) + params.pose_t.xyz);
    if (!finite(world)) { return; }
    var plane: vec4<f32>;
    if (params.level_count > 1u) { plane = multiresolution_plane(world); }
    else { plane = single_resolution_plane(world); }
    // Apply the residual gate only after selecting the finest valid patch.
    // A rejected residual must not trigger a second search at a coarser level.
    // The registration's residual: ((a*x + b*y) + c*z) + d.
    let residual = rounded(rounded(rounded(rounded(plane.x*world.x) + rounded(plane.y*world.y)) + rounded(plane.z*world.z)) + plane.w);
    let length_squared = rounded(rounded(rounded(local.x*local.x) + rounded(local.y*local.y)) + rounded(local.z*local.z));
    let score = 1.0 - divide(0.9*abs(residual), square_root(square_root(max(length_squared, 1e-9))));
    if (!(score > 0.9)) { return; }
    planes[index] = plane;
}
