pub const COMMON: &str = include_str!("common.metal");
pub const HASHMAP: &str = include_str!("hashmap.metal");
pub const LOCALMAP: &str = include_str!("localmap.metal");
pub const PLANES: &str = include_str!("planes.metal");
pub const RADIX_SORT: &str = include_str!("radix_sort.metal");
pub const DOWNSAMPLE: &str = include_str!("downsample.metal");
pub const MULTIRESOLUTION: &str = include_str!("multiresolution.metal");

pub fn full_source() -> String {
    // Order matters: multiresolution.metal uses iso_transform/symmetric_eigen_3x3
    // from common.metal and the Point/PointNormal structs from common.metal.
    format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}",
        COMMON, HASHMAP, LOCALMAP, PLANES, RADIX_SORT, DOWNSAMPLE, MULTIRESOLUTION
    )
}
