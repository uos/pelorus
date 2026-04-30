fn main() {
    let mut build = cc::Build::new();
    build
        .std("c99")
        .include("include")
        .include("src")
        .define("pelorus_pico_init", "pelorus_pico_c_init")
        .define("pelorus_pico_add_imu", "pelorus_pico_c_add_imu")
        .define("pelorus_pico_add_cloud", "pelorus_pico_c_add_cloud")
        .file("src/pp_linalg.c")
        .file("src/pp_voxmap.c")
        .file("src/pp_iekf.c")
        .file("src/pp_imu_int.c")
        .file("src/pp_downsample.c")
        .file("src/pp_undistort.c")
        .file("src/pp_planes.c")
        .file("src/pelorus_pico.c");

    build.compile("pelorus_pico_c_core");
}
