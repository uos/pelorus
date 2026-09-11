use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(vulkan_available)");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_VULKAN_ENABLE");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-env-changed=PATH");
    println!("cargo:rerun-if-env-changed=LD_LIBRARY_PATH");
    println!("cargo:rerun-if-env-changed=DYLD_LIBRARY_PATH");
    println!("cargo:rerun-if-changed=src/vk_radix_sort.h");
    println!("cargo:rerun-if-changed=src/vk_radix_sort_impl.cpp");

    let enabled = env::var_os("CARGO_FEATURE_VULKAN_ENABLE").is_some();
    if !enabled {
        return;
    }

    println!("cargo:rustc-cfg=vulkan_available");

    // Nothing here needs glslang or SPIRV-Tools: `vk_radix_sort.h` carries its
    // shaders as embedded SPIR-V and includes only the Vulkan headers, and the
    // shaderc crate links whatever it needs itself. Linking the system copies on
    // top of the ones inside a source-built `libshaderc_combined` puts two
    // implementations of each into one binary, and which objects the linker takes
    // from which archive then decides whether shaderc calls into a glslang with a
    // different class layout -- a segfault on the first shader compile, on some
    // machines only.

    let mut build = cc::Build::new();
    build.cpp(true);
    build.warnings(false);
    build.extra_warnings(false);
    build.file("src/vk_radix_sort_impl.cpp");
    build.include("src");
    if let Some(sdk) = env::var_os("VULKAN_SDK") {
        let sdk = PathBuf::from(sdk);
        build.include(sdk.join("include"));
    }
    build.compile("vk_radix_sort");

    if pkg_config::probe_library("vulkan").is_err() {
        println!("cargo:rustc-link-lib=vulkan");
    }
}
