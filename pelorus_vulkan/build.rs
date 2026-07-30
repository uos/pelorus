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

    pkg_config::probe_library("SPIRV-Tools").ok();
    pkg_config::probe_library("glslang").ok();
    println!("cargo:rustc-link-lib=SPIRV-Tools");
    println!("cargo:rustc-link-lib=glslang");

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
