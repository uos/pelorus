use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(vulkan_available)");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_VULKAN_ENABLE");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-env-changed=PATH");
    println!("cargo:rerun-if-env-changed=LD_LIBRARY_PATH");
    println!("cargo:rerun-if-env-changed=DYLD_LIBRARY_PATH");

    let enabled = env::var_os("CARGO_FEATURE_VULKAN_ENABLE").is_some();
    if !enabled {
        println!("cargo:warning=Vulkan backend inactive (enable with feature `vulkan_enable`)");
        return;
    }

    // Keep symbols available when feature is enabled; runtime still performs authoritative probing.
    println!("cargo:rustc-cfg=vulkan_available");

    // Link against system spirv-tools and glslang libraries for shaderc compatibility
    pkg_config::probe_library("SPIRV-Tools").ok();
    pkg_config::probe_library("glslang").ok();
    println!("cargo:rustc-link-lib=SPIRV-Tools");
    println!("cargo:rustc-link-lib=glslang");

    // Compile Vulkan Radix Sort wrapper
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

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| "unknown".to_string());
    match detect_loader(&target_os) {
        Some(path) => println!(
            "cargo:warning=Vulkan loader probe: found at {}",
            path.display()
        ),
        None => println!(
            "cargo:warning=Vulkan loader probe: not found for target `{}`; runtime will fallback if unavailable",
            target_os
        ),
    }
}

fn detect_loader(target_os: &str) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Some(sdk) = env::var_os("VULKAN_SDK") {
        let sdk = PathBuf::from(sdk);
        match target_os {
            "windows" => {
                candidates.push(sdk.join("Bin").join("vulkan-1.dll"));
                candidates.push(sdk.join("Lib").join("vulkan-1.lib"));
            }
            "macos" => {
                candidates.push(sdk.join("lib").join("libvulkan.dylib"));
                candidates.push(sdk.join("lib").join("libMoltenVK.dylib"));
            }
            _ => {
                candidates.push(sdk.join("lib").join("libvulkan.so"));
                candidates.push(sdk.join("lib64").join("libvulkan.so"));
            }
        }
    }

    match target_os {
        "linux" => {
            add_env_library_paths(&mut candidates, "LD_LIBRARY_PATH", "libvulkan.so");
            add_default_paths(
                &mut candidates,
                &[
                    "/usr/lib/libvulkan.so",
                    "/usr/lib64/libvulkan.so",
                    "/usr/lib/x86_64-linux-gnu/libvulkan.so",
                    "/lib/x86_64-linux-gnu/libvulkan.so",
                ],
            );
            if pkg_config_detects_vulkan() {
                return Some(PathBuf::from("pkg-config:vulkan"));
            }
        }
        "macos" => {
            add_env_library_paths(&mut candidates, "DYLD_LIBRARY_PATH", "libvulkan.dylib");
            add_default_paths(
                &mut candidates,
                &[
                    "/usr/local/lib/libvulkan.dylib",
                    "/opt/homebrew/lib/libvulkan.dylib",
                    "/usr/local/lib/libMoltenVK.dylib",
                    "/opt/homebrew/lib/libMoltenVK.dylib",
                ],
            );
        }
        "windows" => {
            add_path_executable_candidates(&mut candidates, "vulkan-1.dll");
        }
        _ => {}
    }

    candidates.into_iter().find(|p| p.exists())
}

fn add_default_paths(candidates: &mut Vec<PathBuf>, paths: &[&str]) {
    candidates.extend(paths.iter().map(PathBuf::from));
}

fn add_env_library_paths(candidates: &mut Vec<PathBuf>, env_var: &str, file_name: &str) {
    if let Some(paths) = env::var_os(env_var) {
        candidates.extend(env::split_paths(&paths).map(|p| p.join(file_name)));
    }
}

fn add_path_executable_candidates(candidates: &mut Vec<PathBuf>, file_name: &str) {
    if let Some(path_env) = env::var_os("PATH") {
        candidates.extend(env::split_paths(&path_env).map(|p| p.join(file_name)));
    }
}

fn pkg_config_detects_vulkan() -> bool {
    Command::new("pkg-config")
        .args(["--exists", "vulkan"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}
