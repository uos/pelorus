fn main() {
    println!("cargo::rustc-check-cfg=cfg(metal_available)");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let metal_enable = std::env::var("CARGO_FEATURE_METAL_ENABLE").is_ok();

    if target_os == "macos" {
        if metal_enable {
            println!("cargo:rustc-cfg=metal_available");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=QuartzCore");
        } else {
            println!("cargo:warning=Metal backend inactive");
        }
    }
}
