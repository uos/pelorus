extern crate bindgen;
extern crate pkg_config;

use std::env;

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=icp/src/icp.cpp");
    println!("cargo:rerun-if-changed=icp/include/icp.hpp");

    let eigen = match pkg_config::Config::new().probe("eigen3") {
        Ok(lib) => lib,
        Err(_) => panic!("Eigen3 not found"),
    };

    let tbb = match pkg_config::Config::new().probe("tbb") {
        Ok(lib) => lib,
        Err(_) => panic!("TBB not found"),
    };

    let bindings = bindgen::Builder::default()
        .header("wrapper.hpp")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    let mut build = cc::Build::new();

    build.include("icp/include");
    build.include("icp/small_gicp/include");

    for path in eigen.include_paths {
        build.include(path);
    }

    for lib in eigen.libs {
        build.flag(&format!("-l{}", lib));
    }

    for path in tbb.include_paths {
        build.include(path);
    }

    for lib in tbb.libs {
        build.flag(&format!("-l{}", lib));
    }

    build.warnings(false); // there are countless warnings in small_gicp

    build.cpp(true);
    build.file("icp/src/icp.cpp");
    build.flag_if_supported("-std=c++17");
    build.flag_if_supported("-O3");
    build.compile("lio_icp");

    println!("cargo:rustc-link-lib=static=lio_icp");
    println!(
        "cargo:rustc-link-search=native={}",
        std::env::var("OUT_DIR").unwrap()
    );
}
