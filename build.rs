extern crate bindgen;

use std::env;

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=icp.cpp");

    let bindings = bindgen::Builder::default()
        .header("wrapper.hpp")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("bindings.rs"))
            .expect("Couldn't write bindings!");

        let eigen_include_dir = "/usr/include/eigen3";
    
        cc::Build::new()
            .cuda(true)
            .cpp_link_stdlib("stdc++")
            .files(["icp.cpp"])
            .include(eigen_include_dir)
            .flag_if_supported("-O3")
            .compile("cupcl");
    
        println!("cargo:rustc-link-lib=cupcl");
    
        if cfg!(target_arch = "aarch64") {
            println!("cargo:rustc-link-search=native=/usr/local/cuda/targets/aarch64-linux/lib");
        } else {
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        }
        println!("cargo:rustc-link-lib=cudart");
    
        println!("cargo:rustc-link-lib=stdc++");
    
        println!("cargo:rustc-link-search=native=/usr/local/lib");
        println!("cargo:rustc-link-lib=cudacluster");
        println!("cargo:rustc-link-lib=cudafilter");

}