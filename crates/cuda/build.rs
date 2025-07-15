use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    if let Ok(cuda_path) = env::var("CUDA_HOME") {
        println!("cargo:rustc-cfg=cuda_available");

        let source_files = vec!["src/lib.cu"];
        let include_paths = vec!["src/include"];

        let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_75".to_string());
        let compute = format!("compute_{}", arch.strip_prefix("sm_").unwrap_or("75"));

        println!("cargo:rerun-if-changed=src/lib.cu");
        println!("cargo:rerun-if-changed=src/lib.h");

        // CUDA Toolkit directory, TODO check jetson specific folders
        let cuda_lib_path = PathBuf::from(&cuda_path).join("lib64");

        // --- Multi-Stage Compilation Process ---
        // This is required to solve the `undefined reference to __cudaRegisterLinkedBinary` error.
        // * Compile the .cu file into a relocatable object file (.o).
        // * Perform a "device link" on the object file to resolve device symbols.
        // * Archive both object files into a static library (.a).

        let gencode = format!("arch={},code={}", compute, arch);

        let mut nvcc_compile_cmd = Command::new("nvcc");
        let profile = env::var("PROFILE").unwrap();
        if profile == "release" {
            nvcc_compile_cmd.arg("-O3");
        } else {
            nvcc_compile_cmd.arg("-G").arg("-g");
        }

        for path in &include_paths {
            println!("cargo:rerun-if-changed={}", path);
        }

        // Compile each .cu file into its own object file
        let mut object_files = Vec::new();
        for source_file in &source_files {
            let path = Path::new(source_file);
            // Create a unique name for the object file based on the source file name
            let filename = path.file_stem().unwrap().to_str().unwrap();
            let obj_path = out_dir.join(format!("{}.o", filename));

            for path in &include_paths {
                nvcc_compile_cmd.arg(format!("-I{}", path));
            }
            let status_compile = nvcc_compile_cmd
                .arg("-c") // Compile, don't link
                .arg(source_file)
                .arg("-o")
                .arg(&obj_path)
                .arg("-rdc=true") // Relocatable Device Code
                .arg("-gencode")
                .arg(&gencode)
                .status()
                .expect("Failed to execute nvcc for compilation. Is it in your PATH?");

            if !status_compile.success() {
                panic!("nvcc compilation for {} failed", source_file);
            }
            object_files.push(obj_path);
        }

        // Device link all generated object files
        let dlink_obj_path = out_dir.join("kernel.dlink.o");
        let mut dlink_cmd = Command::new("nvcc");
        dlink_cmd
            .arg("-dlink")
            .arg("-o")
            .arg(&dlink_obj_path)
            .arg("-gencode")
            .arg(&gencode);
        for obj in &object_files {
            dlink_cmd.arg(obj);
        }
        let status_dlink = dlink_cmd
            .status()
            .expect("Failed to execute nvcc for device link.");

        if !status_dlink.success() {
            panic!("nvcc device link (Stage 2) failed");
        }

        // Archive all object files into a single static library
        let lib_path = out_dir.join("libkernel.a");
        let mut ar_cmd = Command::new("ar");
        ar_cmd.arg("crus").arg(&lib_path).arg(&dlink_obj_path);
        for obj in &object_files {
            ar_cmd.arg(obj);
        }
        let status_ar = ar_cmd
            .status()
            .expect("Failed to execute 'ar' to create static library.");

        if !status_ar.success() {
            panic!("'ar' archiving (Stage 3) failed");
        }

        // Instruct cargo to link the compiled static library and the CUDA runtime.
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
        println!("cargo:rustc-link-lib=static=kernel");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=stdc++");

        // Generate bindings using bindgen.
        bindgen::Builder::default()
            .header("src/lib.h")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate bindings")
            .write_to_file(out_dir.join("bindings.rs"))
            .expect("Couldn't write bindings!");

        println!("cargo:info=CUDA library compiled and bindings generated successfully.");
    } else {
        println!("cargo:warning=CUDA_HOME not set. Skipping CUDA build.");

        let bindings_path = out_dir.join("bindings.rs");
        std::fs::write(&bindings_path, "").expect("Could not write empty bindings file.");
    }
}
