use rayon::prelude::*;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use walkdir::WalkDir;

fn main() {
    if env::var("CARGO_FEATURE_CUDA_ENABLE").is_ok() {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

        if let Ok(cuda_path) = env::var("CUDA_HOME") {
            println!("cargo:rustc-cfg=cuda_available");

            let source_files = vec![
                "src/src/lib.cu",
                "src/src/common.cu",
                "src/src/cuda_context.cu",
                "src/src/cuda_debug.cu",
                "src/src/memory_cuda.cu",
                "src/src/exceptions.cu",
                "src/src/cuda_stream.cu",
                "src/src/hashmap.cu",
                "src/src/localmap.cu",
                "src/src/downsample.cu",
            ];
            let use_thrust_sort = env::var("CARGO_FEATURE_CUDA_SORT_THRUST").is_ok();
            let include_paths = vec!["src/include"];

            // orin: 87
            // rtx 40xx: 89
            // jetson nano: 5.3
            // jetson tx2: 6.2
            // jetson xavier: 7.2
            // jetson tegra k1/tk1: 3.2

            let arch_list = env::var("CUDA_ARCHS").unwrap_or_else(|_| "75,80,86,87,89".to_string());

            let arches: Vec<u32> = arch_list
                .split(',')
                .filter_map(|s| s.trim().parse::<u32>().ok())
                .collect();

            // CUDA Toolkit directory, TODO check jetson specific folders
            let cuda_path = PathBuf::from(&cuda_path);
            let cuda_lib_path = cuda_path.join("lib64");
            let cuda_bin_path = cuda_path.join("bin");
            let nvcc_bin = cuda_bin_path.join("nvcc");

            // let gencode = format!("arch={},code={}", compute, arch);
            let gencode_args = if let Ok(_) = env::var("PELORUS_CUDA_NATIVE") {
                vec!["-arch=native".to_string()]
            } else {
                arches
                    .iter()
                    .flat_map(|arch| {
                        let compute = format!("compute_{}", arch);
                        #[allow(unused_variables)]
                        let sm = format!("sm_{}", arch);
                        vec![
                            // format!("-gencode=arch={},code={}", compute, sm),
                            // emit PTX only for JIT compilation of newer GPUs
                            format!("-gencode=arch={},code={}", compute, compute),
                        ]
                    })
                    .collect()
            };

            for path in source_files.iter() {
                println!("cargo:rerun-if-changed={}", path);
            }

            let header_extensions = ["h", "hh", "hpp", "cuh"];
            for path in &include_paths {
                for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
                    let is_file = entry.file_type().is_file();
                    let relevant_file = if let Some(extension) =
                        entry.path().extension().and_then(|s| s.to_str())
                    {
                        header_extensions.contains(&extension)
                    } else {
                        false
                    };
                    if is_file && relevant_file {
                        println!("cargo:rerun-if-changed={}", entry.path().display());
                    }
                }
            }

            let object_files: Vec<PathBuf> = source_files
                .par_iter()
                .map(|source_file| {
                    let path = Path::new(source_file);
                    let filename = path.file_stem().unwrap().to_str().unwrap();
                    let obj_path = out_dir.join(format!("{}.o", filename));

                    let mut nvcc_compile_cmd = Command::new(nvcc_bin.to_string_lossy().to_string());
                    let profile = env::var("PROFILE").unwrap();

                    if profile == "release" || profile == "dbg-release" {
                        nvcc_compile_cmd.arg("-O3");
                    } else {
                        nvcc_compile_cmd.arg("-G").arg("-g");
                    }

                    for path in &include_paths {
                        nvcc_compile_cmd.arg(format!("-I{}", path));
                    }
                    if use_thrust_sort {
                        nvcc_compile_cmd
                            .arg("-DPELORUS_CUDA_SORT_THRUST=1")
                            .arg("--extended-lambda");
                    }

                    let cmd = nvcc_compile_cmd
                        .arg("-c")
                        .arg("-Xcompiler")
                        .arg("-fPIC")
                        .arg(source_file)
                        .arg("-o")
                        .arg(&obj_path)
                        .arg("-rdc=true"); // needed for device‑link later

                    for g in &gencode_args {
                        cmd.arg(g);
                    }

                    let output = cmd
                        .output()
                        .expect("Failed to spawn nvcc – is it on your PATH?");

                    if !output.status.success() {
                        return Err(format!(
                            "nvcc compilation for {} failed:\n{}",
                            source_file,
                            String::from_utf8_lossy(&output.stderr)
                        ));
                    }
                    Ok(obj_path)
                })
                .collect::<Result<Vec<_>, String>>()
                .unwrap_or_else(|err| panic!("{}", err));

            // Device link all generated object files
            let dlink_obj_path = out_dir.join("kernel.dlink.o");
            let mut dlink_cmd = Command::new(nvcc_bin.to_string_lossy().to_string());
            dlink_cmd
                .arg("-dlink")
                .arg("-o")
                .arg(&dlink_obj_path)
                .arg(format!("-L{}", cuda_lib_path.display()))
                .arg("-lcudadevrt"); // link against the device‑runtime library

            for g in &gencode_args {
                dlink_cmd.arg(g);
            }

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

            println!("cargo:rustc-link-search=native={}", out_dir.display());
            println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
            println!("cargo:rustc-link-lib=static=kernel");
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=cuda");
            println!("cargo:rustc-link-lib=cudadevrt");
            println!("cargo:rustc-link-lib=static=cudart_static");

            bindgen::Builder::default()
                .header("src/include/lib.h")
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
}
